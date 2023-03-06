import logging
import random
import json
import string
from transformers.data.data_collator import *
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False
    icil: bool = False
    demo_path: str = None
    adaptive: bool = True # Adaptively change the demo_num.
    demo_dict: Dict[str, List] = None
    decoder_only: bool = False
    has_demo: bool = False

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        self.max_source_length =  min(2048 - self.max_target_length, self.max_source_length)
         
        sources = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 

            task_input = ""

            if not self.icil and self.num_pos_examples == 0:
                task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            # try to add positive examples.
            if not self.icil and self.num_pos_examples != 0:
                
                # Restrain sample number inevitably, if longer than length limit.
                if self.adaptive:
                    demos = self.demo_dict[instance['Task']]

                    def max_index(numbers, N):
                        total = 0
                        for i, num in enumerate(numbers):
                            total += num
                            if total >= N:
                                return i
                        return len(numbers)

                    lens = [len(x) + 2 for x in self.tokenizer(demos)['input_ids']] # 2, for "\n\n"

                    max_length =  self.max_source_length # 2048, which is maximum davinci input length.
                    max_length -= len(self.tokenizer(definition)['input_ids']) + len(self.tokenizer(task_input)['input_ids'])  # target input length
                    max_length -= 2 # last "\n\n"

                    idx = max_index(lens, max_length) 
                    source = "\n\n".join(demos[:idx]) + "\n\n"
 
                    source = definition + task_name + source + task_input

                else:
                    source = "\n\n".join(demos) + "\n\n"
                    source = definition + task_name + source + task_input

            elif not self.icil:
                source = task_name + definition + task_input  
            else:
                task_input = f"Input: {instance['Instance']['input'].strip()}"
                if not task_input[-1] in string.punctuation:
                    task_input += "."
                task_input += "\n"
                task_input += "Output: "

                source = ""

                if self.demo_path or self.has_demo:
                    if self.demo_path:
                        demo_file = json.load(open(self.demo_path))["demo"]
                    elif self.has_demo:
                        demo_file = instance['demo']
                        
                    if self.adaptive:

                        def max_index(numbers, N):
                            total = 0
                            for i, num in enumerate(numbers):
                                total += num
                                if total >= N:
                                    return i
                            return len(numbers)

                        lens = [len(x) + 2 for x in self.tokenizer(demo_file)['input_ids']] # 2, for "\n\n"
                        
                        max_length =  self.max_source_length # 2048, which is maximum davinci input length.
                        max_length -= len(self.tokenizer(definition)['input_ids']) + len(self.tokenizer(task_input)['input_ids'])  # target input length
                        max_length -= 2 # last "\n\n"

                        idx = max_index(lens, max_length) 
                        source = "\n\n".join(demo_file[:idx]) + "\n\n"

                    else:
                        source = "\n\n".join(demo_file) + "\n\n" 

                source = source + task_name + definition + task_input  

            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"inputs": sources}
        elif self.decoder_only:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding='max_length',
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            
        return model_inputs