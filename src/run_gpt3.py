import glob
import json
import openai
import tqdm
import os
import random
from transformers import HfArgumentParser, GPT2TokenizerFast
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field
from itertools import chain
import time
import sys
import spacy
import numpy as np
import random
import re


openai.api_key=os.environ["openai_key"]

@dataclass
class GPT3Arguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="output/gpt3/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    gpt3_temprature: float = field(
        default=0, metadata={"help": "the temprature of GPT3."}
    )
    gpt3_top_p: float = field(
        default=1, metadata={"help": "the top_p parameter of GPT3."}
    )
    engine: str = field(
        default="text-davinci-001", metadata={"help": "the openai GPT3 engine to use."}
    )
    icil: bool = field(
        default=False, metadata={"help": "Either you will use ICIL or not."}
    )
    demo_path: str = field(
        default="", metadata={"help": "Path to file containing demo extracted by clustering."}
    )
    adaptive: bool = field(
        default=False, metadata={"help": "Adaptively change the number of demos. This takes effect only when demo_path is not None"}
    )
    cc_news_path: str = field(
        default="", metadata={"help": "path to cc_news sentences .txt file"}
    )
    irrelevant: bool = field(
        default=False, metadata={"help": "whether to apply OOD demo by inserting random text as Rethinking role... (Min, 2022)"}
    )


if __name__ == "__main__":
    random.seed(123)

    ### WARNING: YOU MAY WANT TO EMPTY THE BELOW FILES. ###
    parser = HfArgumentParser((GPT3Arguments,))
    args, = parser.parse_args_into_dataclasses()
    
    ### Add Positive Demontrastions if original dataset has less than the specified amount.
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    # Bring pos samples.
    testset_filepath = os.path.join(args.data_dir, "test_tasks.txt")
    task_path = args.task_dir
    few_shot_demos = {}

    if args.num_pos_examples != 0:
        id_dict = {}
        for x in raw_datasets['test']:
            if x['Task'] in id_dict:
                id_dict[x['Task']].append(x['id'])
            else:
                id_dict[x['Task']] = [x['id']]


        print("Preparing few-shot Demo...")
        for task in tqdm.auto.tqdm(id_dict):
            f = json.load(open(os.path.join(task_path, f"{task}.json")))
            pos_ex = f['Positive Examples']

            if len(pos_ex) >= args.num_pos_examples or len(f['Instances']) < args.num_pos_examples + args.max_num_instances_per_eval_task: # if less, just use positive examples.
                demo = ['Input: ' + x['input'] + '\nOutput: ' + x['output'] for x in pos_ex[:args.num_pos_examples]]
                few_shot_demos[task] = demo
            else:
                # Bring from instances.
                task_ids = id_dict[task]
                candidates = [f['Instances'][idx] for idx in range(len(f['Instances'])) if f['Instances'][idx]['id'] not in task_ids]
                samples_needed = args.num_pos_examples - len(pos_ex) 
                samples = random.sample(candidates, samples_needed)
                
                demo = ['Input: ' + x['input'] + '\nOutput: ' + x['output'] for x in pos_ex]
                demo += ['Input: ' + x['input'] + '\nOutput: ' + x['output'][0] for x in samples]
                few_shot_demos[task] = demo

    # Replace with Irrelevant Instruction.
    if args.irrelevant:
        # Load cc_news:
        cc_news = [x.strip(", \n\\\'\"").replace('\\"', '\"').replace("\\'", "\'") for x in open(args.cc_news_path, "r").readlines() if x.strip()]
        cc_news_len = [len(x.split()) for x in cc_news]
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])

        # Ask User to specify;
        demos = json.load(open(args.demo_path))["demo"]
        
        def irr_replace(demo):

            demo_sents = []
            not_replace = 0

            for sent_obj in nlp(demo).sents:
                text = str(sent_obj).strip()

                if not text.strip():
                    continue

                if "Input:" in text:
                    not_replace = len(demo_sents)

                text = text.split('\n')
                demo_sents.extend(text)

            # If False, you can manually corrupt instruction (Definition Part.)
            # If True, just input will be automatically corrupted.
            only_input = True

            if only_input:   
                not_replace = [x for x in range(not_replace)]

            else:    
                print(f"Select sentence index that contains answer choice - to not replace, from 0 to {not_replace - 1}. If multiple, put indexes separated by commas.")
                for idx, sent in enumerate(demo_sents[:not_replace]):
                    print(f"[{idx}]\t{sent}")

                not_replace = input("Select Sentence Index: ")
                not_replace = [int(x) for x in not_replace.split(",")]

            final_demo = []

            for idx, sent in enumerate(demo_sents):
                if idx in not_replace:
                    if "Input:" in sent:
                        sent = "\n\n" + sent
                    elif "Output:" in sent:
                        sent = "\n" + sent
                    final_demo.append(sent) # Just append original sentence
                elif "Output:" in sent:    
                    demo_sents[idx] = "\n" + demo_sents[idx].strip()
                    final_demo.extend(demo_sents[idx:])
                    break
                else:
                    length = len(sent.split()) # Randomly select one that has the same length.
                    same_length_text = [text for text, l in zip(cc_news, cc_news_len) if l == length]  

                    rand_sent = random.choice(same_length_text).strip("\'\"")
                    rand_sent = re.sub(r'\\u[0-9a-fA-F]{4}', '', rand_sent)
                    # rand_sent = rand_sent.strip().replace("u201c", "") # Added Later.
                    
                    if "Definition:" in sent:
                        rand_sent = "Definition: " + rand_sent
                    elif "Input:" in sent:
                        rand_sent = "\n\nInput: " + rand_sent

                    final_demo.append(rand_sent) 
            
            final_str =  ' '.join(final_demo) 

            if '\n\nInput: ' not in final_str:
                import pdb
                pdb.set_trace()

            return final_str
        
        # Replace Demo Sample.
        new_demos = {"demo": [irr_replace(demo) for demo in demos]}
        new_demo_path = args.demo_path[:args.demo_path.index(".json")] + "_irrelevant.json"

        json.dump(new_demos, open(new_demo_path, "w"), indent=4)
        args.demo_path = new_demo_path
    

    ## Start of Main Code. ##
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    data_collator = DataCollatorForNI(
        tokenizer,
        model=None,
        padding="max_length" if args.pad_to_max_length else "longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        text_only=True,
        icil=args.icil, # added
        demo_path=args.demo_path,
        adaptive=args.adaptive,
        demo_dict=few_shot_demos
    )

    os.makedirs(args.output_dir, exist_ok=True)


    with open(os.path.join(args.output_dir, "gpt3_run_config.json"), "a") as fout:
        json.dump(args.__dict__, fout)

    existing_requests = {}
    existing_tasks = []

    if os.path.exists(os.path.join(args.output_dir, "predicted_examples.jsonl")):
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl")) as fin:
            for line in fin:
                request_info = json.loads(line)
                existing_requests[request_info["gpt3_input"]] = request_info["gpt3_response"]
                
                # IF already in ... exclude that.
                existing_tasks.append(request_info['id'])

    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "a") as fout:
        for example in tqdm.tqdm(raw_datasets["test"]):
            
            if example['id'] in existing_tasks:
                continue

            encoded_example = data_collator([example])
            example["gpt3_input"] = encoded_example["inputs"][0].strip()
            example["gpt3_target"] = encoded_example["labels"][0].strip()
            length_issue = False

            if example["gpt3_input"] in existing_requests:
                response = existing_requests[example["gpt3_input"]]
            else:
                # call GPT-3 API until result is provided and then return it
                response = None
                received = False   

                while not received:
                    try:
                        # MAKE SURE INPUT IS NOT CUT:
                        if example["gpt3_input"][-7:].strip() != "Output:":
                            length_issue = True
                            break

                        response = openai.Completion.create(
                            engine=args.engine,
                            prompt=example["gpt3_input"],
                            temperature=args.gpt3_temprature,
                            max_tokens=args.max_target_length,
                            top_p=args.gpt3_top_p,
                            frequency_penalty=0,
                            presence_penalty=0,
                            # we set \n\n as the stop sequence
                            stop=["\n\n"],
                        )
                        received = True
                    except:
                        error = sys.exc_info()[0]
                        if error == openai.error.InvalidRequestError: 
                            # something is wrong: e.g. prompt too long
                            print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                            assert False
                        print("API error:", error)
                        time.sleep(2)

            if length_issue:
                print("[Length issue] Skipping...")
                length_issue = False
                continue

            example["gpt3_response"] = response
            # print(response)
            # Note: we cut the generated text at the first period, since the GPT3 language model sometimes generates more than one sentences.
            # Our results show that this won't affect the instruct-GPT3 model very much, but will significantly improve the original GPT3 LM.
            example["prediction"] = example["gpt3_response"]["choices"][0]["text"].strip().split(".")[0]
            print(example["prediction"])
            fout.write(json.dumps(example) + "\n")