import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import glob
import openai
import tqdm
import os
from transformers import HfArgumentParser, GPT2TokenizerFast
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field
import torch
import pdb
import sys
import time

openai.api_key=os.environ["openai_key"]

@dataclass
class GPT3Arguments(DataTrainingArguments):
    supernat: str = field(
        default="data/tasks", metadata={"help": "Path for the directory containing all the task json files."}
    )
    data_dir: str = field(
        default="data/splits/default", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
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
    random_seed: int = field(
        default=42, metadata={"help": "random seed for clustering & PCA"}
    )
    adaptive: bool = field(
        default=False, metadata={"help": "Adaptively change the number of demos. This takes effect only when demo_path is not None"}
    )
    encoder: str = field(
        default="all-MiniLM-L6-v2", metadata={"help": "encoder for clustering demonstrations"}
    )
    demo_num: int = field(
        default=0, metadata={"help": "number of demonstrations per task"}
    )
    num_clusters: int = field(
        default=0, metadata={"help": "number of clusters"}
    )
    task_name: str = field(
        default="task_name", metadata={"help": "task name"}
    )
    axis1: bool = field(
        default=True, metadata={"help": "Retrieval Type: True if Task-wise retrieval / False if Instance-wise retreival"}
    )
    axis2: bool = field(
        default=True, metadata={"help": "Selection Set Type: True if from whole-training set, / False if from only classification with ans. choices."}
    )
    axis3: bool = field(
        default=True, metadata={"help": "Retriever Type: True if SentenceBERT / False if SimCSE"}
    )
    axis4: bool = field(
        default=True, metadata={"help": "Similarity Method: True if Dot Product / False if Cosine Similarity"}
    )
    clf_tasks_file: str = field(
        default="clf_task_list.txt", metadata={"help": "Txt file containng all the clf type task names."}
    )
    
    

# Returns all the texts that are potential candidates of demos.
def get_all_demo_texts(supernat="", task_file_path="", to_exclude=[], task_wise=True):
    if not task_file_path:
        # Unless specified, use all tasks.
        task_list =  [x[:x.index(".json")] for x in os.listdir(supernat) if ".json" in x]
    else:
        task_list = [x.strip() for x in open(task_file_path).readlines() if x.strip()]
    to_exclude_list = [x.strip() for x in open(to_exclude).readlines()]
    filtered_task_file = [ os.path.join(supernat, f"{x}.json") for x in task_list if x not in to_exclude_list]

    elems = []

    # tasks for texts actually making smaples
    tasks = []
    indices = []

    for x in filtered_task_file:

        task_file = x.split("/")[-1]
        task_num = int(task_file[4:task_file.index('_')])

        task = json.load(open(x))
        df = task['Definition'][0]
        instances = task['Instances']

        if task_wise:
            elems.append(df)
            indices.append(task_num)
            inst = random.choice(instances)
            tasks.append("Definition : " + df + '\n\nInput: ' + inst['input'] + '\nOutput: ' + inst['output'][0])
        else:
            for inst in instances[:64]:
                indices.append(task_num)
                elems.append( df + '\n\n' + inst['input'])
                tasks.append("Definition : " + df + '\n\nInput: ' + inst['input'] + '\nOutput: ' + inst['output'][0])

    return elems, tasks, indices

# Returns embeddings from list of texts.
def get_embeddings(encoder, demo_set, embeddings_only=False):
    result = [torch.Tensor(x) for x in encoder.encode(demo_set)]
    
    if embeddings_only:
        return result
    else:
        return list(zip(result, demo_set))

def cosine_sim_matrix(A, B, eps=1e-8):
    result = np.dot(A, B.T) / (np.linalg.norm(A, axis=1, keepdims=True) * np.linalg.norm(B, axis=1, keepdims=True).T)
    return torch.from_numpy(result)

# Dynamically retrieves K nearest demos.
def get_dynamic_demo(tt_embedding, demo_dict, sim_is_dot, task_wise=True, task_set=[], task_indices=[], tokenizer=None, max_length=0):
    
    demo_texts = [x[1] for x in demo_dict]
    demo_embeddings = torch.cat([x[0].unsqueeze(0) for x in demo_dict], dim=0) # N, 768
    tt_embeddings = torch.cat([x.unsqueeze(0) for x in tt_embedding], dim=0) # M, 768

    if sim_is_dot:
        result = torch.matmul(tt_embeddings, demo_embeddings.T)
    else:
        result = cosine_sim_matrix(tt_embeddings, demo_embeddings)
    
    _, max_args = torch.topk(result, axis=-1, k=min(1000, len(demo_embeddings))) # k is 1000, an arbitrary big number, to guarantee different tasks later.
    
    NUM_DEMOS = 8 ## MANUALLY CHANGE THIS
     
    if task_wise: 
        # Retrieve demos.
        demos = []

        for n, idxs in enumerate(max_args):
            tmp = [] 
            curr_len = 0

            for idx in idxs:
                if len(tmp) >= NUM_DEMOS or curr_len >= max_length[n]:
                    break
                if len(tokenizer(task_set[idx])["input_ids"]) <= 256:
                    curr_len += len(tokenizer(task_set[idx])["input_ids"]) + 2 # considering "\n\n"
                    tmp.append(task_set[idx])            

            demos.append(tmp)

    else:
        # Making sure there is no duplicate.
        demos_indices = [[task_indices[idx] for idx in idxs] for idxs in max_args]
        demo_sim_texts = [[task_set[idx] for idx in idxs] for idxs in max_args]

        demos = []
        a = len(demos)

        for demo_num, demo_index in enumerate(demos_indices):
            curr_len = 0
            selected = []
            selected_idx = []

            for i, index in enumerate(demo_index):
                
                if len(selected) >= NUM_DEMOS or curr_len >= max_length[demo_num]:
                    demos.append([demo_sim_texts[demo_num][x] for x in selected_idx])
                    break

                if index not in selected and len(tokenizer(demo_sim_texts[demo_num][i])["input_ids"]) <= 256:
                    curr_len += len(tokenizer(demo_sim_texts[demo_num][i])["input_ids"]) + 2 # considering "\n\n"
                    selected.append(index)
                    selected_idx.append(i)
    
    return demos
     
    


if __name__ == "__main__":
    
    random.seed(123)

    parser = HfArgumentParser((GPT3Arguments,))
    args, = parser.parse_args_into_dataclasses()

    # print(args)
    
    # VARIATION 1. ENCODER TYPE
    if args.axis3:
        args.encoder = "all-MiniLM-L6-v2"
    else:
        args.encoder = "princeton-nlp/sup-simcse-roberta-large"

    encoder = SentenceTransformer(args.encoder)
    encoder.max_seq_length = 512
    
    # VARIATION 2. Retriever SET 
    # supernat: Should be directory 'task' containing all the json files.
    # Test_task_path: each should be path to a .txt file containng list of tasks. if wish to use all, laeve it as "".
    if args.axis2:
        demo_set, task_set, task_indices = get_all_demo_texts(supernat=args.supernat, task_file_path = "", to_exclude = os.path.join(args.data_dir, "test_tasks.txt"), task_wise=args.axis1)
    else:
        demo_set, task_set, task_indices = get_all_demo_texts(supernat=args.supernat, task_file_path = args.clf_tasks_file, to_exclude = os.path.join(args.data_dir, "test_tasks.txt"), task_wise=args.axis1)

    print("Completed Stage 1...")

    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.task_dir,  
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
        demo_num=args.demo_num
    )

    # Get Embeddings of all the demo_set.
    demo_dict = get_embeddings(encoder, demo_set)  # demo_dict: list of (txt, embedding)

    print("Getting embeddings...")

    # Get Embeddings of target. (i.e. the input text to gpt3.)
    if args.axis1:
        test_task = [x['Definition'][0] for x in raw_datasets['test']]
    else:
        test_task = [x['Definition'][0] + "\n\n" + x['Instance']['input'] for x in raw_datasets['test']]
    
    test_task_embedding = get_embeddings(encoder, test_task, embeddings_only=True)

    print("Completed Stage 2...")
    # add embeddings
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # 1920 = 2048 - 128

    MAX_INPUT_LENGTH = min(2048 - 128, args.max_source_length)
    max_length = [MAX_INPUT_LENGTH - len(tokenizer(
                    "Definition: " + x['Definition'][0] + "\n\n" + x['Instance']['input'].strip() + "\nOutput: " + x["Instance"]['output'][0]
                        )['input_ids']) for x in raw_datasets['test']]
    
    
    # Get dynamically demo by finding K Most Nearest Neighbor
    new_demo = get_dynamic_demo(test_task_embedding,  
                                demo_dict, 
                                sim_is_dot=args.axis4, 
                                task_wise=args.axis1,
                                task_set=task_set, 
                                task_indices=task_indices,
                                tokenizer=tokenizer,
                                max_length=max_length)    

    raw_datasets['test'] = raw_datasets['test'].add_column("demo", new_demo)

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
        has_demo=True,
        text_only=True,
        icil=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "gpt3_run_config.json"), "a") as fout:
        json.dump(args.__dict__, fout)

    existing_requests = {}

    # Make sure these files are empty.
    if os.path.exists(os.path.join(args.output_dir, "predicted_examples.jsonl")):
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl")) as fin:
            for line in fin:
                request_info = json.loads(line)
                existing_requests[request_info["gpt3_input"]] = request_info["gpt3_response"]

    t_dict = {}
    
    import pdb
    pdb.set_trace()

    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "a") as fout:
        for example in tqdm.tqdm(raw_datasets["test"]):
            encoded_example = data_collator([example])
            example["gpt3_input"] = encoded_example["inputs"][0].strip()
            example["gpt3_target"] = encoded_example["labels"][0].strip()

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
                            print("ERROR, found")
                            print(example["gpt3_input"])
                            pdb.set_trace()
                            assert False

                        response = openai.Completion.create(
                            engine=args.engine,
                            prompt=example["gpt3_input"],
                            temperature=args.gpt3_temprature,
                            max_tokens=args.max_target_length,
                            top_p=args.gpt3_top_p,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=["\n\n"],
                        )
                        received = True
                    except:
                        error = sys.exc_info()[0]
                        if error == openai.error.InvalidRequestError: 
                            # something is wrong: e.g. prompt too long
                            # print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                            assert False
                        elif error == AssertionError:
                            assert False
                        print("API error:", error)
                        time.sleep(2)

            example["gpt3_response"] = response
            # print(response)
            # Note: we cut the generated text at the first period, since the GPT3 language model sometimes generates more than one sentences.
            # Our results show that this won't affect the instruct-GPT3 model very much, but will significantly improve the original GPT3 LM.
            example["prediction"] = example["gpt3_response"]["choices"][0]["text"].strip().split(".")[0]
            # print(example["prediction"])
            fout.write(json.dumps(example) + "\n")

# if __name__ == "__main__":
#     main()