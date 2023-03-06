import sys
sys.path.append("src")

import random
import os
from transformers import HfArgumentParser, GPT2TokenizerFast
from datasets import load_dataset
from dataclasses import dataclass, field
import json 

@dataclass
class CustomArguments():
    supernat: str = field(
        default="data/tasks", metadata={"help": "Path for the directory containing all the task json files."}
    )
    data_dir: str = field(
        default="data/splits/default", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="demos", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
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
    


if __name__ == "__main__":
    
    random.seed(123)

    parser = HfArgumentParser((CustomArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    demo_set, task_set, task_indices = get_all_demo_texts(supernat=args.supernat, task_file_path = args.clf_tasks_file, to_exclude = os.path.join(args.data_dir, "test_tasks.txt"), task_wise=False)
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.supernat,  
        max_num_instances_per_task=0,
        max_num_instances_per_eval_task=100,
        demo_num=0,
    )

    idxs = []
    elems = []

    for i, elem in enumerate(task_set):
        if ("output: yes" in elem.lower() and elem.lower()[-1] == 's')  or ("output: no" in elem.lower() and elem.lower()[-1] == 'o'):
            if task_indices[i] not in idxs:
                idxs.append(task_indices[i])
                elems.append(elem)
    
    d8 = {"demo": elems[:8]}

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(d8, open(f"{args.output_dir}/eight_overlap_mixed.json", "w"), indent=4)