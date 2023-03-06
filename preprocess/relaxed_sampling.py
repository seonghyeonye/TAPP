import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from transformers import HfArgumentParser, GPT2TokenizerFast
# from run_s2s import DataTrainingArguments
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class CustomArguments():
    supernat: str = field(
        default="data/tasks", metadata={"help": "Path for the directory containing all the task json files."}
    )
    random_seed: int = field(
        default=0, metadata={"help": "random seed for clustering & PCA"}
    )
    data_dir: str = field(
        default="data/splits/default", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="demos/random", metadata={"help": "output path for resulting image"}
    )


if __name__=="__main__":

    parser = HfArgumentParser((CustomArguments,))
    args, = parser.parse_args_into_dataclasses()

    random.seed(args.random_seed)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    files=[]

    with open(os.path.join(args.data_dir, "train_tasks.txt")) as f:
        while True :
            line = f.readline()
            if not line:
                break
            files.append(line.strip())

    accepted = []
    total_output_list = []

    for file in tqdm(files):
        flag = 0
        target_json = json.load(open(os.path.join(args.supernat, file + '.json')))

        # Assume at least 50% of them have the answer inside.
        df = target_json['Definition'][0]
        demo = target_json['Positive Examples']
        instances = target_json['Instances']
        category = target_json['Categories']
        reasoning = target_json['Reasoning']

        rand_int = random.randint(0, len(demo)-1)


        # heuristic 1 -> answer choice should be included in instruction
        target_num = [1 if instance['output'][0].lower().strip() in df.lower() else 0 for instance in instances[:64]] 
        output_list = []
        accepted_elem = {}
        filtered = []
        for instance in instances[:64]:
            output = instance['output'][0]
            if output not in output_list:
                output_list.append(output)

        if sum(target_num) == 64:
            first_sampling = random.sample(instances, 64)
            for instance in first_sampling:
                total_input = "Definition: " + df + '\n\nInput: ' + instance['input'] + '\n' + 'Output: ' + instance['output'][0] + '\n\n'
                input_instance = instance['input'] 
                # heuristic 3 -> demonstration length
                if len(tokenizer(total_input)['input_ids'])<=256:
                    filtered.append(total_input)

            if len(filtered) >= 1: 
                # heuristic 2 -> answer choice overlap
                sampled_instance = random.sample(filtered, 1)
                for output_elem in output_list: 
                    for total_output_list_elem in total_output_list:
                        if output_elem.lower() in [total_output_elem.lower() for total_output_elem in total_output_list_elem]:
                            flag = 1
                            break
                if flag ==0:
                    total_output_list.append(output_list)
                    accepted.append(sampled_instance)

    final_demos = random.sample(accepted, 8)
        
    # Write Json File.
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump({"demo": final_demos}, open(os.path.join(args.output_dir, f"seed_{args.random_seed}.json"), "w"), indent=4)

