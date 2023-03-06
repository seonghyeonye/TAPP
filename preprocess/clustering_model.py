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
import jsonlines
from typing import Tuple, List
from itertools import chain
import numpy as np


@dataclass
class CustomArguments():
    supernat: str = field(
        default="data/tasks", metadata={"help": "Path for the directory containing all the task json files."}
    )
    random_seed: int = field(
        default=42, metadata={"help": "random seed for clustering & PCA"}
    )
    encoder: str = field(
        default="all-MiniLM-L6-v2", metadata={"help": "encoder for clustering demonstrations"}
    )
    output_dir: str = field(
        default="preprocess/output", metadata={"help": "output path for resulting image"}
    ) 
    num_clusters: int = field(
        default=8, metadata={"help": "number of clusters"}
    )
    mode: str = field(
        default="random", metadata={"help": "mode of choosing the demo. either 'random', 'diverse', 'same_cluster'"}
    )
    


if __name__=="__main__":

    parser = HfArgumentParser((CustomArguments,))
    args, = parser.parse_args_into_dataclasses()

    random.seed(args.random_seed)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encoder = SentenceTransformer(args.encoder)
    encoder.max_seq_length = 512

    files=[]

    demo_count = {}
    label_list = []
    total_list = []
    total_dict = {}

    with open('data/splits/default/train_tasks.txt') as f:
        while True :
            line = f.readline()
            if not line:
                break
            files.append(line.strip())

    # files = [x for x in os.listdir(args.supernat) if 'README' not in x]
    accepted = []

    for file in tqdm(files):
        target_json = json.load(open(os.path.join(args.supernat, file + '.json')))

        # Assume at least 50% of them have the answer inside.
        df = target_json['Definition'][0]
        demo = target_json['Positive Examples']
        instances = target_json['Instances']
        category = target_json['Categories']
        reasoning = target_json['Reasoning']


        # rand_int = random.randint(0, len(demo)-1)
        # print(rand_int,"rand")

        target_num = [1 if instance['output'][0].lower().strip() in df.lower() else 0 for instance in instances[:32]] 
        output_list = []
        accepted_elem = {}
        accepted_task_elem= []
        for instance in instances[:32]:
            output = instance['output'][0]
            if output not in output_list:
                output_list.append(output)
        if sum(target_num) == 32:
            sampled_instance = random.sample(instances, 64)
            # sampled_instance = demo
            # random.shuffle(instances)
            for instance in sampled_instance:
                total_input = "Definition: " + df + '\n\nInput: ' + instance['input'] + '\n' + 'Output: ' + instance['output'][0] + '\n\n'
                input_instance = instance['input'] 
                if len(tokenizer(total_input)['input_ids'])<=256:
                    if len(accepted_task_elem) == 64:
                        break
                    accepted_task_elem.append(total_input)
                    accepted.append(total_input)
                    total_dict[total_input] = output_list



    total_list = accepted
    print("length of total_list", len(total_list), len(total_dict))

    
    demo_embeddings = encoder.encode(total_list)

    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=args.num_clusters, random_state=args.random_seed)
    clustering_model.fit(demo_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(args.num_clusters)]

    dist = clustering_model.transform(demo_embeddings)
    clustered_dists = [[] for i in range(args.num_clusters)]
    clustered_idx = [[] for i in range(args.num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(total_list[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)
        
    demos = []
    output_list = []

    for i in range(len(clustered_dists)):
        print("Cluster ", i+1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        print("length is ", len(top_min_dist))
        for element in top_min_dist:
            flag = 0
            min_idx = element[0]

            if len(output_list) == 0:
                demo_element = total_list[clustered_idx[i][min_idx]]
                output_demo = list(total_dict.values())[clustered_idx[i][min_idx]]
                demos.append(demo_element)
                output_list.append(output_demo)
                print(demo_element)
                break
            else: 
                demo_element = total_list[clustered_idx[i][min_idx]]
                output_demo = list(total_dict.values())[clustered_idx[i][min_idx]]
                if flag == 0:
                    demos.append(demo_element)
                    output_list.append(output_demo)
                    print(demo_element)
                    break

    demos = {"demo": demos}

    with open(args.output_dir+'/clustering_demo_revised_seed2.json', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    y_km = clustering_model.fit_predict(demo_embeddings)
    pca_model = PCA(n_components=2, random_state=args.random_seed)
    transformed = pca_model.fit_transform(demo_embeddings)
    centers = pca_model.transform(clustering_model.cluster_centers_)

    plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.4)
    plt.scatter(centers[:, 0],centers[:, 1],
            s=250, marker='*', label='centroids',
            edgecolor='black',
           c=np.arange(0,args.num_clusters),cmap=plt.cm.Paired,)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.output_dir+"/embedding.png", dpi=600)

if __name__ == "__main__":
    main()