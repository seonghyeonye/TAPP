import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
from tqdm.auto import tqdm
import os
from transformers import HfArgumentParser, GPT2TokenizerFast
from dataclasses import dataclass, field
from itertools import chain


@dataclass
class CustomArguments():
    supernat: str = field(
        default="./SuperNat/data/tasks", metadata={"help": "Path for the directory containing all the task json files."}
    )
    random_seed: int = field(
        default=123, metadata={"help": "random seed for clustering & PCA"}
    )
    encoder: str = field(
        default="all-MiniLM-L6-v2", metadata={"help": "encoder for clustering demonstrations"}
    )
    output_dir: str = field(
        default="./output", metadata={"help": "output path for resulting image"}
    ) 
    num_clusters: int = field(
        default=8, metadata={"help": "number of clusters"}
    )
    mode: str = field(
        default="random", metadata={"help": "mode of choosing the demo. either 'random', 'diverse', 'same_cluster'"}
    )
    num_pos_examples: int = field(
        default=8, metadata={"help": "Number of samples to use in demo. i.e. value of K for K-shot."}
    )
    ratio: float = field(
        default=1, metadata={"help": "Ratio of Clf Samples to incorporate."}
    )
    demo_path: str = field(
        default=False, metadata={"help": "path to demo, if there is one from which to modify and insert non-clf demos."}
    )


if __name__=="__main__":

    parser = HfArgumentParser((CustomArguments,))
    args, = parser.parse_args_into_dataclasses()

    random.seed(args.random_seed)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encoder = SentenceTransformer(args.encoder)

    files = [x for x in os.listdir(args.supernat) if 'README' not in x]

    accepted = []
    others = []

    for file in tqdm(files):
        target_json = json.load(open(os.path.join(args.supernat, file)))

        # Assume at least 50% of them have the answer inside.
        df = target_json['Definition'][0]
        instances = target_json['Instances']

        target_num = [1 if instance['output'][0].lower().strip() in df.lower() else 0 for instance in instances]   
        if sum(target_num) == len(instances):
            accepted.append(["Definition: " + df + '\n\n' + x['input'] + '\n' + 'Output: ' + x['output'][0] for x in instances[:32]])
            # accepted.append((file, round(sum(target_num) / len(instances), 4)))
        else:
            others.append(["Definition: " + df + '\n\n' + x['input'] + '\n' + 'Output: ' + x['output'][0] for x in instances[:32]])

        target = list(chain.from_iterable(accepted))
        target2 = list(chain.from_iterable(others))

    MODE = args.mode

    # If user wants to modify from a given CLF demo.
    if args.demo_path:
        final_demos = json.load(open(args.demo_path))["demo"]
        
        clf_num = round(len(final_demos) * args.ratio)
        others_num = args.num_pos_examples - clf_num

        idxs_to_replace = random.sample(range(0, len(final_demos)), others_num)
        other_idxs = random.sample(range(0, len(others)), len(others))
        other_demos = []

        for idx in other_idxs:
            if len(other_demos) >= others_num:
                break

            selected = random.choice(others[idx])
                
            if len(tokenizer(selected)["input_ids"]) <= 256:
                other_demos.append(selected)
            else:
                continue # That particular task is likely to have other instances also longer than 256... so skip.
        
        for i, idx in enumerate(idxs_to_replace):
            final_demos[idx] = other_demos[i]        

    elif MODE == "random":
        # NOTE: Ratio only works for random setting.
        clf_num = round(args.num_pos_examples * args.ratio)
        others_num = args.num_pos_examples - clf_num

        task_idxs = random.sample(range(0, len(accepted)), clf_num)
        clf_demos = [random.choice(accepted[idx]) for idx in task_idxs]

        other_idxs = random.sample(range(0, len(others)), others_num)
        other_demos = [random.choice(others[idx]) for idx in other_idxs]

        final_demos = clf_demos + other_demos

    else:
        ## Set KMeans/PCA training set and encode training set.
        demo_list = target
        demo_input = target
        demo_embeddings = encoder.encode(target)

        # Perform KMeans
        num_clusters = args.num_clusters
        clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
        clustering_model.fit(demo_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]

        dist = clustering_model.transform(demo_embeddings)
        clustered_dists = [[] for i in range(num_clusters)]
        clustered_idx = [[] for i in range(num_clusters)]

        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(demo_list[sentence_id])
            clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
            clustered_idx[cluster_id].append(sentence_id)

        diverse_demos = []

        for i in range(len(clustered_dists)):
            tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
            top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
            for element in top_min_dist:
                min_idx = element[0]

                if len(tokenizer(demo_input[clustered_idx[i][min_idx]])["input_ids"]) <= 256:
                    demo_element = demo_input[clustered_idx[i][min_idx]]
                    diverse_demos.append(demo_element)
                    break

        if MODE == "diverse":
            final_demos = diverse_demos

        elif MODE == "same_cluster":
            # Choose random cluster.
            idx = random.choice(range(0, num_clusters))
            final_demos = random.sample(clustered_sentences[idx], 4)

        else:
            raise NotImplementedError
        
    # Write Json File.
    os.makedirs(args.output_dir, exist_ok=True)
    fname = os.path.join(args.output_dir, f"ICL_ratio_{args.ratio}.json")

    # json.dump({"demo": final_demos}, open(os.path.join(args.output_dir, fname), "w"), indent=4)
    json.dump({"demo": final_demos}, open(fname, "w"), indent=4)



