import os
import json
from datasets import load_dataset
from collections import defaultdict
import random


def load_clustering_results(dataset_id):
    """Load clustering results from workflow.py."""
    current_dir = os.getcwd()
    clustering_file = os.path.join(current_dir, f"look_up/clusters_{(dataset_id.split('/')[-1])}.json")
    if os.path.exists(clustering_file):
        with open(clustering_file, 'r') as f:
            clusters = json.load(f)
        print(f"Loaded clustering results from {clustering_file}")
        return clusters
    else:
        raise FileNotFoundError(f"Clustering results not found at {clustering_file}")

topk_prompt = 10 # number of examples to match to one test example form the same cluster
dataset_id = "ruthchy/syn-length-gen-logo-image"
clusters = load_clustering_results(dataset_id)

dataset_ex = load_dataset(dataset_id)["train"]
print(f"Loaded train split {dataset_id} with {len(dataset_ex)} examples.")

test_ex = load_dataset(dataset_id)["test"]
print(f"Loaded test split {dataset_id} with {len(test_ex)} examples.")

def build_cluster_lookup(clusters):
    test_to_cluster = {}
    train_by_cluster = defaultdict(list)

    for cluster_id, items in clusters.items():
        for item in items:
            split, idx, *_ = item.split("_", 2)
            if split == "test":
                test_to_cluster[f"{split}_{idx}"] = cluster_id
            elif split == "train":
                train_by_cluster[cluster_id].append(idx)

    return test_to_cluster, train_by_cluster

def sample_shared_train_examples(train_by_cluster, topk_prompt):
    shared_train_samples = {}

    for cluster_id, train_indices in train_by_cluster.items():
        if len(train_indices) < topk_prompt:
            # Fallback: sample with replacement if not enough examples
            samples = random.choices(train_indices, k=topk_prompt)
        else:
            samples = random.sample(train_indices, k=topk_prompt)

        shared_train_samples[cluster_id] = samples

    return shared_train_samples

def get_topk_examples_shared(test_ex, dataset_ex, test_to_cluster, shared_train_samples):
    topk_examples = {}

    for test_id in range(len(test_ex)):
        test_key = f"test_{test_id}"
        cluster_id = test_to_cluster.get(test_key)

        if cluster_id is None:
            print(f"Warning: {test_key} not found in any cluster.")
            continue

        train_indices = shared_train_samples.get(cluster_id)
        if not train_indices:
            print(f"Warning: No train samples found for cluster {cluster_id}.")
            continue

        #selected_examples = [dataset_ex[int(idx)] for idx in train_indices]
        train_indices = [int(idx) for idx in train_indices]
        selected_examples = dataset_ex.select(train_indices)
        topk_examples[test_key] = selected_examples

    return topk_examples

print("\nStep 1: Build cluster lookup")
test_to_cluster, train_by_cluster = build_cluster_lookup(clusters)

print("Head of the test_to_cluster dictionary:")
for k, v in list(test_to_cluster.items())[:5]:
    print(f"{k}: {v}")
#print("Head of the train_by_cluster dictionary:")
#for k, v in list(train_by_cluster.items())[:5]:
#    print(f"{k}: {v}")

print("\nStep 2: Sample shared train examples")
shared_train_samples = sample_shared_train_examples(train_by_cluster, topk_prompt)

print("\nStep 3: Get top-k examples for each test example")
topk_results = get_topk_examples_shared(test_ex, dataset_ex, test_to_cluster, shared_train_samples)

print(f"\nFound {len(topk_results)} test examples with top-{topk_prompt} train examples.")