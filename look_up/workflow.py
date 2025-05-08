'''
Implementation of the Preprocessing for the Few-Shot Examples
1. Load the dataset
2. Compute and store the set of Embeddings based on the Description column
    - Use OpenAI's ada embedding model
    - Note: to run the code when the embeddings are not already computed for the dataset you pass the HF-id via the --dataset argument, you need to pass your OpenAI API key 
    for example using a yaml-file (named: open_ai_key.yaml in the look_up folder) which contains: 
        openai_api_key: <sk-...>
3. Cluster the embeddings using Ward’s clustering algorithm (Ward Jr, 1963)
    - Optional: Plot the dendrogram to find the optimal number of clusters 
    - Optional: If the number of clusters is knonw one can specify it via the --n_clusters argument
    - the tree of related examples
    - I think the few-shot examples should be choosen from the cluster given certain criteria as being from train or validation set and if there is a criteria such as being a snowflake or a sequence then this should guide the selection
    - (in the ReGAL paper the next steps would be to topologically sorted and grouped all entries into k batches (ReGAL paper k=5))

Usage:
python look_up/workflow.py --dataset <dataset_id> --plot_dendrogram_only
python look_up/workflow.py --dataset <dataset_id> --n_clusters <n_clusters>
Example:
python look_up/workflow.py --dataset ruthchy/syn-length-gen-logo-image --plot_dendrogram_only
'''
import argparse
import os
import json
import yaml
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from datasets import load_dataset  # Huggingface datasets
from openai import OpenAI
import matplotlib.pyplot as plt


def get_base_dir():
    """Get the base directory of the master-thesis project."""
    current_dir = os.getcwd()
    split_point = current_dir.find("master-thesis")
    if split_point != -1:
        print(f"Base directory: {current_dir}")
        return current_dir[: split_point + len("master-thesis")]
    else:
        raise RuntimeError("Error: 'master-thesis' not found in the current path.")

def load_openai_api_key(base_dir):
    """Load the OpenAI API key from the YAML file."""
    key_file = os.path.join(base_dir, "look_up/open_ai_key.yaml")
    with open(key_file, 'r') as f:
        open_ai_key = yaml.safe_load(f)
    # Access the API key directly from the dictionary
    client = OpenAI(api_key=open_ai_key["openai_api_key"])
    return client

def load_or_compute_dataset_embeddings(dataset_id, base_dir, client):
    """Load or compute embeddings for the dataset."""
    embeddings_folder = os.path.join(base_dir, "look_up/embeddings")
    embedding_file = os.path.join(embeddings_folder, f"embed_{(dataset_id.split('/')[-1])}.json")

    if os.path.exists(embedding_file):
        print(f"Embeddings for dataset {dataset_id} found. Loading embeddings...")
        with open(embedding_file, 'r') as f:
            embeddings = json.load(f)
    else:
        print(f"Embeddings for dataset {dataset_id} not found. Computing embeddings...")
        os.makedirs(embeddings_folder, exist_ok=True)
        dataset = load_dataset(dataset_id)
        embeddings = {}

        for split in dataset.keys():
            print(f"Processing split: {split}")
            descriptions = dataset[split]['Description']
            split_embeddings = {}
            for desc in descriptions:
                response = client.embeddings.create(input=desc, model="text-embedding-ada-002")
                split_embeddings[desc] = response.data[0].embedding
            embeddings[split] = split_embeddings

        with open(embedding_file, 'w') as f:
            json.dump(embeddings, f)
        print(f"Embeddings saved to {embedding_file}")

    return embeddings

# Step 3: Apply clustering algorithm
def plot_dendrogram(embeddings_array, all_descriptions, base_dir, dataset_id):
    """Plot and save the dendrogram."""
    print("Find the optimal number of clusters using a Dendrogram...")
    Z = linkage(embeddings_array, method='ward', optimal_ordering=True)

    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=all_descriptions, leaf_rotation=90)
    plt.title('Dendrogram (Ward Linkage)')
    plt.xlabel('Descriptions')
    plt.ylabel('Distance')

    dendrogram_path = os.path.join(base_dir, f"look_up/dendrogram_{(dataset_id.split('/')[-1])}.png")
    plt.tight_layout()
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Dendrogram saved to {dendrogram_path}")

def apply_clustering(embeddings, n_clusters, base_dir, dataset_id, plot_dendrogram_only=False):
    """Apply clustering and save the results."""
    print("Applying clustering algorithm...")

    # Flatten embeddings and descriptions
    all_embeddings = []
    all_descriptions = []
    for split, split_embeddings in embeddings.items():
        for desc, embedding in split_embeddings.items():
            all_embeddings.append(embedding)
            all_descriptions.append(f"{split}_{desc}")

    embeddings_array = np.array(all_embeddings)

    if args.plot_dendrogram_only:
        plot_dendrogram(embeddings_array, all_descriptions, base_dir, dataset_id)
        return

    print(f"Using specified number of clusters: {n_clusters}")
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(embeddings_array)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(all_descriptions[idx])

    output_file = os.path.join(base_dir, f"look_up/clusters_{(dataset_id.split('/')[-1])}.json")
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=4)
    print(f"Clusters saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Few-Shot Examples Preprocessing')
    parser.add_argument('--dataset', type=str, help='Huggingface dataset name')
    parser.add_argument('--plot_dendrogram_only', action='store_true', help='Only plot the dendrogram without clustering')
    parser.add_argument('--n_clusters', type=int, default=None, help='The number of clusters to form')
    args = parser.parse_args()
    if not args.plot_dendrogram_only and n_clusters is None:
        raise ValueError("Please specify the number of clusters using --n_clusters.")

    dataset_id = args.dataset
    n_clusters = args.n_clusters
    plot_dendrogram_only = args.plot_dendrogram_only

    # Set up path and load OpenAI API key
    base_dir = get_base_dir()
    client = load_openai_api_key(base_dir)

    # Step 1: Compute or load dataset embeddings
    embeddings = load_or_compute_dataset_embeddings(dataset_id, base_dir, client)

    # Step 2: Apply clustering algorithm
    apply_clustering(embeddings, n_clusters, base_dir, dataset_id, plot_dendrogram_only)