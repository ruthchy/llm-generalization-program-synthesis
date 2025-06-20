'''
Implementation of the Preprocessing for the Few-Shot Examples

1. Load the dataset:
    - The dataset is loaded using the Huggingface `datasets` library.
    - The dataset ID is passed via the `--dataset` argument.

2. Compute and store the set of embeddings based on the Description column:
    - Uses OpenAI's `text-embedding-ada-002` model to compute embeddings.
    - Requires an OpenAI API key, which is loaded from a YAML file (`open_ai_key.yaml`) in the `look_up` folder.
    - The embeddings are saved as a JSON file in the `look_up/embeddings` folder.
    - Prints a validation table displaying the number of embeddings matches the number of examples in the dataset for each split (train, validation, test).


3. Analyze the embeddings:
    - Dendrogram: Visualize hierarchical clustering to determine the relationships between examples.
    - Elbow Method: Determine the optimal number of clusters by analyzing the within-cluster sum of squares (WCSS).
    - Silhouette Scores: Evaluate clustering quality for different numbers of clusters.

4. Cluster the embeddings:
    - Uses Ward’s hierarchical clustering algorithm (Ward Jr, 1963).
    - The number of clusters can be specified via the `--n_clusters` argument.
    - Saves the clustering results as a JSON file in the `look_up` folder.

Usage:
    python look_up/workflow.py --dataset <dataset_id> --plot_dendrogram_only
    python look_up/workflow.py --dataset <dataset_id> --plot_elbow --plot_silhouette --n_clusters <n_clusters>
    python look_up/workflow.py --dataset <dataset_id> --n_clusters <n_clusters>

Examples:
    1. Generate a dendrogram to determine the range of clusters:
        python look_up/workflow.py --dataset ruthchy/syn-length-gen-logo-image --plot_dendrogram_only

    2. Generate the elbow method and silhouette score plots:
        python look_up/workflow.py --dataset ruthchy/syn-length-gen-logo-image --plot_elbow --plot_silhouette --n_clusters 10

    3. Perform clustering with a specified number of clusters:
        python look_up/workflow.py --dataset ruthchy/syn-length-gen-logo-image --n_clusters 5

Notes:
    - If no plotting options (`--plot_dendrogram_only`, `--plot_elbow`, `--plot_silhouette`) are specified, the script defaults to applying the clustering algorithm.
    - The script validates that the number of embeddings matches the number of examples in the dataset.
    - Subsampling is applied for dendrograms if the number of embeddings exceeds 500 to improve readability.

Documentation of the final cluster n choosen for each of the datesest
- ruthchy/syn-length-gen-logo-image: 5 (based on shilhouette score: 2 best choice but 5 second best choice while offering more granularity)
- ruthchy/syn-length-gen-logo-image-unbiased-test: 4 (based on shilhouette score: 2 best choice but 4 second best choice while offering more granularity)
- ruthchy/sem-length-gen-logo-image: 5 (based on shilhouette score: 2 best choice but 5 second best choice while offering more granularity)
- ruthchy/sem-length-gen-logo-image-unbiased-test: 5 (based on shilhouette score: 2 best choice but 5 second best choice while offering more granularity)
- ruthchy/mix-match-gen-logo-data-size: 3 (based on shilhouette score: 3 best)
'''
import argparse
import os
import json
import yaml
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from datasets import load_dataset  # Huggingface datasets
from openai import OpenAI
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
import random

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

    dataset = load_dataset(dataset_id)

    if os.path.exists(embedding_file):
        print(f"Embeddings for dataset {dataset_id} found. Loading embeddings...")
        with open(embedding_file, 'r') as f:
            embeddings = json.load(f)
    else:
        print(f"Embeddings for dataset {dataset_id} not found. Computing embeddings...")
        os.makedirs(embeddings_folder, exist_ok=True)
        embeddings = {}

        for split in dataset.keys():
            print(f"Processing split: {split}")
            descriptions = dataset[split]['Description']
            split_embeddings = {}
            for idx, desc in enumerate(descriptions):
                prep_desc = desc.replace(" ", "_")
                unique_key = f"{idx}_{prep_desc}"  # unique key: desc prefixing by index
                try:
                    response = client.embeddings.create(input=desc,
                    model="text-embedding-ada-002")
                    split_embeddings[unique_key] = response.data[0].embedding
                except Exception as e:
                    print(f"Error processing description with key {unique_key}: {e}")
            embeddings[split] = split_embeddings

        with open(embedding_file, 'w') as f:
            json.dump(embeddings, f)
        print(f"Embeddings saved to {embedding_file}")

    # Controll all examples have been processed
    print("\n--- Comparison: Number of exampels and embeddings per split ---")
    for split in dataset:
        print(f"Number of examples in {split}: {len(dataset[split])}")
        print(f"Number of embeddings in {split}: {len(embeddings.get(split, {}))}")
    print("---------------------------------------------------------------\n")    

    return embeddings

# Step 3: Apply clustering algorithm
def prep_embeddings(embeddings):
    """Prepare embeddings for clustering by flattening the dictionary."""
    all_embeddings = []
    all_descriptions = []
    for split, split_embeddings in embeddings.items():
        for desc, embedding in split_embeddings.items():
            all_embeddings.append(embedding)
            all_descriptions.append(f"{split}_{desc}")

    embeddings_array = np.array(all_embeddings)
    return embeddings_array, all_descriptions

def plot_dendrogram(embeddings_array, all_descriptions, base_dir, dataset_id, max_samples=500):
    """Plot and save the dendrogram with optional subsampling."""
    if len(embeddings_array) > max_samples:
        print(f"Subsampling to {max_samples} samples for dendrogram...")
        indices = random.sample(range(len(embeddings_array)), max_samples)
        embeddings_array = embeddings_array[indices]
        all_descriptions = [all_descriptions[i] for i in indices]
    
    # Assign specific colors to splits
    color_map = {
        "train": "#00334f",  # Dark Denim Blue
        "validation": "#57a9d4",  # Hoeth Blue
        "test": "#8B0000"  # Dark Red
    }
    # Assing colors to splits using the thesis color palette
    #color_map = {
    #    "train": "#607196",  
    #    "validation": "#7E8CAC",  
    #    "test": "#FF7B9C"  
    #}

    splits = [desc.split('_')[0] for desc in all_descriptions]
    colors = [color_map[split] for split in splits]

    print("Find the optimal number of clusters using a Dendrogram...")
    Z = linkage(embeddings_array, method='ward', optimal_ordering=True)

    plt.figure(figsize=(24, 12))
    dendrogram(
        Z, 
        labels=all_descriptions,
        leaf_rotation=90,
        truncate_mode='level',  # Truncate the dendrogram
        p=15  # Number of levels or clusters to show
    )
    ax = plt.gca()
    x_labels = ax.get_xticklabels()
    for label in x_labels:
        split = label.get_text().split('_')[0]
        label.set_color(color_map[split])
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=split,
                        markerfacecolor=color, markersize=10)
            for split, color in color_map.items()]
    plt.legend(handles=handles, title="Splits", loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title('Dendrogram (Ward Linkage)')
    plt.xlabel('Descriptions')
    plt.ylabel('Distance')

    dendrogram_path = os.path.join(base_dir, f"look_up/images/dendrogram_{(dataset_id.split('/')[-1])}.png")
    plt.tight_layout()
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Dendrogram saved to {dendrogram_path}")

def plot_elbow_method(embeddings_array, base_dir, dataset_id, max_clusters=10):
    """Plot the elbow method to determine the optimal number of clusters."""
    print(f"Computing the elbow method for up to {max_clusters} clusters...")
    wcss = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings_array)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', color='#607196')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    elbow_path = os.path.join(base_dir, f"look_up/images/elbow_{(dataset_id.split('/')[-1])}.png")
    plt.tight_layout()
    plt.savefig(elbow_path)
    plt.close()
    print(f"Elbow plot saved to {elbow_path}")

def plot_silhouette_scores(embeddings_array, base_dir, dataset_id, max_clusters=10):
    """Plot silhouette scores for different numbers of clusters."""
    print(f"Computing silhouette scores for up to {max_clusters} clusters...")
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):  # Silhouette score requires at least 2 clusters
        clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings_array)
        score = silhouette_score(embeddings_array, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='#607196')
    #plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    silhouette_path = os.path.join(base_dir, f"look_up/images/silhouette_{(dataset_id.split('/')[-1])}.png")
    plt.tight_layout()
    plt.savefig(silhouette_path)
    plt.close()
    print(f"Silhouette plot saved to {silhouette_path}")

def apply_clustering(embeddings_array, all_descriptions, n_clusters, base_dir, dataset_id):
    """Apply clustering and save the results."""
    print("Applying clustering algorithm...")
    print(f"Using specified number of clusters: {n_clusters}")
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(embeddings_array)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        label = int(label) 
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
    parser.add_argument('--plot_elbow', action='store_true', help='Plot the elbow method to determine the optimal number of clusters')
    parser.add_argument('--plot_silhouette', action='store_true', help='Plot silhouette scores to evaluate clustering')
    parser.add_argument('--n_clusters', type=int, default=None, help='The number of clusters to form')
    args = parser.parse_args()
    if not args.plot_dendrogram_only and args.plot_elbow and args.plot_silhouette and args.n_clusters is None:
        raise ValueError("Please specify the number of clusters using --n_clusters.")

    dataset_id = args.dataset
    n_clusters = args.n_clusters

    # Set up path and load OpenAI API key
    base_dir = get_base_dir()
    client = load_openai_api_key(base_dir)

    # Step 1: Compute or load dataset embeddings
    embeddings = load_or_compute_dataset_embeddings(dataset_id, base_dir, client)

    # Step 2: Plot dendrogram, elbow method, or silhouette scores
    embeddings_array, all_descriptions = prep_embeddings(embeddings)
    
    if args.plot_dendrogram_only or args.plot_elbow or args.plot_silhouette:
        print("Generating visual aids to determine the optimal number of clusters...")
        if args.plot_dendrogram_only:
            # Plot dendrogram only
            print("Plotting dendrogram...")
            plot_dendrogram(embeddings_array, all_descriptions, base_dir, dataset_id)
        if args.plot_elbow:
            print("Plotting elbow method...")
            plot_elbow_method(embeddings_array, base_dir, dataset_id, max_clusters=n_clusters)
        if args.plot_silhouette:
            print("Plotting silhouette scores...")
            plot_silhouette_scores(embeddings_array, base_dir, dataset_id, max_clusters=n_clusters)
    else:
        # Step 3: Apply clustering algorithm
        apply_clustering(embeddings_array, all_descriptions, n_clusters, base_dir, dataset_id)