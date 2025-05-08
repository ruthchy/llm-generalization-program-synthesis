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
from sklearn.metrics.pairwise import euclidean_distances

with open("look_up/open_ai_key.yaml", 'r') as f:
    open_ai_key = yaml.safe_load(f)
# Access the API key directly from the dictionary
client = OpenAI(api_key=open_ai_key["openai_api_key"])

# Initialize argument parser
parser = argparse.ArgumentParser(description='Few-Shot Examples Preprocessing')
parser.add_argument('--dataset', type=str, help='Huggingface dataset name')
parser.add_argument('--plot_dendrogram_only', action='store_true', help='Only plot the dendrogram without clustering')
parser.add_argument('--n_clusters', type=int, default=None ,help='The number of clusters to form')
args = parser.parse_args()

dataset_id = args.dataset
n_clusters = args.n_clusters
embeddings_folder = "embeddings"
embedding_file = os.path.join(embeddings_folder, f"embed_{dataset_id}.json")

# Step 1: Load the dataset
dataset = load_dataset(dataset_id)

# Step 2: Compute and store embeddings for all splits
if not os.path.exists(embedding_file):
    print(f"Embeddings for dataset {dataset_id} not found. Computing embeddings...")
    os.makedirs(embeddings_folder, exist_ok=True)

    embeddings = {}
    for split in dataset.keys():  # Iterate over all splits (train, validation, test)
        print(f"Processing split: {split}")
        descriptions = dataset[split]['Description']  # Extract descriptions for the split
        split_embeddings = {}

        # Use OpenAI's ada embedding model
        for desc in descriptions:
            response = client.embeddings.create(input=desc, model="text-embedding-ada-002")
            split_embeddings[desc] = response.data[0].embedding


        embeddings[split] = split_embeddings

    # Save embeddings to a file
    with open(embedding_file, 'w') as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved to {embedding_file}")
else:
    print(f"Embeddings for dataset {dataset_id} found. Loading embeddings...")
    with open(embedding_file, 'r') as f:
        embeddings = json.load(f)

# Step 3: Apply clustering algorithm
print("Applying clustering algorithm...")

# Flatten embeddings from all splits into a single list
all_embeddings = []
all_descriptions = []  # Keep track of descriptions for ordering
for split, split_embeddings in embeddings.items():
    for desc, embedding in split_embeddings.items():
        all_embeddings.append(embedding)
        all_descriptions.append(f"{split}_{desc}")

# Convert to a NumPy array
embeddings_array = np.array(all_embeddings)

# Apply clustering
if args.plot_dendrogram_only:
    print("Find the optimal number of clusters using a Dendrogram...")
    Z = linkage(embeddings_array, method='ward', optimal_ordering = True)

    # Create the plot
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=all_descriptions, leaf_rotation=90)
    plt.title('Dendrogram (Ward Linkage)')
    plt.xlabel('Descriptions')
    plt.ylabel('Distance')

    # Save the plot
    dendrogram_path = os.path.join(embeddings_folder, f"dendrogram_{dataset_id}.png")
    plt.tight_layout()
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Dendrogram saved to {dendrogram_path}")
    sys.exit(0)
else:
    print(f"Using specified number of clusters: {n_clusters}")
clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
clustering.fit(embeddings_array)

# Group examples by cluster and order them by distance
clusters = {}
for idx, label in enumerate(clustering.labels_):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append((all_descriptions[idx], embeddings_array[idx]))

# Save ordered clusters to a file
output_file = os.path.join(embeddings_folder, f"clusters_{dataset_id}.json")
with open(output_file, 'w') as f:
    json.dump(ordered_clusters, f, indent=4)
print(f"Ordered clusters saved to {output_file}")