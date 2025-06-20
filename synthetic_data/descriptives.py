import os
from datasets import load_dataset, DatasetDict
from _6_sem_length import SemLength
from _6_syn_length import SynLength
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

dataset_name_syn = "ruthchy/syn-length-gen-logo-image"
dataset_name_syn_unbiased = "ruthchy/syn-length-gen-logo-image-unbiased-test"
dataset_name_sem = "ruthchy/sem-length-gen-logo-image"
dataset_name_sem_unbiased = "ruthchy/sem-length-gen-logo-image-unbiased-test"

ds_syn = load_dataset(dataset_name_syn)
ds_syn_unbiased = load_dataset(dataset_name_syn_unbiased)
ds_sem = load_dataset(dataset_name_sem)
ds_sem_unbiased = load_dataset(dataset_name_sem_unbiased)


# Helper functions
def calc_max_lengths(ds, syn_length, sem_length):
    # Combine train and validation splits
    train_df = ds['train'].to_pandas().drop(columns=["Image"])
    val_df = ds['validation'].to_pandas().drop(columns=["Image"])
    train_val = pd.concat([train_df, val_df], ignore_index=True)
    # Calculate lengths
    train_val['syn_length'] = train_val['Program'].apply(syn_length.calc_length)
    train_val['sem_length'] = train_val['Program'].apply(sem_length.calc_length)
    max_syn = train_val['syn_length'].max()
    max_sem = train_val['sem_length'].max()
    return max_syn, max_sem

def calc_min_lengths(ds, syn_length, sem_length):
    test = ds['test'].to_pandas().drop(columns=["Image"])
    test['syn_length'] = test['Program'].apply(syn_length.calc_length)
    test['sem_length'] = test['Program'].apply(sem_length.calc_length)
    min_syn = test['syn_length'].min()
    min_sem = test['sem_length'].min()
    return min_syn, min_sem

# Initialize length calculators
syn_length = SynLength()
sem_length = SemLength()

datasets = {
    "Split by Syn. Len.": ds_syn,
    "No Length Bias (Syn. Len.)": ds_syn_unbiased,
    "Split by Sem. Len": ds_sem,
    "No Length Bias (Sem. Len.)": ds_sem_unbiased,

}

for name, ds in datasets.items():
    max_syn, max_sem = calc_max_lengths(ds, syn_length, sem_length)
    min_syn, min_sem = calc_min_lengths(ds, syn_length, sem_length)
    print(f"\n{name}:")
    print(f"  MAX syn_length (train+val): {max_syn}")
    print(f"  MAX sem_length (train+val): {max_sem}")
    print(f"  MIN syn_length (test): {min_syn}")
    print(f"  MIN sem_length (test): {min_sem}")


# Compute global min and max for semantic length
# --- Compute global min/max for syntactic and semantic lengths ---
all_syn_lengths = []
all_sem_lengths = []
for ds in datasets.values():
    for split in ['train', 'validation', 'test']:
        df = ds[split].to_pandas().drop(columns=["Image"])
        df['syn_length'] = df['Program'].apply(syn_length.calc_length)
        df['sem_length'] = df['Program'].apply(sem_length.calc_length)
        all_syn_lengths.append(df['syn_length'])
        all_sem_lengths.append(df['sem_length'])

global_syn_min = pd.concat(all_syn_lengths).min()
global_syn_max = pd.concat(all_syn_lengths).max()
global_sem_min = pd.concat(all_sem_lengths).min()
global_sem_max = pd.concat(all_sem_lengths).max()

shared_syn_bins = np.linspace(global_syn_min, global_syn_max, 31)  # 30 bins
shared_sem_bins = np.logspace(np.log10(global_sem_min), np.log10(global_sem_max), num=30)

for name, ds in datasets.items():
    # Convert splits to pandas and drop Image
    train_df = ds['train'].to_pandas().drop(columns=["Image"])
    val_df = ds['validation'].to_pandas().drop(columns=["Image"])
    test_df = ds['test'].to_pandas().drop(columns=["Image"])

    # Calculate lengths
    train_df['syn_length'] = train_df['Program'].apply(syn_length.calc_length)
    train_df['sem_length'] = train_df['Program'].apply(sem_length.calc_length)
    val_df['syn_length'] = val_df['Program'].apply(syn_length.calc_length)
    val_df['sem_length'] = val_df['Program'].apply(sem_length.calc_length)
    test_df['syn_length'] = test_df['Program'].apply(syn_length.calc_length)
    test_df['sem_length'] = test_df['Program'].apply(sem_length.calc_length)

    ### SYNTACTIC LENGTH DISTRIBUTION ###
    # Plot syntactic length
    all_syn_lengths = pd.concat([train_df['syn_length'], val_df['syn_length'], test_df['syn_length']])
    shared_bins = np.histogram_bin_edges(all_syn_lengths, bins=30)

    plt.figure(figsize=(8, 5))
    plt.hist(train_df['syn_length'], bins=shared_syn_bins, alpha=0.5, label='Train', color='#607196')
    plt.hist(val_df['syn_length'], bins=shared_syn_bins, alpha=0.5, label='Validation', color='#607196')
    plt.hist(test_df['syn_length'], bins=shared_syn_bins, alpha=0.7, label='Test', color='#FF7B9C')
    plt.ylim(0, 1750)   # y-axis: consistent max count
    plt.xlim(0, 25)     # x-axis: match maximum syntactic length
    #plt.title(f"Syntactic Length Distribution ({name})")
    plt.xlabel("Syntactic Length")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{name}_syntactic_length_distribution.png")
    plt.show()
    
    ### SEMANTIC LENGTH DISTRIBUTION ###
    # Define log-spaced bins
    sem_values = pd.concat([train_df['sem_length'], val_df['sem_length'], test_df['sem_length']])
    log_bins = np.logspace(np.log10(sem_values.min()), np.log10(sem_values.max()), num=30)


    # Plot semantic length (log scale)
    plt.figure(figsize=(8, 5))
    plt.hist(train_df['sem_length'], bins=shared_sem_bins, alpha=0.5, label='Train', color='#607196')
    plt.hist(val_df['sem_length'], bins=shared_sem_bins, alpha=0.5, label='Validation', color='#607196')
    plt.hist(test_df['sem_length'], bins=shared_sem_bins, alpha=0.7, label='Test', color='#FF7B9C')
    plt.ylim(0, 2250)         # y-axis: consistent max count
    plt.xlim(1, 1e14)         # x-axis: set log range from 10^0 to 10^14
    plt.xscale('log')
    #plt.title(f"Semantic Length Distribution ({name})")
    plt.xlabel("Semantic Length (log scale)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{name}_semantic_length_distribution_log.png")
    plt.show()