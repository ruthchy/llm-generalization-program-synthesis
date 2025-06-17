'''
Mix-Match Generalization Data Generator
--------------------------------------

This script generates synthetic data that matches the mix-match generalization task:
    - Train data: Contains sequences and snowflakes with single shape arms
    - Validation and test data: Contains snowflakes with sequence arms (longer than 1 shape)

The final dataset contains Description, Program, and Image columns.

Usage:
------
1. Run the script in the thesis_env (requiremnts.txt) environment:
   python synthetic_data/main_mix_match_gen.py

Configuration:
-------------
In the main() function, you can configure:

1. Dataset Sizes:
   - train_size: Number of training examples (default: 8000)
   - val_size: Number of validation examples (default: 1000)
   - test_size: Number of test examples (default: 1000)

2. Sampling Strategy:
   - prioritize: 
     - "ratio": Maintains a 50/50 balance between sequences and snowflakes in training data
                (by oversampling snowflakes if needed because they are rare)
     - "size": Prioritizes reaching target dataset size 
               (will use all available snowflakes and fill rest with sequences)

3. Additional Parameters:
   - batch_size: Number of samples generated per batch (default: 5000)
   - max_iterations: Maximum number of sampling iterations (default: 20)

4. HuggingFace Integration:
   - push_to_hub: Set to True to push dataset to Hugging Face Hub (default: False)
   - Specify hub_name to change the destination repository

Example:
-------
To generate a dataset prioritizing size with 10000 training examples:
```python
train_df, val_df, test_unbiased_df, test_biased_df = generate_mix_match_dataset(
    train_size=10000,
    val_size=1000,
    test_size=1000,
    prioritize="size")
'''

import os
import re
import random
import pandas as pd
from _1_logo_pseudo_code_generator import generateLOGOPseudoCode
from _2_sampler_improved import LOGOProgramSampler
from __parser_pyturtle_pc import ProgramParser

# Set a fixed seed for reproducibility
random.seed(42)

def analyze_shape_distribution():
    """Display an analysis of the possible shape distribution"""
    generator = generateLOGOPseudoCode()
    sampler = LOGOProgramSampler(generator)
    shape_distribution = sampler.count_possible_shapes()

    print("Shape Distribution Analysis:")
    print("===========================")
    for category, data in shape_distribution.items():
        if category != "Total":
            print(f"{category}: {data['count']} shapes ({data['percent']})")
        else:
            print(f"\nTotal possible shapes: {data}\n")

def is_snowflake_with_sequence_arm(description):
    """Check if the program is a snowflake with a sequence as its arms"""
    return "snowflake" in description.lower() and "arms of" in description.lower()

def is_snowflake_with_single_shape_arm(description):
    """Check if the program is a snowflake with a single shape as its arm"""
    return "snowflake" in description.lower() and "arm of" in description.lower()

def is_sequence(description):
    """Check if the program is a sequence"""
    return "sequence of shapes" in description.lower() and "snowflake" not in description.lower()

def analyze_dataset_distribution(df, split_name="Dataset"):
    """Analyze distribution of program types in the dataset"""
    total = len(df)
    if total == 0:
        return pd.DataFrame({
            'Split': [split_name],
            'Sequences (%)': [0],
            'Snowflakes with single shape arms (%)': [0],
            'Snowflakes with sequence arms (%)': [0],
            'Other (%)': [0],
            'Total samples': [0]
        })
        
    # Count program types
    sequences = sum(df["Description"].apply(lambda x: is_sequence(x)))
    single_shape_arms = sum(df["Description"].apply(lambda x: is_snowflake_with_single_shape_arm(x)))
    sequence_arms = sum(df["Description"].apply(lambda x: is_snowflake_with_sequence_arm(x)))
    others = total - (sequences + single_shape_arms + sequence_arms)
    
    # Calculate percentages
    seq_pct = sequences / total * 100
    single_pct = single_shape_arms / total * 100
    seq_arms_pct = sequence_arms / total * 100
    others_pct = others / total * 100
    
    return pd.DataFrame({
        'Split': [split_name],
        'Sequences (%)': [f"{seq_pct:.2f}%"],
        'Snowflakes with single shape arms (%)': [f"{single_pct:.2f}%"],
        'Snowflakes with sequence arms (%)': [f"{seq_arms_pct:.2f}%"],
        'Other (%)': [f"{others_pct:.2f}%"],
        'Total samples': [total]
    })

def generate_mix_match_dataset(train_size=8000, val_size=1000, test_size=1000, batch_size=5000, 
                              max_iterations=20, prioritize="ratio"):
    """
    Generate a dataset for mix-match generalization task with balanced training data.
    
    Parameters:
    - train_size: Target size for training dataset
    - val_size: Target size for validation dataset
    - test_size: Target size for test dataset
    - batch_size: Number of samples to generate in each batch
    - max_iterations: Maximum number of sampling iterations
    - prioritize: Either "ratio" (maintain 50/50 even if dataset is smaller) 
                 or "size" (prioritize total dataset size, allowing imbalanced ratio)
    """
    generator = generateLOGOPseudoCode()
    sampler = LOGOProgramSampler(generator)
    
    single_snowflakes = []
    target_snowflakes = (train_size + val_size + test_size) // 2 
    total_samples = 0

    # Phase 1: Focus on collecting the rare snowflakes with single shape arms
    print("Phase 1: Collecting rare snowflakes with single shape arms...")
    collected_all_snowflakes = False
    
    for i in range(max_iterations):
        if len(single_snowflakes) >= target_snowflakes:
            break
            
        print(f"Batch {i+1}/{max_iterations} for single shape snowflakes")
        batch = sampler.sample(batch_size)
        total_samples += batch_size
        
        for sample in batch:
            description = sample["Description"]
            
            # Focus only on collecting single shape arm snowflakes
            if len(single_snowflakes) < target_snowflakes and is_snowflake_with_single_shape_arm(description):
                single_snowflakes.append(sample)
        
        print(f"Progress: Train (Snowflakes): {len(single_snowflakes)}/{target_snowflakes}")
        
        # Check if we've collected all available snowflakes
        if i >= 2 and len(single_snowflakes) == 180:
            print("All 180 snowflakes with single shape arms collected.")
            collected_all_snowflakes = True
            break

    n_val = int(0.1 * len(single_snowflakes))  # 10% for validation
    n_test = int(0.1 * len(single_snowflakes)) # 10% for test
    val_single_snowflakes = single_snowflakes[:n_val]
    test_single_snowflakes = single_snowflakes[n_val:n_val+n_test]
    train_single_snowflakes = single_snowflakes[n_val+n_test:]

    if prioritize == "ratio":
        # Calculate how many snowflakes and sequences are needed for each split
        n_train_snow = train_size // 2
        n_val_snow = val_size // 2
        n_test_unbiased_snow = test_size // 2

        def oversample(snowflakes, target_count):
            """Oversample snowflakes to reach target count"""
            if len(snowflakes) >= target_count:
                return snowflakes[:target_count]
            else:
                return snowflakes + random.choices(snowflakes, k=target_count - len(snowflakes))
        train_single_snowflakes = oversample(train_single_snowflakes, n_train_snow)
        val_single_snowflakes = oversample(val_single_snowflakes, n_val_snow)
        test_single_snowflakes = oversample(test_single_snowflakes, n_test_unbiased_snow)

    # Split training data into two categories
    train_sequences = []
    val_sequences = []
    test_sequences_unbiased = []
    test_snowflakes_seq_arms_biased = []

    # Calculate target counts for the different datasubsets
    n_train_needed = train_size - len(train_single_snowflakes)
    n_val_needed = val_size - len(val_single_snowflakes)
    n_test_unbiased_needed = test_size - len(test_single_snowflakes)
    n_test_biased_needed = test_size
    
    # Phase 2: Collect sequences and snowflakes with sequence arms
    print("\nPhase 2: Filling splits with sequences and snowflakes with sequence arms...")
    
    # Collect sequences, validation and test data
    for i in range(max_iterations):
        if (len(train_sequences) >= n_train_needed and
            len(val_sequences) >= n_val_needed and
            len(test_sequences_unbiased) >= n_test_unbiased_needed and
            len(test_snowflakes_seq_arms_biased) >= n_test_biased_needed):
            break

            
        print(f"Batch {i+1}/{max_iterations} for remaining categories")
        batch = sampler.sample(batch_size)
        
        for sample in batch:
            description = sample["Description"]
            
            ## Train: fill with sequences
            if len(train_sequences) < n_train_needed and is_sequence(description):
                train_sequences.append(sample)
            # Validation: fill with sequences
            elif len(val_sequences) < n_val_needed and is_sequence(description):
                val_sequences.append(sample)
            # Test (unbiased): fill with sequences
            elif len(test_sequences_unbiased) < n_test_unbiased_needed and is_sequence(description):
                test_sequences_unbiased.append(sample)
            # Test (biased): fill with snowflakes with sequence arms
            elif len(test_snowflakes_seq_arms_biased) < n_test_biased_needed and is_snowflake_with_sequence_arm(description):
                test_snowflakes_seq_arms_biased.append(sample)

        print(f"Progress: Train (Sequences): {len(train_sequences)}/{n_train_needed}, "
              f"Val (Sequences): {len(val_sequences)}/{n_val_needed}, Test (Sequences, unbiased): {len(test_sequences_unbiased)}/{n_test_unbiased_needed}, Test (Snowflakes, biased): {len(test_snowflakes_seq_arms_biased)}/{n_test_biased_needed}")
    

    # Assemble splits
    train_data = train_single_snowflakes + train_sequences
    val_data = val_single_snowflakes + val_sequences
    test_data_unbiased = test_single_snowflakes + test_sequences_unbiased
    test_data_biased = test_snowflakes_seq_arms_biased

    # Shuffle for randomness
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data_unbiased)
    random.shuffle(test_data_biased)

    # Calculate percentages for reporting
    def report_split(split_name, data, seqs, snowflakes):
        total = len(data)
        seq_percent = len(seqs) / total * 100 if total > 0 else 0
        snow_percent = len(snowflakes) / total * 100 if total > 0 else 0
        print(f"- {split_name}: {total} samples")
        print(f"  - Sequences: {len(seqs)} ({seq_percent:.1f}%)")
        print(f"  - Snowflakes: {len(snowflakes)} ({snow_percent:.1f}%)")

    report_split("Train", train_data, train_sequences, train_single_snowflakes)
    report_split("Validation", val_data, val_sequences, val_single_snowflakes)
    report_split("Test (unbiased)", test_data_unbiased, test_sequences_unbiased, test_single_snowflakes)
    print(f"- Test (biased): {len(test_data_biased)} samples (all snowflakes with sequence arms)")


    return pd.DataFrame(train_data)[["Description", "Program"]], pd.DataFrame(val_data)[["Description", "Program"]], pd.DataFrame(test_data_unbiased)[["Description", "Program"]], pd.DataFrame(test_data_biased)[["Description", "Program"]]

def main():
    # Optional shape distribution analysis
    analyze_shape_distribution()

    # Generate the datasets
    train_size = 8000
    val_size = 1000
    test_size = 1000
    prioritize = "ratio" # Use "ratio" or "size" here
    
    train_df, val_df, test_unbiased_df, test_biased_df = generate_mix_match_dataset(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        prioritize=prioritize
    )
    
    # Inspect the datasets instead of saving raw data
    print("\n=== DATASET INSPECTION ===")
    print(f"\nTrain Dataset: {len(train_df)} samples")
    print("\nHead:")
    print(train_df[["Description", "Program"]].head(2))
    print("\nTail:")
    print(train_df[["Description", "Program"]].tail(2))
    
    print(f"\nValidation Dataset: {len(val_df)} samples")
    print("\nHead:")
    print(val_df[["Description", "Program"]].head(2))
    print("\nTail:")
    print(val_df[["Description", "Program"]].tail(2))
    
    print(f"\nTest Dataset (unbiased): {len(test_unbiased_df)} samples")
    print("\nHead:")
    print(test_unbiased_df[["Description", "Program"]].head(2))
    print("\nTail:")
    print(test_unbiased_df[["Description", "Program"]].tail(2))
        
    print(f"\nTest Dataset (biased): {len(test_biased_df)} samples")
    print("\nHead:")
    print(test_biased_df[["Description", "Program"]].head(2))
    print("\nTail:")
    print(test_biased_df[["Description", "Program"]].tail(2))

    # Analyze distribution of program types
    print("\n=== DATASET DISTRIBUTION ===")
    train_dist = analyze_dataset_distribution(train_df, "Train")
    val_dist = analyze_dataset_distribution(val_df, "Validation")
    test_dist_unbiased = analyze_dataset_distribution(test_unbiased_df, "Test")
    test_dist_biased = analyze_dataset_distribution(test_biased_df, "Test")

    
    # Combine into a single dataframe for better visualization
    all_dist = pd.concat([train_dist, val_dist, test_dist_unbiased], ignore_index=True)
    print(all_dist.to_string(index=False))

    all_dist = pd.concat([train_dist, val_dist, test_dist_biased], ignore_index=True)
    print(all_dist.to_string(index=False))

    # Generate images using ProgramParser
    print("\nGenerating images for datasets...")
    
    # Create a dataset dictionary
    from datasets import Dataset, DatasetDict
    dataset_dict_unbiased = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_unbiased_df, preserve_index=False)
    })
    
    dataset_dict_biased = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_biased_df, preserve_index=False)
    })

    # Create an instance of the parser
    parser = ProgramParser(save_dir="logo_graphic/mix_match_gen", save_image=True, eval_mode=False)

    push_to_hub = True
    hub_name = f"ruthchy/mix-match-gen-logo-data-{prioritize}-unbiased-test" 

    ds = parser.wrapper_parse_and_generate_image(
        dataset_dict_unbiased, 
        push_to_hub=push_to_hub, 
        hub_name=hub_name
    )

    hub_name = f"ruthchy/mix-match-gen-logo-data-{prioritize}" 

    ds = parser.wrapper_parse_and_generate_image(
        dataset_dict_biased, 
        push_to_hub=push_to_hub, 
        hub_name=hub_name
    )
    
    print("All images generated successfully!")
    if push_to_hub:
        print("Datasets pushed to Hugging Face Hub")

if __name__ == "__main__":
    main()