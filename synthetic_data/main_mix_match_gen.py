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
                (may result in smaller dataset if snowflakes are limited)
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
train_df, val_df, test_df = generate_mix_match_dataset(
    train_size=10000,
    val_size=1000,
    test_size=1000,
    prioritize="size"
'''

import pandas as pd
import random
import re
from _1_logo_pseudo_code_generator import generateLOGOPseudoCode
from _2_sampler import LOGOProgramSampler
from __parser_pyturtle_pc import ProgramParser
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
    
    # Split training data into two categories
    train_sequences = []
    train_single_snowflakes = []
    val_data = []
    test_data = []
    
    # Calculate target counts for balanced training data
    target_train_sequences = train_size // 2
    target_train_snowflakes = train_size // 2
    
    total_samples = 0
    total_needed = train_size + val_size + test_size
    
    print(f"Generating {total_needed} samples (prioritizing {'ratio' if prioritize=='ratio' else 'total size'})...")
    
    # Phase 1: Focus on collecting the rare snowflakes with single shape arms
    print("Phase 1: Collecting rare snowflakes with single shape arms...")
    collected_all_snowflakes = False
    
    for i in range(max_iterations):
        if len(train_single_snowflakes) >= target_train_snowflakes:
            break
            
        print(f"Batch {i+1}/{max_iterations} for single shape snowflakes")
        batch = sampler.sample(batch_size)
        total_samples += batch_size
        
        for sample in batch:
            description = sample["Description"]
            
            # Focus only on collecting single shape arm snowflakes
            if len(train_single_snowflakes) < target_train_snowflakes and is_snowflake_with_single_shape_arm(description):
                train_single_snowflakes.append(sample)
        
        print(f"Progress: Train (Snowflakes): {len(train_single_snowflakes)}/{target_train_snowflakes}")
        
        # Check if we've collected all available snowflakes
        if i >= 2 and len(train_single_snowflakes) == 180:
            print("Collected all possible snowflakes with single shape arms (180)")
            collected_all_snowflakes = True
            break
    
    # Phase 2: Collect sequences and snowflakes with sequence arms
    print("\nPhase 2: Collecting sequences and snowflakes with sequence arms...")
    
    # Adjust sequence target based on prioritization mode
    if prioritize == "ratio" and len(train_single_snowflakes) < target_train_snowflakes:
        # Keep balanced ratio by limiting sequences to match snowflakes
        actual_target_sequences = len(train_single_snowflakes)
        print(f"Prioritizing ratio: Limiting sequences to {actual_target_sequences} to match snowflakes")
    else:
        # Prioritize size - fill the rest with sequences
        actual_target_sequences = target_train_sequences + (target_train_snowflakes - len(train_single_snowflakes))
        print(f"Prioritizing size: Collecting {actual_target_sequences} sequences to reach target size")
    
    # Collect sequences, validation and test data
    for i in range(max_iterations):
        if (len(train_sequences) >= actual_target_sequences and 
            len(val_data) >= val_size and 
            len(test_data) >= test_size):
            break
            
        print(f"Batch {i+1}/{max_iterations} for remaining categories")
        batch = sampler.sample(batch_size)
        total_samples += batch_size
        
        for sample in batch:
            description = sample["Description"]
            
            # Sort into appropriate categories
            if len(train_sequences) < actual_target_sequences and is_sequence(description):
                train_sequences.append(sample)
            elif len(val_data) < val_size and is_snowflake_with_sequence_arm(description):
                val_data.append(sample)
            elif len(test_data) < test_size and is_snowflake_with_sequence_arm(description):
                test_data.append(sample)
        
        print(f"Progress: Train (Sequences): {len(train_sequences)}/{actual_target_sequences}, "
              f"Val: {len(val_data)}/{val_size}, Test: {len(test_data)}/{test_size}")
    
    # Report status
    categories_complete = []
    if len(train_sequences) >= actual_target_sequences:
        categories_complete.append("Train sequences")
    if collected_all_snowflakes or len(train_single_snowflakes) >= target_train_snowflakes:
        categories_complete.append("Train snowflakes")
    if len(val_data) >= val_size:
        categories_complete.append("Validation data")
    if len(test_data) >= test_size:
        categories_complete.append("Test data")
    
    if len(categories_complete) < 4:
        print(f"\nWARNING: Only completed these categories: {', '.join(categories_complete)}")
    else:
        print("\nAll categories filled successfully!")
    
    # Combine training data
    train_data = train_sequences + train_single_snowflakes
    random.shuffle(train_data)  # Shuffle to mix the two categories
    
    # Calculate percentages for reporting
    total_train = len(train_data)
    seq_percent = len(train_sequences) / total_train * 100 if total_train > 0 else 0
    snow_percent = len(train_single_snowflakes) / total_train * 100 if total_train > 0 else 0
    
    print(f"\nFinal dataset sizes:")
    print(f"- Train: {total_train} samples")
    print(f"  - Sequences: {len(train_sequences)} ({seq_percent:.1f}%)")
    print(f"  - Snowflakes: {len(train_single_snowflakes)} ({snow_percent:.1f}%)")
    print(f"- Validation: {len(val_data)} samples")
    print(f"- Test: {len(test_data)} samples")
    
    if prioritize == "size" and collected_all_snowflakes:
        print(f"\nNote: Used all available snowflakes with single shape arms ({len(train_single_snowflakes)}/180)")
        print(f"Filled the rest with sequences to reach target size")
    
    return pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

def main():
    # Generate the datasets
    train_size = 8000
    val_size = 1000
    test_size = 1000
    
    train_df, val_df, test_df = generate_mix_match_dataset(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        prioritize="size"  # Use "ratio" or "size" here
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
    
    print(f"\nTest Dataset: {len(test_df)} samples")
    print("\nHead:")
    print(test_df[["Description", "Program"]].head(2))
    print("\nTail:")
    print(test_df[["Description", "Program"]].tail(2))

    # Analyze distribution of program types
    print("\n=== DATASET DISTRIBUTION ===")
    train_dist = analyze_dataset_distribution(train_df, "Train")
    val_dist = analyze_dataset_distribution(val_df, "Validation")
    test_dist = analyze_dataset_distribution(test_df, "Test")
    
    # Combine into a single dataframe for better visualization
    all_dist = pd.concat([train_dist, val_dist, test_dist], ignore_index=True)
    print(all_dist.to_string(index=False))
    
    # Generate images using ProgramParser
    print("\nGenerating images for datasets...")
    
    # Create a dataset dictionary
    from datasets import Dataset, DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })
    
    # Create an instance of the parser
    parser = ProgramParser(save_dir="logo_graphic/mix_match", save_image=True, eval_mode=False)
        
    # Generate images for all splits
    dataset_with_images = parser.wrapper_parse_and_generate_image(dataset_dict)
    
    print("All images generated successfully!")
    
    # Push to Hugging Face Hub if needed
    push_to_hub = False  # Set to True to push to hub
    if push_to_hub:
        dataset_with_images.push_to_hub("ruthchy/mix-match-generalization-logo-data")
        print("Dataset pushed to Hugging Face Hub")

if __name__ == "__main__":
    main()