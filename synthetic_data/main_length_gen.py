'''
This script generates synthetic data, joins it with the existing data used in the ReGAL-paper, processes it, and saves it to the Hugging Face Hub.
The script performs the following steps:
    Before starting with the actual steps the path function is used to dynamically set the working directory to the base (master-thesis) of the project. Ensuring that no matter from where the scirpt will be executed the paths will work.
2. Load and preprocess data using the ReGAL paper.
    Before continuninthen the working directory is changed to the a child of the base directory set initally (master-thesis/synthetic_data) 
3. Generate synthetic data (optional): (for 10000 samples about 30 min)
    --generate-synthetic: Flag to generate synthetic data.
    --target-size: Total desired dataset size (ReGAL + synthetic). The script calculates how many synthetic samples to generate.
    --synthetic-path: Path to an existing synthetic dataset file. Used when --generate-synthetic is not set.
4. Generate graphics (optional): 
    --generate-graphics: Flag to generate graphics for both datasets.
    --interpreter-version: Interpreter version (1 or 2, default is 1).
5. Process ASCII (optional): (for 10000 samples under 15 min)
    --process-ascii: Flag to process ASCII data for both datasets.
    --blocks: Number of blocks for ASCII processing (default is 35).
6. Split data by syntactic or semantic length (optional):
    --split-by: "syntactic" or "semantic" to specify the type of split.
7. Save data to Hugging Face Hub (optional):
    --save-hf: Flag to save data to Hugging Face Hub.

Note: 
    - If the generate-synthetic flag isn't set the script will load synthetic data from the path specified by --synthetic-path,
      or fall back to "data/synthetic_data_20250120143151.jsonl" if no path is provided.
    - If the generate-graphics flag isn't set but the process-ascii flag is set, the script will use images generated in an earlier run.
    - When --target-size is provided with --generate-synthetic, the script calculates how many synthetic samples to generate by subtracting the number of ReGAL samples (after duplicates have been removed) from the target size. Default target-size is a total of 10000 samples.

Usage:
    python synthetic_data/main_length_gen.py \
        --generate-synthetic \
        --target-size 15000 \
        --generate-graphics \
            --interpreter-version 1 \
        --process-ascii \
            --blocks 35 \
        --split-by syntactic \
        --save-hf

    # Or use existing synthetic data:
    python synthetic_data/main_length_gen.py \
        --synthetic-path synthetic_data/data/my_synthetic_data.jsonl \
        --process-ascii \
        --split-by semantic \
        --save-hf
'''
import argparse
import json
import pandas as pd
import os
from _1_logo_pseudo_code_generator import generateLOGOPseudoCode
from _2_sampler import LOGOProgramSampler
from _3_executable_logo_primitives import ReGALLOGOPrimitives
from _4_logo_graphic_generator_v1 import PseudoProgramInterpreter as PseudoProgramInterpreter_v1
from _4_logo_graphic_generator_v2 import PseudoProgramInterpreter as PseudoProgramInterpreter_v2
from _5_ascii_processor import ASCIIProcessor
from _6_syntactic_length import SyntacticLength
from _6_semantic_length import ExecutionTimeLength
from datasets import Dataset, DatasetDict

# Global variable for blocks
BLOCKS = 35
def path():
    # Get current working directory
    current_dir = os.getcwd()
    # Find the index of "master-thesis"
    split_point = current_dir.find("master-thesis")
    if split_point != -1:
        # Extract the base path up to "master-thesis"
        base_dir = current_dir[: split_point + len("master-thesis")]
        # Change the working directory to the base path
        os.chdir(base_dir)
        print(f"Working directory set to: {os.getcwd()}")
    else:
        print("Error: 'master-thesis' not found in path.")


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def extract_descriptions_and_programs(data):
    extracted_data = []
    for item in data:
        description = None
        program = None
        if "messages" in item:  # suits the format for train_data and test_data
            for message in item.get('messages', []):
                if message['from'] == 'human':
                    description = message['value']
                elif message['from'] == 'gpt':
                    program = message['value']
        elif "program" in item and "language" in item: # suites the format for dev_data
            program = item['program']
            description = " ".join(item['language'])

        if description and program:
            extracted_data.append([description, program])
    extracted_data = pd.DataFrame(extracted_data, columns=['Description', 'Program'])
    return extracted_data

def combine_and_deduplicate(train_path, dev_path, test_path):
    # Load data from the provided paths
    df_train = extract_descriptions_and_programs(load_data(f"external/dependencies/{train_path}"))
    df_dev = extract_descriptions_and_programs(load_data(f"external/dependencies/{dev_path}"))
    df_test = extract_descriptions_and_programs(load_data(f"external/dependencies/{test_path}"))

    print("df_train shape: ", df_train.shape)
    print("df_dev shape: ", df_dev.shape)
    print("df_test shape: ", df_test.shape)

    # Combine the datasets
    df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)
    print("Dimensions of the combined train, dev, and test data: ", df_all.shape)
    
    # Remove duplicates based on the 'Program' column
    if 'Program' not in df_all.columns:
        raise KeyError("The 'Program' column is missing in the combined DataFrame. Check the input data structure.")
    df_all = df_all.drop_duplicates(subset=['Program'])
    print("Dimensions of the combined data without duplicate Programs: ", df_all.shape)
    
    return df_all

def generate_synthetic_data(generator, df_all, sample_size=9643):
    sampler = LOGOProgramSampler(generator, df_all)
    synthetic_data = sampler.sample(sample_size)
    synthetic_data = pd.DataFrame(synthetic_data, columns=['Description', 'Program'])

    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    output_path = f"synthetic_data/data/synthetic_data_{timestamp}.jsonl"
    synthetic_data.to_json(output_path, orient="records", lines=True)
    print(f"Synthetic data saved to: {output_path}")
    
    return synthetic_data

def generate_graphics(data, output_dir, interpreter_version=1):
    if interpreter_version == 1:
        interpreter = PseudoProgramInterpreter_v1()
    else:
        interpreter = PseudoProgramInterpreter_v2()
    interpreter.process_and_save_graphics(data, output_dir=output_dir)

def process_ascii(data, output_dir):
    dir_images = f"{output_dir}/"
    processor = ASCIIProcessor(n_blocks=BLOCKS, m_blocks=BLOCKS, levels=10)
    ascii_data = processor.store_ascii_input(data, dir_images)
    return ascii_data

def split_data_by_length(df_all_syn, split_by):
    """
    Split data based on specified length metric (syntactic or semantic).
    
    Args:
        df_all_syn (DataFrame): Combined dataset to split
        split_by (str): "syntactic" or "semantic" to specify the type of split
        
    Returns:
        tuple: (train_data, validation_data, test_data) DataFrames
    """
    # Apply appropriate length calculation based on split type
    if split_by == "syntactic":
        print("Calculating syntactic lengths...")
        syn_length = SyntacticLength()
        df_all_syn['Length'] = df_all_syn['Program'].apply(syn_length.calc_syntactic_length)
        length_column = 'Length'
    elif split_by == "semantic":
        print("Calculating semantic lengths...")
        sem_length = ExecutionTimeLength(timeout=5, num_runs=1)
        df_all_syn['Length'] = df_all_syn['Program'].apply(sem_length.calc_execution_time)
        length_column = 'Length'
    else:
        raise ValueError(f"Invalid split_by value: {split_by}. Must be 'syntactic' or 'semantic'.")
    
    # Sort by the calculated length
    df_all_syn = df_all_syn.sort_values(by=length_column).reset_index(drop=True)
    
    # Determine the split index (10% for test)
    test_start_id = int(len(df_all_syn) * 0.9)
    # Ensure we don't split in the middle of examples with the same length
    while (
        test_start_id < len(df_all_syn) and 
        df_all_syn.loc[test_start_id, length_column] == df_all_syn.loc[test_start_id - 1, length_column]
    ):
        test_start_id += 1
    
    # Create train and test splits
    train_data = df_all_syn.iloc[:test_start_id]
    test_data = df_all_syn.iloc[test_start_id:]
    
    # Create a validation dataset from the training data with the same size as the test set
    rs = 42  # Fixed random seed for reproducibility
    length = int(len(test_data))
    validation_data = train_data.sample(n=length, random_state=rs)  
    train_data = train_data.drop(validation_data.index)
    
    print(f"Split dataset: \nTrain={len(train_data)}\nValidation={len(validation_data)}\nTest={len(test_data)}")
    return train_data, validation_data, test_data

def save_to_hf_hub(train_data, validation_data, test_data, args):
    # Ensure all datasets have required columns
    for df in [train_data, validation_data, test_data]:
        if 'ASCII-Art' not in df.columns:
            print("Warning: ASCII-Art column not found, adding empty column")
            df['ASCII-Art'] = ""
        
        # Fill any null values with empty strings
        for col in ['Description', 'ASCII-Art', 'Program']:
            df[col] = df[col].fillna("")
    
    def create_dataset(df):
        return Dataset.from_pandas(df[['Description', 'ASCII-Art', 'Program']], preserve_index=False)

    dataset_dict = DatasetDict({
        split: create_dataset(df)  
        for split, df in zip(["train", "validation", "test"], [train_data, validation_data, test_data])
    })
    
    dataset_dict.push_to_hub(f"ruthchy/{args.split_by}-length-gen-logo-data-desc-ascii_{BLOCKS}", private=False)

def main(args):
    path()  # Set the working directory before anything else to master-thesis directory
    print(os.getcwd())
    # Paths to the ReGAL data
    train_path = "logo_data/python/train_200_dataset.jsonl"
    dev_path = "logo_data/python/dev_100.jsonl"
    test_path = "logo_data/python/test_dataset.jsonl"

    # Load and preprocess ReGAL data
    df_all = combine_and_deduplicate(train_path, dev_path, test_path)

    # Change the working directory to the synthetic data directory
    #current_dir = os.getcwd()
    #base_dir = f"{current_dir}/synthetic_data"
    #os.chdir(base_dir)

    # Handle synthetic data
    if args.generate_synthetic:
        print("Generating synthetic data...")
        generator = generateLOGOPseudoCode()
        
        # Calculate sample size based on target size if provided
        if args.target_size:
            regal_count = len(df_all)
            sample_size = max(0, args.target_size - regal_count)
            print(f"Target size: {args.target_size}, ReGAL data count: {regal_count}")
            print(f"Will generate {sample_size} synthetic samples to reach target")
        else:
            sample_size = 9643  # Default sample size
            print(f"Generating default {sample_size} synthetic samples...")
            
        synthetic_data = generate_synthetic_data(generator, df_all, sample_size=sample_size)
    else:
        # Use the specified path if provided, otherwise use the default
        synthetic_file_path = args.synthetic_path if args.synthetic_path else "synthetic_data/data/synthetic_data_20250120143151.jsonl"
        print(f"Loading synthetic data from: {synthetic_file_path}")
        synthetic_data = pd.read_json(synthetic_file_path, orient="records", lines=True)
    
    # Optional: Generate graphics
    if args.generate_graphics:
        print("Generating graphics...")
        regal_output_dir = "synthetic_data/logo_graphic/all_ReGAL"
        synthetic_output_dir = f"synthetic_data/logo_graphic/synthetic_v{args.interpreter_version}"
        # Generate images for both datasets
        generate_graphics(df_all, regal_output_dir, args.interpreter_version)
        generate_graphics(synthetic_data, synthetic_output_dir, args.interpreter_version)

    # Optional: Process ASCII
    if args.process_ascii:
        print("Processing ASCII data...")
        # Generate ASCII representations fo the images
        df_all = process_ascii(df_all, "synthetic_data/logo_graphic/all_ReGAL")
        synthetic_data = process_ascii(synthetic_data, f"synthetic_data/logo_graphic/synthetic_v{args.interpreter_version}")

    # Join datasets
    df_all_syn = pd.concat([df_all, synthetic_data], ignore_index=True)

    # Optional: Split data
    if args.split_by:
        print(f"Splitting data by {args.split_by} length...")
        train_data, validation_data, test_data = split_data_by_length(df_all_syn, args.split_by)
        if args.save_hf:
            save_to_hf_hub(train_data, validation_data, test_data, args)  # Remember to pass args
            print("Data saved to Hugging Face Hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data and process it.")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic data.")
    parser.add_argument("--synthetic-path", type=str, help="Path to load previously generated synthetic data from.")
    parser.add_argument("--target-size", type=int, help="Target size for synthetic data generation.")
    parser.add_argument("--generate-graphics", action="store_true", help="Generate graphics for both datasets.")
    parser.add_argument("--interpreter-version", type=int, default=1, help="Interpreter version (1 or 2).")
    parser.add_argument("--process-ascii", action="store_true", help="Process ASCII data for both datasets.")
    parser.add_argument("--blocks", type=int, default=BLOCKS, help="Number of blocks for ASCII processing.")
    parser.add_argument("--split-by", choices=["syntactic", "semantic"], help="Split data by syntactic or semantic length.")
    parser.add_argument("--save-hf", action="store_true", help="Save data to Hugging Face Hub.")
    args = parser.parse_args()
    BLOCKS = args.blocks  # Update global BLOCKS variable if specified
    main(args)