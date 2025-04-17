import os
import pandas as pd
import gc
from datasets import load_dataset, Dataset
from _1_logo_pseudo_code_generator import generateLOGOPseudoCode
from _2_sampler import LOGOProgramSampler
from _6_length import Length
from main_length_gen import generate_synthetic_data, generateLOGOPseudoCode

def generate_valid_synthetic_data(combined_dataset, threshold, required_samples):
    """
    Generate synthetic programs until the required number of valid samples is reached.

    Args:
        combined_dataset (DataFrame): Combined dataset (train, validation, test) to avoid duplicates.
        threshold (int): Maximum allowed length for programs in the new test split.
        required_samples (int): Number of valid samples needed.

    Returns:
        DataFrame: A DataFrame containing exactly the required number of valid samples.
    """
    generator = generateLOGOPseudoCode()
    sampler = LOGOProgramSampler(generator, combined_dataset)
    length_calculator = Length()
    valid_synthetic_data = []

    while len(valid_synthetic_data) < required_samples:
        # Generate a batch of synthetic programs
        batch_size = required_samples - len(valid_synthetic_data)
        synthetic_data = sampler.sample(n=batch_size)
        synthetic_data = pd.DataFrame(synthetic_data, columns=['Description', 'Program'])

        # Calculate lengths and filter programs below the threshold
        synthetic_data['Length'] = synthetic_data['Program'].apply(length_calculator.calc_length)
        valid_batch = synthetic_data[synthetic_data['Length'] <= threshold]
        print(f"Filtered down to {len(valid_batch)} valid programs...")

        # Add valid programs to the result
        valid_synthetic_data.extend(valid_batch.to_dict(orient="records"))

        print(f"Collected {len(valid_synthetic_data)} valid programs so far...")

    print(f"The longest program in the new test set is: {max([item['Length'] for item in valid_synthetic_data])}\nThreshold is: {threshold}")
    # Convert the valid programs back to a DataFrame
    return pd.DataFrame(valid_synthetic_data[:required_samples])


# MAIN
if __name__ == "__main__":
    # Step 1: Load the dataset
    dataset_name = "ruthchy/length-gen-logo-image"
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    # Step 2: Combine all splits (train, validation, test) into a single dataset
    print("Combining train, validation, and test splits into a single dataset...")
    combined_dataset = pd.concat(
        [ds['train'].to_pandas(), ds['validation'].to_pandas(), ds['test'].to_pandas()],
        ignore_index=True
    )
    print(f"Combined dataset size: {len(combined_dataset)}")

    # Step 3: Determine the threshold (shortest program length in the current test set)
    length_calculator = Length()
    ds['test'] = ds['test'].to_pandas()  # Convert to pandas DataFrame for easier manipulation
    ds['test']['Length'] = ds['test']['Program'].apply(length_calculator.calc_length)
    threshold = ds['test']['Length'].min()
    print(f"Threshold (shortest program length in test set): {threshold}")

    # Step 4: Generate exactly 997 valid synthetic programs
    required_samples = 997  # Desired number of new programs
    valid_synthetic_data = generate_valid_synthetic_data(combined_dataset, threshold, required_samples)

    print(f"Generated {len(valid_synthetic_data)} valid synthetic programs.")

    # Step 5: Remove the 'Length' column and add a Image column
    valid_synthetic_data = valid_synthetic_data.drop(columns=['Length'])

    # Create an instance of the parser
    from __parser_pyturtle_pc import ProgramParser 
    parser = ProgramParser(save_dir="logo_graphic/len_gen_dataset", save_image=True, eval_mode=False)

    # Replace the test split with the valid synthetic data
    ds['test'] = Dataset.from_pandas(valid_synthetic_data)

    print(f"Dataset Structure: {ds}") # missing image column shows that replacement was successful
    
    # remove unused data to free up memory
    del combined_dataset
    del valid_synthetic_data
    gc.collect()

    push_to_hub = True  # Change to True if you want to upload
    hub_name = "ruthchy/length-gen-logo-image-unbiased-test" 
    ds = parser.wrapper_parse_and_generate_image(ds, push_to_hub=push_to_hub, hub_name=hub_name)