import os
from datasets import load_dataset, DatasetDict, Features, Image, Value
create_dataset_with_image_column = False
if create_dataset_with_image_column:
    from __parser_pyturtle_pc import ProgramParser  # Import the parser class

    # Load the dataset
    dataset_name = "ruthchy/semantic-length-generalization-logo-data-desc-ascii_35"
    ds = load_dataset(dataset_name)
    ds = ds.remove_columns("ASCII-Art")
    print(ds)
    # Select the first 100 rows of the train dataset for testing purposes
    #ds['train'] = ds['train'].select(range(100))

    # Create an instance of the parser
    parser = ProgramParser(save_dir="logo_graphic/len_gen_dataset", save_image=True, eval_mode=False)

    push_to_hub = True  # Change to True if you want to upload
    hub_name = "ruthchy/semantic-length-generalization-logo-image" 
    ds = parser.wrapper_parse_and_generate_image(ds, push_to_hub=push_to_hub, hub_name=hub_name)

#### Apply ASCII transformation to Image-column
apply_ascii_transformation = True
if apply_ascii_transformation:
    from __adapt_ascii_processor import AdaptiveASCIIProcessor

    dataset_name = "ruthchy/semantic-length-generalization-logo-image"
    ds = load_dataset(dataset_name)
    # select the first rows of the train dataset 
    ds = ds["train"].select(range(10))
    ascii_processor = AdaptiveASCIIProcessor(levels=10, black_threshold=150) # levels represents the range from 0 to 9; 0=no black pixels vs 9=all black pixels the threshold determines when a pixel is considered black

    ds = ascii_processor.process_dataset(ds)

    # Print the dataset with ASCII art
    if isinstance(ds, DatasetDict):
        print(ds['train'][1])
    else:
        print(ds[1])
