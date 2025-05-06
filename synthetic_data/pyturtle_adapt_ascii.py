'''
This script is used to:
1. Create a dataset with an image column by replacing the "ASCII-Art" column with generated images.
2. Apply an ASCII transformation to the images in a dataset.

### How to Use:
#### 1. Create a Dataset with an Image Column:
- Set `create_dataset_with_image_column = True`.
- Update the following variables:
  - `dataset_name`: Name of the dataset to load (e.g., "ruthchy/syn-length-gen-logo-data-desc-ascii_35").
  - `hub_name`: Name of the dataset to push to the Hugging Face Hub (e.g., "ruthchy/syn-length-gen-logo-image").
  - `push_to_hub`: Set to `True` to upload the dataset to the Hub.
- Run the script:
  python synthetic_data/pyturtle_adapt_ascii.py

#### 2. Apply ASCII Transformation:
- Set apply_ascii_transformation = True.
- Update the following variables:
  - dataset_name: Name of the dataset with the image column (e.g., "ruthchy/syn-length-gen-logo-image").
- Optionally configure the AdaptiveASCIIProcessor:
  - levels: Range of black pixel levels (default: 10).
  - black_threshold: Pixel intensity threshold for black (default: 220).
  - block_size: Block size for ASCII conversion (optional).
  - crop_to_size: Crop image to this size (optional).
  - drop_images: Set to True to drop the image column after transformation.
- Run the script:
  python synthetic_data/pyturtle_adapt_ascii.py
#### 3. Notes:
- For testing, you can uncomment the line to process only a subset of the dataset: 
# ds['train'] = ds['train'].select(range(100))
'''
import os
from datasets import load_dataset, DatasetDict, Features, Image, Value
create_dataset_with_image_column = True
if create_dataset_with_image_column:
    from __parser_pyturtle_pc import ProgramParser  # Import the parser class

    # Load the dataset
    dataset_name = "ruthchy/syn-length-gen-logo-data-desc-ascii_35"
    ds = load_dataset(dataset_name)
    ds = ds.remove_columns("ASCII-Art")
    print(ds)
    # Select the first 100 rows of the train dataset for testing purposes
    #ds['train'] = ds['train'].select(range(100))

    # Create an instance of the parser
    parser = ProgramParser(save_dir="logo_graphic/len_gen_dataset", save_image=True, eval_mode=False)

    push_to_hub = True  # Change to True if you want to upload
    hub_name = "ruthchy/syn-length-gen-logo-image" 
    ds = parser.wrapper_parse_and_generate_image(ds, push_to_hub=push_to_hub, hub_name=hub_name)

#### Apply ASCII transformation to Image-column
apply_ascii_transformation = False
if apply_ascii_transformation:
    from __adapt_ascii_processor import AdaptiveASCIIProcessor

    dataset_name = "ruthchy/semantic-length-generalization-logo-image"
    ds = load_dataset(dataset_name)
    # select the first rows of the train dataset 
    ds = ds["train"].select(range(10))
    ascii_processor = AdaptiveASCIIProcessor(levels=10, black_threshold=220, block_size=32, crop_to_size=512, drop_images=True) # levels represents the range from 0 to 9; 0=no black pixels vs 9=all black pixels the threshold determines when a pixel is considered black; if a block_size can be given but is optional if it isn't the user has 2 mintues to choose a block size from the list of common divisors; crop_to_size this is an optional parameter to cropt the image around the center; last argument specifies if the image column is kept or dropped

    ds = ascii_processor.process_dataset(ds)

    # Print the dataset with ASCII art
    if isinstance(ds, DatasetDict):
        print(ds['train'][1])
    else:
        print(ds[1])
