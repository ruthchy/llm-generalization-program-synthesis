# Synthetic Length Generalization Dataset

This repository contains scripts to generate synthetic datasets for different generalization criteria. These datasets are used to evaluate model capabilities for length and mix-match generalization tasks.

## Structure

- **Main Generation Scripts**
  - `main_length_gen.py`: Generates and splits datasets based on length generalization criteria (syntactic or semantic). Creates data with Description, Program, and ASCII-Art columns.
  - `main_mix_match_gen.py`: Generates and splits datasets based on mix-match generalization criteria. Creates data with Description, Program, and Image columns.

- **Core Components**
  - `_1_logo_pseudo_code_generator.py`: Generates LOGO pseudocode for various shapes and patterns.
  - `_2_sampler.py`: Samples LOGO programs from the generator.
  - `_3_executable_logo_primitives.py`: Contains executable LOGO primitives.
  - `_4_logo_graphic_generator_v1.py`: First version of the graphic generator.
  - `_4_logo_graphic_generator_v2.py`: Second version of the graphic generator.
  - `_5_ascii_processor.py`: Processes images into ASCII art representations.
  - `_6_semantic_length.py`: Calculates semantic length (execution time) of programs.
  - `_6_syntactic_length.py`: Calculates syntactic length of programs.

- **Utility Scripts**
  - `pyturtle_adapt_ascii.py`: Transforms datasets with ASCII-Art columns to include Image columns instead.
  - `__parser_pyturtle_pc.py`: Parser for working with PyTurtle for image generation.
  - `__adapt_ascii_processor.py`: Applies adaptive ASCII transformation.
  - `__dataset_direction_modifier.py`: Modifies a percentage of programs to use right() instead of left() (or vice versa).

- **Archived**
  - `main.ipynb`: The original notebook for dataset generation (now archived).
  - `forkstate_test.ipynb` and `synthetic_data_forkstate_test.ipynb`: Experimental notebooks for the forkstate approach.

## Usage


```bash
# Generate a dataset split by syntactic length
python synthetic_data/main_length_gen.py \
    --generate-synthetic \
    --target-size 10000 \
    --generate-graphics \
    --interpreter-version 1 \
    --process-ascii \
    --blocks 35 \
    --split-by syntactic \
    --save-hf

# Using existing synthetic data and splitting by semantic length
python synthetic_data/main_length_gen.py \
    --synthetic-path synthetic_data/data/my_synthetic_data.jsonl \  # optional: if not set default synthetic data (synthetic_data_20250120143151.jsonl) is used 
    --process-ascii \
    --split-by semantic \
    --save-hf

# Generate the dataset by mix and match generalization criterion
python synthetic_data/main_mix_match_gen.py