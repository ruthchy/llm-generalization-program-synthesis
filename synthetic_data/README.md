# Synthetic Length Generalization Dataset

This repository contains scripts and notebooks to generate a synthetic length generalization dataset. The main notebook, `main.ipynb`, orchestrates the generation process using various Python scripts.

## Structure

- **Main Notebook**
  - `main.ipynb`: This notebook is used to generate the synthetic length generalization dataset using all the Python scripts starting with `_number_name.py`.

- **Python Scripts**
  - `_1_logo_pseudo_code_generator.py`
  - `_2_sampler.py`
  - `_3_executable_logo_primitives.py`
  - `_4_logo_graphic_generator_v1.py`
  - `_4_logo_graphic_generator_v2.py`
  - `_5_ascii_processor.py`
  - `_6_semantic_length.py`
  - `_6_syntactic_length.py`

  These scripts are used in conjunction with `main.ipynb` to generate the dataset.

- **Additional Scripts**
  - `pyturtle_adapt_ascii.py`: This script uses `__parser_pyturtle_pc.py` to add an Image column to the synthetic dataset or this `__adapt_ascii_processor.py` to apply an adaptive ASCII transformation to the Image column.
  - `__dataset_direction_modifier.py`: this script contains a method which can be used to modify XX% of the Programs contained in a the datasets to use the direction right() instead of left() (vice versa). 

- **Forkstate Scripts**
  - `forkstate_test.ipynb`
  - `synthetic_data_forkstate_test.ipynb`

  These scripts are not yet integrated into the any workflow. They might serve as a starting point to replace the `embed()` function from the ReGAL paper with `forkstate()` form the PBE paper.

## Usage

1. **Generate Dataset**
   - Run `main.ipynb` to generate the synthetic length generalization dataset.

2. **Add Image Column or Apply ASCII Transformation**
   - Use `pyturtle_adapt_ascii.py` to add an Image column to the dataset or apply an adaptive ASCII transformation.

3. **Explore Forkstate Integration**
   - Investigate the `forkstate_test.ipynb` and `synthetic_data_forkstate_test.ipynb` notebooks for potential integration of the `forkstate()` function.

## Requirements

- activate the conda ReGAL_env
