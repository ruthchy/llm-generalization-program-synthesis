# Evaluating the Generalization Capabilities of Large Language Models in Program Synthesis
This repository accompanies my master thesis:  **Evaluating the Generalization Capabilities of Large Language Models in Program Synthesis**. The main goal is to assess how well a LLM (here a CodeLlama model) can solve programming-by-example (PBE) tasks, especially when the elements in the target LOGO graphic have not been encountered during training. 

## Overview
- **Exploration of the Domain:**
    The `programs_DSL.ipynb` notebook was used during the initial exploration of the domain. It includes experiments and visualizations related to the pseudo-LOGO Python code and the `PyTurtle` class.

- **Synthetic Data Generation:**
    The `synthetic_data` directory contains scripts to generate training, validation, and test datasets for LOGO tasks. Each task consists of a LOGO graphic from which the LLM must synthesize the corresponding LOGO-pseudo-python program.
    Datasets are also available on my HuggingFace (https://huggingface.co/ruthchy).

- **Pipeline and Evaluation:**
    The central script is `pipeline.py`, which orchestrates fine-tuning, inference, and evaluation. Supporting scripts and resources include:
    - `hp_optuna.py` for hyperparameter tuning.
    - `detailed_eval.py` for in-depth evaluation of program synthesis and image generation metrics.
    - `__LOGO_image.py` for visual inspection of generated and reference LOGO-images.

- **External Dependencies:**
    The `external` directory contains a submodule that is a fork of my supervisor's repository, which itself is a fork of the repository corresponding to the ReGAL paper:
    - **ReGAL: Refactoring Programs to Discover Generalizable Abstractions**  
      Stengel-Eskin, E., A. Prasad, and M. Bansal (2024, June).  
      [arXiv:2401.16467](https://arxiv.org/abs/2401.16467)
    - This submodule provides additional resources and dependencies used in the pipeline and evaluation processes. The most important resource used from the submodule is the `PyTurtle` class, which instantiates the pseudo-LOGO Python code for generating and visualizing LOGO graphics.

## Pipeline and Evaluation

### Pipeline Modes
The `pipeline.py` script can be executed in three modes:
1. **Fine-tuning:**
    - Trains the model on the synthetic dataset.
    - Example command:
      ```bash
      python pipeline.py --fine_tune --sample_fraction 1.0 --config config.yaml
      ```
2. **Inference:**
    - Runs inference using a fine-tuned model or a model from the hub.
    - Example command:
      ```bash
      python pipeline.py --inference_hub_model --sample_fraction 1.0 --config config.yaml
      ```
3. **Evaluation:**
    - Evaluates existing inference results.
    - Example command:
      ```bash
      python pipeline.py --eval_inf_dir <results_dir> --config <results_dir>/config.yaml
      ```

### Hyperparameter Tuning
- `hp_optuna.py` is used for hyperparameter tuning of the pipeline via Optuna.
    - The search algorithm used is **Bayesian Optimization**, implemented through Optuna's **TPE Sampler** (Tree-structured Parzen Estimator). This allows for efficient exploration of the hyperparameter space compared to traditional grid search.
    - `hyperparameter_grid.yaml` contains the hyperparameters and their corresponding values to explore.
    - Example batch script: `hp-job-pipeline.sh`.

### Few-Shot Example Selection (`look_up` Directory)
The `look_up` directory contains resources for selecting few-shot examples during inference. It includes:
- **Clustering Files:** Precomputed clusters of image-program pairs in the training, validation, and test datasets. These clusters are based on similarity in their descriptions, calculated using OpenAI's `ada-002` embeddings.
- **Workflow Script:** The `workflow.py` script generates hierarchical clusters to facilitate few-shot example selection.

These resources are used to improve inference performance by providing relevant examples to the model.

### In-depth Evaluation
- `detailed_eval.py` performs detailed analysis of program synthesis and image generation metrics for a given experiment.
    - Computes summary statistics overall and per LOGO-abstraction categories (e.g., Basic Shapes, Snowflake).
    - Generates visualizations (e.g., histograms, violin plots) for metric distributions.
    - Exports pre-defined metric behavior cases to an Excel file for further inspection.
    - Example command:
      ```bash
      python detailed_eval.py --eval_dir /path/to/eval_dir
      ```

### Visual Inspection
- `__LOGO_image.py` generates images for both the generated program (completion) and the ground truth (reference) program from a predictions file.
    - Example command:
      ```bash
      python __LOGO_image.py --eval_dir ./eval_results --ID 0_1
      ```

## Quickstart

1. **Install dependencies:**
   ```bash
   conda create -n thesis_env 
   conda activate thesis_env
   pip install -r requirements.txt
   ```
2. **Run fine-tuning:**
   ```bash
   python pipeline.py --fine_tune --sample_fraction 1.0 --config config.yaml
   ```
3. **Run inference with a model from the hub:**
   ```bash
   python pipeline.py --inference_hub_model --sample_fraction 1.0 --config config.yaml
   ```
4. **Evaluate existing inference results:**
   ```bash
   python pipeline.py --eval_inf_dir <results_dir> --config <results_dir>/config.yaml
   ```
5. **In-depth Evaluation:**
   ```bash
   python detailed_eval.py --eval_dir /path/to/eval_dir
   ```
6. **Visual inspection of generated and reference LOGO-images:**
   ```bash
   python __LOGO_image.py --eval_dir ./eval_results --ID 0_1
   ```

## Inputs & Outputs

### Pipeline Inputs
- `config.yaml`: Main configuration file for pipeline and training.
- Synthetic datasets (from HuggingFace or generated locally).
- Optionally, clustering files in `look_up/` for few-shot selection.

### Pipeline Outputs
- Results and metrics in the `results/` directory.
- Model checkpoints and logs.
- Evaluation summaries (`evaluation.json`), detailed metrics (`detailed_metrics.jsonl`), and predictions (`predictions.json`).

### Evaluation Inputs
- `detailed_metrics.jsonl`, `predictions.json`, and `config.yaml` in the specified evaluation directory.

### Evaluation Outputs
- Summary statistics and plots in the evaluation directory.
- Excel file with metric behavior cases.
- JSON files mapping abstractions to IDs.
- Text report with all results and comparisons.

## Datasets

- [Synthetic LOGO datasets on HuggingFace](https://huggingface.co/ruthchy)

## Example Results

See the `results/` directory for example outputs and evaluation summaries.