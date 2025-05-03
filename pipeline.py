'''
Steps:
    1. Load the YAML Configuration
    2. Load the Model and Tokenizer
    3. Prepare the Dataset
    4. Wandb initalization for the fine-tuning run 
    5. Training Preperation
    6. Training the model with custom PLW-Trainer
    7. Preparing Inference
    8.1 Inference using the recently fine-tuned model with zero-shot prompting and one answer per-prompt (num_return_sequences == 0) only
    8.2 Inference using model from hub with zero-shot (topk_prompt == 0) or few-shot prompting (topk_prompt > 0) and with a search budget > 1 (num_return_sequences > 1)
    9. Evaluation

Note: 
- The function inference_from_hub() is used to run inference with a model from the hub. Beside this it has two functionalities which are not implemented in the inference(): 
    - It can handle both zero-shot and few-shot prompting, depending on the value of topk_prompt in the config.yaml file.
    - Also, it can handle if the num_return_sequences > 1, meaning the llm generates n completions to each given prompt.


Run script in conda thesis_env (can be gererated using the requirements.txt file)
To fine-tune the model, conduct inference and evaluate the results, run the following command:
    python pipeline.py  --fine_tune  --sample_fraction 1.0  --config config.yaml
To run inference with a model from the hub, use:
    python pipeline.py  --inference_hub_model  --sample_fraction 1.0  --config config.yaml
To only run the evaluation on already generated inference results, use:
    python pipeline.py  --eval_inf_dir <directory>  --config <directory>/config.yaml

python pipeline.py 
    --fine_tune       # if the flag is set the model will be fine-tune the model and then do inf and eval if it loads a model and starts with inf followed by eval
    --inference_hub_model  # if the flag is set the model will load a model from the hub and do inference and eval
    --eval_inf_dir <directory>  # non of the above flags is set and the directory to already generated inference results is given to evaluate
    --config <directory>config.yaml  # Path to config file
    --sample_fraction 1.0   # the entire dataset is used or by setting the number < 1 a random sample of the dataset is used
'''
############################################################################################################
# Housekeeping - single GPU unsloth setup
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # new
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:64,expandable_segments:True" # ensure that the GPU memory is not fragmented and that PyTorch is less conservative with memory allocation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # ensures deterministic behavior

# Decide the mode in which the script should run fine-tuning and/or inference; in testing or production mode
import argparse
parser = argparse.ArgumentParser(description='Run fine-tuning and/or inference pipeline')
parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the model')
parser.add_argument('--inference_hub_model', action='store_true', help='Whether to run inference with a model from the hub')
parser.add_argument('--eval_inf_dir', type=str, help='Whether to only run the evaluation on inference results. Provide the directory where the predictions.json file is located')
parser.add_argument('--sample_fraction', type=float, default=1.0, help='Fraction of dataset to use (for debugging)')
parser.add_argument('--config', type=str, default="config.yaml", help='Path to config file')
parser.add_argument('--wb_type', type=str, help='Added to the inference run name to distinguish between different runs')
parser.add_argument('--model_for_parsing', type=str, default=None, help='Model to use for parsing during evaluation. Defaults to None.')
args = parser.parse_args()
        
fine_tune = args.fine_tune
inference_hub_model = args.inference_hub_model
inf_dir = args.eval_inf_dir
sample_fraction = args.sample_fraction
config_file = args.config
wb_type = args.wb_type
model_for_parsing = args.model_for_parsing


# Ensure mutual exclusivity of fine_tune, inference_hub_model, and eval_inf_dir
if sum([fine_tune, inference_hub_model, bool(inf_dir)]) != 1:
    raise ValueError("You must specify exactly one of the following: --fine_tune, --inference_hub_model, or --eval_inf_dir.")

if fine_tune:
    print(f"⚙️ Fine-tuning model with sample_fraction={sample_fraction} and {config_file}")
elif inference_hub_model:
    print(f"⚙️ Running inference with model from hub with sample_fraction={sample_fraction} and {config_file}")
else:
    print(f"⚙️ Running the evaluation on inference results in {inf_dir} and {config_file}")

# Step 1: Load the YAML Configuration
# Standard library imports
import traceback
import gc
import hashlib # new to create the program-id while logging the detailed metrics during fine-tuning
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
# Third-party imports
import yaml
import torch
import wandb
import numpy as np
from torch.cuda import memory_summary, reset_peak_memory_stats
#from torch.amp import autocast 
#torch.set_default_dtype(torch.float16)
from torch.nn import CrossEntropyLoss
from Levenshtein import distance as levenshtein_distance
from datasets import load_dataset, DatasetDict
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported, UnslothTrainer
    print("Using unsloth library.")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Unsloth not installed. Using transformers library instead.")
from transformers import (
    TrainingArguments, 
    Trainer, 
    get_scheduler, 
    EarlyStoppingCallback, 
    TrainerCallback, 
    TrainerState, 
    TrainerControl
)
from transformers.trainer_utils import EvalPrediction
from bitsandbytes.optim import AdamW8bit
#from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Custom imports
from __eval import LLMCodeEvaluator
# Initialize the evaluator
evaluator = LLMCodeEvaluator(model_for_parsing=model_for_parsing)

dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16 #torch.float16
print(f"⚙️ Using dtype: {dtype}")

@dataclass
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: List[str]

@dataclass
class TrainingConfig:
    prompt_loss_weight: float
    max_seq_length: int
    learning_rate: float
    lr_scheduler_type: str
    train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    random_seed: int
    shuffle: bool
    gradient_checkpointing: bool
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None

@dataclass
class ModelConfig:
    model_id: str
    topk_train: int
    topk_prompt: int
    num_return_sequences: int
    top_k: int
    temperature: float
    max_new_tokens: int

@dataclass
class LoggingConfig:
    use_wandb: bool

@dataclass
class DataConfig:
    dataset_id: str
    use_forkstate: bool
    include_desc: bool
    include_ascii: bool
    mix_directions: bool
    image_to_ascii: bool
    ascii_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptConfig:
    include_sys_prompt_fn: bool = False
    include_sys_prompt_inf: bool = False
    _system_prompt: str = """Your task is to draw simple black and white graphics with the custom library. DO NOT USE THE BUILT-IN TURTLE LIBRARY.
You will use a custom turtle library, similar to the built-in library, which is sufficient for all tasks.

Here are all the available functions in the custom turtle library:
- forward(x): move forward x pixels
- left(theta): rotate left by theta degrees
- right(theta): rotate right by theta degrees
- penup(): stop drawing
- pendown(): start drawing
- teleport(x, y, theta): move to position (x, y) with angle theta
- heading(): get the current angle of the turtle
- isdown(): check if the pen is down
- embed(program, local vars): runs the code in program using the current context and teleports back to the original position. Allows you to nest programs. Implementationally, embed gets the turtle state (is down, x, y, heading), executes program, then returns to the original state."""
    
    @property
    def system_prompt(self) -> str:
        """Property for backward compatibility"""
        return self._system_prompt
    
    def get_prompt_template(self, include_desc: bool, include_ascii: bool) -> str:
        """Generates the prompt template based on configuration flags"""
        task_description = ""
        if include_ascii and include_desc:
            task_description = "Here is a gray scale image described as containing {Description}. The image is represented with integer values 0-9:\n{ASCII-Art}\nPlease write a Python program that generates this image using our custom turtle module."
        elif include_ascii:
            task_description = "Here is a gray scale image represented with integer values 0-9:\n{ASCII-Art}\nPlease write a Python program that generates this image using our custom turtle module."
        elif include_desc:
            task_description = "Here is a gray scale image described as containing {Description}\nPlease write a Python program that generates this image using our custom turtle module."
        else:
            raise ValueError("At least one of include_ascii or include_desc must be True")
        
        return task_description

@dataclass
class ScriptArguments:
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    completion_template: str = field(default="{Program}")

@dataclass
class Config:
    """Main configuration class with type validation"""
    def __init__(self, config_dict: dict):
        self.lora = LoraSettings(
            rank=int(config_dict["lora"]["rank"]),
            alpha=int(config_dict["lora"]["alpha"]),
            dropout=float(config_dict["lora"]["dropout"]),
            target_modules=list(config_dict["lora"]["target_modules"])
        )
        self.training = TrainingConfig(
            prompt_loss_weight=float(config_dict["training"]["prompt_loss_weight"]),
            max_seq_length=int(config_dict["training"]["max_seq_length"]),
            learning_rate=float(config_dict["training"]["learning_rate"]),
                        lr_scheduler_type=str(config_dict["training"]["lr_scheduler_type"]),
            train_epochs=int(config_dict["training"]["train_epochs"]),
            per_device_train_batch_size=int(config_dict["training"]["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(config_dict["training"]["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(config_dict["training"]["gradient_accumulation_steps"]),
            save_steps=int(config_dict["training"]["save_steps"]),
            eval_steps=int(config_dict["training"]["eval_steps"]),
            logging_steps=int(config_dict["training"]["logging_steps"]),
            random_seed=int(config_dict["training"]["random_seed"]),
            shuffle=bool(config_dict["training"]["shuffle"]),
            gradient_checkpointing=bool(config_dict["training"].get("gradient_checkpointing", False)),
            warmup_steps=int(config_dict["training"]["warmup_steps"]) if config_dict["training"]["warmup_steps"] not in [None, 'None'] else None,
            warmup_ratio=float(config_dict["training"]["warmup_ratio"]) if config_dict["training"]["warmup_ratio"] not in [None, 'None'] else None
        )
        self.model = ModelConfig(
            model_id=str(config_dict["model"]["model_id"]),
            topk_train=int(config_dict["model"]["topk_train"]),
            topk_prompt=int(config_dict["model"]["topk_prompt"]),
            num_return_sequences=int(config_dict["model"]["num_return_sequences"]),
            top_k=int(config_dict["model"]["top_k"]),
            temperature=float(config_dict["model"]["temperature"]),
            max_new_tokens=int(config_dict["model"]["max_new_tokens"])
        )
        self.logging = LoggingConfig(
            use_wandb=bool(config_dict.get("logging", {}).get("use_wandb", False))
        )
        self.data = DataConfig(
            dataset_id=str(config_dict["data"]["dataset_id"]),
            use_forkstate=bool(config_dict["data"]["use_forkstate"]),
            include_desc=bool(config_dict["data"]["include_desc"]),
            include_ascii=bool(config_dict["data"]["include_ascii"]),
            mix_directions=bool(config_dict["data"]["mix_directions"]),
            image_to_ascii=bool(config_dict["data"]["image_to_ascii"]),
            ascii_parameters=config_dict["data"].get("ascii_parameters", {})
        )
        self.prompt = PromptConfig(
            include_sys_prompt_fn=bool(config_dict["data"]["include_sys_prompt_fn"]),
            include_sys_prompt_inf=bool(config_dict["data"]["include_sys_prompt_inf"])
        )

def load_config(source_config: str, fine_tune: bool) -> Tuple[Config, str, str, str, str]:
    """Load training configuration from yaml file and store a copy in results directory"""    
    if os.path.exists(source_config):
        with open(source_config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Validate and convert configuration with the Config class
        config = Config(config_dict)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Determine generalization type
        gen_type = ""
        dataset_id = config.data.dataset_id

        if "-gen-" in dataset_id:
            dataset_name = dataset_id.split('/')[-1]  # Get last part after "/"
            parts = dataset_name.split('-')
            if "gen" in parts:
                gen_index = parts.index("gen")
                if gen_index > 0:  # Ensure there's a valid preceding part
                    gen_type = parts[gen_index - 1]

        # Short model name
        model_type = config.model.model_id.split('/')[-1]
        model_type_short = model_type.split('-')[0]

        # Result directory
        if fine_tune:
            result_dir = f"results/{gen_type}/{model_type_short}_{timestamp}"
        else: # inference loading model from hub
            if model_type_short.startswith(f"{gen_type}_"):
                model_type_short = model_type_short[len(f"{gen_type}_"):]
            result_dir = f"results/{gen_type}/{model_type_short}/inference/{timestamp}"
        os.makedirs(result_dir, exist_ok=True)

        # Copy the config to results directory
        with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f)
    else:
        raise FileNotFoundError(
            "No config.yaml file found in current directory. "
            "Please create a config.yaml file with your training configuration."
        )

    # Check for conflicting parameters
    if config.training.warmup_steps is not None and config.training.warmup_ratio is not None:
        raise ValueError("Both 'warmup_steps' and 'warmup_ratio' are set. Please set only one of them.")

    print(f"⚙️ Loaded configuration from {source_config}\n{result_dir}")
    return config, timestamp, gen_type, model_type_short, result_dir

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For CUDA operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    print(f"🌱 Random seed set to: {seed}")

# Step 2: Load the Model and Tokenizer
def load_model_and_tokenizer(config: Config):
    model, tokenizer = FastLanguageModel.from_pretrained(
            config.model.model_id,
            dtype=dtype,
            load_in_4bit=True,
            device_map='auto',
        )

    model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            use_gradient_checkpointing=config.training.gradient_checkpointing,
            random_state=config.training.random_seed,
            )

    # Enable gradient checkpointing if specified in the config
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for the model.")

    return  model, tokenizer

# Step 3: Prepare the Dataset
def apply_modifications(dataset, config: Config, modifier=None, ascii_processor=None):
    if config.data.use_forkstate:
        def transform(example):
            if "Program" in example:
                # Transform the Program column to use fork_state
                example["Program"] = transform_program(
                    example["Program"],
                    embed_to_fork=True,  # True: embed() => with fork_state()
                    fork_to_embed=False
                )
            return example

        # Apply the transformation to the dataset
        dataset = dataset.map(transform, desc="Transforming Program column with fork_state")

    if config.data.mix_directions:
        dataset, summary_df = modifier.replace_direction(
            dataset=dataset,
            source_dir="left", 
            target_dir="right",
            field="Program",
            proportion=0.5,
            return_overview=True
        )
        # Check if applied correctly
        print("Direction modification summary:")
        print(summary_df)

    if config.data.image_to_ascii:
        dataset = ascii_processor.process_dataset(dataset)
    return dataset

def prepare_dataset(config: Config, tokenizer, sample_fraction = 1.0, modifier=None, ascii_processor=None):
    # Only load train and validation splits
    dataset = load_dataset(config.data.dataset_id, split=["train", "validation"])
    dataset = DatasetDict({
        "train": dataset[0],
        "validation": dataset[1]
    })
    
    if sample_fraction < 1.0:
        dataset["train"] = dataset["train"].shuffle(seed=config.training.random_seed).select(range(int(len(dataset["train"]) * sample_fraction)))
        dataset["validation"] = dataset["validation"].shuffle(seed=config.training.random_seed).select(range(int(len(dataset["validation"]) * sample_fraction)))

    # Apply modifications
    dataset = apply_modifications(dataset, config, modifier=modifier, ascii_processor=ascii_processor)

    def format_prompt(example, split_type, idx):
        """Formats a single example with the template"""
        prompt = config.prompt.get_prompt_template(
            include_desc=config.data.include_desc,
            include_ascii=config.data.include_ascii
        )
        
        formatted_prompt = prompt.format(**example)
        completion = example['Program']
        
        # Create messages list based on configuration
        messages = []
        
        # For train/validation datasets
        if config.prompt.include_sys_prompt_fn:
            messages.append({"role": "system", "content": config.prompt._system_prompt})
        messages.append({"role": "user", "content": formatted_prompt})
        messages.append({"role": "assistant", "content": completion})
        
        # For prompt_length calculation, we need to include everything that's part of the input
        # but not part of what the model should generate (i.e., not the assistant's response)
        prompt_messages = []
        
        # Include system message if it exists
        if messages and messages[0]["role"] == "system":
            prompt_messages.append(messages[0])
        
        # Always include user message
        for msg in messages:
            if msg["role"] == "user":
                prompt_messages.append(msg)
        
        # Calculate prompt_length from system + user messages
        prompt_text = tokenizer.apply_chat_template(
            conversation=prompt_messages,
            tokenize=False, 
            add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=False)

        # DEBUG: Print formatted examples
        if idx < 1:  # Only print the first few examples for each split
            print(f"\n--- DEBUG: {split_type} EXAMPLE #{idx} ---")
            print(f"System prompt included: {config.prompt.include_sys_prompt_fn if split_type != 'test' else config.prompt.include_sys_prompt_inf}")
            #print(f"Messages format: {messages}")
            #print(f"PROMPT TEXT (for length calculation):\n{prompt_text}")
            print(f"FULL TEXT (with completion):\n{full_text}")
            print("--- END DEBUG ---\n")

        return {
            "text": full_text, #tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
            "prompt_length": len(prompt_text)
        }

    def format_dataset(dataset: DatasetDict):
        def map_format_prompt(example, idx, split_type):
            return format_prompt(example, split_type, idx)
        
        formatted_splits = {}
        for split in dataset.keys(): 
            formatted_splits[split] = dataset[split].map(
                lambda ex, idx: map_format_prompt(ex, idx, split),
                with_indices=True,
                desc=f"Formatting {split} prompts",
                batched=False
            )
        return formatted_splits

    formatted_dataset = format_dataset(dataset)

    def tokenize_and_mask(examples):   
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.training.max_seq_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        
        prompt_masks = []
        completion_masks = []
        
        for offsets, length in zip(tokenized["offset_mapping"], examples["prompt_length"]):
            prompt_mask = [1 if offset[1] <= length else 0 for offset in offsets]
            completion_mask = [0 if offset[1] <= length else 1 for offset in offsets]
            prompt_masks.append(prompt_mask)
            completion_masks.append(completion_mask)
        
        tokenized["prompt_mask"] = torch.tensor(prompt_masks, dtype=torch.bool)  # Store as bool 
        tokenized["completion_mask"] = torch.tensor(completion_masks, dtype=torch.bool)  # Store as bool
        tokenized["labels"] = torch.tensor(tokenized["input_ids"]).clone() # labels are grund truth
        
        del tokenized["offset_mapping"]
        del examples["text"]
        gc.collect()  # Force garbage collection to free up memory
        torch.cuda.empty_cache()  # Clear CUDA cache
        return tokenized

    # Add this function to calculate and print token statistics
    def print_token_statistics(tokenized_dataset):
        """Prints an overview of token input sizes for train and validation datasets."""
        pad_token_id = tokenizer.pad_token_id or tokenizer.unk_token_id  # Use <unk> if no pad_token is defined

        for split in tokenized_dataset.keys():
            token_lengths = []
            for input_ids in tokenized_dataset[split]["input_ids"]:
                # Exclude padding tokens
                actual_length = sum(1 for token_id in input_ids if token_id != pad_token_id)
                token_lengths.append(actual_length)

            print(f"\n--- {split.upper()} TOKEN STATISTICS ---")
            print(f"Max token length (without padding): {max(token_lengths)}")
            print(f"Min token length (without padding): {min(token_lengths)}")
            print(f"Average token length (without padding): {sum(token_lengths) / len(token_lengths):.2f}")
            print(f"Number of examples: {len(token_lengths)}")
            print(f"Max_seq_length parameter: {config.training.max_seq_length}")
        print("\n")

    tokenized_dataset = {}
    for split in formatted_dataset.keys():
        tokenized_dataset[split] = formatted_dataset[split].map(
            tokenize_and_mask,
            batched=True,
            remove_columns=formatted_dataset[split].column_names, 
            desc=f"Tokenizing {split} set",
            load_from_cache_file=False
        )
    
    # Print token statistics
    print_token_statistics(tokenized_dataset)

    return tokenized_dataset

# Step 4: WandB and Logging
def init_wandb(config: Config, timestamp: str, gen_type: str, model_type_short: str, wb_type: Optional[str] = None):
    """Initialize wandb with configuration"""
    # Generate a unique experiment name
    experiment_name = f"{gen_type}_{model_type_short}_{timestamp}"
    
    # Make config JSON serializable
    def make_json_serializable(obj):
        if hasattr(obj, '__dict__'):
            return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
    
    config_dict = make_json_serializable(config)
    
    tags = [gen_type, model_type_short]
    if wb_type: 
        tags.append(wb_type)
    if config.data.use_forkstate:
        tags.append("with-forkstate")
    else:
        tags.append("with-embed()")

    # Init wandb
    wandb.init(
        project="master-thesis",
        name=experiment_name,
        config=config_dict,
        tags=tags
    )

def log_metrics_per_example(
    true_program, pred_program, normalized_distances, crystalbleu_scores, image_metrics, 
    split, detailed_metrics_path, 
    current_epoch=None, current_batch_count=None, current_eval_step=None
):
    log_entries = []
    try:
        for idx, (gt_prog, pred_prog, norm_lev, bleu, img_metrics) in enumerate(zip(true_program, pred_program, normalized_distances, crystalbleu_scores, image_metrics)):
            # Initialize the entry dictionary
            entry = {
                "program_id": f"{idx}_{hashlib.md5(gt_prog.encode('utf-8')).hexdigest()}",
                "predicted_program": pred_prog,
                "norm_lev_dist": norm_lev,
                "crystalbleu_score": bleu,
                "image_metrics": img_metrics,
            }
            # Add split-specific context
            if split == 'train':
                entry["epoch"] = current_epoch
                entry["batch_count"] = current_batch_count
            else:
                entry["eval_step_count"] = current_eval_step
            log_entries.append(entry)
    except Exception as e:
        print(f"Error processing entry {idx}: {e}")

    try:
        with open(detailed_metrics_path, "a") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Error writing to file {detailed_metrics_path}: {e}") 

# Step 5: Training Preperation
# Custom Metrics
### Initialize global counters which are used to give context in the detailed_metrics.jsonl file of the train and validation set
current_epoch = 1 
current_batch = 0
current_eval_step = 0

def prepare_compute_metrics(dataset: DatasetDict, tokenizer, evaluator):
    """Prepare custom metrics function for Trainer"""
    val_prompt_mask = np.array([x["prompt_mask"] for x in dataset['validation']])
    val_completion_mask = np.array([x["completion_mask"] for x in dataset['validation']])

    # Store masks in a dictionary for each split
    masks = {'validation': 
        {
        'prompt_mask': val_prompt_mask,
        'completion_mask': val_completion_mask # uses numpy arrays (on CPU)
        }
    }

    # uses numpy arrays (on CPU)
    def compute_metrics(data, split='validation'):
        # Get masks based on split
        if split == 'validation':
            # Use precomputed validation masks
            prompt_mask = masks[f'{split}']['prompt_mask']
            completion_mask = masks[f'{split}']['completion_mask']
        else:
            batch_labels = data.label_ids
            batch_size, seq_len = batch_labels.shape
            if hasattr(data, 'inputs') and 'prompt_mask' in data.inputs and 'completion_mask' in data.inputs:
                prompt_mask = data.inputs['prompt_mask'].detach().cpu().numpy() if isinstance(data.inputs['prompt_mask'], torch.Tensor) else data.inputs['prompt_mask']
                completion_mask = data.inputs['completion_mask'].detach().cpu().numpy() if isinstance(data.inputs['completion_mask'], torch.Tensor) else data.inputs['completion_mask']
        
        # Process predictions (token_preds, token_losses)
        token_preds, token_losses = data.predictions
        
        # Move tensors to CPU before NumPy operations
        if isinstance(token_preds, torch.Tensor):
            token_preds = token_preds.detach().cpu().numpy()
        if isinstance(token_losses, torch.Tensor):
            token_losses = token_losses.detach().cpu().numpy()
            
        # Ensure labels are on CPU
        labels = data.label_ids
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            
        # shift labels and masks
        labels = labels[..., 1:]
        shift_prompt_mask = prompt_mask[..., 1:]
        shift_comp_mask = completion_mask[..., 1:]

        # For training data, we need to ensure we're only using the batch data
        if split == 'train':
            # Make sure we're only using the amount of data in the current batch
            batch_size = labels.shape[0]
            shift_prompt_mask = shift_prompt_mask[:batch_size]
            shift_comp_mask = shift_comp_mask[:batch_size]

        # Check shapes before operations
        if token_losses.reshape(-1).shape[0] != shift_prompt_mask.reshape(-1).shape[0]:
            print(f"Shape mismatch in {split}: token_losses={token_losses.reshape(-1).shape}, prompt_mask={shift_prompt_mask.reshape(-1).shape}")
            
        else:
            # Normal calculation if shapes match
            prompt_loss = np.sum(token_losses.reshape(-1) * shift_prompt_mask.reshape(-1)) / (shift_prompt_mask.sum() or 1)
            completion_loss = np.sum(token_losses.reshape(-1) * shift_comp_mask.reshape(-1)) / (shift_comp_mask.sum() or 1)

        # Program comparison: Levenshtein distance, CrystalBLEU score and Image Comparison
        try:
            true_comp_tokens = [l[m == 1] for l, m in zip(labels, shift_comp_mask)]
            pred_comp_tokens = [p[m == 1] for p, m in zip(token_preds, shift_comp_mask)]
            
            true_program = tokenizer.batch_decode(true_comp_tokens, skip_special_tokens=True)
            pred_program = tokenizer.batch_decode(pred_comp_tokens, skip_special_tokens=True)

            # Clean code before Levenshtein calculation (remove py comments)
            clean_true_program = [evaluator.clean_python_code(code) for code in true_program]
            clean_pred_program = [evaluator.clean_python_code(code) for code in pred_program]

            # Calculate normalized Levenshtein distances using cleaned code (1.0 = perfect match, 0.0 = no match)
            try:
                normalized_distances = []
                exact_prog_match_count = 0
                for pred, true in zip(clean_pred_program, clean_true_program):
                    max_len = max(len(pred), len(true))
                    if max_len == 0:  # Handle edge case of empty strings
                        normalized_distances.append(1.0)
                    else:
                        lev_dist = levenshtein_distance(pred, true)
                        norm_lev_dist = 1.0 - (lev_dist / max_len)
                        normalized_distances.append(norm_lev_dist)
                        if norm_lev_dist == 1.0:
                            exact_prog_match_count += 1
                
                avg_norm_levenshtein_dist = np.mean(normalized_distances)
                std_norm_levenshtein_dist = np.std(normalized_distances)
            except Exception as e:
                print(f"Error computing normalized Levenshtein distance: {str(e)}")
                avg_norm_levenshtein_dist = 0.0
                std_norm_levenshtein_dist = 0.0
                exact_prog_match_count = 0
            # CrystalBLEU score
            try:
                crystalbleu_scores = []
                for pred, true in zip(clean_true_program, clean_pred_program):
                    crystalbleu_result = evaluator.check_crystalbleu_similarity(true, pred)
                    crystalbleu_scores.append(crystalbleu_result["crystalbleu_score"])
                
                avg_crystalbleu = np.mean([score for score in crystalbleu_scores if score is not None])
            except Exception as e:
                print(f"Error computing CrystalBLEU score: {str(e)}")
                bleu_score = 0.0
            # image comparison
            try:
                image_metrics = []
                exact_image_match_count = 0
                for pred, true in zip(true_program, pred_program):
                    is_executable, _, pred_image = evaluator.code_execution_pyturtle(pred)
                    _, _, true_image = evaluator.code_execution_pyturtle(true)
                    if is_executable and pred_image and true_image:
                        comp_image = evaluator.compare_images(pred_image, true_image)
                        image_metrics.append(comp_image)
                        if comp_image["pixel_f1"] == 1.0:
                            exact_image_match_count += 1
                    else:
                        # Append NaN for cases where images cannot be generated
                        image_metrics.append({
                            "ssim_score": np.nan,
                            "dreamsim_score": np.nan,
                            "pixel_precision": np.nan,
                            "pixel_recall": np.nan,
                            "pixel_f1": np.nan
                        })
                if image_metrics and any(not np.isnan(m["ssim_score"]) for m in image_metrics):
                    avg_ssim = np.nanmean([m["ssim_score"] for m in image_metrics])
                    avg_dreamsim = np.nanmean([m["dreamsim_score"] for m in image_metrics])
                    avg_precision = np.nanmean([m["pixel_precision"] for m in image_metrics])
                    avg_recall = np.nanmean([m["pixel_recall"] for m in image_metrics])
                    avg_f1 = np.nanmean([m["pixel_f1"] for m in image_metrics])
                else:
                    avg_ssim = np.nan
                    avg_dreamsim = np.nan
                    avg_precision = np.nan
                    avg_recall = np.nan
                    avg_f1 = np.nan
            except Exception as e:
                print(f"Error computing Image metrics: {str(e)}")
                avg_ssim = np.nan
                avg_dreamsim = np.nan
                avg_precision = np.nan
                avg_recall = np.nan
                avg_f1 = np.nan
        except Exception as e:
            print(f"Error in program and image comparison: {str(e)}")
            avg_norm_levenshtein_dist = 0.0
            std_norm_levenshtein_dist = 0.0
            avg_crystalbleu = 0.0
            exact_prog_match_count = 0
            avg_ssim = np.nan
            avg_dreamsim = np.nan
            avg_precision = np.nan
            avg_recall = np.nan
            avg_f1 = np.nan
            exact_image_match_count = 0

        # Clean up to free memory, especially important for training batches
        if split == 'train':
            # Force cleanup of large arrays
            del token_preds, token_losses, labels, shift_prompt_mask, shift_comp_mask

        global current_epoch, current_batch, current_eval_step
        try:
            # Log metrics per example
            if split == 'train':
                current_batch += config.training.logging_steps
                if current_batch % (len(dataset['train']) // config.training.per_device_train_batch_size) == 0:
                    current_epoch += 1
                detailed_metrics_path = os.path.join(result_dir, f"train_detailed_metrics.jsonl")
                
                log_metrics_per_example(
                    true_program=true_program,
                    pred_program=pred_program,
                    normalized_distances=normalized_distances,
                    crystalbleu_scores=crystalbleu_scores,
                    image_metrics=image_metrics,
                    split='train',
                    detailed_metrics_path=detailed_metrics_path,
                    current_epoch=current_epoch,
                    current_batch_count=current_batch
                )
            elif split == 'validation':
                current_eval_step += 1
                detailed_metrics_path = os.path.join(result_dir, f"val_detailed_metrics.jsonl")
                
                log_metrics_per_example(
                    true_program=true_program,
                    pred_program=pred_program,
                    normalized_distances=normalized_distances,
                    crystalbleu_scores=crystalbleu_scores,
                    image_metrics=image_metrics,
                    split='validation',
                    detailed_metrics_path=detailed_metrics_path, 
                    current_eval_step=current_eval_step
                )
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")

        image_metrics.clear()

        # Return metrics with Python native types
        return {
            'comp_loss': float(completion_loss),
            'prompt_loss': float(prompt_loss),
            # Program Comparison
            'avg_norm_levenshtein_dist': float(avg_norm_levenshtein_dist),
            'std_norm_levenshtein_dist': float(std_norm_levenshtein_dist),
            'avg_crystalbleu': float(avg_crystalbleu),
            'exact_prog_match': int(exact_prog_match_count),
            # Image Comparison
            'avg_ssim': float(avg_ssim),
            'avg_dreamsim': float(avg_dreamsim),
            'avg_precision': float(avg_precision),
            'avg_recall': float(avg_recall),
            'avg_f1': float(avg_f1),
            'exact_image_match': int(exact_image_match_count),
        }
        
    return compute_metrics

# uses PyTorch tensors (on GPU)
def preprocess_logits_for_metrics(logits, labels):
    # get predictions
    token_preds = logits.argmax(-1)[..., :-1]

    # compute per-token losses
    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

    # pass predictions and losses to compute_metrics function (above)
    predictions = (token_preds, token_losses)
    return predictions

# Step 6. Training the model with custom PLW-Trainer
def train_model(model, tokenizer, dataset, result_dir: str, config: Config, timestamp: str, gen_type: str, model_type_short: str, wb_type: Optional[str] = None):
    """Training loop with configurable prompt and completion weights"""

    # Only if set True in config logging to wandb will be enabled
    if config.logging.use_wandb:
        init_wandb(config, timestamp, gen_type, model_type_short, wb_type)
        report_to = "wandb"
    else:
        report_to = "none"
    
    class PLWTrainer(Trainer):
        def __init__(self, *args, prompt_loss_weight=1.0, shuffle=False, **kwargs):
            self.processor = kwargs.pop('tokenizer', None)
            super().__init__(*args, **kwargs)
            self.prompt_loss_weight = prompt_loss_weight
            self.shuffle = shuffle
            self.distributed_training = torch.distributed.is_initialized() # not used in single GPU setup

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # num_items_in_batch(batch_size*gradient_accumulation_steps) not used
            # Debug: Check memory before forward pass
            #print(f"Before forward pass: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
            #print(f"Before forward pass: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            if not model.training:
                with torch.no_grad():
                    outputs = model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"])
                torch.cuda.empty_cache()
            else:
                outputs = model(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"])
            # Debug: Check memory after forward pass
            #print(f"After forward pass: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
            #print(f"After forward pass: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
        
            logits = outputs.get("logits")
            labels = inputs.pop("labels")

            # Optimize weights computation
            weights = torch.where(
                inputs["completion_mask"].bool(),
                torch.tensor(1.0, dtype=logits.dtype, device=logits.device),
                torch.tensor(self.prompt_loss_weight, dtype=logits.dtype, device=logits.device)
            )

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            # Debug: Before loss
            #print(f"Before loss computation: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
            #print(f"Before loss computation: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

            # Flatten tensors
            B, T, V = shift_logits.size()
            flat_logits = shift_logits.view(-1, V)       # [(B*T), V]
            flat_labels = shift_labels.view(-1)          # [(B*T)]
            flat_weights = shift_weights.view(-1)        # [(B*T)]

            # Compute loss
            loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
            token_losses = loss_fct(flat_logits, flat_labels)  # [(B*T)]

            # Mask pad tokens
            mask = flat_labels != tokenizer.pad_token_id
            token_losses = token_losses[mask]
            flat_weights = flat_weights[mask]

            # Weighted average loss
            loss = (token_losses * flat_weights).sum() / flat_weights.sum()

            # Clean up
            #del loss, logits, labels
            #torch.cuda.empty_cache()
            #print(f"After loss computation: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
            #print(f"After loss computation: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Training requires a train_dataset.")

            sampler=self._get_train_sampler()
            use_generator = isinstance(sampler, torch.utils.data.RandomSampler)
            if use_generator:
                generator = torch.Generator()
                generator.manual_seed(config.training.random_seed)
            else:
                generator = None

            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self._get_train_sampler(),
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                generator=generator,
            )

            return dataloader

        # this allows us to toggle on/off data shuffling, which can sometimes cause 'staircase' effects in training loss
        def _get_train_sampler(self):
            #if self.distributed_training: # not used in single GPU setup
            #    return torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=self.shuffle) 
            if self.shuffle:
                return torch.utils.data.RandomSampler(self.train_dataset)
            return torch.utils.data.SequentialSampler(self.train_dataset)


    class MetricsLoggingCallback(TrainerCallback):
        """Custom callback to compute metrics for both training and validation"""
        
        def __init__(self, trainer, eval_dataset, train_dataset, tokenizer, compute_metrics_fn, preprocess_fn, logging_steps):
            self.trainer = trainer
            self.eval_dataset = eval_dataset
            self.train_dataset = train_dataset
            self.tokenizer = tokenizer
            self.compute_metrics_fn = compute_metrics_fn
            self.preprocess_fn = preprocess_fn
            self.logging_steps = logging_steps
            self.is_currently_logging = False  # Flag to prevent recursive calls
            
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            # Prevent recursive calls
            if self.is_currently_logging:
                return control
            
            # Skip if not on a logging step or if we're just starting
            if state.global_step % self.logging_steps != 0 or state.global_step == 0:
                return control
                
            # Set flag to prevent recursive calls
            self.is_currently_logging = True
            
            try:
                # Calculate which step these metrics belong to (the previous step)
                metrics_step = state.global_step - 1
                train_dataloader = self.trainer.get_train_dataloader()
                try:
                    batch = next(iter(train_dataloader))
                except StopIteration:
                    self.is_currently_logging = False
                    return control  # No batches available
                        
                # Move batch to correct device
                batch = self.trainer._prepare_inputs(batch)
                
                # Store a copy of the masks for later
                prompt_mask = batch.pop("prompt_mask", None)
                completion_mask = batch.pop("completion_mask", None)
                    
                # No gradient computation needed for metrics
                with torch.no_grad():
                    # Run model
                    outputs = self.trainer.model(**batch)                            
                    # Process logits
                    if self.preprocess_fn is not None:
                        predictions = self.preprocess_fn(outputs.logits, batch["labels"])
                    else:
                        predictions = outputs.logits.argmax(-1)
                            
                    # Create EvalPrediction object
                    eval_prediction = EvalPrediction(
                        predictions=predictions,
                        label_ids=batch["labels"]
                        )
                        
                    # Add the masks back to the inputs dictionary
                    batch["prompt_mask"] = prompt_mask
                    batch["completion_mask"] = completion_mask
                    
                    # Store the full batch with masks in the inputs attribute
                    eval_prediction.inputs = batch

                    # Compute metrics for training data
                    train_metrics = self.compute_metrics_fn(eval_prediction, split='train')
                    
                    # Add train/ prefix to all metrics
                    train_metrics_prefixed = {f"train/{k}": v for k, v in train_metrics.items()}
                        
                # Log directly to wandb or other trackers without going through trainer.log to avoid recursion
                if self.trainer.is_world_process_zero():
                    if hasattr(args, "report_to") and "wandb" in args.report_to:
                        wandb.log(train_metrics_prefixed, step=metrics_step)
                    
            except Exception as e:
                print(f"Error in metrics callback: {e}")
                traceback.print_exc()
            finally:
                # Always reset the flag, even if an error occurred
                self.is_currently_logging = False
                
                # Clean up to free memory
                if 'batch' in locals():
                    del batch
                if 'outputs' in locals():
                    del outputs
                if 'predictions' in locals():
                    del predictions
                if 'eval_prediction' in locals():
                    del eval_prediction
                    
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            return control


    # Training arguments based on config
    training_args = TrainingArguments(
        output_dir=f"{result_dir}/fn_model", 
        num_train_epochs=config.training.train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        #eval_accumulation_steps=1,
        remove_unused_columns=False,
        learning_rate=config.training.learning_rate,
        #warmup_steps=config.training.warmup_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        fp16=not is_bfloat16_supported(), #True,
        bf16=is_bfloat16_supported(), #False,
        tf32=True,              # bc set in run_plw.py
        seed=config.training.random_seed, 
        report_to=report_to,    # dynamic wandb reporting
        gradient_checkpointing=config.training.gradient_checkpointing,     # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        weight_decay=0.001,     # Set weight_decay to match DeepSpeed config 
        max_grad_norm=0.3,      # bc set in run_plw.py
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="comp_loss",
        greater_is_better=False,  # Whether a higher metric value is better
        push_to_hub = True,
        hub_model_id = f"{gen_type}_{model_type_short}_{timestamp}"
    )    

    # Initialize trainer with configurable weights
    trainer = PLWTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        shuffle=config.training.shuffle,
        prompt_loss_weight=config.training.prompt_loss_weight,
        compute_metrics=prepare_compute_metrics(dataset, tokenizer, evaluator), 
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.optimizer = AdamW8bit(trainer.model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    
    print("🚀 Optimizer:", trainer.optimizer)

    trainer.lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=trainer.optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.num_train_epochs * len(trainer.get_train_dataloader())
    )
    print("🚀 Scheduler:", trainer.lr_scheduler)

    # Add custom callback for logging metrics
    trainer.add_callback(MetricsLoggingCallback(
        trainer=trainer,
        eval_dataset=dataset["validation"],
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        compute_metrics_fn=prepare_compute_metrics(dataset, tokenizer, evaluator),
        preprocess_fn=preprocess_logits_for_metrics,
        logging_steps=config.training.logging_steps
    ))

    # Train the model
    trainer.train()
    
    # Logging is only disabled if it was enabled before
    if config.logging.use_wandb:
        wandb.finish()
    
    return model

# Step 7: Preparing Inference
def clear_training_memory(dataset=None):
    """
    Clear training and validation data from memory to free resources for inference
    
    Args:
        dataset: Optional dataset to explicitly delete
    """
    # Explicitly delete the dataset if provided
    if dataset is not None:
        print("Removing training and validation datasets from memory...")
        if 'train' in dataset:
            del dataset['train']
        if 'validation' in dataset:
            del dataset['validation']
        del dataset
    
    # Clear Python memory
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Cleared training data from memory")

def init_wandb_for_inf(config: Config, model_id: str, inference_type: str):
    tags = [model_id, inference_type, "inference"]
    # Add "zero-shot" or "few-shot" based on topk_prompt
    if config.model.topk_prompt == 0:
        tags.append("zero-shot")
    elif config.model.topk_prompt > 0:
        tags.append("few-shot")
    # Add "with-forkstate" if use_forkstate is True
    if config.data.use_forkstate:
        tags.append("with-forkstate")
    else:
        tags.append("with-embed()")
    # Add the search budget if used (num_return_sequences > 0)
    if config.model.num_return_sequences > 0:
        tags.append(f"{config.model.num_return_sequences}_samples")

    wandb.init(
        project="master-thesis--inference", 
        name=f"{model_id}-{inference_type}",
        tags=tags,
        config={
            "model_id": config.model.model_id,
            "include_sys_prompt": config.prompt.include_sys_prompt_inf,
            "include_ascii": config.data.include_ascii,
            "include_desc": config.data.include_desc,
            "temperature": config.model.temperature,
            "max_new_tokens": 250,
            "inference_type": inference_type
        }
    )

# Step 8.1: Inference using the recently fine-tuned model
def inference(model, tokenizer, config: Config, result_dir: str, inference_type: str, sample_fraction = 1.0, modifier=None, ascii_processor=None):
    # Initialize WandB for inference
    if config.logging.use_wandb:
        model_id = config.model.model_id.split("/")[-1]
        init_wandb_for_inf(config, model_id, inference_type)
   
    print(f'Begin inference on test dataset')
    inf_dir = os.path.join(result_dir, "inference")
    os.makedirs(inf_dir, exist_ok=True)

    model.eval()  # model in evaluation mode (PyTorch)
    
    try:
        model = FastLanguageModel.for_inference(model)
        print("Using Unsloth's optimized inference")
    except Exception as e:
        print(f"Could not enable Unsloth's optimized inference: {e}")
    
    # Load the raw test dataset
    dataset = load_dataset(config.data.dataset_id)["test"]
    if sample_fraction < 1.0:
        dataset = dataset.shuffle(seed=config.training.random_seed).select(range(int(len(dataset) * sample_fraction)))

    # Apply modifications
    dataset = apply_modifications(dataset, config, modifier=modifier, ascii_processor=ascii_processor)

    results = []
    
    # Get prompt template
    prompt_template = config.prompt.get_prompt_template(
        include_desc=config.data.include_desc,
        include_ascii=config.data.include_ascii
    )
    
    # Process in smaller batches to avoid memory issues
    batch_size = config.training.per_device_eval_batch_size
    num_examples = len(dataset)

    example_id = 0 # Global counter for example IDs

    with torch.no_grad():
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(num_examples-1)//batch_size + 1} " +
                 f"(examples {batch_start}-{batch_end-1})")
            
            for i in batch_indices:
                example = dataset[i]
                
                # Format prompt
                formatted_prompt = prompt_template.format(**example)
                
                # Create messages list with system prompt if configured
                messages = []
                if config.prompt.include_sys_prompt_inf:
                    messages.append({"role": "system", "content": config.prompt._system_prompt})
                messages.append({"role": "user", "content": formatted_prompt})
                
                # Apply chat template for consistent formatting with training
                chat_formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Generate with the model - using the chat template
                tokenized_input = tokenizer(chat_formatted_prompt, return_tensors="pt", padding=True)
                input_ids = tokenized_input.input_ids.cuda()
                attention_mask = tokenized_input.attention_mask.cuda()
                
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.model.max_new_tokens,
                    temperature=config.model.temperature,
                    top_k=config.model.top_k,
                    num_return_sequences=config.model.num_return_sequences,
                    do_sample=True
                )
                
                # Decode generation
                decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # Group outputs per input
                grouped_outputs = [
                    decoded_outputs[i:i + config.model.num_return_sequences]
                    for i in range(0, len(decoded_outputs), config.model.num_return_sequences)
                ]

                for example_idx, completions in enumerate(grouped_outputs):
                    cleaned_completions = []

                    for i, comp in enumerate(completions):
                        # Extract the completion part - adapting to different model formats
                        if "<|assistant|>" in comp:
                            completion = comp.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
                        elif "[/INST]" in comp:
                            completion = comp.split("[/INST]")[-1].strip()
                        elif "assistant:" in comp.lower():
                            completion = comp.split("assistant:", 1)[-1].strip()
                        else:
                            user_content = messages[-1]["content"]
                            if user_content in comp:
                                completion = comp.split(user_content, 1)[-1].strip()
                            else:
                                completion = comp.strip()  # Fallback

                        cleaned_completions.append(completion)

                    # Get ground truth
                    ground_truth = example["Program"]

                    # Store result for this example
                    result = {
                        "id": example_id,
                        "prompt": chat_formatted_prompt,
                        "ground_truth": ground_truth,
                    }

                    for idx, completion in enumerate(cleaned_completions, start=1):
                        result[f"completion_{idx}"] = completion

                    results.append(result)
                    example_id += 1  # Increment the global counter

            
            if results:
                with open(os.path.join(inf_dir, "predictions.json"), "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved {len(results)} predictions after batch {batch_start//batch_size + 1}")
            
            if config.logging.use_wandb:
                wandb.log({
                    "progress/examples_processed": len(results),
                    "progress/batch_number": batch_start//batch_size + 1
                })
                
            # Free memory after each batch
            torch.cuda.empty_cache()
    
    # After all batches are processed, save the final results
    if results:
        # Save all predictions in a single file
        with open(os.path.join(inf_dir, "predictions.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Inference completed. Generated {len(results)} predictions.")
    else:
        print("Inference completed but no results were generated.")
    
    if config.logging.use_wandb:
        if results:
            # Log summary metrics and artifacts
            wandb.log({
                "total_examples": len(results)
            })
            
            # Save predictions as an artifact
            predictions_artifact = wandb.Artifact(
                name=f"predictions-{inference_type}", 
                type="predictions"
            )
            predictions_artifact.add_file(os.path.join(inf_dir, "predictions.json"))
            wandb.log_artifact(predictions_artifact)
        else:
            wandb.log({
                "total_examples": 0
            })
        
        wandb.finish()
    return results, inf_dir

# Step 8.2: Inference using model from hub
def inference_from_hub(config: Config, result_dir: str, inference_type: str, sample_fraction = 1.0, modifier=None, ascii_processor=None):
    """
    Runs inference using a model loaded directly from the Hugging Face Hub.
    
    Args:
        config: Configuration object
        result_dir: Directory to save results
        inference_type: Type of inference being performed
        sample_fraction: Fraction of test dataset to use
    """
    # Initialize WandB for inference
    if config.logging.use_wandb:
        hub_model_name = config.model.model_id.split("/")[-1]
        init_wandb_for_inf(config, hub_model_name, inference_type)
   
    print(f'Begin inference on test dataset using model from hub: {config.model.model_id}')

    inf_dir = result_dir
    os.makedirs(inf_dir, exist_ok=True)

    # Load model and tokenizer from Hub
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            config.model.model_id,
            max_seq_length=config.training.max_seq_length,  # allows longer token seq. needed for few-shot prompting (as the default set by unsloth)
            dtype=dtype,
            load_in_4bit=True,
        )

        model = FastLanguageModel.for_inference(model)
        print(f"Loaded model {config.model.model_id} using Unsloth's optimized inference")
    except Exception as e:
        print(f"Could not load with Unsloth, falling back to HF Transformers: {e}")
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            torch_dtype=dtype,
            device_map='auto',
        )
        print(f"Loaded model {config.model.model_id} using standard HF Transformers")
    
    # Load the raw test dataset
    dataset = load_dataset(config.data.dataset_id)["test"]
    if sample_fraction < 1.0:
        dataset = dataset.shuffle(seed=config.training.random_seed).select(range(int(len(dataset) * sample_fraction)))
    
    # Apply modifications
    dataset = apply_modifications(dataset, config, modifier=modifier, ascii_processor=ascii_processor)

    if config.model.topk_prompt > 0:  # few-shot inference
        dataset_ex = load_dataset(config.data.dataset_id)["train"]
        dataset_ex = apply_modifications(dataset_ex, config, modifier=modifier, ascii_processor=ascii_processor)
    else:                       # zero-shot inference
        dataset_ex = None

    results = []
    
    # Get prompt template
    prompt_template = config.prompt.get_prompt_template(
        include_desc=config.data.include_desc,
        include_ascii=config.data.include_ascii
    )
    
    # Process in smaller batches to avoid memory issues
    batch_size = config.training.per_device_eval_batch_size
    num_examples = len(dataset)

    example_id = 0  # Initialize a global counter for unique IDs
    
    with torch.no_grad():
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(num_examples-1)//batch_size + 1} " +
                 f"(examples {batch_start}-{batch_end-1})")
            
            for i in batch_indices:
                example = dataset[i]
                
                # Format prompt
                formatted_prompt = prompt_template.format(**example)
                
                # Create messages list with system prompt if configured
                messages = []
                if config.prompt.include_sys_prompt_inf:
                    messages.append({"role": "system", "content": config.prompt._system_prompt})
                
                # Randomly select few-shot examples without replacement
                used_examples = set() # tracking of already sampled examples
                if dataset_ex and config.model.topk_prompt > 0:
                    dataset_shuffled = dataset_ex.shuffle(seed=config.training.random_seed)
                    available_indices = list(range(len(dataset_shuffled)))
                    sampled_examples = []
                    for _ in range(config.model.topk_prompt):
                        remaining_indices = [idx for idx in available_indices if idx not in used_examples]
                        if remaining_indices:
                            selected_idx = random.choice(remaining_indices)
                            sampled_examples.append(dataset_shuffled[selected_idx])
                            used_examples.add(selected_idx)
                        else:
                            print("Not enough examples in the train Dataset to ensure that no example has been used priviously")
                            break
                    for ex in sampled_examples:
                        train_prompt = prompt_template.format(**ex)
                        train_completion = ex["Program"]
                        messages.append({"role": "user", "content": train_prompt})
                        messages.append({"role": "assistant", "content": train_completion})
                
                messages.append({"role": "user", "content": formatted_prompt})
                
                # Apply chat template for consistent formatting
                chat_formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                #print(f"Chat formatted prompt: {chat_formatted_prompt}") # Debug

                # Generate with the model
                tokenized_input = tokenizer(chat_formatted_prompt, return_tensors="pt", padding=True)
                input_ids = tokenized_input.input_ids.to(model.device)
                attention_mask = tokenized_input.attention_mask.to(model.device)

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.model.max_new_tokens,
                    temperature=config.model.temperature,
                    top_k=config.model.top_k,
                    num_return_sequences=config.model.num_return_sequences,
                    do_sample=True
                )

                # Decode generation
                decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # Group outputs per input
                grouped_outputs = [
                    decoded_outputs[i:i + config.model.num_return_sequences]
                    for i in range(0, len(decoded_outputs), config.model.num_return_sequences)
                ]

                for example_idx, completions in enumerate(grouped_outputs):
                    cleaned_completions = []

                    for i, comp in enumerate(completions):
                        # Extract the completion part - adapting to different model formats
                        if "<|assistant|>" in comp:
                            completion = comp.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
                        elif "[/INST]" in comp:
                            completion = comp.split("[/INST]")[-1].strip()
                        elif "assistant:" in comp.lower():
                            completion = comp.split("assistant:", 1)[-1].strip()
                        else:
                            user_content = messages[-1]["content"]
                            if user_content in comp:
                                completion = comp.split(user_content, 1)[-1].strip()
                            else:
                                completion = comp.strip()  # Fallback

                        cleaned_completions.append(completion)

                    # Get ground truth
                    ground_truth = example["Program"]

                    # Store results
                    result = {
                        "id": example_id,  # Use the global counter
                        "prompt": chat_formatted_prompt,
                        "ground_truth": ground_truth
                    }

                    for idx, comp in enumerate(cleaned_completions, start=1):
                        result[f"completion_{idx}"] = comp

                    results.append(result)
                    example_id += 1  # Increment the global counter
            
            if results:
                with open(os.path.join(inf_dir, "predictions.json"), "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved {len(results)} predictions after batch {batch_start//batch_size + 1}")
            
            if config.logging.use_wandb:
                wandb.log({
                    "progress/examples_processed": len(results),
                    "progress/batch_number": batch_start//batch_size + 1
                })
                
            # Free memory after each batch
            torch.cuda.empty_cache()
    
    # After all batches are processed, save the results
    if results:
        # Save all predictions in a single file
        with open(os.path.join(inf_dir, "predictions.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Hub inference completed. Generated {len(results)} predictions.")
    else:
        print("Hub inference completed but no results were generated.")
    
    if config.logging.use_wandb:
        if results:
            # Log total examples and save predictions as an artifact
            wandb.log({"total_examples": len(results)})
            
            predictions_artifact = wandb.Artifact(
                name=f"predictions-{inference_type}-{hub_model_name}", 
                type="predictions"
            )
            predictions_artifact.add_file(os.path.join(inf_dir, "predictions.json"))
            wandb.log_artifact(predictions_artifact)
        else:
            wandb.log({"total_examples": 0})
        
        wandb.finish()
    
    # Clean up resources
    del model
    torch.cuda.empty_cache()
    
    return results, inf_dir

# Step 9: Evaluation
def evaluation(inf_dir: str, n_completions: int, fork_state: bool = False):
    """
    Evaluate model predictions using the LLMCodeEvaluator class.
    
    Args:
        inf_dir (str): Directory containing predictions.json
        
    Returns:
        tuple: (metrics, summary)
    """    
    print(f"Starting evaluation on predictions in {inf_dir}")
    
    try:
        # Run the evaluation pipeline
        metrics, summary = evaluator.evaluate_and_summarize(inf_dir, n_completions=n_completions, fork_state=fork_state)
        return metrics, summary
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()
        return None, None

# MAIN
if __name__ == "__main__":
    try:
        # Load configuration
        config, timestamp, gen_type, model_type_short, result_dir = load_config(config_file, fine_tune)
        # Import and initalize components based on the configuration
        modifier = None
        ascii_processor = None
        if config.data.use_forkstate:
            from synthetic_data.transform_data_to_forkstate_custom import transform_program
        if config.data.mix_directions:
            from synthetic_data.__dataset_direction_modifier import DatasetDirectionModifier
            modifier = DatasetDirectionModifier(random_seed=config.training.random_seed)
        if config.data.image_to_ascii:
            from synthetic_data.__adapt_ascii_processor import AdaptiveASCIIProcessor
            black_threshold = config.data.ascii_parameters.get("black_threshold", 128)
            block_size = config.data.ascii_parameters.get("block_size", None)
            crop_to_size = config.data.ascii_parameters.get("crop_to_size", None)
            ascii_processor = AdaptiveASCIIProcessor(levels=10,
                                                        black_threshold=black_threshold,
                                                        block_size=block_size,
                                                        crop_to_size=crop_to_size,drop_images=True
                                                    )
        # Set random seed for reproducibility
        set_random_seeds(config.training.random_seed)

        if fine_tune:
            # Prep
            model, tokenizer = load_model_and_tokenizer(config)
            dataset = prepare_dataset(config, tokenizer, sample_fraction = sample_fraction, modifier=modifier, ascii_processor=ascii_processor) 
            # Training
            model = train_model(model, tokenizer, dataset, result_dir, config, timestamp, gen_type, model_type_short, wb_type)
            clear_training_memory(dataset) # Free up memory after training
            # Inference after Finetuning
            results, inf_dir = inference(model, tokenizer, config, result_dir, inference_type=f"{wb_type}_{timestamp}", sample_fraction = sample_fraction, modifier=modifier, ascii_processor=ascii_processor)
        elif inference_hub_model:
            # Inference with Model from Hub
            results, inf_dir = inference_from_hub(config, result_dir, inference_type=f"{wb_type}_hub_{timestamp}", sample_fraction = sample_fraction, modifier=modifier, ascii_processor=ascii_processor)
        # Evaluation
        if 'model' in locals() and model is not None:
            del model
            torch.cuda.empty_cache()
        metrics, summary = evaluation(inf_dir, n_completions=config.model.num_return_sequences, fork_state=config.data.use_forkstate)
        print("Pipeline completed successfully! 🎉")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()