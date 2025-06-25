"""
__pipeline_hash.py

Generates a mapping from hashed program completions to their descriptions for the validation split, reproducing the exact hashing logic used during fine-tuning withe the 'pipeline.py', including all preprocessing steps such as
direction mixing, forkstate, and ASCII-art conversion.
This ensures consistent matching between the precalculated validation metrics and dataset entries. This is a helper script for the detailed_eval.py script.

Usage (as script):
    python __pipeline_hash.py --config_file /path/to/config.yaml

Usage (as module):
    from __pipeline_hash import get_val_dataset_hash
    val_dataset_hash = get_val_dataset_hash(config_file)
"""

import argparse
import hashlib
import traceback
import gc
import os
import numpy as np
import hashlib # new to create the program-id while logging the detailed metrics during fine-tuning
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
# Third-party imports
import yaml
import torch
from collections import defaultdict
from unsloth import FastLanguageModel, is_bfloat16_supported, UnslothTrainer

fine_tune = True  # because we recreate the hash which was created during fine-tuning

dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16 #torch.float16

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
    adapter_path: Optional[str] = None

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
    use_forkstate: bool = False
    _system_prompt: str = field(init=False)

    def __post_init__(self):
        base_sys_prompt = """Your task is to draw simple black and white graphics with the custom library. DO NOT USE THE BUILT-IN TURTLE LIBRARY.
You will use a custom turtle library, similar to the built-in library, which is sufficient for all tasks.

Here are all the available functions in the custom turtle library:
- forward(x): move forward x pixels
- left(theta): rotate left by theta degrees
- right(theta): rotate right by theta degrees
- penup(): stop drawing
- pendown(): start drawing
- teleport(x, y, theta): move to position (x, y) with angle theta
- heading(): get the current angle of the turtle
- isdown(): check if the pen is down"""

        if self.use_forkstate:
            embed_fork_exp = """
- with fork_state(): a context manager that runs the code in the block using the current context and restores the original state afterwards. Allows you to nest programs. Internally, fork_state saves the turtle state (is_down, x, y, heading), executes the block, then restores the original state."""
        else:
            embed_fork_exp = """
- embed(program, local vars): runs the code in program using the current context and teleports back to the original position. Allows you to nest programs. Implementationally, embed gets the turtle state (is down, x, y, heading), executes program, then returns to the original state."""
        self._system_prompt = base_sys_prompt + embed_fork_exp

    
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

def parse_adapter_path(val):
    if val in [None, "None", "null", ""]:
        return None
    return str(val)

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
            adapter_path=parse_adapter_path(config_dict["model"].get("adapter_path", None)),
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
            include_sys_prompt_inf=bool(config_dict["data"]["include_sys_prompt_inf"]),
            use_forkstate=self.data.use_forkstate
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
        if config.model.adapter_path is not None: # config.model.adapter_path not in ["None", "null", ""]:
            model_type = config.model.adapter_path.split('/')[2]
        else:
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



def build_val_dataset_hash(val_dataset, config, tokenizer):
    """
    Build a mapping from the hash (as used in val_detailed_metrics.jsonl) to the Description,
    by reproducing the exact tokenization and detokenization pipeline.
    """
    val_dataset_hash = {}

    prompt_template = config.prompt.get_prompt_template(
        include_desc=config.data.include_desc,
        include_ascii=config.data.include_ascii
    )

    for idx, row in enumerate(val_dataset):
        # 1. Format prompt and completion
        formatted_prompt = prompt_template.format(**row)
        completion = row['Program']

        # 2. Build messages as in pipeline
        messages = []
        if config.prompt.include_sys_prompt_fn:
            messages.append({"role": "system", "content": config.prompt._system_prompt})
        messages.append({"role": "user", "content": formatted_prompt})
        messages.append({"role": "assistant", "content": completion})

        # 3. Build prompt_messages for prompt_length calculation
        prompt_messages = []
        if messages and messages[0]["role"] == "system":
            prompt_messages.append(messages[0])
        for msg in messages:
            if msg["role"] == "user":
                prompt_messages.append(msg)

        prompt_text = tokenizer.apply_chat_template(
            conversation=prompt_messages,
            tokenize=False, 
            add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 4. Tokenize and mask
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=config.training.max_seq_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        offsets = tokenized["offset_mapping"]
        prompt_length = len(prompt_text)
        completion_mask = [0 if offset[1] <= prompt_length else 1 for offset in offsets]

        # 5. Extract completion tokens
        input_ids = tokenized["input_ids"]
        completion_token_ids = [tid for tid, m in zip(input_ids, completion_mask) if m == 1]

        # 6. Detokenize
        decoded_completion = tokenizer.decode(completion_token_ids, skip_special_tokens=True)

        # 7. Hash and map to description
        hash_val = hashlib.md5(decoded_completion.encode("utf-8")).hexdigest()
        val_dataset_hash[hash_val] = row["Description"]

    return val_dataset_hash

def get_val_dataset_hash(config_file, val_dataset=None):
    """
    Given a config file and (optionally) a validation dataset,
    returns the hash-to-description mapping as used in val_detailed_metrics.jsonl.
    """
    # Load configuration
    config, _, _, _, _ = load_config(config_file, fine_tune=False)
    _, tokenizer = load_model_and_tokenizer(config)

    # Load validation dataset if not provided
    if val_dataset is None:
        from datasets import load_dataset
        val_dataset = load_dataset(config.data.dataset_id, split='validation')
    
    # Apply modifications
    val_dataset = apply_modifications(val_dataset, config, modifier=modifier, ascii_processor=ascii_processor)


    # Build and return the hash mapping
    return build_val_dataset_hash(val_dataset, config, tokenizer)

def get_val_dataset_hash(config_file, val_dataset=None):
    """
    Given a config file and (optionally) a validation dataset,
    returns the hash-to-description mapping as used in val_detailed_metrics.jsonl.
    """
    # Load configuration
    config, _, _, _, _ = load_config(config_file, fine_tune=False)
    _, tokenizer = load_model_and_tokenizer(config)

    # Load validation dataset if not provided
    if val_dataset is None:
        from datasets import load_dataset
        val_dataset = load_dataset(config.data.dataset_id, split='validation')

    # --- Initialize modifier and ascii_processor here ---
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
        resize_img = config.data.ascii_parameters.get("resize_img", None)
        ascii_processor = AdaptiveASCIIProcessor(
            levels=10,
            black_threshold=black_threshold,
            block_size=block_size,
            crop_to_size=crop_to_size,
            resize_img=resize_img,
            drop_images=True
        )
    # ---------------------------------------------------

    # Apply modifications
    val_dataset = apply_modifications(val_dataset, config, modifier=modifier, ascii_processor=ascii_processor)

    # Build and return the hash mapping
    return build_val_dataset_hash(val_dataset, config, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build hash mapping for validation set.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    try:
        config_file = args.config_file
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
            resize_img = config.data.ascii_parameters.get("resize_img", None)
            ascii_processor = AdaptiveASCIIProcessor(levels=10,
                                                        black_threshold=black_threshold,
                                                        block_size=block_size,
                                                        crop_to_size=crop_to_size,
                                                        resize_img=resize_img,
                                                        drop_images=True
                                                    )
        # Set random seed for reproducibility
        set_random_seeds(config.training.random_seed)
        
        val_dataset_hash = get_val_dataset_hash(config_file, val_dataset=None)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()