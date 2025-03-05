"""
This file contains a pipeline for finetuning and inference of a model.
It is designed to handle single GPU training and inference using unsloth and also 
multi-GPU training and inference using torch with prompt loss weighting support.
"""
# 1. GPU Setup and Utilities - housekeeping related to single/multi-GPU training
# Initialize the distributed environment
import os
import torch
if 'LOCAL_RANK' in os.environ and torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)  # Get major and minor version
    cuda_arch = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch
    print(f"Setting TORCH_CUDA_ARCH_LIST={cuda_arch}")
import yaml
import json
from datetime import datetime
# Basic imports that both setups need
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from dataclasses import dataclass, field
from typing import List, Set, Union, Optional, Dict, Any
from enum import Enum
import wandb
from Levenshtein import distance as levenshtein_distance
import numpy as np

# Import appropriate backend
if 'LOCAL_RANK' in os.environ:
    import deepspeed
    deepspeed.init_distributed("nccl")
else:
    from unsloth import FastLanguageModel #, is_bfloat16_supported

def is_bfloat16_supported():
    """Check if bfloat16 is supported on the current device"""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0]
        return capability >= 8  # Ampere and newer GPUs support bf16
    return False

print("all imports done")
################
# Setup
################
# 1.) Argument parsing
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
    warmup_steps: int
    lr_scheduler_type: str
    train_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    random_seed: int
    shuffle: bool

@dataclass
class ModelConfig:
    model_id: str
    topk_train: int
    topk_prompt: int

@dataclass
class LoggingConfig:
    use_wandb: bool

@dataclass
class DataConfig:
    dataset_id: str
    include_desc: bool
    include_ascii: bool

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PromptConfig:
    include_sys_prompt: bool = True
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
    def system_prompt(self) -> Optional[str]:
        return self._system_prompt if self.include_sys_prompt else None
    
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
        
        # Wrap the task description and system prompt with appropriate tokens
        prompt = "[INST]"
        if self.include_sys_prompt:
            prompt += f"[SYS]{self.system_prompt}[/SYS]"
        prompt += task_description + "[/INST]"
        return prompt

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
            warmup_steps=int(config_dict["training"]["warmup_steps"]),
            lr_scheduler_type=str(config_dict["training"]["lr_scheduler_type"]),
            train_epochs=int(config_dict["training"]["train_epochs"]),
            per_device_batch_size=int(config_dict["training"]["per_device_batch_size"]),
            gradient_accumulation_steps=int(config_dict["training"]["gradient_accumulation_steps"]),
            save_steps=int(config_dict["training"]["save_steps"]),
            eval_steps=int(config_dict["training"]["eval_steps"]),
            logging_steps=int(config_dict["training"]["logging_steps"]),
            random_seed=int(config_dict["training"]["random_seed"]),
            shuffle=bool(config_dict["training"]["shuffle"])
        )
        self.model = ModelConfig(
            model_id=str(config_dict["model"]["model_id"]),
            topk_train=int(config_dict["model"]["topk_train"]),
            topk_prompt=int(config_dict["model"]["topk_prompt"])
        )
        self.logging = LoggingConfig(
            use_wandb=bool(config_dict.get("logging", {}).get("use_wandb", False))
        )
        self.data = DataConfig(
            dataset_id=str(config_dict["data"]["dataset_id"]),
            include_desc=bool(config_dict["data"]["include_desc"]),
            include_ascii=bool(config_dict["data"]["include_ascii"])
        )
        self.prompt = PromptConfig(
            include_sys_prompt=bool(config_dict.get("prompt", {}).get("include_sys_prompt", True))
        )

def load_config(model_name):
    """Load training configuration from yaml file and store a copy in results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/data/{model_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    source_config = "config.yaml"
    target_config = os.path.join(result_dir, "config.yaml")
    
    if os.path.exists(source_config):
        with open(source_config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validiere und konvertiere Konfiguration mit der Config-Klasse
        config = Config(config_dict)
        
        # Copy the config to results directory
        with open(target_config, 'w') as f:
            yaml.dump(config_dict, f)
    else:
        raise FileNotFoundError(
            "No config.yaml file found in current directory. "
            "Please create a config.yaml file with your training configuration."
        )
    print(f"Loaded configuration from {source_config}\n{result_dir}")
    return config, result_dir

# 2.) GPU setup
def is_main():
    """Check if this is the main process (rank 0) or single GPU"""
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    return True

def main(func):
    """Decorator to run function only on main process"""
    def wrapper(*args, **kwargs):
        if is_main():
            result = func(*args, **kwargs)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return result
        else:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return None
    return wrapper

@main
def print_gpus():
    """Print available GPU information"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"Number of CUDA devices: {num_gpus}")
        for idx, name in enumerate(gpu_names):
            print(f"GPU {idx}: {name}")
    else:
        print("CUDA is not available.")

# 3.) Model setup
def setup_model(config: Config, multi_gpu=False):
    """Setup model based on single/multi GPU configuration"""
    # Let SLURM handle GPU visibility
    n_gpus = torch.cuda.device_count()
    if is_main():
        print(f"Number of visible GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if multi_gpu and n_gpus > 1:
        # Multi-GPU setup with DeepSpeed
        print("Using DeepSpeed for multi-GPU training")
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if is_main():
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.model_id,
                trust_remote_code=True,
            )
    else:
        # Force single GPU for unsloth
        if n_gpus > 1:
            print("Multiple GPUs detected. Forcing single GPU (0) for unsloth")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            torch.cuda.set_device(0)
        
        print("Using unsloth for single-GPU training")
        model, tokenizer = FastLanguageModel.from_pretrained(
            config.model.model_id,
            load_in_4bit=True,
        )
    
    # Add dedicated padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if not multi_gpu:  # Unsloth handles embedding resize internally
            model.resize_token_embeddings(len(tokenizer))
    
    if is_main():
        print("Initialized model and tokenizer")
        print(f"Padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    
    return model, tokenizer

###############
# Data Prep.
###############
if 'LOCAL_RANK' in os.environ:
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist

def prepare_dataset(dataset, config: Config, tokenizer, multi_gpu=False):
    """Prepares the dataset according to configuration"""
    if is_main():
        print(f"Preparing dataset with {len(dataset)} examples")
    def format_prompt(example, idx):
        """Formats a single example with the template"""
        prompt = config.prompt.get_prompt_template(
            include_desc=config.data.include_desc,
            include_ascii=config.data.include_ascii
        )
        
        formatted_prompt = prompt.format(**example)
        completion = example['Program']
        
        full_text = f"{formatted_prompt}\n{completion}"
        prompt_length = len(formatted_prompt) + 1  # +1 for newline
        
        return {
            "text": full_text,
            "prompt_length": prompt_length
        }

    # Format dataset with distributed-aware mapping
    formatted_dataset = dataset.map(
        format_prompt,
        with_indices=True,
        desc="Formatting prompts",
        load_from_cache_file=False  # Important for distributed setup
    )
    
    def tokenize_and_mask(examples):   
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.training.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        prompt_masks = []
        completion_masks = []
        
        for offsets, length in zip(tokenized["offset_mapping"], examples["prompt_length"]):
            prompt_mask = [1 if offset[1] <= length else 0 for offset in offsets]
            completion_mask = [0 if offset[1] <= length else 1 for offset in offsets]
            prompt_masks.append(prompt_mask)
            completion_masks.append(completion_mask)
        
        tokenized["prompt_mask"] = prompt_masks
        tokenized["completion_mask"] = completion_masks
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        del tokenized["offset_mapping"]
        del examples["text"]  # Remove after tokenization
    return tokenized
    
    # Tokenize with consistent settings
    tokenized_dataset = formatted_dataset.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=formatted_dataset.column_names,
        desc="Tokenizing and masking",
        load_from_cache_file=False
    )
    
    if is_main():
        print(f"Dataset preparation complete. Size: {len(tokenized_dataset)}")
    
    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(tokenized_dataset, shuffle=True) if multi_gpu else None

    dataloader = DataLoader(
    tokenized_dataset,
    batch_size=config.training.per_device_batch_size,
    sampler=sampler,
    shuffle=(sampler is None),
    num_workers=2,
    pin_memory=True
    )
    torch.cuda.empty_cache()
    return dataloader

###############
# Finetuning
###############
def prepare_training(model, config: Config, multi_gpu=False):
    """Prepare model for training with appropriate configuration"""
    if multi_gpu:
        # For multi-GPU, we'll let the Trainer handle DeepSpeed initialization
        return model
    else:
        # Single-GPU using unsloth
        return FastLanguageModel.get_peft_model(
            model,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            use_gradient_checkpointing=True
        )

def init_wandb(config: Config):
    """Initialize wandb with configuration"""
    if not is_main():
        return 
    gen_type = ""
    if "generalization" in config.data.dataset_id:
        dataset_name = config.data.dataset_id.split('/')[-1]
        parts = dataset_name.split('-')
        for i, part in enumerate(parts):
            if "generalization" in part:
                gen_type = parts[i-1]
                break
    model_type = config.model.model_id.split('/')[-1]
    model_type_short = model_type.split('-')[0]

    # Generate a unique experiment name
    experiment_name = f"{gen_type}_{model_type_short}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
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
    
    # Init wandb
    wandb.init(
        project="master-thesis",
        name=experiment_name,
        config=config_dict
    )

# Custom Metrics
from torch.nn import CrossEntropyLoss

def prepare_compute_metrics(eval_dataset, tokenizer):
    """Prepare custom metrics function for Trainer"""

    # extract prompt/completion masks
    prompt_mask = np.array([x["prompt_mask"] for x in eval_dataset])
    completion_mask = np.array([x["completion_mask"] for x in eval_dataset])

    # uses numpy arrays (on CPU)
    def compute_metrics(data):

        # data.predictions contains the tuple (token_preds, token_losses)
        # from the preprocess_logits_for_metrics function (below)
        token_preds, token_losses = data.predictions

        # shift labels and masks
        labels = data.label_ids[..., 1:]
        shift_prompt_mask = prompt_mask[..., 1:]
        shift_comp_mask = completion_mask[..., 1:]

        # average both losses (prompt and completion) over their respective tokens
        prompt_loss = token_losses.reshape(-1) @ shift_prompt_mask.reshape(-1) / shift_prompt_mask.sum()
        completion_loss = token_losses.reshape(-1) @ shift_comp_mask.reshape(-1) / shift_comp_mask.sum()

        # compute levenshtein-distance (edit-distance) between pred-programs and grund-truth programs
        # 1st: identify completion tokens
        true_comp_tokens = [l[m == 1] for l, m in zip(labels, shift_comp_mask)]
        pred_comp_tokens = [p[m == 1] for p, m in zip(token_preds, shift_comp_mask)]
        # 2nd: decode completion tokens => program 
        true_program = tokenizer.batch_decode(true_comp_tokens, skip_special_tokens=True)
        pred_program = tokenizer.batch_decode(pred_comp_tokens, skip_special_tokens=True)
        # 3rd: compute levenshtein-distance
        distances = [levenshtein_distance(pred, true) for pred, true in zip(pred_program, true_program)]


        # return metrics
        return {
            'comp_loss': completion_loss,
            'prompt_loss': prompt_loss,
            'levenshtein_dist': distances,
        }
    return compute_metrics

# uses PyTorch tensors (on GPU)
def preprocess_logits_for_metrics(logits, labels):
    # get predictions
    token_preds = logits.argmax(-1)[..., :-1]

    # compute per-token losses
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

    # pass predictions and losses to compute_metrics function (above)
    predictions = (token_preds, token_losses)
    return predictions



# Custom Trainer
def train_model(model, tokenizer, train_dataset, eval_dataset, config: Config, multi_gpu=False):
    """Training loop with configurable prompt and completion weights"""
    
    # Only if set True in config logging to wandb will be enabled
    if config.logging.use_wandb and is_main():
        init_wandb(config)
        report_to = "wandb"
    else:
        report_to = "none"
    
    class PLWTrainer(Trainer):
        def __init__(self, *args, prompt_loss_weight=1.0, shuffle=False, **kwargs):
            self.processor = kwargs.pop('tokenizer', None)
            super().__init__(*args, **kwargs)
            self.prompt_loss_weight = prompt_loss_weight
            self.shuffle = shuffle
            self.distributed_training = torch.distributed.is_initialized()

        def compute_loss(self, model, inputs, return_outputs=False):
            print(f"compute_loss called with num_items_in_batch: {num_items_in_batch}, kwargs: {kwargs}") # DEBUG

            # get outputs without computing loss (by not passing in labels)
            outputs = model(input_ids=inputs["input_ids"], 
                            attention_mask=inputs["attention_mask"])
            logits = outputs.get("logits")
            labels = inputs.pop("labels")

            # compute per-token weights
            weights = self.prompt_loss_weight * inputs["prompt_mask"] + inputs["completion_mask"]

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)

            # per-token losses
            loss_fct = CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                    shift_labels.view(-1))

            # Compute weighted average of losses
            loss = (token_losses.float() @ shift_weights.view(-1).float()) / shift_weights.sum()
            return (loss, outputs) if return_outputs else loss
        
        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Training requires a train_dataset.")

            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self._get_train_sampler(),
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        # this allows us to toggle on/off data shuffling, which can sometimes cause 'staircase' effects in training loss
        def _get_train_sampler(self):
            if self.distributed_training:
                return torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=self.shuffle)
            if self.shuffle:
                return torch.utils.data.RandomSampler(self.train_dataset)
            return torch.utils.data.SequentialSampler(self.train_dataset)



    # Training arguments based on config
    training_args = TrainingArguments(
        output_dir=f"results/models/{config.model.model_id}", 
        num_train_epochs=config.training.train_epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        remove_unused_columns = False,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        deepspeed=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "zero3_decay.json") if multi_gpu else None,
        seed=config.training.random_seed, 
        report_to=report_to,  # dynamic wandb reporting
        gradient_checkpointing=True,     # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": True},
        weight_decay=0.01  # Set weight_decay to match DeepSpeed config
    )

    # Initialize trainer with configurable weights
    trainer = PLWTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        shuffle=config.training.shuffle,
        prompt_loss_weight=config.training.prompt_loss_weight,
        #processing_class=tokenizer,
        compute_metrics=prepare_compute_metrics(eval_dataset, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train the model
    trainer.train()
    
    # Logging is only disabled if it was enabled before
    if config.logging.use_wandb and is_main():
        wandb.finish()
    
    return model

###############
# Inference
###############
def run_inference(model, tokenizer, prompt):
    """Run inference on trained model"""
    # ... inference implementation ...
    pass

###############
# Evaluation
###############
def evaluate_model(model, tokenizer, eval_dataset):
    """Evaluate model performance"""
    # ... evaluation implementation ...
    pass



###############
# Main
###############
if __name__ == "__main__":
    try:
        # Check if running with torchrun
        multi_gpu = torch.cuda.device_count() > 1 and 'LOCAL_RANK' in os.environ
        if multi_gpu:
            print("Multi-GPU training with DeepSpeed")
        else:
            print("Single-GPU training with unsloth")
                
        # Print GPU information
        if is_main():
            print_gpus()
        
        # Load configuration
        if is_main():
            print("Loading configuration...")
        config, result_dir = load_config("config.yaml")
        if is_main():
            print("Configuration loaded.")
        
        # Setup model
        if is_main():
            print("Setting up model...")
        model, tokenizer = setup_model(config, multi_gpu)
        if is_main():
            print("Model setup complete.")
        
        # Broadcast tokenizer to all processes
        if multi_gpu:
            if is_main():
                tokenizer_data = tokenizer.save_pretrained(".")
            torch.distributed.barrier()
            if not is_main():
                tokenizer = AutoTokenizer.from_pretrained(".")
        
        # Prepare model for training
        if is_main():
            print("Preparing model for training...")
        model = prepare_training(model, config, multi_gpu)
        if is_main():
            print("Model preparation complete.")
        
        # Load training and evaluation datasets
        if is_main():
            print("Loading datasets...")
        from datasets import load_dataset
        dataset = load_dataset(config.data.dataset_id)
        #train_dataset = dataset['train']
        #eval_dataset = dataset['validation']        
        # Select a sample of 0.02% of the data
        sample_fraction = 0.0002
        train_dataset = dataset['train'].shuffle(seed=config.training.random_seed).select(range(int(len(dataset['train']) * sample_fraction)))
        eval_dataset = dataset['validation'].shuffle(seed=config.training.random_seed).select(range(int(len(dataset['validation']) * sample_fraction)))
        if is_main():
            print("Sampled datasets.")
        
        # Prepare datasets
        if is_main():
            print("Preparing datasets...")
        train_dataloader = prepare_dataset(train_dataset, config, tokenizer, multi_gpu)
        eval_dataloader = prepare_dataset(eval_dataset, config, tokenizer, multi_gpu)
        if is_main():
            print("Datasets prepared.")
        
        # Train the model
        if is_main():
            print("Training model...")
        model = train_model(model, tokenizer, train_dataloader, eval_dataloader, config, multi_gpu)
        if is_main():
            print("Model training complete.")
        
        # Inference and evaluation (placeholders)
        # test_dataset = dataset['test']
        # prompt = ""
        # run_inference(model, tokenizer, prompt)
        # evaluate_model(model, tokenizer, eval_dataset)
    except Exception as e:
        if is_main():
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()