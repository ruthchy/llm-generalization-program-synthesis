"""
This file contains a pipeline for finetuning and inference of a model.
It is designed to handle single GPU training and inference using unsloth and also 
multi-GPU training and inference using deepspeed with prompt loss weighting support.
"""
import os
import yaml
from datetime import datetime
import json
import torch

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

# Determine multi-GPU setup first
multi_gpu = 'LOCAL_RANK' in os.environ

# Import appropriate backend
if multi_gpu:
    import deepspeed
else:
    from unsloth import FastLanguageModel, is_bfloat16_supported

def is_bfloat16_supported():
    """Check if bfloat16 is supported on the current device"""
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()[0]
        return compute_capability >= 8  # Ampere and newer GPUs support bf16
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
    completion_loss_weight: float
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
        if include_ascii and include_desc:
            return "[INST] Here is a gray scale image described as containing {Description}. The image is represented with integer values 0-9:\n{ASCII-Art}\nPlease write a Python program that generates this image using our custom turtle module. [/INST]"
        elif include_ascii:
            return "[INST] Here is a gray scale image represented with integer values 0-9:\n{ASCII-Art}\nPlease write a Python program that generates this image using our custom turtle module. [/INST]"
        elif include_desc:
            return "[INST] Here is a gray scale image described as containing {Description}\nPlease write a Python program that generates this image using our custom turtle module. [/INST]"
        else:
            raise ValueError("At least one of include_ascii or include_desc must be True")

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
            completion_loss_weight=float(config_dict["training"]["completion_loss_weight"]),
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
def prepare_dataset(dataset, config: Config, tokenizer):
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
        if is_main():
            print(f"Processing batch of size {len(examples['text'])}")
            
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
        torch.cuda.empty_cache()  # Clear GPU cache periodically
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
    
    return tokenized_dataset

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

def train_model(model, tokenizer, train_dataset, eval_dataset, config: Config, multi_gpu=False):
    """Training loop with configurable prompt and completion weights"""
    
    # Only if set True in config logging to wandb will be enabled
    if config.logging.use_wandb:
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

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Compute weighted loss for prompts and completions
            
            Args:
                model: The model to compute loss for
                inputs: Dictionary containing input tensors
                return_outputs: Whether to return model outputs
                num_items_in_batch: Number of items in batch (used by Unsloth)
            """
            outputs = model(input_ids=inputs["input_ids"], 
                          attention_mask=inputs["attention_mask"])
            logits = outputs.get("logits")
            labels = inputs.pop("labels")

            # Ensure masks are on the correct device
            prompt_mask = inputs["prompt_mask"].to(logits.device)
            completion_mask = inputs["completion_mask"].to(logits.device)
            
            # Regular loss computation with proper device placement
            weights = self.prompt_loss_weight * prompt_mask + completion_mask
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = (token_losses.float() @ shift_weights.view(-1).float()) / shift_weights.sum()

            # Log metrics if wandb is enabled
            if self.processor and wandb.run:
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    pred_texts = self.processor.batch_decode(predictions, skip_special_tokens=True)
                    label_texts = self.processor.batch_decode(labels, skip_special_tokens=True)
                    
                    pred_programs = [text.split("\n")[-1] for text in pred_texts]
                    label_programs = [text.split("\n")[-1] for text in label_texts]
                    
                    distances = [levenshtein_distance(pred, label) 
                               for pred, label in zip(pred_programs, label_programs)]
                    avg_distance = np.mean(distances)
                    
                    wandb.log({
                        "train/levenshtein_distance": avg_distance,
                        "train/loss": loss.item()
                    }, step=self.state.global_step)

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

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """Override prediction_step to handle missing masks in eval"""
            if "prompt_mask" not in inputs or "completion_mask" not in inputs:
                # During evaluation, create default masks if they don't exist
                seq_length = inputs["input_ids"].shape[1]
                device = inputs["input_ids"].device
                
                # Default to treating everything as completion
                inputs["prompt_mask"] = torch.zeros((inputs["input_ids"].shape[0], seq_length), device=device)
                inputs["completion_mask"] = torch.ones((inputs["input_ids"].shape[0], seq_length), device=device)
            
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


    # Define compute_metrics and preprocess_logits_for_metrics here or import them
    def preprocess_logits_for_metrics(logits, labels):
        """Preprocess logits before metric computation (on GPU)"""
        # Get predictions for next token
        token_preds = logits.argmax(-1)[..., :-1]
        
        # Compute per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        
        # Return both predictions and losses
        return (token_preds, token_losses)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        """Compute metrics keeping operations on GPU as long as possible"""
        token_preds, token_losses = eval_pred.predictions
        labels = eval_pred.label_ids[..., 1:]  # Shift labels
        
        # Move to GPU and ensure proper dimensions
        device = torch.cuda.current_device()
        token_preds = torch.tensor(token_preds, device=device)
        token_losses = torch.tensor(token_losses, device=device)
        labels = torch.tensor(labels, device=device)
        
        # Create masks on GPU (shifted to match predictions)
        seq_length = labels.shape[1]
        batch_size = labels.shape[0]
        
        # Default mask treating everything as completion
        completion_mask = torch.ones((batch_size, seq_length), 
                                   dtype=torch.bool, 
                                   device=device)
        
        # Compute metrics on GPU
        metrics = {}
        
        try:
            # Compute losses for completion
            completion_losses = token_losses.masked_select(completion_mask)
            if completion_losses.numel() > 0:
                metrics["completion_loss"] = completion_losses.mean().item()
            
            # Get predictions for non-padding tokens
            valid_mask = labels != tokenizer.pad_token_id
            valid_preds = token_preds.masked_select(valid_mask)
            valid_labels = labels.masked_select(valid_mask)
            
            # Move to CPU only for final text processing
            valid_preds = valid_preds.cpu()
            valid_labels = valid_labels.cpu()
            
            # Decode and compute distances
            pred_texts = tokenizer.batch_decode(valid_preds.unsqueeze(1), skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(valid_labels.unsqueeze(1), skip_special_tokens=True)
            
            # Extract programs and compute metrics
            pred_programs = [text.split("\n")[-1] for text in pred_texts]
            label_programs = [text.split("\n")[-1] for text in label_texts]
            
            distances = [levenshtein_distance(p, l) for p, l in zip(pred_programs, label_programs)]
            metrics["levenshtein_dist"] = float(np.mean(distances))
            metrics["levenshtein_dist_std"] = float(np.std(distances))
            metrics["exact_match"] = 100 * float(np.mean([p == l for p, l in zip(pred_programs, label_programs)]))
            
        except Exception as e:
            print(f"Error in metric computation: {str(e)}")
            metrics = {
                "levenshtein_dist": -1,
                "levenshtein_dist_std": -1,
                "exact_match": 0.0,
                "completion_loss": float('inf')
            }
        
        finally:
            # Always clear GPU cache
            torch.cuda.empty_cache()
        
        return metrics

    # Training arguments based on config
    training_args = TrainingArguments(
        output_dir=f"results/models/{config.model.model_id}", 
        num_train_epochs=config.training.train_epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
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
    )

    # Initialize trainer with configurable weights
    trainer = PLWTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        shuffle=config.training.shuffle,
        prompt_loss_weight=config.training.prompt_loss_weight,
        processing_class=tokenizer,  # Updated from tokenizer to processing_class
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train the model
    trainer.train()
    
    # Logging is only disabled if it was enabled before
    if config.logging.use_wandb:
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
    # Check if running with torchrun
    multi_gpu = 'LOCAL_RANK' in os.environ
    if multi_gpu:
        print("Multi-GPU training with DeepSpeed")
    else:
        print("Single-GPU training with unsloth")
    
    # Set GPU environment
    multi_gpu = torch.cuda.device_count() > 1 and 'LOCAL_RANK' in os.environ
    
    # Print GPU information
    print_gpus()
    
    # Load configuration
    config, result_dir = load_config("config.yaml")
    
    # Setup model
    model, tokenizer = setup_model(config, multi_gpu)
    
    # Prepare model for training
    model = prepare_training(model, config, multi_gpu)
    
    # Load training and evaluation datasets (Beispiel)
    # Hier kannst du deine eigenen Datensätze laden
    from datasets import load_dataset
    dataset = load_dataset(config.data.dataset_id)
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    # Datensätze vorbereiten
    train_dataset = prepare_dataset(train_dataset, config, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, config, tokenizer)
    
    # Modell trainieren
    model = train_model(model, tokenizer, train_dataset, eval_dataset, config, multi_gpu)
    
    # Inferenz und Evaluierung (Platzhalter)
    prompt = f"Given this ASCII art:\n{ascii}\nGenerate the Logo program that draws this pattern:"
    run_inference(model, tokenizer, prompt)
    evaluate_model(model, tokenizer, eval_dataset)