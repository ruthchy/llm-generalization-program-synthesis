'''
Steps:
    1. Load the YAML Configuration
    2. Load the Model and Tokenizer
    3. Prepare the Dataset
    4. Wandb initalization for the fine-tuning run
    5. Training Preperation
    6. Training the model with custom PLW-Trainer
    7. Preparing Inference
    8.1 Inference using the recently fine-tuned model
    8.2 Inference using model from hub
    9. Evaluation                                       # not yet implemented
    
Run script in conda thesis_env (can be gererated using the requirements.txt file)
python pipeline.py --fine_tune False --sample_fraction 0.1

python pipeline.py 
    --fine_tune False       # False = only inf with a model from the hub (Step 1, 2, 7, 8.2, 9)
    --fine_tune True        # True = fine-tune the model and do inf (Step 1, 2, 3, 4, 5, 6, 7, 8.1, 9)
    --sample_fraction 1.0   # the entire dataset is used or by setting the number < 1 a random sample of the dataset is used
'''
############################################################################################################
# Housekeeping - single GPU unsloth setup
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # new

# Decide the mode in which the script should run fine-tuning and/or inference; in testing or production mode
import argparse
parser = argparse.ArgumentParser(description='Run fine-tuning and/or inference pipeline')
parser.add_argument('--fine_tune', type=lambda x: x.lower() == 'true', default=False, help='Whether to fine-tune the model (True) or run inference only (False)')
parser.add_argument('--sample_fraction', type=float, default=1.0, help='Fraction of dataset to use (for debugging)')
parser.add_argument('--config', type=str, default="config.yaml", help='Path to config file')
args = parser.parse_args()
        
fine_tune = args.fine_tune
sample_fraction = args.sample_fraction
config_file = args.config
print(f"Running with fine_tune={fine_tune}, sample_fraction={sample_fraction}, config={config_file}")

# Step 1: Load the YAML Configuration
import yaml
import torch
import wandb
import json
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from unsloth import FastLanguageModel, is_bfloat16_supported
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments, Trainer, get_scheduler, EarlyStoppingCallback
#from transformers import default_data_collator
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from Levenshtein import distance as levenshtein_distance

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
    top_k: int
    temperature: float
    max_new_tokens: int

@dataclass
class LoggingConfig:
    use_wandb: bool

@dataclass
class DataConfig:
    dataset_id: str
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
            top_k=int(config_dict["model"]["top_k"]),
            temperature=int(config_dict["model"]["temperature"]),
            max_new_tokens=int(config_dict["model"]["max_new_tokens"])
        )
        self.logging = LoggingConfig(
            use_wandb=bool(config_dict.get("logging", {}).get("use_wandb", False))
        )
        self.data = DataConfig(
            dataset_id=str(config_dict["data"]["dataset_id"]),
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
        if "generalization" in dataset_id:
            dataset_name = dataset_id.split('/')[-1]
            parts = dataset_name.split('-')
            for i, part in enumerate(parts):
                if "generalization" in part and i > 0:
                    gen_type = parts[i - 1]
                    break

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

    print(f"Loaded configuration from {source_config}\n{result_dir}")
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
    print(f"Random seed set to: {seed}")

# Step 2: Load the Model and Tokenizer
def load_model_and_tokenizer(config: Config):
    model, tokenizer = FastLanguageModel.from_pretrained(
            config.model.model_id,
            load_in_4bit=True,
        )

    model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            use_gradient_checkpointing=True)
   
    return  model, tokenizer

# Step 3: Prepare the Dataset
def prepare_dataset(config: Config, tokenizer, sample_fraction = 1.0):
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
    if config.data.mix_directions:
        from synthetic_data.__dataset_direction_modifier import DatasetDirectionModifier
        modifier = DatasetDirectionModifier(random_seed=config.training.random_seed)
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
        from synthetic_data.__adapt_ascii_processor import AdaptiveASCIIProcessor
        black_threshold = config.data.ascii_parameters.get("black_threshold", 150)
        block_size = config.data.ascii_parameters.get("block_size", None)
        crop_to_size = config.data.ascii_parameters.get("crop_to_size", None)
        ascii_processor = AdaptiveASCIIProcessor(levels=10,
                                                 black_threshold=black_threshold,
                                                 block_size=block_size,
                                                 crop_to_size=crop_to_size,drop_images=True)
        dataset = ascii_processor.process_dataset(dataset)


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
        if idx < 3:  # Only print the first few examples for each split
            print(f"\n--- DEBUG: {split_type} EXAMPLE #{idx} ---")
            print(f"System prompt included: {config.prompt.include_sys_prompt_fn if split_type != 'test' else config.prompt.include_sys_prompt_inf}")
            print(f"Messages format: {messages}")
            print(f"PROMPT TEXT (for length calculation):\n{prompt_text}")
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
        
        tokenized["prompt_mask"] = prompt_masks
        tokenized["completion_mask"] = completion_masks
        tokenized["labels"] = tokenized["input_ids"].copy() # labels are grund truth
        
        del tokenized["offset_mapping"]
        del examples["text"]  # Remove after tokenization
        return tokenized
    
    tokenized_dataset = {}
    for split in formatted_dataset.keys():
        tokenized_dataset[split] = formatted_dataset[split].map(
            tokenize_and_mask,
            batched=True,
            remove_columns=formatted_dataset[split].column_names, 
            desc=f"Tokenizing {split} set",
            load_from_cache_file=False
        )
    
    return tokenized_dataset

# Step 4: WandB 
def init_wandb(config: Config, timestamp: str, gen_type: str, model_type_short: str):
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
    
    # Init wandb
    wandb.init(
        project="master-thesis",
        name=experiment_name,
        config=config_dict
    )
# Step 5: Training Preperation
# Custom Metrics
from torch.nn import CrossEntropyLoss
from Levenshtein import distance as levenshtein_distance

def prepare_compute_metrics(dataset: DatasetDict, tokenizer):
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
        from __eval import LLMCodeEvaluator

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

        # Compute Levenshtein distance
        try:
            true_comp_tokens = [l[m == 1] for l, m in zip(labels, shift_comp_mask)]
            pred_comp_tokens = [p[m == 1] for p, m in zip(token_preds, shift_comp_mask)]
            
            true_program = tokenizer.batch_decode(true_comp_tokens, skip_special_tokens=True)
            pred_program = tokenizer.batch_decode(pred_comp_tokens, skip_special_tokens=True)

            # Clean code before Levenshtein calculation (remove py comments)
            clean_true_program = [LLMCodeEvaluator.clean_python_code(code) for code in true_program]
            clean_pred_program = [LLMCodeEvaluator.clean_python_code(code) for code in pred_program]

            # Calculate normalized Levenshtein distances using cleaned code (1.0 = perfect match, 0.0 = no match)
            normalized_distances = []
            for pred, true in zip(clean_pred_program, clean_true_program):
                max_len = max(len(pred), len(true))
                if max_len == 0:  # Handle edge case of empty strings
                    normalized_distances.append(1.0)
                else:
                    lev_dist = levenshtein_distance(pred, true)
                    normalized_distances.append(1.0 - (lev_dist / max_len))
            
            avg_norm_levenshtein_dist = np.mean(normalized_distances)
            std_norm_levenshtein_dist = np.std(normalized_distances)
        except Exception as e:
            print(f"Error computing normalized Levenshtein distance: {str(e)}")
            avg_norm_levenshtein_dist = 0.0
            std_norm_levenshtein_dist = 0.0
        
        # Clean up to free memory, especially important for training batches
        if split == 'train':
            # Force cleanup of large arrays
            del token_preds, token_losses, labels, shift_prompt_mask, shift_comp_mask
            
        # Return metrics with Python native types
        return {
            'comp_loss': float(completion_loss),
            'prompt_loss': float(prompt_loss),
            'avg_norm_levenshtein_dist': float(avg_norm_levenshtein_dist),
            'std_norm_levenshtein_dist': float(std_norm_levenshtein_dist),
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

# Step 6. Training the model with custom PLW-Trainer
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl

def train_model(model, tokenizer, dataset, result_dir: str, config: Config, timestamp: str, gen_type: str, model_type_short: str):
    """Training loop with configurable prompt and completion weights"""
    
    # Only if set True in config logging to wandb will be enabled
    if config.logging.use_wandb:
        init_wandb(config, timestamp, gen_type, model_type_short)
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
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])
            logits = outputs.get("logits")
            labels = inputs.pop("labels")

            # Compute per-token weights
            weights = self.prompt_loss_weight * inputs["prompt_mask"] + inputs["completion_mask"]

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            # Move tensors to correct device
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)

            # Compute per-token loss
            loss_fct = CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Compute weighted average of losses
            loss = (token_losses.float() @ shift_weights.view(-1).float()) / shift_weights.sum()

            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Training requires a train_dataset.")

            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self._get_train_sampler(),
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
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
                    from transformers.trainer_utils import EvalPrediction
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
                    train_metrics = self.compute_metrics_fn(eval_prediction,split='train')
                    
                    # Add train/ prefix to all metrics
                    train_metrics_prefixed = {f"train/{k}": v for k, v in train_metrics.items()}
                        
                # Log directly to wandb or other trackers without going through trainer.log to avoid recursion
                if self.trainer.is_world_process_zero():
                    if hasattr(args, "report_to") and "wandb" in args.report_to:
                        wandb.log(train_metrics_prefixed, step=metrics_step)
                    
            except Exception as e:
                print(f"Error in metrics callback: {e}")
                import traceback
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
                    
                import gc
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
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        tf32=True,              # bc set in run_plw.py
        seed=config.training.random_seed, 
        report_to=report_to,    # dynamic wandb reporting
        gradient_checkpointing=True,     # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": True},
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
        #processing_class=tokenizer,
        compute_metrics=prepare_compute_metrics(dataset, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.optimizer = AdamW(trainer.model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
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
        compute_metrics_fn=prepare_compute_metrics(dataset, tokenizer),
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
    import gc
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Cleared training data from memory")

def init_wandb_for_inf(config: Config, model_id: str, inference_type: str):
    wandb.init(
        project="master-thesis--inference", 
        name=f"{model_id}-{inference_type}",
        tags=[model_id, inference_type, "inference"],
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
def inference(model, tokenizer, config: Config, result_dir: str, inference_type: str, sample_fraction = 1.0):
    # Initialize WandB for inference
    if config.logging.use_wandb:
        model_id = config.model.model_id.split("/")[-1]
        init_wandb_for_inf(config, model_id, inference_type)
   
    print(f'Begin inference on test dataset')
    inf_dir = os.path.join(result_dir, "inference")
    os.makedirs(inf_dir, exist_ok=True)

    model.eval()  # model in evaluation mode (PyTorch)
    
    try:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.for_inference(model)
        print("Using Unsloth's optimized inference")
    except Exception as e:
        print(f"Could not enable Unsloth's optimized inference: {e}")
    
    # Load the raw test dataset
    raw_test_dataset = load_dataset(config.data.dataset_id)["test"]
    if sample_fraction < 1.0:
        raw_test_dataset = raw_test_dataset.shuffle(seed=config.training.random_seed).select(range(int(len(raw_test_dataset) * sample_fraction)))

    # Apply modifications
    if config.data.mix_directions:
        from synthetic_data.__dataset_direction_modifier import DatasetDirectionModifier
        modifier = DatasetDirectionModifier(random_seed=config.training.random_seed)
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
        from synthetic_data.__adapt_ascii_processor import AdaptiveASCIIProcessor
        black_threshold = config.data.ascii_parameters.get("black_threshold", 150)
        block_size = config.data.ascii_parameters.get("block_size", None)
        crop_to_size = config.data.ascii_parameters.get("crop_to_size", None)
        ascii_processor = AdaptiveASCIIProcessor(levels=10,
                                                 black_threshold=black_threshold,
                                                 block_size=block_size,
                                                 crop_to_size=crop_to_size,drop_images=True)
        dataset = ascii_processor.process_dataset(dataset)

    results = []
    
    # Get prompt template
    prompt_template = config.prompt.get_prompt_template(
        include_desc=config.data.include_desc,
        include_ascii=config.data.include_ascii
    )
    
    # Process in smaller batches to avoid memory issues
    batch_size = config.training.per_device_eval_batch_size
    num_examples = len(raw_test_dataset)
    
    with torch.no_grad():
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(num_examples-1)//batch_size + 1} " +
                 f"(examples {batch_start}-{batch_end-1})")
            
            for i in batch_indices:
                example = raw_test_dataset[i]
                
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
                    do_sample=True
                )
                
                # Decode generation
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract the completion part
                if "<|assistant|>" in generated_text:
                    # Extract between assistant token and end token
                    completion = generated_text.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
                else:
                    # Fall back to [/INST] pattern
                    completion = generated_text.split("[/INST]")[-1].strip()
                
                # Get ground truth
                ground_truth = example["Program"]
                
                # Store result for this example
                result = {
                    "id": i,
                    "prompt": chat_formatted_prompt,
                    "completion": completion,
                    "ground_truth": ground_truth,
                }
                
                results.append(result)
            
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
def inference_from_hub(config: Config, result_dir: str, inference_type: str, sample_fraction = 1.0):
    """
    Runs inference using a model loaded directly from the Hugging Face Hub.
    
    Args:
        config: Configuration object
        result_dir: Directory to save results
        inference_type: Type of inference being performed
        sample_fraction: Fraction of test dataset to use
    """
    print(f"DEBUG - Temperature from config: {config.model.temperature}, type: {type(config.model.temperature)}")

    # Initialize WandB for inference
    if config.logging.use_wandb:
        hub_model_name = config.model.model_id.split("/")[-1]
        init_wandb_for_inf(config, hub_model_name, inference_type)
   
    print(f'Begin inference on test dataset using model from hub: {config.model.model_id}')

    inf_dir = result_dir
    os.makedirs(inf_dir, exist_ok=True)

    # Load model and tokenizer from Hub
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            config.model.model_id,
            load_in_4bit=True,
        )
        model = FastLanguageModel.for_inference(model)
        print(f"Loaded model {config.model.model_id} using Unsloth's optimized inference")
    except Exception as e:
        print(f"Could not load with Unsloth, falling back to HF Transformers: {e}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            device_map="auto",
        )
        print(f"Loaded model {config.model.model_id} using standard HF Transformers")
    
    # Load the raw test dataset
    raw_test_dataset = load_dataset(config.data.dataset_id)["test"]
    if sample_fraction < 1.0:
        raw_test_dataset = raw_test_dataset.shuffle(seed=config.training.random_seed).select(range(int(len(raw_test_dataset) * sample_fraction)))
    
    # Apply modifications
    if config.data.mix_directions:
        from synthetic_data.__dataset_direction_modifier import DatasetDirectionModifier
        modifier = DatasetDirectionModifier(random_seed=config.training.random_seed)
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
        from synthetic_data.__adapt_ascii_processor import AdaptiveASCIIProcessor
        black_threshold = config.data.ascii_parameters.get("black_threshold", 150)
        block_size = config.data.ascii_parameters.get("block_size", None)
        crop_to_size = config.data.ascii_parameters.get("crop_to_size", None)
        ascii_processor = AdaptiveASCIIProcessor(levels=10,
                                                 black_threshold=black_threshold,
                                                 block_size=block_size,
                                                 crop_to_size=crop_to_size,drop_images=True)
        dataset = ascii_processor.process_dataset(dataset)

    results = []
    
    # Get prompt template
    prompt_template = config.prompt.get_prompt_template(
        include_desc=config.data.include_desc,
        include_ascii=config.data.include_ascii
    )
    
    # Process in smaller batches to avoid memory issues
    batch_size = config.training.per_device_eval_batch_size
    num_examples = len(raw_test_dataset)
    
    with torch.no_grad():
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(num_examples-1)//batch_size + 1} " +
                 f"(examples {batch_start}-{batch_end-1})")
            
            for i in batch_indices:
                example = raw_test_dataset[i]
                
                # Format prompt
                formatted_prompt = prompt_template.format(**example)
                
                # Create messages list with system prompt if configured
                messages = []
                if config.prompt.include_sys_prompt_inf:
                    messages.append({"role": "system", "content": config.prompt._system_prompt})
                messages.append({"role": "user", "content": formatted_prompt})
                
                # Apply chat template for consistent formatting
                chat_formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                
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
                    do_sample=True
                )
                
                # Decode generation
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract the completion part - adapting to different model formats
                if "<|assistant|>" in generated_text:
                    # Extract between assistant token and end token
                    completion = generated_text.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
                elif "[/INST]" in generated_text:
                    # Llama format
                    completion = generated_text.split("[/INST]")[-1].strip()
                elif "assistant:" in generated_text.lower():
                    # Generic assistant format
                    completion = generated_text.split("assistant:", 1)[-1].strip()
                else:
                    # Fallback - just take everything after the user's last message
                    user_content = messages[-1]["content"]
                    if user_content in generated_text:
                        completion = generated_text.split(user_content, 1)[-1].strip()
                    else:
                        completion = generated_text  # Last resort
                
                # Get ground truth
                ground_truth = example["Program"]
                
                # Store result for this example (no evaluation metrics)
                result = {
                    "id": i,
                    "prompt": chat_formatted_prompt,
                    "completion": completion,
                    "ground_truth": ground_truth
                }
                
                results.append(result)
            
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
def evaluation(inf_dir: str):
    """
    Evaluate model predictions using the LLMCodeEvaluator class.
    
    Args:
        inf_dir (str): Directory containing predictions.json
        
    Returns:
        tuple: (metrics, summary)
    """
    from __eval import LLMCodeEvaluator
    
    print(f"Starting evaluation on predictions in {inf_dir}")
    
    # Initialize the evaluator
    evaluator = LLMCodeEvaluator()
    
    try:
        # Run the evaluation pipeline
        metrics, summary = evaluator.evaluate_and_summarize(inf_dir)
        
        # Save the evaluation results
        with open(os.path.join(inf_dir, "evaluation.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        with open(os.path.join(inf_dir, "detailed_metrics.json"), "w") as f:
            # Convert any non-serializable values to strings
            serializable_metrics = []
            for metric in metrics:
                serializable_metric = {}
                for k, v in metric.items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        serializable_metric[k] = v
                    else:
                        serializable_metric[k] = str(v)
                json.dump(serializable_metrics, f, indent=2)
        
        print(f"Evaluation complete. Results saved to {inf_dir}/evaluation.json")
        return metrics, summary
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# MAIN
if __name__ == "__main__":
    try:
        config, timestamp, gen_type, model_type_short, result_dir = load_config(config_file, fine_tune)
        set_random_seeds(config.training.random_seed)
        if fine_tune:
            # Prep
            model, tokenizer = load_model_and_tokenizer(config)
            dataset = prepare_dataset(config, tokenizer, sample_fraction = sample_fraction) 
            # Training
            model = train_model(model, tokenizer, dataset, result_dir, config, timestamp, gen_type, model_type_short)
            clear_training_memory(dataset) # Free up memory after training
            # Inference after Finetuning
            results, inf_dir = inference(model, tokenizer, config, result_dir, inference_type=f"test_{timestamp}", sample_fraction = sample_fraction)
        else:
            # Inference with Model from Hub
            results, inf_dir = inference_from_hub(config, result_dir, inference_type=f"test_hub_{timestamp}", sample_fraction = sample_fraction)
        metrics, summary = evaluation(inf_dir)
        print("Pipeline completed successfully! 🎉")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()