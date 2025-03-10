'''
python test.py
torchrun --nproc_per_node=2 test.py
'''
############################################################################################################
# Housekeeping - single GPU unsloth setup
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Step 1: Load the YAML Configuration
import yaml
import torch
import wandb
import json
import numpy as np
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
        prompt = "" #[INST]"
        if self.include_sys_prompt:
            prompt += f"[SYS]{self.system_prompt}[/SYS]"
        prompt += task_description + "\n"#"[/INST]"
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
            include_sys_prompt=bool(config_dict["data"]["include_sys_prompt"])
        )

def load_config(model_name: str) -> Tuple[Config, str, str, str, str]:
    """Load training configuration from yaml file and store a copy in results directory"""
    source_config = "config.yaml"
    
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
        result_dir = f"results/{gen_type}/{model_type_short}_{timestamp}"
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
    # Optional: Set seeds for other libraries if used
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
from queue import PriorityQueue
from itertools import chain
import random

def spfhp(seq_lens, chunk_length=2048):
    q = PriorityQueue()
    q.put((0, []))
    idx = seq_lens.argsort()[::-1]
    for i in idx:
        n, pack = seq_lens[i], q.get()
        if n + pack[0] > chunk_length:
            q.put(pack)
            pack = (0, [])
        q.put((n + pack[0], pack[1] + [i]))
    return list(q.queue)

def pack(sample, chunk_length=2048, pad_token_id=0):
    # Compute packing arrangement
    seq_lens = np.array([len(t) for t in sample["input_ids"]])
    chunks = spfhp(seq_lens, chunk_length=chunk_length)
    random.shuffle(chunks)

    # Pack sequences according to arrangement
    result = {}
    for k in sample.keys():
        result[k] = []
        pad_id = pad_token_id if k == "input_ids" else 0
        for chunk in chunks:
            item = list(chain(*[sample[k][i] for i in chunk[1]], [pad_id] * (chunk_length - chunk[0])))
            result[k].append(item)

    # Add labels (same as input_ids)
    result["labels"] = result["input_ids"].copy()
    return result
    
from datasets import DatasetDict, Sequence, Value
from functools import partial

# Function to tokenize and encode a batch of samples, and create prompt/completion masks
def tokenize_batch(batch, tokenizer):
    # Tokenize and encode text
    tokenized_text = tokenizer(batch["text"], add_special_tokens=False, return_offsets_mapping=True)
    data = {k: tokenized_text[k] for k in tokenized_text.keys()}

    # Use offset_mappings to make prompt/completion masks (idx marks the start of each completion)
    prompt_masks, completion_masks = [], []
    for offset_mapping, idx in zip(data["offset_mapping"], batch["prompt_length"]):
        prompt_masks.append([1 if o[1] < idx else 0 for o in offset_mapping])
        completion_masks.append([0 if o[1] < idx else 1 for o in offset_mapping])

    data["prompt_mask"] = prompt_masks
    data["completion_mask"] = completion_masks
    del data["offset_mapping"]
    return data

# Tokenize and pack dataset
def tokenize_and_pack(dataset, tokenizer, max_seq_length):
    # Tokenize dataset
    tokenized_dataset = dataset.map(partial(tokenize_batch, tokenizer=tokenizer), 
                                    batched=True, 
                                    remove_columns=list(dataset.features))

    # Recast mask columns to int8
    for column in ["prompt_mask", "completion_mask"]:
        tokenized_dataset = tokenized_dataset.cast_column(column, Sequence(Value("int8")))

    # Filter out rows of tokenized_dataset that are too long
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_length)

    # Sequence packing to save space
    packed_dataset = tokenized_dataset.map(partial(pack, 
                                                   chunk_length=max_seq_length, 
                                                   pad_token_id=tokenizer.pad_token_id),
                                           batched=True)
    # Recast labels column to int32
    packed_dataset = packed_dataset.cast_column("labels", Sequence(Value("int32")))

    return packed_dataset

def prepare_dataset(config: Config, tokenizer, sample_fraction=1.0):
    dataset = load_dataset(config.data.dataset_id)
    if sample_fraction < 1.0:
        dataset["train"] = dataset["train"].shuffle(seed=config.training.random_seed).select(range(int(len(dataset["train"]) * sample_fraction)))
        dataset["validation"] = dataset["validation"].shuffle(seed=config.training.random_seed).select(range(int(len(dataset["validation"]) * sample_fraction)))
        dataset["test"] = dataset["test"].shuffle(seed=config.training.random_seed).select(range(int(len(dataset["test"]) * sample_fraction)))

    def format_prompt(example, split_type, idx):
        """Formats a single example with the template"""
        prompt = config.prompt.get_prompt_template(
            include_desc=config.data.include_desc,
            include_ascii=config.data.include_ascii
        )
        
        formatted_prompt = prompt.format(**example)
        completion = example['Program']
        
        if split_type == "test":
            messages = [{"role": "user", "content": formatted_prompt}]
        else:
            messages = [{"role": "user", "content": formatted_prompt},
                        {"role": "assistant", "content": completion }]
        prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)        
        return {
            "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
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

    # Tokenize and pack dataset
    tokenized_dataset = {}
    for split in formatted_dataset.keys():
        tokenized_dataset[split] = tokenize_and_pack(formatted_dataset[split], tokenizer, config.training.max_seq_length)
    
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

    # extract prompt/completion masks
    prompt_mask = np.array([x["prompt_mask"] for x in dataset['validation']])
    completion_mask = np.array([x["completion_mask"] for x in dataset['validation']])

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
        avg_levenshtein_dist = np.mean(distances)
        std_levenshtein_dist = np.std(distances)

        # return metrics
        return {
            'comp_loss': completion_loss,
            'prompt_loss': prompt_loss,
            'avg_levenshtein_dist': avg_levenshtein_dist,
            'std_levenshtein_dist': std_levenshtein_dist,
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

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # num_items_in_batch can be used to normalize the loss but I'm not using it here since the loss is calculated per token
            #print(f"num_items_in_batch: {num_items_in_batch}")
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

            # Print a sample batch for debugging
            #first_batch = next(iter(dataloader))
            #print("🚀 First batch keys:", first_batch.keys())
            #print("🚀 First batch shapes:")
            #for key, value in first_batch.items():
            #    print(f"  - {key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")

            return dataloader


        # this allows us to toggle on/off data shuffling, which can sometimes cause 'staircase' effects in training loss
        def _get_train_sampler(self):
            #if self.distributed_training: # not used in single GPU setup
            #    return torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=self.shuffle) 
            if self.shuffle:
                return torch.utils.data.RandomSampler(self.train_dataset)
            return torch.utils.data.SequentialSampler(self.train_dataset)

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
        greater_is_better=False  # Whether a higher metric value is better
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
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

    # Train the model
    trainer.train()
    
    # Logging is only disabled if it was enabled before
    if config.logging.use_wandb:
        wandb.finish()
    
    return model


# MAIN
if __name__ == "__main__":
    try:
        config, timestamp, gen_type, model_type_short, result_dir = load_config("config.yaml")
        set_random_seeds(config.training.random_seed)
        model, tokenizer = load_model_and_tokenizer(config)
        dataset = prepare_dataset(config, tokenizer)
        # Training
        model = train_model(model, tokenizer, dataset, result_dir, config, timestamp, gen_type, model_type_short)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()