### load parameters from config.yaml
import yaml
def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

cuda_devices = config["cuda"]["devices"]
# data
data_dir = config["data"]
# model
model_name = config["model"]["name"]
#lora
rank = int(config["lora"]["rank"])
alpha = int(config["lora"]["alpha"])

# training
max_seq_length = config["training"]["max_seq_length"]
learning_rate = float(config["training"]["learning_rate"])
train_epochs = config["training"]["train_epochs"]
per_device_batch_size = config["training"]["per_device_batch_size"]
gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
save_steps = config["training"]["save_steps"]
eval_steps = config["training"]["eval_steps"]
logging_steps = config["training"]["logging_steps"]
random_seed = config["training"]["random_seed"]


### set the cuda device(s)
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

### load libraries and model from HF 
import torch
import pandas as pd
from functools import partial
import wandb
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM # to train model only on the generated prompts
#from trl.trainer import ConstantLengthDataset
from datasets import load_dataset
from _1_prompt_temp_v1 import instruction_format, conversational_format, sys_prompt
from unsloth import FastLanguageModel, is_bfloat16_supported, apply_chat_template

timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M")
# Initialize WandB (ensure you've logged in using `wandb login`)
wandb.init(project="code-llama-finetuning", 
           name=f"fine-tune-{model_name.split('/')[-1]}-{data_dir.split('/')[-1]}_{timestamp}",
           config={"learning_rate": learning_rate, "num_train_epochs": train_epochs, "max_seq_length": max_seq_length, "learning_rate": learning_rate})

#######################
# Model configuration #
#######################
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=rank,
    lora_alpha=alpha,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state= random_seed
)

########
# Data #
########
dataset = load_dataset(data_dir)

for split in dataset:
    dataset[split] = dataset[split].map(lambda x: instruction_format(x, include_description=True, include_ascii=True))

train_dataset, val_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

# Print out a sample to confirm the changes
print(train_dataset["conversations"][0])

### define and apply chat template
chat_template = """### Instruction:
{INPUT}
### Python Program:
{OUTPUT}"""

dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    default_system_message = sys_prompt, # optional
)
##########################
# Training configuration # 
##########################
#check https://huggingface.co/docs/trl/sft_trainer#extending-sfttrainer-for-vision-language-models for more details on integrating images into the training

trainer_1 = SFTTrainer(
    model,
    tokenizer=tokenizer,            # tokenizer or collator (bc collator 
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=None, # (Callable[[transformers.EvalPrediction], dict], optional defaults to None) — The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values. If not specified, only the loss will be computed during evaluation.
    args = SFTConfig(
        output_dir = f"./results/fine-tune-{model_name.split('/')[-1]}-{data_dir.split('/')[-1]}_{timestamp}",
        # pre-processing
        max_seq_length = max_seq_length,
        packing = False,
        dataset_num_proc =4,
        # training parameters
        learning_rate = learning_rate,
        num_train_epochs = train_epochs,
        gradient_accumulation_steps = gradient_accumulation_steps,
        per_device_train_batch_size = per_device_batch_size,
        per_device_eval_batch_size = per_device_batch_size,
        # reporting and logging
        report_to = ["wandb"],
        push_to_hub = True,
        hub_model_id = f"fine-tune-{model_name.split('/')[-1]}-{data_dir.split('/')[-1]}_{timestamp}",
        logging_strategy = "steps",
        logging_steps = logging_steps,
        # checkpointing
        save_strategy = "steps",
        save_steps = save_steps,
        # evaluation
        eval_strategy = "steps",
        eval_steps = eval_steps,
        # optimization
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),    
        # ensure reproducibility
        seed = random_seed,
        data_seed = random_seed,
        full_determinism=True
        )
    )


####################################
### Training on completions only ###
# https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only

### orients on the instruction_format_v1 function from _1_prompt_temp_v2.py but excludes the include_description and include_ascii arguments
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["Description"])):
        formated_input = f"Generate a python program producing the graphic, which is described and depicted as follows:\n    The Program draws {example['Description'][i]}\n    Graphic:\n{example['ASCII-Art'][i]}\n"
        text = f"### Instruction: {formated_input}\n### Python Program: {example['Program'][i]}"
        output_texts.append(text)
    return output_texts


response_template = "### Python Program:" 
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer_2 = SFTTrainer(
    model,
    data_collator=collator,          
    formatting_func=formatting_prompts_func, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=None, # (Callable[[transformers.EvalPrediction], dict], optional defaults to None) — The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values. If not specified, only the loss will be computed during evaluation.
    args = SFTConfig(
        output_dir = f"./results/fine-tune-{model_name.split('/')[-1]}-{data_dir.split('/')[-1]}_{timestamp}",
        # pre-processing
        max_seq_length = max_seq_length,
        packing = False,
        dataset_num_proc =4,
        # training parameters
        learning_rate = learning_rate,
        num_train_epochs = train_epochs,
        gradient_accumulation_steps = gradient_accumulation_steps,
        per_device_train_batch_size = per_device_batch_size,
        per_device_eval_batch_size = per_device_batch_size,
        # reporting and logging
        report_to = ["wandb"],
        push_to_hub = True,
        hub_model_id = f"fine-tune-{model_name.split('/')[-1]}-{data_dir.split('/')[-1]}_{timestamp}",
        logging_strategy = "steps",
        logging_steps = logging_steps,
        # checkpointing
        save_strategy = "steps",
        save_steps = save_steps,
        # evaluation
        eval_strategy = "steps",
        eval_steps = eval_steps,
        # optimization
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),    
        # ensure reproducibility
        seed = random_seed,
        data_seed = random_seed,
        full_determinism=True
        )
    )

if config["training"]["trainer_output_only"]:
    trainer_2.train()
else:
    trainer_1.train()