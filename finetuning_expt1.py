# set the GPU to use
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# load libraries and model from HF
import torch
import pandas as pd
import wandb
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported

model_name = "codellama/CodeLlama-7b-hf"
max_seq_length = 2048

timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M")
# Initialize WandB (ensure you've logged in using `wandb login`)
wandb.init(project="code-llama-finetuning", 
           name=f"fine-tune-semantic-length-generalization-ascii_{timestamp}",
           config={"learning_rate": 5e-5, "num_train_epochs": 3, "max_seq_length": max_seq_length, "num_epochs": 3,})

# Model configuration
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
# if tokenizer.pad_token is None then an error will be raised therfore set it
if tokenizer.pad_token == None:
    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
    if tokenizer.pad_token == '[PAD]':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"Added padding token: {tokenizer.pad_token}")

# LoRA (16-bit) for PEFT => this means I need 16GB of VRAM when training the 7B-codelama model (Alternative: QLoRA could be specified with 8GB VRAM)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

# load the dataset and access the splits
# Input-Experiment 1: ASCII-art 
dataset = load_dataset("ruthchy/semantic-length-generalization-logo-data-ascii")
train_dataset, val_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

# Tokenize the datasets
def preprocess_function(examples):
    return tokenizer(
        examples["Input"],  
        text_pair=examples["Program"],  
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

# Apply the tokenizer to the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, num_proc=4)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=4)

# Define a base TrainingArguments
training_args = TrainingArguments(
    output_dir="./results/01_expt",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps = 100,
    learning_rate=5e-5,
    max_new_tokens=300, # the longest Program in test is 244 so 300 is enough
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    report_to=["wandb"],
    hub_model_id="ruthchy/01_expt_code-llama-ascii", 
    push_to_hub=True,
    logging_steps = 1,
)

# Create trainers
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    max_seq_length=max_seq_length,
)

# Train the model
trainer.train()
# Evaluate the model
results = trainer.evaluate(tokenized_test_dataset)
wandb.log({"Experiment 1 Results": results})
print("Experiment 1 Results:", results)