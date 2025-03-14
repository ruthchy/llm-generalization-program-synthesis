# load libraries and model from HF
import yaml
import os
import torch
import pandas as pd
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from transformers import TextStreamer
from _1_prompt_temp_v1 import conversational_format_PBE_zeroshot_inference

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

#cuda_devices = config["cuda"]["devices"]
# data
data_dir = config["data"]["dataset_id"]
# model
model_name = config["model"]["model_id"]

max_seq_length = config["training"]["max_seq_length"]

### set the cuda device(s)
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
#print("cuda devices:", cuda_devices)


# Initialize WandB
timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M")
# Initialize WandB (ensure you've logged in using `wandb login`)
wandb.init(project="code-llama-finetuning", 
           name=f"inference-{model_name.split('/')[-1]}-{data_dir.split('/')[-1]}_{timestamp}")

#######################
# Model configuration #
#######################
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

######################
# Data configuration #
######################

from _1_prompt_temp_v1 import conversational_format_PBE_zeroshot_inference # lacks assistant response

# load the datasets and access the splits
test_dataset = load_dataset(data_dir, split="test")
test_dataset = test_dataset.map(conversational_format_PBE_zeroshot_inference)
#print(test_dataset["conversations"][0])

#############
# Inference #
#############
#Alternative: https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-10.-train-the-model
# Set the model to evaluation mode and move to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

FastLanguageModel.for_inference(model) 

def generate_response_streaming(conversation):
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, padding=True)
    
    # Move the inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Initialize the text streamer
    streamer = TextStreamer(tokenizer)
    
    # Generate the response and stream the output
    model.generate(**inputs, streamer=streamer)
    
    # The streaming will automatically handle the output tokens
    response = "".join([streamer.decode() for token in streamer.stream()])
    return response

def generate_response_streaming(conversation_text):
    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, padding=True)
    
    # Move inputs to GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Initialize the text streamer
    text_streamer = TextStreamer(tokenizer)
    
    # Generate the response and stream the output
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=64)
    
    # Collect the response from the streamer
    response = text_streamer.text
    return response

def generate_response_streaming(conversation_text):
    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, padding=True)
    
    # Move inputs to GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Initialize the text streamer
    text_streamer = TextStreamer(tokenizer)
    
    try:
        # Generate the response and stream the output
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=64)
        
        # If the model didn't generate any tokens or response, handle it
        if not text_streamer.text:  # If no tokens were generated or text is empty
            return "No response generated."

        # Return the response from the streamer
        return text_streamer.text

    except Exception as e:
        # Handle the case where something goes wrong (e.g., no tokens generated)
        return f"Error generating response: {str(e)}"


for conversation in test_dataset["conversations"][:2]:
    conversation_text = "".join([f"{entry['role']}: {entry['content']}\n" for entry in conversation])
    response = generate_response_streaming(conversation_text)
    print(f"Response for conversation: {response}")