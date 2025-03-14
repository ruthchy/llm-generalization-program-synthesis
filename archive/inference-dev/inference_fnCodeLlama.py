# load libraries and model from HF
import yaml
import os
import torch
import pandas as pd
import wandb
from datasets import load_dataset
from datetime import datetime
from unsloth import FastLanguageModel
from _1_prompt_temp_v1 import conversational_format_PBE_zeroshot_inference
import json

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def init_wandb(model_name: str, data_dir: str):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M")
    # Initialize WandB (ensure you've logged in using `wandb login`)
    wandb.init(project="code-llama-finetuning", 
            name=f"inference-{model_name.split('/')[-1]}-{data_dir.split('/')[-1]}_{timestamp}")

def model_config(model_name, max_seq_length):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer

def generate_response_streaming(conversation_text, model, tokenizer, device):
    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, padding=True)
    
    # Move inputs to GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    try:
        # Generate the response
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        
        # Decode the generated tokens
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        response = response[input_length:].strip()
        
        if not response:  # If no tokens were generated or text is empty
            return "No response generated."

        return response

    except Exception as e:
        # Handle the case where something goes wrong
        return f"Error generating response: {str(e)}"

def process_example(example, model, tokenizer, device):
    conversation_text = "".join([f"{entry['role']}: {entry['content']}\n" for entry in example['conversations']])
    response = generate_response_streaming(conversation_text, model, tokenizer, device)
    return {"Pred. Program": response}

def save_predictions(dataset, model_name, timestamp):
    results_dir = f"results/length/inference/{model_name.split('/')[-1]}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    predictions = {
        "model_info": {
            "name": model_name,
            "timestamp": timestamp,
            "config": {
                "topktrain": dataset[0].get('topktrain', 0),
                "topkprompt": dataset[0].get('topkprompt', 0)
            }
        },
        "predictions": [
            {
                "original": example['conversations'][0]['content'],  # Preserves all whitespace
                "prediction": example['Pred. Program'],
                "metadata": {
                    "topktrain": example.get('topktrain', 0),
                    "topkprompt": example.get('topkprompt', 0)
                }
            }
            for example in dataset
        ]
    }
    
    # Save as JSON with proper indentation
    predictions_path = os.path.join(results_dir, 'predictions.json')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved predictions to {predictions_path}")
    return predictions_path

def main():
    try:
        config = load_config("config.yaml")
        data_dir = config["data"]["dataset_id"]
        model_name = config["model"]["model_id"]
        max_seq_length = config["training"]["max_seq_length"]

        # Load the test dataset and format it
        test_dataset = load_dataset(data_dir, split="test")
        test_dataset = test_dataset.map(conversational_format_PBE_zeroshot_inference)
        test_dataset = test_dataset.select(range(10)) # For testing purposes

        model, tokenizer = model_config(model_name, max_seq_length)
        
        # Set the model to evaluation mode and move to GPU (if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        FastLanguageModel.for_inference(model)

        # Initialize WandB
        init_wandb(model_name, data_dir)

        # Process all examples in the dataset
        print(f"Processing {len(test_dataset)} examples...")
        test_dataset = test_dataset.map(lambda example: process_example(example, model, tokenizer, device))
        print("Finished processing examples")

        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        save_predictions(test_dataset, model_name, timestamp)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



