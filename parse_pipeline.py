'''
# Using LLMs to parse generated completions of the main pipline
'''
# Housekeeping - single GPU unsloth setup
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # new
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import re
from unsloth import FastLanguageModel, is_bfloat16_supported

dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

def load_predictions(self, inf_dir):
    """
    Load predictions from JSON file.
        
    Args:
        inf_dir (str): Directory containing predictions.json
            
    Returns:
        list: Loaded predictions
    """
    with open(os.path.join(inf_dir, "predictions.json"), "r", encoding='utf-8') as f:
            predictions = json.load(f)
    return predictions

def init_model(model_name = "codellama/CodeLlama-7b-Instruct-hf"):
    """
    Initialize the model and tokenizer.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=1024,
        dtype=dtype,
        load_in_4bit=True,  # Optional: Use 4-bit quantization if supported
        device_map="auto",  # Automatically map model to available GPUs
    )
    model = FastLanguageModel.for_inference(model)

    return model, tokenizer


def extract_program(response, model, tokenizer):
    """
    Extract the Python program from a model's response using an LLM.

    Args:
        response (str): The response from the model.
        model (str): The LLM model to use for extraction.

    Returns:
        str: The extracted Python program.
    """
    # Define the prompt to extract the program
    messages = [
        {"role": "system", "content": "You are an assistant that extracts Python programs from text."},
        {"role": "user", "content": (
            f"The input text may contain a description, metadata, or other non-code content before the actual program starts. Your task is to identify and extract only the Python program from the input text. Do not include any markdown formatting (e.g., ```) or additional comments. If the input text does not contain a Python program, return an empty string.\n\nHere is the input text:\n{response}\n\nExtracted Python program:"
        )}
    ]

    # Format the prompt using the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Prompt:\n{prompt}\n{'-' * 80}\n")

    # Call the LLM to extract the program
    try:
        tokenized_input = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized_input.input_ids.to(model.device)
        attention_mask = tokenized_input.attention_mask.to(model.device)

        generated_id = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=250,
            temperature=0.7,
            top_k = 50,
        )
        
        new_tokens = generated_id[0][input_ids.shape[1]:]
        extracted_program = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return extracted_program
    except Exception as e:
        print(f"Error extracting program: {e}")
        return ""


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run code extraction pipeline')
    parser.add_argument('--pre_dict', type=str, help='Provide the directory where the predictions.json file is located')

    args = parser.parse_args()
    pre_dict = args.pre_dict

    print(f"This prediction file is used: {pre_dict}")

    # Load the model and tokenizer
    model_name = "codellama/CodeLlama-7b-Instruct-hf" #"Salesforce/codegen-2B-mono"
    model, tokenizer = init_model(model_name)
    
    # Load predictions
    predictions = load_predictions(pre_dict)
    
    output_file = "extracted_programs.jsonl"
    with open(output_file, "w") as f:
        for i, pred in enumerate(predictions):
            example_id = pred["id"] 
            n_completions = len([key for key in pred.keys() if key.startswith("completion_")])
            
            for idx in range(1, n_completions + 1):  
                completion_key = f"completion_{idx}"
                if completion_key in pred:
                    completion = pred[completion_key]
                    print(f"Example ID: {example_id}, Completion {idx}: {completion}")
                    program = extract_program(completion, model, tokenizer)
                    print(f"Extracted Program {idx}:\n{program}\n{'-' * 80}\n")

                    json_line = {"generated": completion, "program": program}
                    f.write(json.dumps(json_line) + "\n")

    print(f"Results saved to {output_file}")