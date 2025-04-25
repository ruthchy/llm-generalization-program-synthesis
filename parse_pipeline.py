'''
Using LLMs to parse generated completions of the main pipline:
 1. Load the predictions from the JSON file.
 2. Initialize the model and tokenizer.    
 3. For each prediction, extract the Python program using the LLM.
 4. Save the extracted programs to a JSONL file.

The script can be run with the following command:
    python parse_pipeline_small.py --pre_dict <path_to_predictions.json> --few_shot
    
    python parse_pipeline_small.py --pre_dict "results/length/CodeLlama/inference/copy" --few_shot

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
from unsloth import is_bfloat16_supported
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

def load_predictions(inf_dir):
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

def init_model(model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"): 
    """
    Initialize the model and tokenizer.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    # Define the BitsAndBytesConfig for 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype
    )
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto", 
                quantization_config=quantization_config,
                trust_remote_code=True,           
    )

    return model, tokenizer


def extract_program(response, model, tokenizer, few_shot_prompt=False): 
    """
    Extract the Python program from a model's response using an LLM.

    Args:
        response (str): The response from the model.
        model (str): The LLM model to use for extraction.

    Returns:
        str: The extracted Python program.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        if few_shot_prompt:
            # Use the chat template for few-shot prompting
            messages = [
                {"role": "system", "content": "You are an assistant that extracts Python programs from text."},
                {"role": "user", "content": (
                    f"The input text may contain a description, metadata, or other non-code content before the actual program starts. Your task is to identify and extract only the Python program from the input text. Do not include any markdown formatting (e.g., ```) or additional comments. If the input text does not contain a Python program, return an empty string.\n\nHere is the input text:\nHere is a Python program that generates the gray scale image using the custom turtle module:\n```\nfrom turtle import forward, left, right, penup, pendown, teleport, heading, isdown, embed\n\ndef draw_image():\n    # Draw a line\n    forward(10)\n    left(90)\n    # Draw a square\n    for i in range(4):\n        forward(10)\n        left(90)\n    # Draw a circle\n    penup()\n    teleport(10, 10, 0)\n    right(90)\n    for i in range(4):\n        forward(10)\n        left(90)\n    pendown()\n\ndef draw_image():\n    # Draw the image\n    draw_image()\n\n# Display the image\nshow()\n```\nThis program uses the `forward`, `left`, `right`, `penup`, `pendown`, `teleport`, `heading`, `isdown`, and `embed` functions from the custom turtle library to draw the image. The\n\nExtracted Python program:"
                )},
                {"role": "assistant", "content": (
                    f"from turtle import forward, left, right, penup, pendown, teleport, heading, isdown, embed\n\ndef draw_image():\n    # Draw a line\n    forward(10)\n    left(90)\n    # Draw a square\n    for i in range(4):\n        forward(10)\n        left(90)\n    # Draw a circle\n    penup()\n    teleport(10, 10, 0)\n    right(90)\n    for i in range(4):\n        forward(10)\n        left(90)\n    pendown()\n\ndef draw_image():\n    # Draw the image\n    draw_image()\n\n# Display the image\nshow()"
                )},
                {"role": "user", "content": (
                    f"The input text may contain a description, metadata, or other non-code content before the actual program starts. Your task is to identify and extract only the Python program from the input text. Do not include any markdown formatting (e.g., ```) or additional comments. If the input text does not contain a Python program, return an empty string.\n\nHere is the input text:\nfor j in range(8):\n    embed(\"\"\"forward(2)\nleft(180)\nfor i in range(4):\n    forward(4)\n    left(90.0)\"\"\", locals())\n    forward(0)\n    left(45.0)\n\nExtracted Python program:"
                )},
                {"role": "assistant", "content": (
                    f"for j in range(8):\n    embed(\"\"\"forward(2)\nleft(180)\nfor i in range(4):\n    forward(4)\n    left(90.0)\"\"\", locals())\n    forward(0)\n    left(45.0)"
                )},
                {"role": "user", "content": (
                    f"The input text may contain a description, metadata, or other non-code content before the actual program starts. Your task is to identify and extract only the Python program from the input text. Do not include any markdown formatting (e.g., ```) or additional comments. If the input text does not contain a Python program, return an empty string.\n\nHere is the input text:\n### Response:\n# the following program draws a small square connected to a big line and a small 5 gon as arms\nfor i in range(4):\n    forward(2)\n    left(90.0)\nforward(4)\nleft(0.0)\nfor i in range(5):\n    forward(4)\n    left(72.0)\n\nExtracted Python program:"
                )},
                {"role": "assistant", "content": (
                    f"for i in range(4):\n    forward(2)\n    left(90.0)\nforward(4)\nleft(0.0)\nfor i in range(5):\n    forward(4)\n    left(72.0)"
                )},
                {"role": "user", "content": (
                    f"The input text may contain a description, metadata, or other non-code content before the actual program starts. Your task is to identify and extract only the Python program from the input text. Do not include any markdown formatting (e.g., ```) or additional comments. If the input text does not contain a Python program, return an empty string.\n\nHere is the input text:\n{response}\n\nExtracted Python program:"
                )}
            ]

        else:   
            messages = [
                {"role": "system", "content": "You are an assistant that extracts Python programs from text."},
                {"role": "user", "content": (
                    f"The input text may contain a description, metadata, or other non-code content before the actual program starts. Your task is to identify and extract only the Python program from the input text. Do not include any markdown formatting (e.g., ```) or additional comments. If the input text does not contain a Python program, return an empty string.\n\nHere is the input text:\n{response}\n\nExtracted Python program:"
                )}
            ]
        # Format the prompt using the tokenizer's chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    else:
        instruction = "Extracts only the Python programs from the input text. Do not include any markdown formatting (e.g., ```) or additional comments. If the input text does not contain a Python program, return an empty string."
        prompt = f"{instruction}\n\nInput:\n{response}\n\nOutput:\n"
        if few_shot_prompt:
            examples = [
                {
                    "input": "Here is a Python program that generates the gray scale image using the custom turtle module:\n```\nfrom turtle import forward, left, right, penup, pendown, teleport, heading, isdown, embed\n\ndef draw_image():\n    # Draw a line\n    forward(10)\n    left(90)\n    # Draw a square\n    for i in range(4):\n        forward(10)\n        left(90)\n    # Draw a circle\n    penup()\n    teleport(10, 10, 0)\n    right(90)\n    for i in range(4):\n        forward(10)\n        left(90)\n    pendown()\n\ndef draw_image():\n    # Draw the image\n    draw_image()\n\n# Display the image\nshow()\n```\nThis program uses the `forward`, `left`, `right`, `penup`, `pendown`, `teleport`, `heading`, `isdown`, and `embed` functions from the custom turtle library to draw the image. The",
                    "output": "from turtle import forward, left, right, penup, pendown, teleport, heading, isdown, embed\n\ndef draw_image():\n    # Draw a line\n    forward(10)\n    left(90)\n    # Draw a square\n    for i in range(4):\n        forward(10)\n        left(90)\n    # Draw a circle\n    penup()\n    teleport(10, 10, 0)\n    right(90)\n    for i in range(4):\n        forward(10)\n        left(90)\n    pendown()\n\ndef draw_image():\n    # Draw the image\n    draw_image()\n\n# Display the image\nshow()"
                },
                {
                    "input": "for j in range(8):\n    embed(\"\"\"forward(2)\nleft(180)\nfor i in range(4):\n    forward(4)\n    left(90.0)\"\"\", locals())\n    forward(0)\n    left(45.0)",
                    "output": "for j in range(8):\n    embed(\"\"\"forward(2)\nleft(180)\nfor i in range(4):\n    forward(4)\n    left(90.0)\"\"\", locals())\n    forward(0)\n    left(45.0)"
                },
                {
                    "input": "### Response:\n# the following program draws a small square connected to a big line and a small 5 gon as arms\nfor i in range(4):\n    forward(2)\n    left(90.0)\nforward(4)\nleft(0.0)\nfor i in range(5):\n    forward(4)\n    left(72.0)",
                    "output": "for i in range(4):\n    forward(2)\n    left(90.0)\nforward(4)\nleft(0.0)\nfor i in range(5):\n    forward(4)\n    left(72.0)"
                }
            ]
            few_shot_prompt = instruction + "\n\n"
            for example in examples:
                few_shot_prompt += f"Input:\n{example['input']}\n\nOutput:\n{example['output']}\n\n"

            few_shot_prompt += f"Input:\n{response}\n\nOutput:\n"



    print(f"Prompt:\n{prompt}\n{'-' * 80}\n")

    # Call the LLM to extract the program
    try:
        #tokenized_input = tokenizer(prompt, return_tensors="pt")
        #input_ids = tokenized_input.input_ids.to(model.device)
        #attention_mask = tokenized_input.attention_mask.to(model.device)
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)

        #generated_id = model.generate(
        #    input_ids=input_ids,
        #    attention_mask=attention_mask,
        #    max_new_tokens=250,
        #    temperature=0.7,
        #    top_k = 50,
        #)

        generated_id = model.generate(
            **tokenized_input,
            max_new_tokens=150,
            temperature=0.5,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        #new_tokens = generated_id[0][input_ids.shape[1]:]
        new_tokens = generated_id[0][tokenized_input.input_ids.shape[1]:]
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
    parser.add_argument('--few_shot', action='store_true', help='Whether the model should be prompted with few-shot examples')

    args = parser.parse_args()
    pre_dict = args.pre_dict
    few_shot_prompt = args.few_shot

    print(f"This prediction file is used: {pre_dict}\n")
    print(f"Few-shot prompt enabled: {few_shot_prompt}\n")

    # Load the model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    #"codellama/CodeLlama-7b-Instruct-hf" #"Salesforce/codegen-2B-mono" #"microsoft/phi-1" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0" # "bigcode/tiny_starcoder_py"
    # "bigcode/starcoderbase-1b" (access revoced)
    model, tokenizer = init_model(model_name)
    
    # Load predictions
    predictions = load_predictions(pre_dict)
    
    if few_shot_prompt:
        file_name = f"extracted_programs_few_shot_{model_name.replace('/', '_')}_few-shot.jsonl"
    else:
        file_name = f"extracted_programs_{model_name.replace('/', '_')}.jsonl"
    output_file = os.path.join(pre_dict, file_name)
    with open(output_file, "w") as f:
        for i, pred in enumerate(predictions):
            example_id = pred["id"] 
            n_completions = len([key for key in pred.keys() if key.startswith("completion_")])
            
            for idx in range(1, n_completions + 1):  
                completion_key = f"completion_{idx}"
                if completion_key in pred:
                    completion = pred[completion_key]
                    print(f"Example: {example_id}_{completion_key}")
                    program = extract_program(completion, model, tokenizer, few_shot_prompt=few_shot_prompt)
                    print(f"Extracted Program {example_id}_{completion_key}:\n{program}\n{'-' * 80}\n")

                    json_line = {
                        "id": example_id,
                        f"{completion_key}": completion, 
                        "program": program}
                    f.write(json.dumps(json_line) + "\n")

    print(f"Results saved to {output_file}")