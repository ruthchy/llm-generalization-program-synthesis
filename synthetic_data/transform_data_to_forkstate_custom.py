import re
import json
from datasets import load_dataset

# Regex pattern
embed_pattern = re.compile(
    r'^(?P<indent>\s*)embed\("""(?P<code>.*?)""",\s*locals\(\)\)',
    re.MULTILINE | re.DOTALL
)

fork_pattern = re.compile(
    r'^(?P<indent>\s*)with fork_state\(\):\n(?P<code>(?:\1 {4}.*\n?)*)',
    re.MULTILINE
)

def embed_to_fork_state(match):
    """
    Regex callback that transforms:
        <indent>embed(\"\"\" code... \"\"\", locals())
    into:
        <indent>with fork_state():
            <indent>    code...
    """
    indent = match.group("indent")
    code_block = match.group("code")

    # Split the embedded code into lines
    lines = code_block.split("\n")

    # Re-indent each line
    reindented_lines = [(indent + "    " + line if line.strip() else line)
                        for line in lines]

    # Join the re-indented lines with newlines
    transformed_code = "\n".join(reindented_lines)

    # Return a `with fork_state():` block
    return f"{indent}with fork_state():\n{transformed_code}"

def fork_state_to_embed(match):
    """
    Regex callback that transforms:
        <indent>with fork_state():
            <indent>    code...
    into:
        <indent>embed(\"\"\" code... \"\"\", locals())
    """
    indent = match.group("indent")  # Indentation of `with fork_state():`
    code_block = match.group("code")  # Code inside `with fork_state():`

    # Calculate the full indentation to remove (indent + 4 spaces)
    full_indent = indent + " " * 4

    # Split the code block into individual lines
    lines = code_block.split("\n")

    # Dedent each line by removing `full_indent`
    dedented_lines = [
        line[len(full_indent):] if line.startswith(full_indent) else line for line in lines
    ]
    
    dedented_code = "\n".join(dedented_lines).rstrip()
    # Return the `embed(""" """, locals())` block
    return f'{indent}embed("""{dedented_code}""", locals())\n'


### TRANFORMATION
def transform_program(example, embed_to_fork, fork_to_embed):
    if embed_to_fork:
        return embed_pattern.sub(embed_to_fork_state, example)
    elif fork_to_embed:
        return fork_pattern.sub(fork_state_to_embed, example)
    else:
        raise ValueError("No transformation specified. Use --embed_to_fork or --fork_to_embed.") 

# Hugging Face dataset transformation
def transform_huggingface_dataset(data_id, embed_to_fork, fork_to_embed):
    ds = load_dataset(data_id)
    def transform_example(example):
        if "Program" in example:
            example["Program"] = transform_program(example["Program"], embed_to_fork, fork_to_embed)
        return example
    transformed_ds = ds.map(transform_example)
    if embed_to_fork:
        transformation = "fork"
    elif fork_to_embed:
        transformation = "embed"
    transformed_ds.push_to_hub(f"{data_id}_{transformation}")
    print(f"Transformed dataset pushed to Hugging Face Hub: {data_id}_transformed")

# JSON file transformation
def transform_json_file(json_file_path, embed_to_fork, fork_to_embed):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        if "ground_truth" in entry:
            entry["ground_truth"] = transform_program(entry["ground_truth"], embed_to_fork, fork_to_embed)
        for key in entry:
            if key.startswith("completion_"):
                entry[key] = transform_program(entry[key], embed_to_fork, fork_to_embed)
    if embed_to_fork:
        transformation = "fork"
    elif fork_to_embed:
        transformation = "embed"
    output_file = f"{json_file_path.rsplit('.', 1)[0]}_{transformation}.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Transformed JSON saved to: {output_file}")

### MAIN
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description='Run fine-tuning and/or inference pipeline')
    parser.add_argument('--embed_to_fork', action='store_true', help='Program should contain `with fork_state():` instead of `embed(...)`')
    parser.add_argument('--fork_to_embed', action='store_true', help='Program should contain `embed(...)` instead of `with fork_state():`')
    parser.add_argument('--data_id', type=str, default=None, help='Hugging Face dataset ID to load the dataset which the alteration should be applied to')
    parser.add_argument('--json_file_path', type=str, default=None, help='Path to the predictions file')
    args = parser.parse_args()

    embed_to_fork = args.embed_to_fork
    fork_to_embed = args.fork_to_embed
    data_id = args.data_id
    json_file_path = args.json_file_path

    if not data_id and not json_file_path:
        raise ValueError("You must specify exactly one of --data_id or --json_file_path.")
    if data_id and json_file_path:
        raise ValueError("You cannot specify both --data_id and --json_file_path.")
    if embed_to_fork and fork_to_embed:
        raise ValueError("You cannot specify both --embed_to_fork and --fork_to_embed.")
    if not embed_to_fork and not fork_to_embed:
        raise ValueError("You must specify either --embed_to_fork or --fork_to_embed.")

    if data_id:
        transform_huggingface_dataset(data_id, embed_to_fork, fork_to_embed)
    if json_file_path:
        transform_json_file(json_file_path, embed_to_fork, fork_to_embed)