import re
from datasets import load_dataset

# 1. Load your dataset
ds = load_dataset("ruthchy/semantic-length-generalization-logo-data-desc-ascii_35")

# 2. Define the same regex pattern
pattern = re.compile(
    r'^(?P<indent>\s*)embed\("""(?P<code>.*?)""",\s*locals\(\)\)',
    re.MULTILINE | re.DOTALL
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

def transform_completion(example):
    """Transform the program field in each example"""
    if 'Program' in example:
        program = example['Program']
    elif 'completion' in example:
        program = example['completion']
    else:
        return example
        
    new_program = pattern.sub(embed_to_fork_state, program)
    example['Program'] = new_program
    return example

# Optional: Push to HF (commented out)
ds_transformed = ds.map(transform_completion)
print(ds["train"]["Program"][8:10])
print("\n")
print(ds_transformed["train"]["Program"][8:10])
# ds_transformed.push_to_hub("ruthchy/semantic-length-generalization-logo-data-desc-ascii_35_fork")