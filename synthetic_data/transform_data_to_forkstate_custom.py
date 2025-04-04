import re
from datasets import load_dataset

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

def transform(example):
    if "Program" in example:
        old_prog = example["Program"]
        new_prog = pattern.sub(embed_to_fork_state, old_prog)
        example["Program"] = new_prog  
    return example

if __name__ == "__main__":
    # 1. Load your dataset
    ds = load_dataset("ruthchy/length-gen-logo-image")

    # 3. Apply the transformation to the dataset
    ds_fork = ds.map(transform)

    # 4. Push the transformed dataset to Hugging Face under a new name
    ds_fork.push_to_hub("ruthchy/length-gen-logo-image-prog-fork")