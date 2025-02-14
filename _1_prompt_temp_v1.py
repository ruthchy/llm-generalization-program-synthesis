import yaml
def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

include_ascii = config["input"]["include_ascii"]
include_desc = config["input"]["include_desc"]

sys_prompt = """Your task is to draw simple black and white graphics with the custom library. DO NOT USE THE BUILT-IN TURTLE LIBRARY.
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

# Instruction Format 
def instruction_format(example):
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    formated_input = "Generate a python program producing the graphic, which is"
    if include_desc and include_ascii:
        formated_input += " described and depicted"
    elif include_desc:
        formated_input += " described"
    elif include_ascii:
        formated_input += " depicted"
    formated_input += " as follows:\n"
    if include_desc:
        formated_input += f"    The Program draws {example['Description']}\n"
    if include_ascii:
        formated_input += f"    Graphic:\n{example['ASCII-Art']}\n"

    prompt = f"""### Instruction: {formated_input}
### Python Program: {example["Program"]}"""
    return {"conversations": prompt}


# Conversational Format
def conversational_format(example):
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    if include_desc and include_ascii:
        return {
        "conversations": [
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described and depicted as follows:\n The Program draws {example['Description']}\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_desc:
        return {
        "conversations": [
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described as follows:\n The Program draws {example['Description']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_ascii:
        return {
        "conversations": [
            {"role": "user", "content": f"Generate a python program producing the graphic, which is depicted as follows:\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }

# Conversational Format inc system prompt
def conversational_format_inc_sys(example):
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    if include_desc and include_ascii:
        return {
        "conversations": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described and depicted as follows:\n The Program draws {example['Description']}\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_desc:
        return {
        "conversations": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described as follows:\n The Program draws {example['Description']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_ascii:
        return {
        "conversations": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Generate a python program producing the graphic, which is depicted as follows:\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }

# Format in Is-PBE
# Conversational Format
def conversational_format_PBE_zeroshot(example):
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    if include_desc and include_ascii:
        return {
        "conversations": [
            {"role": "user", "content": f"Here is a gray scale image described as containing {example['Description']}. The image is represented with integer values 0-9.\n{example['ASCII-Art']}\nPlease, write a Python program that generates this image using our own custom turtle module."},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_desc:
        return {
        "conversations": [
            {"role": "user", "content": f"Here is a gray scale image described as containing {example['Description']}\nPlease, write a Python program that generates this image using our own custom turtle module."},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_ascii:
        return {
        "conversations": [
            {"role": "user", "content": f"Here is a gray scale image represented with integer values 0-9.\n{example['ASCII-Art']}\nPlease, write a Python program that generates this image using our own custom turtle module."},
            {"role": "assistant", "content": example["Program"]}
        ]
    }

def conversational_format_PBE_zeroshot_inference(example):
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    if include_desc and include_ascii:
        return {
        "conversations": [
            {"role": "user", "content": f"Here is a gray scale image described as containing {example['Description']}. The image is represented with integer values 0-9.\n{example['ASCII-Art']}\nPlease, write a Python program that generates this image using our own custom turtle module."},
            #{"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_desc:
        return {
        "conversations": [
            {"role": "user", "content": f"Here is a gray scale image described as containing {example['Description']}\nPlease, write a Python program that generates this image using our own custom turtle module."},
            #{"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_ascii:
        return {
        "conversations": [
            {"role": "user", "content": f"Here is a gray scale image represented with integer values 0-9.\n{example['ASCII-Art']}\nPlease, write a Python program that generates this image using our own custom turtle module."},
            #{"role": "assistant", "content": example["Program"]}
        ]
    }

######################## test the functions ############################
# for the completionsonly.py
def formatting_prompts_func(example):
    output_texts = []
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    for i in range(len(example["Description"])):
        if include_ascii and include_desc:
            formated_input = f"Generate a python program producing the graphic, which is described and depicted as follows:\n    The Program draws {example['Description'][i]}\n    Graphic:\n{example['ASCII-Art'][i]}\n"
        elif include_desc:
            formated_input = f"Generate a python program producing the graphic, which is described as follows:\n    The Program draws {example['Description'][i]}\n"
        elif include_ascii:
            formated_input = f"Generate a python program producing the graphic, which is depicted as follows:\n    Graphic:\n{example['ASCII-Art'][i]}\n"

        text = f"### Instruction: {formated_input}\n### Python Program: {example['Program'][i]}"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_PBE(example):
    output_texts = []
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    for i in range(len(example["Description"])):
        if include_ascii and include_desc:
            formated_input = f"Here is a gray scale image described as containing {example['Description'][i]}. The image is represented with integer values 0-9.\n{example['ASCII-Art'][i]}\nPlease, write a Python program that generates this image using our own custom turtle module.\n"
        elif include_desc:
            formated_input = f"Here is a gray scale image described as containing {example['Description'][i]}\nPlease, write a Python program that generates this image using our own custom turtle module.\n"
        elif include_ascii:
            formated_input = f"Here is a gray scale image represented with integer values 0-9.\n{example['ASCII-Art'][i]}\nPlease, write a Python program that generates this image using our own custom turtle module.\n"

        text = f"### Instruction: {formated_input}\n### Python Program: {example['Program'][i]}"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_PBE_INSTtok(example):
    output_texts = []
    assert include_ascii or include_desc, "At least one of include_ascii or include_desc must be True."
    for i in range(len(example["Description"])):
        if include_ascii and include_desc:
            formated_input = f"Here is a gray scale image described as containing {example['Description'][i]}. The image is represented with integer values 0-9.\n{example['ASCII-Art'][i]}\nPlease, write a Python program that generates this image using our own custom turtle module.\n"
        elif include_desc:
            formated_input = f"Here is a gray scale image described as containing {example['Description'][i]}\nPlease, write a Python program that generates this image using our own custom turtle module.\n"
        elif include_ascii:
            formated_input = f"Here is a gray scale image represented with integer values 0-9.\n{example['ASCII-Art'][i]}\nPlease, write a Python program that generates this image using our own custom turtle module.\n"

        text = f"[INST]### Instruction: {formated_input}[/INST]\n### Python Program: {example['Program'][i]}"
        output_texts.append(text)
    return output_texts