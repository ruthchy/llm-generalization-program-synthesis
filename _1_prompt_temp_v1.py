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
def instruction_format(example, include_description=False, include_ascii=False):
    if not (include_description or include_ascii):
        raise ValueError("At least one of include_description or include_ascii must be True.")
    formated_input = "Generate a python program producing the graphic, which is"
    if include_description and include_ascii:
        formated_input += " described and depicted"
    elif include_description:
        formated_input += " described"
    elif include_ascii:
        formated_input += " depicted"
    formated_input += " as follows:\n"
    if include_description:
        formated_input += f"    The Program draws {example['Description']}\n"
    if include_ascii:
        formated_input += f"    Graphic:\n{example['ASCII-Art']}\n"

    prompt = f"""### Instruction: {formated_input}
### Python Program: {example["Program"]}"""
    return {"conversations": prompt}

# Conversational Format
def conversational_format(example, include_description=False, include_ascii=False):
    if not (include_description or include_ascii):
        raise ValueError("At least one of include_description or include_ascii must be True.")
    if include_description and include_ascii:
        return {
        "conversations": [
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described and depicted as follows:\n The Program draws {example['Description']}\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_description:
        return {
        "conversations": [
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described as follows:\n The Program draws {example['Description']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_description:
        return {
        "conversations": [
            {"role": "user", "content": f"Generate a python program producing the graphic, which is depicted as follows:\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }

# Conversational Format inc system prompt
def conversational_format(example, include_description=False, include_ascii=False):
    if not (include_description or include_ascii):
        raise ValueError("At least one of include_description or include_ascii must be True.")
    if include_description and include_ascii:
        return {
        "conversations": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described and depicted as follows:\n The Program draws {example['Description']}\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_description:
        return {
        "conversations": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Generate a python program producing the graphic, which is described as follows:\n The Program draws {example['Description']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
    elif include_description:
        return {
        "conversations": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Generate a python program producing the graphic, which is depicted as follows:\n Graphic:\n{example['ASCII-Art']}\n"},
            {"role": "assistant", "content": example["Program"]}
        ]
    }
