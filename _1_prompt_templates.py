import re
##############################################################
# ReGAL-Input
regal_prompt=f"""
Your task is to draw simple figures using python Turtle graphics.\n
You will use a custom turtle library, similar to the built-in library, which is sufficient for all tasks.\n
\n
Here's a description of the custom turtle library:\n
- forward(x): move forward x pixels\n
- left(theta): rotateleft by the theta a degrees\n
- right(theta): rotate right by the theta a degrees\n
- penup(): stop drawing\n
- pendown(): start drawing\n
- teleport(x, y, theta): move to position(x, y) with angle the theta\n
- heading(): get the current angle of the turtle\n
- isdown(): check if the pen is down\n
- embed(program, local vars): runs the code in program using the current context and teleports back to the original position. Allows you to next programs. Implementationally, embed gets the turtle state (is down, x, y, heading), executes program, then returns to the original state.\n
\n
You will be given a query and have to produce a program. Begin your program with a comment that explains your reasoning. For example, you might write:\n
# Thought: the query asks for a line, so I will use the forward() function.\n
Examples:\n
\n
Please generate ONLY the code to produce the answer and nothing else.\n
Query: Draw PLACEHOLDER_DESCRIPTION\n
Program:\n
"""

# PBE-Input
pbe_prompt=f"""
Your task is to draw simple black and white graphics with the custom library. DO NOT USE THE BUILT-IN TURTLE LIBRARY.\n
You will use a custom turtle library, similar to the built-in library, which is sufficient for all tasks.\n
\n
Here are all the available functions in the custom turtle library:\n
- forward(x): move forward x pixels\n
- left(theta): rotate left by theta degrees\n
- right(theta): rotate right by theta degrees\n
- penup(): stop drawing\n
- pendown(): start drawing\n
- teleport(x, y, theta): move to position (x, y) with angle theta\n
- heading(): get the current angle of the turtle\n
- isdown(): check if the pen is down\n
- embed(program, local vars): runs the code in program using the current context and teleports back to the original position. Allows you to nest programs. Implementationally, embed gets the turtle state (is down, x, y, heading), executes program, then returns to the original state.\n
\n
Graphic:
Python program: draw an interesting graphic using our own custom turtle library.\n
# the following program draws PLACEHOLDER_DESCRIPTION:\n
Program: \n
"""
##############################################################
##############################################################
# System Prompt - based on the ReGAL and PBE prompt-templates
sys_prompt = """
Your task is to draw simple black and white graphics with the custom library. DO NOT USE THE BUILT-IN TURTLE LIBRARY.
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

token_dict = {
    "B_SYS": ["<s>", "[INST]", "### System Prompt", "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"],
    "E_SYS": [None, None, "[/INST]", None],
    "B_USER": [None, "[INST]", "### User Prompt", "<|start_header_id|>user<|end_header_id|>"],
    "E_USER": [None, None, "[/INST]", None],
    "B_ASSISTANT": [None, "[PY_CODE]", "### Assistant Output", "<|start_header_id|>assistant<|end_header_id|>"],
    "E_ASSISTANT": ["</s>", "[/PY_CODE]", None, "<|end_of_text|>"]
}

# Corrected get_token function to ensure it handles the token_dict correctly
def get_token(token_dict, token_key, position):
    tokens = token_dict.get(token_key, ["", ""])
    return tokens[position] if position < len(tokens) and tokens[position] is not None else ""

def remove_extra_newlines(text):
    # Replace multiple newlines with a single one
    return re.sub(r'\n\s*\n', '\n', text)

# Corrected format_prompt function with proper token_dict handling
def format_prompt(description, ascii_art, include_description=False, include_ascii=False, token_dict=None, token_position=0):
    if token_dict is None:
        raise ValueError("token_dict must be provided.")

    # User Instruction
    if not (include_description or include_ascii):
        raise ValueError("At least one of include_description or include_ascii must be True.")
    
    user_instruction = "Generate the python program producing the graphic, which is"
    if include_description and include_ascii:
        user_instruction += " described and depicted"
    elif include_description:
        user_instruction += " described"
    elif include_ascii:
        user_instruction += " depicted"
    
    user_instruction += " as follows:\n"
    if include_description:
        user_instruction += f"    The Program draws {description}\n"
    if include_ascii:
        user_instruction += f"    Graphic:\n{ascii_art}\n"
    
    formatted_text = f"""{get_token(token_dict, "B_SYS", token_position)}
{sys_prompt}
{get_token(token_dict, "E_SYS", token_position)}
{get_token(token_dict, "B_USER", token_position)}
{user_instruction}
{get_token(token_dict, "E_USER", token_position)}
{get_token(token_dict, "B_ASSISTANT", token_position)}
Python program:
{get_token(token_dict, "E_ASSISTANT", token_position)}"""
    
    formatted_text = remove_extra_newlines(formatted_text)

    return formatted_text


def format_data(df, include_description=False, include_ascii=False, token_dict=token_dict, token_position=0):
    def format_row(example):
        description = example["Description"]
        ascii_art = example["ASCII-Art"]

        input_prompt = format_prompt(
            description, ascii_art, 
            include_description=include_description, include_ascii=include_ascii, 
            token_dict=token_dict, token_position=token_position
        )

        # Return the formatted row as a dictionary
        return {"Prompt": input_prompt}

    # Apply the formatting function across the entire dataset using map
    formatted_data = df.map(format_row, remove_columns=["Description", "ASCII-Art"])

    return formatted_data
