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
Query: Draw {description}\n
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
# the following program draws {description}:\n
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
    "B_SYS": ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<s>"],
    "E_SYS": [None, None],
    "B_USER": ["<|start_header_id|>user<|end_header_id|>", None],
    "E_USER": [None, None],
    "B_ASSISTANT": ["<|start_header_id|>assistant<|end_header_id|>", None],
    "E_ASSISTANT": ["<|end_of_text|>", "</s>"]
}
def get_token(token_key, position):
    tokens = token_dict.get(token_key, ["", ""])
    return tokens[position] if tokens[position] is not None else ""

# Function to format each prompt based on input type
def format_prompt(description, ascii_art, include_description=False, include_ascii=False, token_dict=None, token_position=0):
    # User Instruction
    if not (include_description or include_ascii):
        raise ValueError("At least one of include_description or include_ascii must be True.")
    
    user_instruction = "Use the following"
    if include_description and include_ascii:
        user_instruction += " description and ASCII-art representing the target graphic"
    elif include_description:
        user_instruction += " description of the target graphic"
    elif include_ascii:
        user_instruction += " ASCII-art representing the target graphic"
    
    user_instruction += " to generate the python program:\n\n"
    
    if include_description:
        user_instruction += f"    Description: {description}\n"
    if include_ascii:
        user_instruction += f"    ASCII-Art:\n{ascii_art}\n"

    
    formatted_text = f"""
{get_token("B_SYS", token_position)}
{sys_prompt}
{get_token("E_SYS", token_position)}
{get_token("B_USER", token_position)}
{user_instruction}
{get_token("E_USER", token_position)}
{get_token("B_ASSISTANT", token_position)}
{get_token("E_ASSISTANT", token_position)}
"""
    return formatted_text

