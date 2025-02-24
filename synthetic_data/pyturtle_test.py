import sys
import os
import re
import matplotlib.pyplot as plt
from __parser_pyturtle_pc import ProgramParser
# Add the root directory of external dependencies to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dependencies_path = os.path.join(repo_root, 'external/dependencies')
sys.path.append(dependencies_path)

# Import dependencies
from program_refactoring.domains.logos.pyturtle_pc import PyTurtle, HALF_INF, EPS_DIST, EPS_ANGLE
from datasets import load_dataset

# Load the dataset
ds = load_dataset("ruthchy/semantic-length-generalization-logo-data-desc-ascii_35")

# Get the program from validation set line 18
program = ds['validation'][18]['Program']
print("Program to execute:")
print(program)

# Create PyTurtle instance
turtle = PyTurtle()

# Extract and execute the parsed program
exec_scope = {
    'turtle': turtle,
    'HALF_INF': HALF_INF,
    'EPS_DIST': EPS_DIST,
    'EPS_ANGLE': EPS_ANGLE,
    'embed': turtle.embed,  # Embed should call the method from PyTurtle
    'penup': turtle.penup,
    'pendown': turtle.pendown,
    'forward': turtle.forward,
    'left': turtle.left
}

try:
    exec(program, exec_scope)  # Execute the extracted program with PyTurtle
except Exception as e:
    print(f"Error executing program: {e}")

# Show the output
plt.show()

# Save the result
output_path = 'validation_18_output.png'
turtle.save(output_path)
print(f"\nOutput saved to {output_path}")
########################################################################################################################
import os
from datasets import load_dataset, DatasetDict, Features, Image, Value
from __parser_pyturtle_pc import ProgramParser  # Import the parser class

# Load the dataset
dataset_name = "ruthchy/semantic-length-generalization-logo-data-desc-ascii_35"
ds = load_dataset(dataset_name)
ds = ds.remove_columns("ASCII-Art")
print(ds)
# Select the first 100 rows of the train dataset for testing purposes
#ds['train'] = ds['train'].select(range(100))

# Create an instance of the parser
parser = ProgramParser(save_dir="logo_graphic/len_gen_dataset", save_image=True, eval_mode=False)

push_to_hub = True  # Change to True if you want to upload
hub_name = "ruthchy/semantic-length-generalization-logo-image" 
ds = parser.wrapper_parse_and_generate_image(ds, push_to_hub=push_to_hub, hub_name=hub_name)