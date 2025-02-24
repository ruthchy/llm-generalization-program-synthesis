import os
import sys
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
# uses the PyTurtle class from the ReGAL repo (only slightly modified by excluding the codebase method which requires an openai api key)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dependencies_path = os.path.join(repo_root, 'external/dependencies')
sys.path.append(dependencies_path)
from program_refactoring.domains.logos.pyturtle_pc import PyTurtle, HALF_INF, EPS_DIST, EPS_ANGLE

class ProgramParser:
    """Class to parse and execute logo programs, generating images using PyTurtle."""

    def __init__(self, save_dir="logo_graphic", save_image=False, eval_mode=False):
        self.save_dir = save_dir
        self.save_image = save_image
        self.eval_mode = eval_mode
        if not eval_mode and not save_image:
            os.makedirs(self.save_dir, exist_ok=True)

    def parse_and_generate_image(self, example, split_name=None):
        """Parses the program, executes it, and stores the generated image path."""
        if "Program" not in example:
            print(f"Error: 'Program' key not found in example: {example}")
            example["Image"] = None
            return example

        program = example["Program"]
        turtle = PyTurtle()  # Create a new PyTurtle instance

        # Define execution context
        exec_scope = {
            "turtle": turtle,
            "HALF_INF": HALF_INF,
            "EPS_DIST": EPS_DIST,
            "EPS_ANGLE": EPS_ANGLE,
            "forward": turtle.forward,
            "left": turtle.left,
            "right": turtle.right,
            "teleport": turtle.teleport,
            "penup": turtle.penup,
            "pendown": turtle.pendown,
            "heading": turtle.heading,
            "embed": turtle.embed
        }

        try:
            exec(program, exec_scope)  # Execute the program
        except Exception as e:
            print(f"Error executing program for ID {example.get('id', 'unknown')}: {e}")
            example["Image"] = None  # Store None if execution fails
            return example

        # Save the result as a PIL image
        turtle.fig.canvas.draw()
        pil_img = PILImage.frombytes('RGB', turtle.fig.canvas.get_width_height(), turtle.fig.canvas.tostring_rgb())
        plt.close(turtle.fig)  # Close the figure to free up memory

        # Start with the id followed by the description
        description = example.get('Description', example.get('id', 'unknown')).replace(' ', '_')
        if self.eval_mode:
            example["Image"] = pil_img  # Store PIL image temporarily
        elif self.save_image:
            example["Image"] = pil_img  # Store PIL image in dataset
        else:
            if split_name is None:
                image_path = os.path.join(self.save_dir, f"{example.get('id', 'unknown')}_{description}.png")
            else:
                image_path = os.path.join(self.save_dir, f"{split_name}/{example.get('id', 'unknown')}_{description}.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the directory exists
            pil_img.save(image_path)
            example["Image"] = image_path

        return example

    def wrapper_parse_and_generate_image(self, ds, push_to_hub=False, hub_name=None, split_name=None):
        """Wrapper function to apply the parse_and_generate_image function to a dataset with or without splits."""
        if isinstance(ds, DatasetDict):
            for split_name in ds.keys():
                ds[split_name] = ds[split_name].map(lambda example: self.parse_and_generate_image(example, split_name))
        else:
            ds = ds.map(lambda example: self.parse_and_generate_image(example, split_name))

        if push_to_hub and hub_name:
            ds.push_to_hub(hub_name, private=False)
            print(f"Dataset pushed to Hub: {hub_name}")
        else:
            print("Dataset not pushed to Hub.")
        
        return ds

