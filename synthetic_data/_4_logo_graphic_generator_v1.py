# Version 1 - no bbox calculation
import os
import matplotlib.pyplot as plt
from _3_executable_logo_primitives import ReGALLOGOPrimitives

# Interpreter class
class PseudoProgramInterpreter:
    def __init__(self):
        self.state = ReGALLOGOPrimitives()
        self.circle_vars = {
            "EPS_ANGLE": 1.0,    # incrementing by 1° at each step   
            "EPS_DIST": 0.03490481287456702, 
            "HALF_INF": 180    # half-circle has 180°        
            }


    def execute(self, program, local_vars=None):
        """
        Execute a program sequence.
        program: str, the program text
        local_vars: dict, the local variables for `embed` functionality
        """
        local_scope = {"forward": self.state.forward,
                       "left": self.state.left,
                       "right": self.state.right,
                       "penup": self.state.penup,
                       "pendown": self.state.pendown,
                       "teleport": self.state.teleport,
                       "heading": self.state.heading,
                       "isdown": self.state.isdown,
                       "embed": self.embed}
        
        # Update the local scope with circle variables
        local_scope.update(self.circle_vars)  # Add circle variables
        if local_vars:
            local_scope.update(local_vars)
        
        exec(program, {}, local_scope)

    def embed(self, subprogram, local_vars):
        """
        Executes an embedded subprogram with access to the given locals.
        """
        self.execute(subprogram, local_vars)

            
    def save_graphics(self, filename="output.png", line_width=5.0):
        """
        Saves the generated graphics as an image file.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='datalim')

        # Draw paths
        for (x1, y1), (x2, y2) in self.state.path:
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=1.0, linewidth=line_width)  # Pen-down lines in black

        # Draw pen-up paths for visualization (optional)
        for (x1, y1), (x2, y2) in self.state.pen_up_path:
            ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.0)  # Pen-up lines in dashed red (if alpha > 0.0)

        # Hide axes for a clean output
        ax.axis('off')
        # Save the figure with tight bounding box
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def process_and_save_graphics(self, df, output_dir, filename="output.png"):
        """
        Executes a program and saves the resulting graphics as an image file.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Second pass: Save graphics with consistent scaling
        for i, row in df.iterrows():
            self.reset_state()
            self.execute(row["Program"])
            description = row["Description"].replace(" ", "_")  # Sanitize description for filenames
            filename = os.path.join(output_dir, f"{i}_{description}.png")
            self.save_graphics(filename)

    def reset_state(self):
        """
        Resets the graphics state for a new drawing.
        """
        self.state = ReGALLOGOPrimitives()
