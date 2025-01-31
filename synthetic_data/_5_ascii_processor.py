import os
import numpy as np
from PIL import Image
import pandas as pd

class ASCIIProcessor:
    def __init__(self, n_blocks=35, m_blocks=35, levels=10):
        """
        Initialize the ASCII Processor with default parameters.
        Args:
            n_blocks: Number of blocks along the height of the image.
            m_blocks: Number of blocks along the width of the image.
            levels: Number of quantization levels for ASCII representation.
        """
        self.n_blocks = n_blocks
        self.m_blocks = m_blocks
        self.levels = levels

    def return_divisor(self, n):
        divisors = []
        for i in range(1, n + 1):
            if n % i == 0:
                divisors.append(i)
        return divisors

    def image_to_ascii(self, image_path):
        """
        Convert an image to an ASCII matrix.
        Args:
            image_path: Path to the input image.
        Returns:
            ascii_matrix: A matrix representing the ASCII art.
        """
        # Load the image and convert it to grayscale
        image = Image.open(image_path).convert("L")
        image = np.array(image)
        
        # Get the dimensions of the image
        height, width = image.shape
        
        # Ensure image size matches the required dimensions
        assert height == 525 and width == 525, "Image dimensions must be 525x525 pixels."
        
        # Determine block size
        block_height = height // self.n_blocks  
        block_width = width // self.m_blocks
        divisors_height = self.return_divisor(height)
        divisors_width = self.return_divisor(width)
        common_divisors = list(set(divisors_height).intersection(set(divisors_width)))
        if not common_divisors:
            raise ValueError(f"Block sizes {self.n_blocks}x{self.m_blocks} must divide the image size of {height}x{width}.\nPossible Block sizes hight: {self.return_divisor(height)}\nPossible Block sizes width: {self.return_divisor(width)}\n Common divisors: {common_divisors}")

        
        # Initialize matrix for ASCII levels
        ascii_matrix = np.zeros((self.n_blocks, self.m_blocks), dtype=int)
        
        # Loop through the blocks
        for i in range(self.n_blocks):
            for j in range(self.m_blocks):
                # Get the block
                block = image[
                    i * block_height : (i + 1) * block_height,
                    j * block_width : (j + 1) * block_width
                ]
                
                # Calculate the density of black pixels (assuming black is 0 intensity)
                density = np.mean(block < 128)  # Threshold for black pixel
                
                # Quantize the density into levels
                quantized_value = int(density * self.levels)
                ascii_matrix[i, j] = min(quantized_value, self.levels - 1)  # Ensure within range
        
        return ascii_matrix

    @staticmethod # a static method operates on the class itself, not on instances of the class
    def ascii_matrix_to_string(ascii_matrix):
        """
        Convert an ASCII matrix into a string representation.
        Args:
            ascii_matrix: The matrix containing ASCII values.
        Returns:
            ascii_string: A string representation of the matrix.
        """
        return "\n".join("".join(str(val) for val in row) for row in ascii_matrix)

    def store_ascii_input(self, df, image_dir):
        """
        Process a DataFrame of descriptions and generate ASCII art for each.
        Args:
            df: DataFrame containing a 'Description' column.
            image_dir: Directory containing the images.
        Returns:
            Updated DataFrame with an 'ascii_input' column.
        """
        df_copy = df.copy() 
        ascii_inputs = []

        for i, row in df_copy.iterrows():
            description = row["Description"].replace(" ", "_")  
            image_path = os.path.join(image_dir, f"{i}_{description}.png")
            
            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                ascii_inputs.append(None)
                continue
            
            # Process the image to ASCII
            ascii_matrix = self.image_to_ascii(image_path)
            ascii_art = self.ascii_matrix_to_string(ascii_matrix)
            ascii_inputs.append(ascii_art)
        
        # Update the DataFrame
        df_copy["ASCII-Art"] = ascii_inputs
        return df_copy