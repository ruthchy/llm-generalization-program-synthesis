import os
import numpy as np
from PIL import Image
from datasets import DatasetDict

class AdaptiveASCIIProcessor:
    def __init__(self, levels=10, black_threshold=150):
        """
        Initialize the Adaptive ASCII Processor with default parameters.
        Args:
            levels: Number of quantization levels for ASCII representation.
        """
        self.levels = levels
        self.black_threshold = black_threshold # 128
        self.block_size = None

    def return_divisors(self, n):
        divisors = []
        for i in range(1, n + 1):
            if n % i == 0:
                divisors.append(i)
        return divisors

    def determine_block_size(self, images):
        """
        Determine a common block size for all images if they share the same dimensions.
        Args:
            images: List of PIL images.
        Returns:
            Selected block size.
        """
        unique_sizes = set((img.height, img.width) for img in images)

        if len(unique_sizes) > 1:
            raise ValueError("Images have different dimensions. Ensure all images are the same size.")

        height, width = unique_sizes.pop()

        # Determine the common divisors of the image dimensions
        divisors_height = self.return_divisors(height)
        divisors_width = self.return_divisors(width)
        common_divisors = list(set(divisors_height).intersection(set(divisors_width)))
        if not common_divisors:
            raise ValueError(f"No common divisors found for image size {height}x{width}.")
        
        common_divisors.sort()
        print(f"Common divisors for image size {height}x{width}: {common_divisors}")

        while True:
            try:
                block_size = int(input("Enter a block size from the list above: "))
                if block_size in common_divisors:
                    self.block_size = block_size
                    break
                else:
                    print("Invalid block size. Please select from the list above.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

        print(f"Selected block size: {self.block_size}")
        return self.block_size

    def image_to_ascii(self, image):
        """
        Convert an image to an ASCII matrix.
        Args:
            image: PIL Image object.
        Returns:
            ascii_matrix: A matrix representing the ASCII art.
        """
        if self.block_size is None:
            raise ValueError("Block size has not been set. Call determine_block_size() first.")

        # Convert the image to grayscale
        image = image.convert("L")
        image = np.array(image)
        height, width = image.shape

        # Set the block size
        n_blocks = m_blocks = self.block_size
        block_height = height // n_blocks
        block_width = width // m_blocks
        
        # Initialize matrix for ASCII levels
        ascii_matrix = np.zeros((n_blocks, m_blocks), dtype=int)
        
        # Loop through the blocks
        for i in range(n_blocks):
            for j in range(m_blocks):
                # Get the block
                block = image[
                    i * block_height : (i + 1) * block_height,
                    j * block_width : (j + 1) * block_width
                ]
                
                # Calculate the density of black pixels (assuming black is 0 intensity)
                density = np.mean(block < self.black_threshold)  # Threshold for when pixels are determined as black
                
                # Quantize the density into levels
                quantized_value = int(density * self.levels)
                ascii_matrix[i, j] = min(quantized_value, self.levels - 1)  # Ensure within range
        
        return ascii_matrix

    @staticmethod
    def ascii_matrix_to_string(ascii_matrix):
        """
        Convert an ASCII matrix into a string representation.
        Args:
            ascii_matrix: The matrix containing ASCII values.
        Returns:
            ascii_string: A string representation of the matrix.
        """
        return "\n".join("".join(str(val) for val in row) for row in ascii_matrix)

    def store_ascii_input(self, example):
        """
        Process an example and generate ASCII art for the image.
        Args:
            example: A single example from the dataset.
        Returns:
            Updated example with an 'ASCII-Art' field.
        """
        # Get the image from the example
        image = example["Image"]
        
        # Process the image to ASCII
        ascii_matrix = self.image_to_ascii(image)
        ascii_art = self.ascii_matrix_to_string(ascii_matrix)
        example["ASCII-Art"] = ascii_art
        
        return example

    def process_dataset(self, ds):
        """
        Process an HF dataset and apply the ASCII conversion efficiently.
        Ensures block size is determined only once if all images share dimensions.
        """
        # Extract images
        if isinstance(ds, DatasetDict):
            all_images = [img for split in ds.keys() for img in ds[split]["Image"]]
        else:
            all_images = ds["Image"]

        # Determine block size once
        self.determine_block_size(all_images)

        # Apply processing
        if isinstance(ds, DatasetDict):
            for split in ds.keys():
                ds[split] = ds[split].map(lambda example: self.store_ascii_input(example))
        else:
            ds = ds.map(lambda example: self.store_ascii_input(example))

        return ds