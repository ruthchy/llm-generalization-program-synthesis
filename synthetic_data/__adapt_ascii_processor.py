import os
import numpy as np
import signal
from PIL import Image
from datasets import DatasetDict

class TimeoutError(Exception):
    """Raised when a timeout occurs during user input."""
    pass

def timeout_handler(signum, frame):
    """Handler for SIGALRM signal."""
    raise TimeoutError("No block size was selected within the time limit")

class AdaptiveASCIIProcessor:
    def __init__(self, levels=10, black_threshold=150, block_size=None, timeout=120, drop_images=False):
        """
        Initialize the Adaptive ASCII Processor with default parameters.
        Args:
            levels: Number of quantization levels for ASCII representation.
            black_threshold: Threshold for determining black pixels (0-255).
            block_size: Optional predefined block size. If None, will be determined interactively.
            timeout: Timeout in seconds for user input when selecting block size.
            drop_images: Whether to drop the Image column after processing to save memory.
        """
        self.levels = levels
        self.black_threshold = black_threshold
        self.predefined_block_size = block_size
        self.block_size = None
        self.timeout = timeout
        self.drop_images = drop_images

    def return_divisors(self, n):
        """Find all divisors of n."""
        divisors = []
        for i in range(1, n + 1):
            if n % i == 0:
                divisors.append(i)
        return divisors

    def determine_block_size(self, images):
        """
        Determine block size for ASCII conversion, with optional timeout for user input.
        Args:
            images: List of PIL images.
        Returns:
            Selected block size.
        Raises:
            TimeoutError: If no block size is selected within timeout period.
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

        # If a block size was provided during initialization, check if it's valid
        if self.predefined_block_size is not None:
            if self.predefined_block_size in common_divisors:
                self.block_size = self.predefined_block_size
                print(f"Using predefined block size: {self.block_size}")
                return self.block_size
            else:
                print(f"Warning: Predefined block size {self.predefined_block_size} is not a common divisor. Please select from the list.")
        
        # Set up timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)  # Set alarm for timeout seconds
        
        try:
            while True:
                try:
                    block_size = int(input(f"Enter a block size from the list (you have {self.timeout} seconds): "))
                    if block_size in common_divisors:
                        self.block_size = block_size
                        break
                    else:
                        print("Invalid block size. Please select from the list above.")
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
        except TimeoutError as e:
            print(f"Timeout: {str(e)}")
            # Use a reasonable default - middle of the list or power of 10
            default_options = [d for d in common_divisors if d in (10, 20, 25, 50, 100)]
            if default_options:
                self.block_size = default_options[0]
            else:
                # Choose a divisor in the middle of the list
                self.block_size = common_divisors[len(common_divisors) // 2]
            print(f"Automatically selected block size: {self.block_size}")
        finally:
            # Cancel the alarm
            signal.alarm(0)

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
        
        # Optionally drop the image to save memory
        if self.drop_images and "Image" in example:
            del example["Image"]
        
        return example

    def process_dataset(self, ds):
        """
        Process an HF dataset and apply the ASCII conversion efficiently.
        Ensures block size is determined only once if all images share dimensions.
        
        Args:
            ds: HuggingFace dataset or DatasetDict to process
            
        Returns:
            Processed dataset with ASCII-Art field added and optionally Image field removed
        """
        # Extract images
        if isinstance(ds, DatasetDict):
            # Take just a few images to determine block size (for performance)
            sample_images = []
            for split in ds.keys():
                if len(ds[split]) > 0:
                    sample_images.append(ds[split][0]["Image"])
                    if len(sample_images) >= 5:  # 5 samples should be enough to check dimension consistency
                        break
            # If we still need more images, sample from the largest split
            if not sample_images:
                raise ValueError("No images found in dataset")
        else:
            if len(ds) == 0:
                raise ValueError("Empty dataset provided")
            sample_images = [ds[0]["Image"]]  # Just use the first image to determine block size

        # Determine block size once
        self.determine_block_size(sample_images)

        # Apply processing
        if isinstance(ds, DatasetDict):
            modified_ds = DatasetDict()
            for split in ds.keys():
                modified_ds[split] = ds[split].map(lambda example: self.store_ascii_input(example))
            return modified_ds
        else:
            return ds.map(lambda example: self.store_ascii_input(example))