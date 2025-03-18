import os
import numpy as np
import signal
from PIL import Image
from datasets import DatasetDict
from functools import partial

class TimeoutError(Exception):
    """Raised when a timeout occurs during user input."""
    pass

def timeout_handler(signum, frame):
    """Handler for SIGALRM signal."""
    raise TimeoutError("No block size was selected within the time limit")

class AdaptiveASCIIProcessor:
    def __init__(self, levels=10, black_threshold=150, block_size=None, crop_to_size=None, timeout=120, drop_images=False):
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
        self.block_size = block_size       # Set directly if provided
        self.crop_to_size = crop_to_size
        self.timeout = timeout
        self.drop_images = drop_images

    def center_crop(self, image, target_size):
        """
        Crop the image to the target size from the center.
        
        Args:
            image: PIL Image object
            target_size: Tuple (width, height) for the desired crop size
        
        Returns:
            Cropped PIL Image
        """
        width, height = image.size
        target_width = target_height = target_size
        
        # Ensure target size is not larger than original image
        target_width = min(width, target_width)
        target_height = min(height, target_height)
        
        # Calculate crop coordinates
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        # Crop and return
        return image.crop((left, top, right, bottom))

    def return_divisors(self, n):
        """Find all divisors of n efficiently."""
        divisors = []
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:  # Avoid duplicates for perfect squares
                    divisors.append(n // i)
        return sorted(divisors)

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
        # If block size is already set, just return it
        if self.block_size is not None:
            return self.block_size
        
        # Apply optional center cropping to sample image for determining block size
        if self.crop_to_size is not None:
            sample_image = self.center_crop(images[0], self.crop_to_size)
            height, width = sample_image.height, sample_image.width
        else:
            unique_sizes = set((img.height, img.width) for img in images)
            if len(unique_sizes) > 1:
                raise ValueError("Images have different dimensions. Ensure all images are the same size.")
            height, width = unique_sizes.pop()

        # Determine the common divisors of the image dimensions
        divisors_height = self.return_divisors(height)
        divisors_width = self.return_divisors(width)
        common_divisors = sorted(set(divisors_height).intersection(set(divisors_width)))
        
        if not common_divisors:
            raise ValueError(f"No common divisors found for image size {height}x{width}.")
        
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

    def image_to_ascii_optimized(self, image):
        """
        Convert an image to ASCII matrix using optimized vectorized operations.
        Args:
            image: PIL Image object.
        Returns:
            ascii_matrix: A matrix representing the ASCII art.
        """
        if self.block_size is None:
            raise ValueError("Block size has not been set. Call determine_block_size() first.")

        # Apply center cropping if specified
        if self.crop_to_size is not None:
            image = self.center_crop(image, self.crop_to_size)

        # Convert the image to grayscale
        image = image.convert("L")
        image_array = np.array(image)
        height, width = image_array.shape

        # Calculate block dimensions
        n_blocks = m_blocks = self.block_size
        block_height = height // n_blocks
        block_width = width // m_blocks

        # Reshape the image into blocks - this is much faster than using nested loops
        # First, ensure the image dimensions are divisible by the block size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # Crop the image to make dimensions divisible
            new_height = (height // self.block_size) * self.block_size
            new_width = (width // self.block_size) * self.block_size
            image_array = image_array[:new_height, :new_width]

        # Now reshape to get blocks
        blocks = image_array.reshape(n_blocks, block_height, m_blocks, block_width)
        
        # Calculate the density of each block (mean of pixels < threshold)
        # This is the vectorized equivalent of the nested loops
        black_pixels = blocks < self.black_threshold
        densities = np.mean(black_pixels, axis=(1, 3))  # Average across height and width dimensions of each block
        
        # Quantize the densities into levels
        quantized_values = np.minimum((densities * self.levels).astype(int), self.levels - 1)
        
        return quantized_values

    def image_to_ascii(self, image):
        """Wrapper for the optimized function to maintain compatibility."""
        return self.image_to_ascii_optimized(image)

    @staticmethod
    def ascii_matrix_to_string(ascii_matrix):
        """
        Convert an ASCII matrix into a string representation, optimized.
        Args:
            ascii_matrix: The matrix containing ASCII values.
        Returns:
            ascii_string: A string representation of the matrix.
        """
        # Faster join operations
        return '\n'.join(''.join(map(str, row)) for row in ascii_matrix)

    def process_batch(self, examples):
        """
        Process a batch of examples - optimized for speed.
        Args:
            examples: A batch of examples from the dataset.
        Returns:
            Updated examples with 'ASCII-Art' field.
        """
        images = examples["Image"]
        ascii_arts = []
        
        for img in images:
            ascii_matrix = self.image_to_ascii(img)
            ascii_art = self.ascii_matrix_to_string(ascii_matrix)
            ascii_arts.append(ascii_art)
            
        examples["ASCII-Art"] = ascii_arts
        
        # Optionally drop the images to save memory
        if self.drop_images:
            del examples["Image"]
            
        return examples

    def process_dataset(self, ds, batch_size=16, num_proc=4):
        """
        Process an HF dataset and apply the ASCII conversion efficiently.
        Uses batched processing and multiple CPU cores for speed.
        
        Args:
            ds: HuggingFace dataset or DatasetDict to process
            batch_size: Number of examples to process in each batch
            num_proc: Number of processes to use for parallel processing
            
        Returns:
            Processed dataset with ASCII-Art field added and optionally Image field removed
        """
        # Extract images for block size determination
        if isinstance(ds, DatasetDict):
            # Take just a few images to determine block size
            sample_images = []
            for split in ds.keys():
                if len(ds[split]) > 0:
                    sample_images.append(ds[split][0]["Image"])
                    if len(sample_images) >= 5:
                        break
            if not sample_images:
                raise ValueError("No images found in dataset")
        else:
            if len(ds) == 0:
                raise ValueError("Empty dataset provided")
            sample_images = [ds[0]["Image"]]

        # Only determine block size if it's not already set
        if self.block_size is None:
            self.determine_block_size(sample_images)

        # Apply processing using batched map with multiple processes
        if isinstance(ds, DatasetDict):
            modified_ds = DatasetDict()
            for split in ds.keys():
                modified_ds[split] = ds[split].map(
                    self.process_batch,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=num_proc
                )
            return modified_ds
        else:
            return ds.map(
                self.process_batch,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc
            )