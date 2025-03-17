import random
from datasets import DatasetDict, Dataset
from typing import Dict, Set, Tuple, List, Union, Optional
import pandas as pd

class DatasetDirectionModifier:
    """
    Class for modifying 'left' and 'right' directions in dataset programs.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the dataset modifier.
        
        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        
    def replace_direction(self, 
                          dataset: Union[Dataset, DatasetDict], 
                          source_dir: str = "left", 
                          target_dir: str = "right", 
                          field: str = "Program",
                          proportion: float = 0.5,
                          splits: Optional[List[str]] = None,
                          return_overview: bool = True) -> Union[Dataset, DatasetDict, Tuple[Union[Dataset, DatasetDict], pd.DataFrame]]:
        """
        Replace directional terms in the dataset and optionally return modification overview.
        
        Args:
            dataset: The dataset to modify (Dataset or DatasetDict)
            source_dir: Source direction to replace (default: "left")
            target_dir: Target direction to replace with (default: "right")
            field: Field containing the text to modify (default: "Program")
            proportion: Proportion of examples to modify (default: 0.5)
            splits: List of splits to modify (default: all splits)
            return_overview: Whether to return an overview of modifications (default: True)
            
        Returns:
            If return_overview is False: Modified dataset with the same structure as input
            If return_overview is True: Tuple of (modified dataset, overview DataFrame)
        """
        # First, count occurrences in the original dataset
        original_counts = self.count_directions(
            dataset, 
            directions=[source_dir, target_dir], 
            field=field
        )
        
        # Create modified dataset
        if isinstance(dataset, DatasetDict):
            # Handle DatasetDict case
            modified_ds = DatasetDict()
            used_splits = splits if splits is not None else dataset.keys()
            
            for split_name in dataset.keys():
                if split_name in used_splits:
                    modified_ds[split_name] = self._modify_split(
                        dataset[split_name], source_dir, target_dir, field, proportion
                    )
                else:
                    modified_ds[split_name] = dataset[split_name]
        else:
            # Handle single Dataset case
            modified_ds = self._modify_split(dataset, source_dir, target_dir, field, proportion)
        
        # If overview not requested, just return the modified dataset
        if not return_overview:
            return modified_ds
        
        # Count occurrences in the modified dataset
        modified_counts = self.count_directions(
            modified_ds, 
            directions=[source_dir, target_dir], 
            field=field
        )
        
        # Create a summary DataFrame
        data = []
        
        if isinstance(dataset, DatasetDict):
            for split_name in dataset.keys():
                orig_source = original_counts[split_name][source_dir]
                orig_target = original_counts[split_name][target_dir]
                mod_source = modified_counts[split_name][source_dir]
                mod_target = modified_counts[split_name][target_dir]
                
                data.append([split_name, orig_source, orig_target, mod_source, mod_target])
        else:
            # Single dataset case
            orig_source = original_counts["dataset"][source_dir]
            orig_target = original_counts["dataset"][target_dir]
            mod_source = modified_counts["dataset"][source_dir]
            mod_target = modified_counts["dataset"][target_dir]
            
            data.append(["dataset", orig_source, orig_target, mod_source, mod_target])
        
        # Create the DataFrame
        column_names = [
            "Split", 
            f"Orig {source_dir}", 
            f"Orig {target_dir}", 
            f"Mod {source_dir}", 
            f"Mod {target_dir}"
        ]
        
        overview_df = pd.DataFrame(data, columns=column_names)
        
        # Return both the modified dataset and the overview
        return modified_ds, overview_df
        
    def _modify_split(self, 
                     split: Dataset, 
                     source_dir: str, 
                     target_dir: str, 
                     field: str, 
                     proportion: float) -> Dataset:
        """
        Modify a single dataset split by replacing directional terms.
        
        Args:
            split: Dataset split to modify
            source_dir: Source direction to replace
            target_dir: Target direction to replace with
            field: Field containing the text to modify
            proportion: Proportion of examples to modify
            
        Returns:
            Modified dataset split
        """
        num_examples = len(split)
        num_to_modify = int(num_examples * proportion)
        selected_indices = set(random.sample(range(num_examples), k=num_to_modify))
        
        def _replace_direction(example, idx):
            if idx in selected_indices:
                example[field] = example[field].replace(source_dir, target_dir)
            return example
        
        return split.map(_replace_direction, with_indices=True)
        
    def count_directions(self, 
                         dataset: Union[Dataset, DatasetDict], 
                         directions: List[str] = ["left", "right"], 
                         field: str = "Program",
                         splits: Optional[List[str]] = None) -> Dict:
        """
        Count occurrences of directional terms in the dataset.
        
        Args:
            dataset: Dataset to analyze
            directions: List of directional terms to count
            field: Field to search for terms
            splits: List of splits to analyze (default: all splits)
            
        Returns:
            Dictionary with counts for each direction by split
        """
        results = {}
        
        if isinstance(dataset, DatasetDict):
            used_splits = splits if splits is not None else dataset.keys()
            
            for split_name in used_splits:
                if split_name in dataset:
                    results[split_name] = self._count_in_split(dataset[split_name], directions, field)
        else:
            # Single dataset case
            results["dataset"] = self._count_in_split(dataset, directions, field)
        
        return results
    
    def _count_in_split(self, split: Dataset, directions: List[str], field: str) -> Dict[str, int]:
        """Count examples containing each direction at least once in a split."""
        texts = split[field]  # Extract column directly
        
        # Count examples that contain each direction at least once
        counts = {direction: sum(1 for text in texts if direction in text) for direction in directions}
        return counts

    def modification_overview(self, original_dataset: Union[Dataset, DatasetDict], 
                         modified_dataset: Union[Dataset, DatasetDict],
                         source_dir: str = "left",
                         target_dir: str = "right",
                         field: str = "Program") -> pd.DataFrame:
        """
        Generate a summary of modifications showing counts of directional terms before and after.
        
        Args:
            original_dataset: The original dataset before modification
            modified_dataset: The modified dataset after replacement
            source_dir: Source direction that was replaced (default: "left")
            target_dir: Target direction that was replaced with (default: "right")
            field: Field that was modified (default: "Program")
            
        Returns:
            DataFrame showing counts of each direction in each split before and after modification
        """
        # Count directions in both datasets
        original_counts = self.count_directions(
            original_dataset, 
            directions=[source_dir, target_dir], 
            field=field
        )
        modified_counts = self.count_directions(
            modified_dataset, 
            directions=[source_dir, target_dir], 
            field=field
        )
        
        # Create a summary DataFrame
        data = []
        
        if isinstance(original_dataset, DatasetDict):
            for split_name in original_dataset.keys():
                orig_source = original_counts[split_name][source_dir]
                orig_target = original_counts[split_name][target_dir]
                mod_source = modified_counts[split_name][source_dir]
                mod_target = modified_counts[split_name][target_dir]
                
                data.append([split_name, orig_source, orig_target, mod_source, mod_target])
        else:
            # Single dataset case
            orig_source = original_counts["dataset"][source_dir]
            orig_target = original_counts["dataset"][target_dir]
            mod_source = modified_counts["dataset"][source_dir]
            mod_target = modified_counts["dataset"][target_dir]
            
            data.append(["dataset", orig_source, orig_target, mod_source, mod_target])
        
        # Create and return the DataFrame
        column_names = [
            "Split", 
            f"Orig {source_dir}", 
            f"Orig {target_dir}", 
            f"Mod {source_dir}", 
            f"Mod {target_dir}"
        ]
        
        return pd.DataFrame(data, columns=column_names)