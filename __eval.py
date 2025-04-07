'''
This file contains the evaluation functions. 
'''
import json
import tempfile
import subprocess
import os
import re
import numpy as np
import gc
from PIL import Image
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
from crystalbleu import corpus_bleu
from synthetic_data.transform_data_to_forkstate_custom import transform_program
try: 
    import dreamsim
    import torch
    from torchvision.transforms import ToTensor
except ImportError:
    print("dreamsim package not found. Please install it using 'pip install dreamsim'")
            

class LLMCodeEvaluator:
    """
    Class for evaluating code completions from LLMs.
    Handles syntax validation, similarity metrics, execution testing, and image comparison.
    """
    
    def __init__(self, repo_root=None):
        """
        Initialize the evaluator.
        
        Args:
            repo_root (str, optional): Repository root path. If None, uses current directory.
        """
        import sys  # Add this import at the beginning of the method
        
        self.repo_root = repo_root or os.getcwd()
        # Configure dependencies path
        self.dependencies_path = os.path.join(self.repo_root, 'external/dependencies')
        if self.dependencies_path not in sys.path:
            sys.path.append(self.dependencies_path)
        
        # Check for GPU availability - making sure it uses the GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(f"Using device: {self.device}")

        # Initialize DreamSim model
        self.dsim_model, self.dsim_preprocess = dreamsim.dreamsim(pretrained=True, device=self.device)

    def load_predictions(self, inf_dir):
        """
        Load predictions from JSON file.
        
        Args:
            inf_dir (str): Directory containing predictions.json
            
        Returns:
            list: Loaded predictions
        """
        with open(os.path.join(inf_dir, "predictions.json"), "r", encoding='utf-8') as f:
            predictions = json.load(f)
        return predictions
    
    def evaluate_completions(self, predictions, n_completions=1, fork_state=False, inf_dir=None):
        """
        Evaluate a set of code completions and store metrics incrementally to a JSONL file.

        Args:
            predictions (list): List of prediction dictionaries with completions and ground truths.
            n_completions (int): Number of completions per prediction.
            fork_state (bool): Whether to transform programs to fork_state format.
            inf_dir (str): Directory to save detailed_metrics.jsonl incrementally.

        Returns:
            None
        """
        print(f"[DEBUG] evaluate_completions() called with n_completions={n_completions}, fork_state={fork_state}")

        detailed_metrics_path = os.path.join(inf_dir, "detailed_metrics.jsonl")
        checkpoint_path = os.path.join(inf_dir, "checkpoint.txt")
        batch_size = 5  # Number of examples to process before clearing memory

        # Load the last processed index from the checkpoint
        start_index = self.load_checkpoint(checkpoint_path)
        print(f"[DEBUG] Resuming evaluation from index {start_index}")

        for i, prediction in enumerate(predictions[start_index:], start=start_index):
            example_id = prediction["id"]
            ground_truth = prediction["ground_truth"]
            if fork_state:
                ground_truth = transform_program(ground_truth, embed_to_fork=False, fork_to_embed=True)

            # Iterate over all completions (completion_1, completion_2, ..., completion_n)
            print(f"[DEBUG] Evaluating example_id={example_id} with available completions: {list(prediction.keys())}")

            for idx in range(1, n_completions + 1):
                completion_key = f"completion_{idx}"
                if completion_key in prediction:
                    completion = prediction[completion_key]
                    print(f"[DEBUG] Evaluating {completion_key} for example_id={example_id}")

                    if fork_state:
                        completion = transform_program(completion, embed_to_fork=False, fork_to_embed=True)
                    completion_clean = self.clean_python_code(completion)

                    # Evaluate the completion (your existing logic here)
                    is_valid_syntax, format_message, details_dict = self.check_formatting(completion_clean)
                    is_executable, execution_message, image = self.code_execution_pyturtle(completion_clean)

                    # Image comparison for executed code
                    ssim_score = np.nan
                    pixel_similarity = np.nan
                    dreamsim_score = np.nan
                    pixel_precision = np.nan
                    pixel_recall = np.nan
                    pixel_f1 = np.nan
                    if is_executable:
                        gt_executable, _, gt_image = self.code_execution_pyturtle(ground_truth)
                        if gt_executable:
                            image_comparison = self.compare_images(image, gt_image)
                            ssim_score = image_comparison.get('ssim_score', np.nan)
                            pixel_similarity = image_comparison.get('pixel_similarity', np.nan)
                            dreamsim_score = image_comparison.get('dreamsim_score', np.nan)
                            pixel_precision = image_comparison.get('pixel_precision', np.nan)
                            pixel_recall = image_comparison.get('pixel_recall', np.nan)
                            pixel_f1 = image_comparison.get('pixel_f1', np.nan)

                    # Compile all metrics
                    result = {
                        "id": f"{example_id}_{idx}",  # Append _n to the ID
                        "syntactically_valid": is_valid_syntax,
                        "outer_valid": details_dict["outer_valid"],
                        "any_embed_call": details_dict.get("embed_usage", {}).get("any_embed_call", False),
                        "correctly_formed_embed": details_dict.get("embed_usage", {}).get("correctly_formed", False),
                        "alternative_embed_patterns": details_dict.get("embed_usage", {}).get("alternative_patterns", []),
                        "embed_blocks_count": len(details_dict.get("embed_blocks", [])),
                        "embed_blocks_all_valid": all(block.get("valid", False) for block in details_dict.get("embed_blocks", [])),
                        "format_message": format_message,
                        "executable": is_executable,
                        "execution_message": execution_message,
                        "ssim_score": ssim_score,
                        "pixel_similarity": pixel_similarity,
                        "dreamsim_score": dreamsim_score,
                        "pixel_precision": pixel_precision,
                        "pixel_recall": pixel_recall,
                        "pixel_f1": pixel_f1,
                        **self.check_lev_similarity(completion_clean, ground_truth),
                        **self.check_basic_similarity(completion_clean, ground_truth),
                        **self.check_crystalbleu_similarity(completion_clean, ground_truth),
                    }

                    # Write the result to the JSONL file
                    with open(detailed_metrics_path, "a") as f:
                        f.write(json.dumps(result) + "\n")

            # Save progress to the checkpoint file after processing each batch
            if (i + 1) % batch_size == 0:
                self.save_checkpoint(checkpoint_path, i + 1)
                torch.cuda.empty_cache()
                gc.collect()

        # Save the final checkpoint
        self.save_checkpoint(checkpoint_path, len(predictions))
        print(f"[DEBUG] Evaluation completed and metrics saved to {detailed_metrics_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path):
        """
        Load the last processed index from the checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            int: Last processed index.
        """
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                return int(f.read().strip())
        return 0

    @staticmethod
    def save_checkpoint(checkpoint_path, index):
        """
        Save the index of the last processed entry to a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            index (int): Index of the last processed entry.
        """
        with open(checkpoint_path, "w") as f:
            f.write(str(index))

    def generate_summary(self, metrics):
        summary = {
            "total_samples": len(metrics),
            "valid_code": {
                "syntactically_valid": sum(1 for m in metrics if m["syntactically_valid"]),
                "outer_valid": sum(1 for m in metrics if m["outer_valid"])
            },
            "embed_usage": {
                "any_embed_call": sum(1 for m in metrics if m.get("any_embed_call", False)),
                "correctly_formed": sum(1 for m in metrics if m.get("correctly_formed_embed", False)),
                "alternative_patterns": sum(1 for m in metrics if len(m.get("alternative_embed_patterns", [])) > 0)
            },
            "similarity": {
                "exact_matches": sum(1 for m in metrics if m["exact_match"]),
                "avg_normalized_lev_distance": sum(m["normalized_lev_distance"] for m in metrics) / len(metrics),
                "avg_line_similarity": sum(m["line_similarity"] for m in metrics) / len(metrics),
                "avg_crystalbleu_score": sum(m["crystalbleu_score"] for m in metrics if m["crystalbleu_score"] is not None) / len(metrics)
            },
            "execution": {
                "executable_count": sum(1 for m in metrics if m["executable"])
            }
        }

        # Add AST similarity data
        ast_metrics = [m for m in metrics if "normalized_ast_similarity" in m]
        if ast_metrics:
            summary["similarity"]["avg_ast_similarity"] = sum(m["normalized_ast_similarity"] for m in ast_metrics) / len(ast_metrics)
            summary["similarity"]["ast_available_count"] = len(ast_metrics)

        # Add image comparison data
        executable_metrics = [m for m in metrics if m["executable"]]

        if executable_metrics:
            # SSIM
            ssim_scores = [m["ssim_score"] for m in executable_metrics if m.get("ssim_score") is not None]
            summary["execution"]["avg_ssim"] = np.nanmean(ssim_scores) if ssim_scores else np.nan
            summary["execution"]["ssim_available_count"] = len(ssim_scores)
            summary["execution"]["perfect_ssim_count"] = sum(1 for m in executable_metrics if m.get("ssim_score") == 1.0)

            # Pixel Similarity
            pixel_similarities = [m["pixel_similarity"] for m in executable_metrics if m.get("pixel_similarity") is not None]
            summary["execution"]["avg_pixel_similarity"] = np.nanmean(pixel_similarities) if pixel_similarities else np.nan
            summary["execution"]["pixel_similarity_available_count"] = len(pixel_similarities)
            summary["execution"]["perfect_pixel_count"] = sum(1 for m in executable_metrics if m.get("pixel_similarity") == 1.0)

            # DreamSim
            dreamsim_scores = [m["dreamsim_score"] for m in executable_metrics if m.get("dreamsim_score") is not None]
            summary["execution"]["avg_dreamsim"] = np.nanmean(dreamsim_scores) if dreamsim_scores else np.nan
            summary["execution"]["dreamsim_available_count"] = len(dreamsim_scores)
            summary["execution"]["zero_dreamsim_count"] = sum(1 for m in executable_metrics if m.get("dreamsim_score") == 0.0)

        # Add metric for perfect agreement across all three metrics
        executable_with_all_metrics = [m for m in executable_metrics if 
                                       m.get("ssim_score") is not None and 
                                       m.get("pixel_similarity") is not None and 
                                       m.get("dreamsim_score") is not None]
        if executable_with_all_metrics:
            perfect_agreement_count = sum(1 for m in executable_with_all_metrics 
                                          if m["ssim_score"] == 1.0 and 
                                          m["pixel_similarity"] == 1.0 and 
                                          m["dreamsim_score"] == 0.0)
            summary["execution"]["perfect_agreement_count"] = perfect_agreement_count
            summary["execution"]["all_metrics_available_count"] = len(executable_with_all_metrics)

        # Precision-Recall
        executable_with_precision_recall = [m for m in executable_metrics if "pixel_precision" in m]
        if executable_with_precision_recall:
            pixel_precisions = [m["pixel_precision"] for m in executable_with_precision_recall]
            pixel_recalls = [m["pixel_recall"] for m in executable_with_precision_recall]
            pixel_f1s = [m["pixel_f1"] for m in executable_with_precision_recall]

            summary["execution"]["avg_pixel_precision"] = np.nanmean(pixel_precisions)
            summary["execution"]["avg_pixel_recall"] = np.nanmean(pixel_recalls)
            summary["execution"]["avg_pixel_f1"] = np.nanmean(pixel_f1s)
            summary["execution"]["precision_recall_available_count"] = len(executable_with_precision_recall)
        else:
            summary["execution"]["avg_pixel_precision"] = np.nan
            summary["execution"]["avg_pixel_recall"] = np.nan
            summary["execution"]["avg_pixel_f1"] = np.nan
            summary["execution"]["precision_recall_available_count"] = 0

        return summary
    
    def print_summary(self, summary):
        """
        Print a detailed summary of metrics to the console.

        Args:
            summary (dict): Summary dictionary generated by `generate_summary`.
        """
        print("\n--- SUMMARY ---")

        # Total samples
        print(f"Total samples: {summary['total_samples']}")

        # Valid code
        valid_code = summary["valid_code"]
        print("\n--- Valid Code ---")
        print(f"Syntactically valid: {valid_code['syntactically_valid']}")
        print(f"Outer valid: {valid_code['outer_valid']}")

        # Embed usage
        embed_usage = summary["embed_usage"]
        print("\n--- Embed Usage ---")
        print(f"Programs with any embed() call: {embed_usage['any_embed_call']}")
        print(f"Programs with correctly formed embed(): {embed_usage['correctly_formed']}")
        print(f"Programs with alternative embed() patterns: {embed_usage['alternative_patterns']}")

        # Similarity metrics
        similarity = summary["similarity"]
        print("\n--- Similarity Metrics ---")
        print(f"Exact matches: {similarity['exact_matches']}")
        print(f"Average normalized Levenshtein distance: {similarity['avg_normalized_lev_distance']:.4f}")
        print(f"Average line similarity: {similarity['avg_line_similarity']:.4f}")
        print(f"Average CrystalBLEU score: {similarity['avg_crystalbleu_score']:.4f}")
        if "avg_ast_similarity" in similarity:
            print(f"Average AST-based similarity: {similarity['avg_ast_similarity']:.4f} from {similarity['ast_available_count']} samples")


        # Execution metrics
        execution = summary["execution"]
        print("\n--- Execution Results ---")
        print(f"Executable code samples: {execution['executable_count']}")
        if "avg_ssim" in execution:
            print(f"Average SSIM: {execution['avg_ssim']:.4f} (from {execution['ssim_available_count']} samples)")
            print(f"Perfect SSIM count: {execution['perfect_ssim_count']}")
        if "avg_pixel_similarity" in execution:
            print(f"Average pixel similarity: {execution['avg_pixel_similarity']:.4f} (from {execution['pixel_similarity_available_count']} samples)")
            print(f"Perfect pixel similarity count: {execution['perfect_pixel_count']}")
        if "avg_dreamsim" in execution:
            print(f"Average DreamSim similarity: {execution['avg_dreamsim']:.4f} (from {execution['dreamsim_available_count']} samples)")
            print(f"Zero DreamSim count: {execution['zero_dreamsim_count']}")
        if "perfect_agreement_count" in execution:
            print(f"Perfect agreement count: {execution['perfect_agreement_count']} (SSIM, Pixel Sim, and DreamSIM available for {execution['all_metrics_available_count']} samples)")
        if "avg_pixel_precision" in execution:
            print(f"Average pixel precision: {execution['avg_pixel_precision']:.4f}")
            print(f"Average pixel recall: {execution['avg_pixel_recall']:.4f}")
            print(f"Average pixel F1 score: {execution['avg_pixel_f1']:.4f}")
            print(f"Precision-recall metrics available for {execution['precision_recall_available_count']} samples")

    def evaluate_and_summarize(self, inf_dir, n_completions=1, fork_state=False):
        """
        Complete evaluation pipeline: load predictions, evaluate, and summarize.

        Args:
            inf_dir (str): Directory containing predictions.json
            n_completions (int): Number of completions per prediction.
            fork_state (bool): Whether to transform programs to fork_state format.

        Returns:
            tuple: (metrics, summary)
        """
        print(f"[DEBUG] evaluate_and_summarize() called with n_completions={n_completions}, fork_state={fork_state}")

        # Step 1: Load predictions
        predictions = self.load_predictions(inf_dir)
        print(f"[DEBUG] Loaded {len(predictions)} predictions from {inf_dir}")

        # Step 2: Evaluate completions and save metrics incrementally
        self.evaluate_completions(predictions, n_completions=n_completions, fork_state=fork_state, inf_dir=inf_dir)
        print(f"[DEBUG] Evaluation completed and metrics saved incrementally to detailed_metrics.jsonl")

        # Step 3: Load metrics from detailed_metrics.jsonl
        detailed_metrics_path = os.path.join(inf_dir, "detailed_metrics.jsonl")
        with open(detailed_metrics_path, "r") as f:
            metrics = [json.loads(line) for line in f]
        print(f"[DEBUG] Loaded {len(metrics)} metrics from {detailed_metrics_path}")

        # Step 4: Generate summary
        summary = self.generate_summary(metrics)
        self.print_summary(summary)

        # Step 5: Save the evaluation summary to evaluation.json
        evaluation_path = os.path.join(inf_dir, "evaluation.json")
        with open(evaluation_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[DEBUG] Saved evaluation summary to {evaluation_path}")

        return metrics, summary
    
    @staticmethod
    def clean_python_code(code):
        """
        Remove comments and empty lines from Python code.
        
        Args:
            code (str): Python code as string
            
        Returns:
            str: Cleaned code
        """
        cleaned_lines = []
        for line in code.split('\n'):
            # Remove comments
            if '#' in line:
                line = line.split('#', 1)[0]
            # Only add non-empty lines
            if line.strip():
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def check_formatting(completion):
        """
        Check if the completion is syntactically complete, including code inside embed() function
        Also detects different patterns of embed() usage
        Returns (is_valid, message, details_dict)
        """
        import re
        import ast
        
        if not completion:
            return False, "Empty code", {"outer_valid": False, "embed_valid": False}
        
        # Track validity of outer code and embed code
        details = {
            "outer_valid": False,
            "embed_blocks": [],
            "all_valid": False,
            "embed_usage": {
                "any_embed_call": False,  # Any use of embed() function
                "correctly_formed": False,  # Specifically embed("""...""", locals())
                "alternative_patterns": []  # Other patterns used with embed()
            }
        }
        
        # Initialize both counters
        details["valid_embed_blocks_count"] = 0
        details["total_embed_blocks_count"] = 0
        
        # First check the entire code's syntax
        try:
            ast.parse(completion)
            details["outer_valid"] = True
        except SyntaxError as e:
            return False, f"Syntax error in outer code: {str(e)}", details
        except Exception as e:
            return False, f"Other error in outer code: {str(e)}", details
        
        # Check for any usage of embed function (basic pattern detection)
        if "embed(" in completion:
            details["embed_usage"]["any_embed_call"] = True
            details["total_embed_blocks_count"] = len(re.findall(r'embed\(', completion))
            
            # Find correctly formed embed calls
            correct_embed_pattern = r'embed\(\s*("""|\'\'\')(.*?)("""|\'\'\')\s*,\s*locals\(\)\s*\)'
            correct_embed_matches = re.findall(correct_embed_pattern, completion, re.DOTALL)
            if correct_embed_matches:
                details["embed_usage"]["correctly_formed"] = True
                details["valid_embed_blocks_count"] = len(correct_embed_matches)
                
                # Check each embedded code block in standard format
                all_embed_valid = True
                for i, match in enumerate(correct_embed_matches):
                    embed_code = match[1]  # The code inside triple quotes
                    
                    # Validate this embedded code block
                    try:
                        ast.parse(embed_code)
                        embed_status = {"valid": True, "message": "Valid"}
                    except SyntaxError as e:
                        embed_status = {"valid": False, "message": f"Syntax error: {str(e)}"}
                        all_embed_valid = False
                    except Exception as e:
                        embed_status = {"valid": False, "message": f"Other error: {str(e)}"}
                        all_embed_valid = False
                    
                    # Store results for this embed block
                    embed_status["code"] = embed_code
                    details["embed_blocks"].append(embed_status)
            
            # Now look for alternative embed patterns
            # This captures the content of all embed() calls
            alternative_pattern = r'(embed\(\s*.*?\s*\))'
            
            try:
                all_embed_calls = re.findall(alternative_pattern, completion, re.DOTALL)
                
                # Filter out correctly formed patterns to find alternative ones
                for full_call in all_embed_calls:
                    # Check if this is NOT a correctly formed embed call
                    if not re.match(r'embed\(\s*("""|\'\'\').*?("""|\'\'\')\s*,\s*locals\(\)\s*\)', full_call, re.DOTALL):
                        # This is an alternative pattern
                        # Format as needed
                        formatted_pattern = ""
                        if len(full_call) > 30:  # If pattern is long, truncate it
                            start_content = full_call[:15].strip()  # First 15 chars including "embed("
                            end_content = full_call[-15:].strip()   # Last 15 chars
                            formatted_pattern = f"{start_content}...{end_content}"  # Without extra quotes
                        else:
                            formatted_pattern = full_call  # Without extra quotes
                        
                        details["embed_usage"]["alternative_patterns"].append(formatted_pattern)
            except Exception:
                # If regex fails, just ignore
                pass

        # Determine overall validity
        details["all_valid"] = details["outer_valid"] and all(block.get("valid", True) for block in details["embed_blocks"])
        
        # Prepare return message
        if details["all_valid"]:
            message = "All code is syntactically valid"
        elif details["outer_valid"]:
            if details["embed_blocks"]:
                message = "Outer code valid but embed code has errors"
            else:
                message = "Outer code valid, alternative embed patterns found"
        else:
            message = "Outer code has syntax errors"
        
        return details["all_valid"], message, details

    @staticmethod
    def check_basic_similarity(completion, ground_truth):
        """
        Calculate basic similarity metrics between completion and ground truth.
        
        Args:
            completion (str): Completion code
            ground_truth (str): Ground truth code
            
        Returns:
            dict: Basic similarity metrics
        """
        # Check for exact match
        exact_match = completion.strip() == ground_truth.strip()
        
        # Line-by-line analysis
        completion_lines = completion.strip().split('\n')
        ground_truth_lines = ground_truth.strip().split('\n')
        
        # Count of matching lines
        matching_lines = sum(1 for cl, gl in zip(completion_lines, ground_truth_lines) if cl == gl)
        
        # Line-level similarity
        total_lines = max(len(completion_lines), len(ground_truth_lines))
        line_similarity = matching_lines / total_lines if total_lines > 0 else 1.0
        
        return {
            "exact_match": exact_match,
            "matching_lines": matching_lines,
            "total_lines": total_lines,
            "line_similarity": line_similarity
        }

    @staticmethod
    def check_lev_similarity(completion, ground_truth):
        """
        Calculate Levenshtein-based similarity metrics.
        
        Args:
            completion (str): Completion code
            ground_truth (str): Ground truth code
            
        Returns:
            dict: Levenshtein similarity metrics
        """
        from Levenshtein import distance as levenshtein_distance
        
        # Basic Levenshtein distance
        lev_distance = levenshtein_distance(completion, ground_truth)
        
        # Normalized Levenshtein similarity (0 to 1 where 1 is identical)
        max_len = max(len(completion), len(ground_truth))
        if max_len == 0:
            normalized_lev_distance = 1.0  # Both strings are empty
        else:
            normalized_lev_distance = 1.0 - (lev_distance / max_len)
        
        return {
            "levenshtein_distance": lev_distance,
            "normalized_lev_distance": normalized_lev_distance
        }

    @staticmethod
    def check_ast_similarity(completion, ground_truth):
        """
        Calculate similarity between two code snippets based on their AST structure.
        
        Args:
            completion (str): Completion code
            ground_truth (str): Ground truth code
            
        Returns:
            dict: AST similarity metrics
        """
        import ast
        import zss  # Requires package: pip install zss

        # Helper functions for zss
        def get_children(node):
            return [child for child in ast.iter_child_nodes(node)]
        
        def get_label(node):
            return type(node).__name__
        
        # Function to convert AST to a format suitable for tree edit distance
        def convert_ast_to_zss_tree(node):
            if node is None:
                return None
            return zss.Node(
                get_label(node),
                [convert_ast_to_zss_tree(child) for child in get_children(node)]
            )
        
        try:
            # Parse both code snippets into AST
            completion_ast = ast.parse(completion)
            ground_truth_ast = ast.parse(ground_truth)
            
            # Convert ASTs to zss trees
            completion_tree = convert_ast_to_zss_tree(completion_ast)
            ground_truth_tree = convert_ast_to_zss_tree(ground_truth_ast)
            
            # Calculate tree edit distance
            ast_distance = zss.simple_distance(completion_tree, ground_truth_tree)
            
            # Calculate node counts for normalization
            completion_node_count = sum(1 for _ in ast.walk(completion_ast))
            ground_truth_node_count = sum(1 for _ in ast.walk(ground_truth_ast))
            max_nodes = max(completion_node_count, ground_truth_node_count)
            
            # Normalize the distance (0 to 1, where 1 means identical)
            if max_nodes == 0:
                normalized_ast_similarity = 1.0  # Both ASTs empty
            else:
                normalized_ast_similarity = 1.0 - (ast_distance / max_nodes)
            
            return {
                "ast_distance": ast_distance,
                "normalized_ast_similarity": normalized_ast_similarity,
                "completion_node_count": completion_node_count,
                "ground_truth_node_count": ground_truth_node_count
            }
            
        except SyntaxError:
            # If code has syntax errors, can't compare ASTs
            return {
                "ast_distance": -1,
                "normalized_ast_similarity": 0.0,
                "completion_node_count": -1,
                "ground_truth_node_count": -1
            }  

    def check_crystalbleu_similarity(self, completion, ground_truth):
        """
        Calculate CrystalBLEU similarity between completion and ground truth code.
        
        Args:
            completion (str): Completion code
            ground_truth (str): Ground truth code
            
        Returns:
            dict: CrystalBLEU similarity metric
        """
        try:            
            # Calculate CrystalBLEU score 
            bleu_score = corpus_bleu([[ground_truth]], [completion])
            
            return {
                "crystalbleu_score": bleu_score,
            }
        except ImportError as e:
            print(f"Import error in check_crystalbleu_similarity: {e}")
            return {
                "crystalbleu_score": None,
                "crystalbleu_error": "crystalbleu package not installed"
            }
        except Exception as e:
            print(f"Exception in check_crystalbleu_similarity: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "crystalbleu_score": 0.0,
                "crystalbleu_error": str(e)
            }

    def code_execution_pyturtle(self, program):
        """
        Execute a single program and check if it runs correctly.
        
        Args:
            program (str): Python code to execute
            
        Returns:
            tuple: (is_executable, message, image)
        """
        import sys
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image as PILImage
        
        # Get current directory
        current_dir = os.getcwd()
        
        # Add dependencies path to sys.path
        dependencies_path = os.path.join(current_dir, 'external/dependencies')
        if dependencies_path not in sys.path:
            sys.path.append(dependencies_path)
        
        turtle = None
        try:
            from program_refactoring.domains.logos.pyturtle_pc import PyTurtle, HALF_INF, EPS_DIST, EPS_ANGLE
            
            # Create PyTurtle instance
            turtle = PyTurtle()
            
            # Create execution scope with all necessary variables
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
            
            # Execute the code
            exec(program, exec_scope)
            
            # Get the image - UPDATED TO USE buffer_rgba()
            turtle.fig.canvas.draw()
            width, height = turtle.fig.canvas.get_width_height()
            
            # Get RGBA buffer and convert to PIL Image
            buffer = turtle.fig.canvas.buffer_rgba()
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
            
            # Convert RGBA to RGB for consistency with previous code
            image = PILImage.fromarray(image_array[:, :, :3])
            
            plt.close(turtle.fig)  # Close figure to free memory
            
            return True, "Program executed successfully", image
            
        except ImportError as e:
            if turtle and hasattr(turtle, 'fig'):
                plt.close(turtle.fig)
            return False, f"Import error: {str(e)}", None
        except Exception as e:
            if turtle and hasattr(turtle, 'fig'):
                plt.close(turtle.fig)
            return False, f"Execution error: {str(e)}", None

    def compare_images(self, image_pred, image_gr):
        """
        Compare two images using SSIM, pixel-wise comparison, and precision/recall.
        
        Args:
            image_pred (PIL.Image): Predicted image
            image_gr (PIL.Image): Ground truth image
            
        Returns:
            dict: Similarity metrics
        """
        # Calculate SSIM
        ssim_score = None
        try:
            ssim_score = self._calculate_ssim(image_pred, image_gr)
        except ImportError:
            pass  # SSIM couldn't be calculated
            
        # Calculate pixel similarity
        pixel_similarity = None
        try:
            pixel_similarity = self._calculate_pixel_similarity(image_pred, image_gr)
        except ImportError:
            pass  # Pixel similarity couldn't be calculated

        # Calculate precision and recall for black pixels
        pixel_precision_recall = {}
        try:
            pixel_precision_recall = self._calculate_pixel_precision_recall(image_pred, image_gr)
        except Exception as e:
            print(f"Error calculating pixel precision/recall: {e}")
        
        # Calculate DreamSim
        dreamsim_score = None
        try:
            dreamsim_score = self._calculate_dreamsim(image_pred, image_gr)
        except Exception:
            pass  # DreamSim couldn't be calculated
        
        # Create the result dictionary
        result = {
            "ssim_score": ssim_score,
            "pixel_similarity": pixel_similarity,
            "dreamsim_score": dreamsim_score,
        }
        
        # Add precision/recall metrics if available
        if pixel_precision_recall:
            result.update(pixel_precision_recall)
        
        return result
    
    @staticmethod
    def _calculate_ssim(image1, image2):
        """
        Calculate Structural Similarity Index Measure between two images.
        
        Args:
            image1 (PIL.Image): First image
            image2 (PIL.Image): Second image
            
        Returns:
            float: SSIM score
        """
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        
        # Convert PIL images to numpy arrays
        img1_np = np.array(image1.convert('L'))  # Convert to grayscale
        img2_np = np.array(image2.convert('L'))
            
        # Resize if dimensions don't match
        if img1_np.shape != img2_np.shape:
            from PIL import Image
            # Resize the second image to match the first
            image2_resized = image2.resize(image1.size, Image.LANCZOS)
            img2_np = np.array(image2_resized.convert('L'))
            
        # Calculate SSIM
        ssim_score = ssim(img1_np, img2_np)
        return ssim_score
    
    @staticmethod
    def _calculate_pixel_similarity(image1, image2):
        """
        Calculate pixel-wise similarity between two images.
        
        Args:
            image1 (PIL.Image): First image
            image2 (PIL.Image): Second image
            
        Returns:
            float: Similarity score (0-1)
        """
        import numpy as np
        from PIL import ImageChops, ImageStat, Image

        # Resize if dimensions don't match
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.LANCZOS)
            
        # Calculate difference
        diff = ImageChops.difference(image1.convert('RGB'), image2.convert('RGB'))
        stat = ImageStat.Stat(diff)
        diff_ratio = sum(stat.mean) / (255 * 3)  # Average difference across RGB channels
            
        # Convert to similarity (0 = completely different, 1 = identical)
        pixel_similarity = 1.0 - diff_ratio
        return pixel_similarity
    
    def _calculate_dreamsim(self, image1, image2):
        """
        Calculate DreamSim similarity between two images.
        
        Args:
            image1 (PIL.Image): First image
            image2 (PIL.Image): Second image
            
        Returns:
            float: DreamSim similarity score
        """
        try:
            # Resize image2 if dimensions don't match
            if image1.size != image2.size:
                from PIL import Image
                image2 = image2.resize(image1.size, Image.LANCZOS)

            # Preprocess images - move to the correct device
            tensor1 = self.dsim_preprocess(image1).to(self.device)
            tensor2 = self.dsim_preprocess(image2).to(self.device)

            # Compute DreamSim score
            with torch.no_grad():
                # Get all return values but only use the first one
                # This handles the "too many values to unpack" error
                result = self.dsim_model(tensor1, tensor2)
                dreamsim_score = result.item()
                
                return dreamsim_score

        except Exception as e:
            print(f"Error calculating DreamSim: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_pixel_precision_recall(self, image_pred, image_gr):
        """
        Calculate precision and recall for black pixels between two images.
        
        Args:
            image_pred (PIL.Image): Predicted image
            image_gr (PIL.Image): Ground truth image
            
        Returns:
            dict: Precision and recall metrics
        """
        import numpy as np
        from PIL import Image
        
        # Resize if dimensions don't match
        if image_pred.size != image_gr.size:
            image_gr = image_gr.resize(image_pred.size, Image.LANCZOS)
        
        # Convert to grayscale
        img_pred_gs = np.array(image_pred.convert('L'))  # Predicted image as grayscale
        img_gr_gs = np.array(image_gr.convert('L'))      # Ground truth image as grayscale
        
        # Apply threshold to convert to binary (black/white)
        threshold = 200  # Adjust threshold if needed
        img_pred_bw = img_pred_gs < threshold
        img_gr_bw = img_gr_gs < threshold
        # Count black and white pixels
        pred_black_pixels = np.sum(img_pred_bw)
        pred_white_pixels = img_pred_bw.size - pred_black_pixels
        gr_black_pixels = np.sum(img_gr_bw)
        gr_white_pixels = img_gr_bw.size - gr_black_pixels
        
        print(f"Predicted image: {pred_black_pixels} black pixels, {pred_white_pixels} white pixels")
        print(f"Ground truth image: {gr_black_pixels} black pixels, {gr_white_pixels} white pixels")
        
        # Calculate true positives, false positives, false negatives
        true_positives = np.sum(img_pred_bw & img_gr_bw)
        false_positives = np.sum(img_pred_bw & ~img_gr_bw)
        false_negatives = np.sum(~img_pred_bw & img_gr_bw)
        
        print(f"True positives (black pixels correctly predicted): {true_positives}")
        print(f"False positives (black pixels incorrectly predicted): {false_positives}")
        print(f"False negatives (missed black pixels): {false_negatives}")
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"F1 Score: {f1_score}")
        
        return {
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_f1": f1_score
        }

    @staticmethod
    def clean_result_path(result_path):
        """
        Clean up result path by standardizing format.
        
        Args:
            result_path (str): Original result path
            
        Returns:
            str: Cleaned result path
        """
        # Split the path into directory and filename
        result_dir, model_id = os.path.split(result_path)

        # Remove "length_" prefix if it exists
        if model_id.startswith("length_"):
            model_id = model_id[len("length_"):]
        
        # Find all timestamps (sequences of digits)
        timestamps = re.findall(r'\d{8}_\d{4}', model_id)

        # If there are two timestamps, remove the last one
        if len(timestamps) > 1:
            model_id = model_id.rsplit("_", 2)[0]  # Remove last timestamp
        
        # Reconstruct the full path
        cleaned_path = os.path.join(result_dir, model_id)
        return cleaned_path

# Simple usage example
if __name__ == "__main__":
    import sys
    
    evaluator = LLMCodeEvaluator()
    
    if len(sys.argv) > 1:
        inf_dir = sys.argv[1]
    else:
        inf_dir = "results/length/CodeLlama_20250311_1601/inference"
    
    metrics, summary = evaluator.evaluate_and_summarize(inf_dir)
    
    # Save results if needed
    # with open(os.path.join(inf_dir, "evaluation.json"), "w") as f:
    #    json.dump(summary, f, indent=2)