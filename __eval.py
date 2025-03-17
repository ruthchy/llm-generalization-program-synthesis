'''
This file contains the evaluation functions. 
'''
import json
import tempfile
import subprocess
import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance

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
    
    def evaluate_completions(self, predictions):
        """
        Evaluate a set of code completions.
        
        Args:
            predictions (list): List of prediction dictionaries with completions and ground truths
            
        Returns:
            list: Metrics for each completion
        """
        metrics = []
        for idx, prediction in enumerate(predictions):
            completion = prediction["completion"]
            ground_truth = prediction["ground_truth"]

            completion_clean = self.clean_python_code(completion)
            
            # Check syntax
            is_valid_syntax, format_message, details_dict = self.check_formatting(completion_clean)
            
            # Check execution
            is_executable, execution_message, image = self.code_execution_pyturtle(completion_clean)
            
            # Image comparison for executed code
            ssim_score = None
            pixel_similarity = None
            if is_executable:
                gt_executable, _, gt_image = self.code_execution_pyturtle(ground_truth)
                if gt_executable:
                    image_comparison = self.compare_images(image, gt_image) 
                    ssim_score = image_comparison.get('ssim_score')
                    pixel_similarity = image_comparison.get('pixel_similarity')
            
            # Extract embed usage
            embed_usage = details_dict.get("embed_usage", {})
            
            # Compile all metrics
            result = {
                "id": idx,
                "syntactically_valid": is_valid_syntax,
                "outer_valid": details_dict["outer_valid"],
                "any_embed_call": embed_usage.get("any_embed_call", False),
                "correctly_formed_embed": embed_usage.get("correctly_formed", False),
                "alternative_embed_patterns": embed_usage.get("alternative_patterns", []),
                "embed_blocks_count": len(details_dict.get("embed_blocks", [])),
                "embed_blocks_all_valid": all(block.get("valid", False) for block in details_dict.get("embed_blocks", [])),
                "format_message": format_message,
                "executable": is_executable,
                "execution_message": execution_message,
                "ssim_score": ssim_score,
                "pixel_similarity": pixel_similarity,
                **self.check_lev_similarity(completion_clean, ground_truth),
                **self.check_basic_similarity(completion_clean, ground_truth),
            }
            
            # Add AST similarity if both codes are syntactically valid
            try:
                import ast
                try:
                    ast.parse(completion_clean)
                    ast.parse(ground_truth)
                    result.update(self.check_ast_similarity(completion_clean, ground_truth))
                except SyntaxError:
                    result["ast_error"] = "Syntax error prevents AST comparison"
            except ImportError:
                result["ast_error"] = "AST comparison requires zss package"
            
            metrics.append(result)
        
        return metrics
    
    def generate_summary(self, metrics):
        """
        Generate summary statistics from metrics.
        
        Args:
            metrics (list): List of metrics for each completion
            
        Returns:
            dict: Summary statistics
        """
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
                "avg_line_similarity": sum(m["line_similarity"] for m in metrics) / len(metrics)
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
        executable_with_ssim = [m for m in metrics if m["executable"] and m.get("ssim_score") is not None]
        if executable_with_ssim:
            summary["execution"]["avg_ssim"] = sum(m["ssim_score"] for m in executable_with_ssim) / len(executable_with_ssim)
            summary["execution"]["ssim_available_count"] = len(executable_with_ssim)
        
        executable_with_pixel_sim = [m for m in metrics if m["executable"] and m.get("pixel_similarity") is not None]
        if executable_with_pixel_sim:
            summary["execution"]["avg_pixel_similarity"] = sum(m["pixel_similarity"] for m in executable_with_pixel_sim) / len(executable_with_pixel_sim)
            summary["execution"]["pixel_similarity_available_count"] = len(executable_with_pixel_sim)
        
        return summary
    
    def print_summary(self, metrics):
        """
        Print a detailed summary of metrics to console.
        
        Args:
            metrics (list): List of metrics for each completion
        """
        # Helper function to print percentages
        def pct(count, total):
            return f"{count}/{total} ({count/total*100:.2f}%)"
        
        total = len(metrics)
        
        print("\n--- Valid Code ---")
        valid_count = sum(1 for m in metrics if m["syntactically_valid"])
        outer_valid_count = sum(1 for m in metrics if m["outer_valid"])
        print(f"Completely valid code samples: {pct(valid_count, total)}")
        print(f"Valid outer code: {pct(outer_valid_count, total)}")

        # Embed usage stats
        any_embed_usage = sum(1 for m in metrics if m.get("any_embed_call", False))
        correctly_formed_embed = sum(1 for m in metrics if m.get("correctly_formed_embed", False))
        alternative_embed = sum(1 for m in metrics if len(m.get("alternative_embed_patterns", [])) > 0)
        print(f"Programs with any embed() call: {pct(any_embed_usage, total)}")
        print(f"Programs with correctly formed embed(): {pct(correctly_formed_embed, total)}")
        print(f"Programs with alternative embed() patterns: {pct(alternative_embed, total)}")

        # Embed validity for correctly formed embeds
        programs_with_correct_embed = [m for m in metrics if m.get("embed_blocks_count", 0) > 0]
        if programs_with_correct_embed:
            valid_embed_in_programs_with_embed = sum(1 for m in programs_with_correct_embed if m.get("embed_blocks_all_valid", False))
            print(f"Valid embed code: {pct(valid_embed_in_programs_with_embed, len(programs_with_correct_embed))}")
        else:
            print("No programs with correctly formed embed blocks found")

        print("\n--- SIMILARITY METRICS ---")
        # Levenshtein similarity
        avg_normalized_lev_distance = sum(m["normalized_lev_distance"] for m in metrics) / total
        print(f"Average normalized levenshtein similarity: {avg_normalized_lev_distance:.4f}")
        
        # AST similarity
        ast_metrics = [m for m in metrics if "normalized_ast_similarity" in m and m["normalized_ast_similarity"] >= 0]
        if ast_metrics:
            avg_ast_similarity = sum(m["normalized_ast_similarity"] for m in ast_metrics) / len(ast_metrics)
            print(f"Average AST-based similarity: {avg_ast_similarity:.4f}")
            print(f"AST similarity available for {len(ast_metrics)}/{total} samples")
        
        # Exact matches
        exact_matches = sum(1 for m in metrics if m["exact_match"])
        print(f"Exact matches: {pct(exact_matches, total)}")

        print("\n--- EXECUTION RESULTS ---")
        # Execution stats
        executable_count = sum(1 for m in metrics if m["executable"])
        print(f"Executable code samples: {pct(executable_count, total)}")

        # Image comparison stats for executable code
        executable_metrics = [m for m in metrics if m["executable"]]
        if executable_metrics:
            # SSIM stats
            valid_ssim = [m for m in executable_metrics if m.get("ssim_score") is not None]
            if valid_ssim:
                avg_ssim = sum(m["ssim_score"] for m in valid_ssim) / len(valid_ssim)
                print(f"Average structural similarity: {avg_ssim:.4f} (from {len(valid_ssim)}/{len(executable_metrics)} executable samples)")
            else:
                print("No valid SSIM scores available")
                
            # Pixel similarity stats
            valid_pixel = [m for m in executable_metrics if m.get("pixel_similarity") is not None]
            if valid_pixel:
                avg_pixel = sum(m["pixel_similarity"] for m in valid_pixel) / len(valid_pixel)
                print(f"Average pixel similarity: {avg_pixel:.4f} (from {len(valid_pixel)}/{len(executable_metrics)} executable samples)")
            else:
                print("No valid pixel similarity scores available")
        else:
            print("No executable code samples to calculate similarity measures")

    def evaluate_and_summarize(self, inf_dir):
        """
        Complete evaluation pipeline: load predictions, evaluate, and summarize.
        
        Args:
            inf_dir (str): Directory containing predictions.json
            
        Returns:
            tuple: (metrics, summary)
        """
        predictions = self.load_predictions(inf_dir)
        print(f"Loaded {len(predictions)} predictions from {inf_dir}")
        
        metrics = self.evaluate_completions(predictions)
        print(f"Evaluated {len(metrics)} completions")
        
        summary = self.generate_summary(metrics)
        self.print_summary(metrics)
        
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
        else:
            details["all_valid"] = details["outer_valid"]
            return details["all_valid"], "No embed() calls found", details
        
        # Extract code blocks inside embed() function calls - standard pattern with triple quotes
        correct_embed_pattern = r'embed\(\s*("""|\'\'\')(.*?)("""|\'\'\')\s*,\s*locals\(\)\s*\)'
        
        # Find all correct embed calls
        correct_embed_matches = re.findall(correct_embed_pattern, completion, re.DOTALL)
        if correct_embed_matches:
            details["embed_usage"]["correctly_formed"] = True
            
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
        alternative_pattern = r'embed\(\s*(.*?)\s*\)'
        
        try:
            all_embed_calls = re.findall(alternative_pattern, completion, re.DOTALL)
            
            # Filter out correctly formed patterns to find alternative ones
            for call_content in all_embed_calls:
                # Check if this is NOT a correctly formed embed call
                if not re.match(r'\s*("""|\'\'\').*?("""|\'\'\')\s*,\s*locals\(\)\s*$', call_content, re.DOTALL):
                    # This is an alternative pattern
                    details["embed_usage"]["alternative_patterns"].append(call_content.strip())
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
        lev_dist = levenshtein_distance(completion, ground_truth)
        
        # Normalized Levenshtein similarity (0 to 1 where 1 is identical)
        max_len = max(len(completion), len(ground_truth))
        if max_len == 0:
            normalized_lev_distance = 1.0  # Both strings are empty
        else:
            normalized_lev_distance = 1.0 - (lev_dist / max_len)
        
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
        Compare two images using SSIM and pixel-wise comparison.
        
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
            
        return {
            "ssim_score": ssim_score,
            "pixel_similarity": pixel_similarity
        }
    
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