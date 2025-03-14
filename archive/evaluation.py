import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
from datasets import load_dataset
import json
import os
import matplotlib.pyplot as plt
from synthetic_data._5_ascii_processor import ASCIIProcessor
from synthetic_data._4_logo_graphic_generator_v1 import PseudoProgramInterpreter
import tempfile
import subprocess
from Levenshtein import distance as levenshtein_distance


def load_config(yaml_file):
    """Load configuration from yaml file"""
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)

def load_and_match_data(predictions_path, data_dir):
    # Load predictions
    with open(predictions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        predictions = data.get("predictions", [])  # handle new metadata structure
    
    # Load test dataset
    test_dataset = load_dataset(data_dir)["test"]
    
    # Debug prints to see data structure
    print("\nFirst test example:", test_dataset[0])
    print("\nFirst prediction:", predictions[0] if predictions else "No predictions")
    
    # Match predictions with test examples
    matched_data = []
    for pred in predictions:
        try:
            test_idx = next(i for i, test_example in enumerate(test_dataset) 
                          if str(test_example["Program"]) == str(pred.get("Program", "")))
            matched_data.append({
                "prediction": pred.get("Pred. Program", ""),
                "ground_truth": pred.get("Program", ""),
                "prompt": pred.get("Prompt", ""),
                "ascii_art": test_dataset[test_idx].get("ASCII-Art", ""),
                "metadata": data.get("metadata", {})  # include metadata from predictions
            })
        except StopIteration:
            print(f"Warning: No matching program found in test dataset: {pred.get('Program', '')[:100]}...")
        except TypeError as e:
            print(f"Error processing example: {e}")
            print(f"Test example type: {type(test_dataset[0])}")
            print(f"Prediction type: {type(pred)}")
            break
    
    print(f"\nMatched {len(matched_data)} examples")
    return matched_data

def clean_and_parse_program(predicted_program):
    """Clean and parse the predicted program, fixing common completion issues"""
    if not predicted_program:
        return False, "Empty program"
    
    # Fix unclosed parentheses
    if predicted_program.count('(') > predicted_program.count(')'):
        predicted_program += ')' * (predicted_program.count('(') - predicted_program.count(')'))
    
    # Fix unclosed triple quotes in all embed() functions
    start_idx = 0
    while True:
        # Find next embed( occurrence
        embed_pos = predicted_program.find('embed(', start_idx)
        if embed_pos == -1:
            break
            
        # Check if it starts with triple quotes
        quote_start = predicted_program.find('"""', embed_pos)
        if quote_start != -1 and quote_start == embed_pos + 6:  # 6 is len('embed(')
            # Find the corresponding closing quotes before , locals())
            locals_pos = predicted_program.find(', locals()', embed_pos)
            if locals_pos != -1:
                # If no closing quotes found before , locals(), add them
                section = predicted_program[quote_start + 3:locals_pos]
                if section.count('"""') == 0:
                    predicted_program = (
                        predicted_program[:locals_pos] + 
                        '"""' + 
                        predicted_program[locals_pos:]
                    )
        
        # Move to check next embed()
        start_idx = embed_pos + 1
    
    # Create a temporary file with imports and program execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
from synthetic_data._4_logo_graphic_generator_v1 import PseudoProgramInterpreter

# Initialize interpreter
interpreter = PseudoProgramInterpreter()

# Execute program
program = '''{}'''
interpreter.execute(program)
""".format(predicted_program))
        temp_path = f.name
    
    try:
        # Try to execute the program to check for syntax errors
        p = subprocess.Popen(
            ["python", temp_path], 
            stderr=subprocess.PIPE, 
            stdout=subprocess.PIPE
        )
        _, errs = p.communicate()
        errs = errs.decode()
        
        if errs:
            return False, f"Execution error: {errs}"
        return True, predicted_program
            
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

def evaluate_predictions(matched_data):
    eval_mode = True  # or False depending on whether you want to save images
    interpreter = PseudoProgramInterpreter()
    ascii_processor = ASCIIProcessor()
    
    results = {
        "exact_match": 0,
        # "partial_match": 0,  # Commented out for now
        "no_match": 0,
        "visual_match": 0,
        "levenshtein_distance": [],
        "total": len(matched_data),
        "execution_errors": 0
    }
    
    for idx, example in enumerate(matched_data):
        actual = example["ground_truth"]
        predicted = example["prediction"]
        target_ascii = example["ascii_art"]
        
        if not predicted or not actual:
            results["no_match"] += 1
            continue
            
        if predicted.strip() == actual.strip():
            results["exact_match"] += 1
        else:
            results["no_match"] += 1
        
        # Calculate Levenshtein distance
        lev_dist = levenshtein_distance(predicted.strip(), actual.strip())
        results["levenshtein_distance"].append(lev_dist)
            
        try:
            # Execute predicted program
            interpreter.reset_state()
            interpreter.execute(predicted)
            
            # Create the figure (similar to save_graphics but don't save)
            target_size_pixels = 525
            dpi = 100
            figsize = (target_size_pixels / dpi, target_size_pixels / dpi)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.set_aspect('equal', adjustable='datalim')
            
            # Draw paths
            for (x1, y1), (x2, y2) in interpreter.state.path:
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=1.0, linewidth=3.0)
            
            # Configure figure
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.axis('off')
            
            # Convert directly to ASCII
            pred_ascii = ascii_processor.figure_to_ascii(fig)
            pred_ascii_str = ascii_processor.ascii_matrix_to_string(pred_ascii)
            
            # Compare ASCII representations
            if pred_ascii_str.strip() == target_ascii.strip():
                results["visual_match"] += 1
            
            # Optionally save the figure if you want to keep it for reference
            if eval_mode:
                interpreter.save_graphics(f"pred_{idx}.png", eval_mode=True)
            
            plt.close(fig)
                
        except Exception as e:
            results["execution_errors"] += 1
            print(f"Error executing predicted program {idx}: {e}")
    
    # Calculate percentages
    total = float(results["total"])
    results["exact_match_percentage"] = (results["exact_match"] / total) * 100
    # results["partial_match_percentage"] = (results["partial_match"] / total) * 100  # Commented out for now
    results["no_match_percentage"] = (results["no_match"] / total) * 100
    results["avg_levenshtein_distance"] = np.mean(results["levenshtein_distance"])
    results["std_levenshtein_distance"] = np.std(results["levenshtein_distance"])
    results["visual_match_percentage"] = (results["visual_match"] / total) * 100
    results["execution_errors_percentage"] = (results["execution_errors"] / total) * 100
    
    return results

def print_aggregated_results(results):
    """Prints aggregated results in a formatted way"""
    print("=== Aggregated Results (averages) ===")
    for (topktrain, topkprompt), (correctness, accuracy) in sorted(results.items()):
        print(f"topktrain={topktrain}, topkprompt={topkprompt} | "
              f"correctness={correctness:.3f}, accuracy={accuracy:.3f}")

def plot_heatmap(results):
    """Creates a heatmap visualization of the results"""
    import matplotlib.pyplot as plt
    
    # Filter out entries where topkprompt in {0, 30}
    filtered_results = {
        (train, prompt): (corr, acc)
        for (train, prompt), (corr, acc) in results.items()
        if prompt not in [0, 30]
    }
    
    # Get unique values for axes
    unique_trains = sorted({k[0] for k in filtered_results.keys()})
    unique_prompts = sorted({k[1] for k in filtered_results.keys()})
    
    # Create heatmap data matrix
    heatmap_data = np.full((len(unique_trains), len(unique_prompts)), np.nan)
    
    for (train, prompt), (corr, acc) in filtered_results.items():
        i = unique_trains.index(train)
        j = unique_prompts.index(prompt)
        heatmap_data[i, j] = acc  # Using accuracy for visualization
    
    # Plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, cmap="viridis", aspect="auto", 
                    interpolation="nearest", origin="lower")
    
    plt.colorbar(im, label="Accuracy")
    plt.xticks(range(len(unique_prompts)), unique_prompts, rotation=45)
    plt.yticks(range(len(unique_trains)), unique_trains)
    
    plt.ylabel("# samples for finetuning")
    plt.xlabel("# n-shot prompts")
    plt.title("Accuracy Heatmap")
    
    # Add value annotations
    for i in range(len(unique_trains)):
        for j in range(len(unique_prompts)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                color = "white" if val > np.nanmax(heatmap_data)/2 else "black"
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color=color)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load config
    config = load_config("config.yaml")
    model_name = config["model"]["model_id"]  # same as in inference.py
    
    # Construct the exact path for the predictions
    results_dir = "results/length/inference"
    model_dir = model_name.split('/')[-1]  # matches the directory created in inference.py
    predictions_path = os.path.join(results_dir, model_dir, 'predictions.json')
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"No predictions found for model {model_dir} at {predictions_path}")
    
    print(f"Processing predictions from: {predictions_path}")
    
    # Process the predictions
    matched_data = load_and_match_data(predictions_path, config["data"]["dataset_id"])
        
    # Evaluate matched data
    results = evaluate_predictions(matched_data)
        
    # Save metrics
    metrics_path = os.path.join(results_dir, model_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()