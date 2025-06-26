'''
This script generates images from program completions and ground truth using the ProgramParser.
It expects a directory containing predictions in JSON format and an ID to specify which row and completion to use.
The ID should be formatted as "row_idx_completion_idx", where row_idx is the index of the row in the predictions and completion_idx is the index of the completion within that row. This ID corresponds to the format used in the detaile_metrics.jsonl and serves for a visual insepction of the generated and ground truth program evaluated.


Usage:
python __LOGO_image.py --eval_dir <path_to_eval_dir> --ID <row_idx_completion_idx>

Where:
- `<path_to_eval_dir>` is the directory containing the `predictions.json` file.
- `<row_idx_completion_idx>` is the ID formatted as described above.    

Example:
python __LOGO_image.py --eval_dir ./eval_results --ID 0_1
This will generate images for the first row and the first completion in the predictions file.

python __LOGO_image.py --eval_dir results/length/CodeLlama_20250531_0054/inference/20250603_1637 --ID 0_1
'''
import os
import json
import argparse
from synthetic_data.__parser_pyturtle_pc import ProgramParser

def load_predictions(eval_dir):
    with open(os.path.join(eval_dir, 'predictions.json'), 'r') as f:
        predictions = json.load(f)
    return predictions

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--eval_dir', type=str, required=True)
    argparser.add_argument('--ID', type=str, required=True)
    args = argparser.parse_args()

    eval_dir = args.eval_dir
    ID = args.ID

    # Parse ID
    try:
        row_idx_str, completion_idx_str = ID.split('_')
        row_idx = int(row_idx_str)
        completion_idx = int(completion_idx_str)
    except Exception as e:
        print(f"Error parsing ID '{ID}': {e}")
        return

    predictions = load_predictions(eval_dir)
    if not (0 <= row_idx < len(predictions)):
        print(f"Row index {row_idx} out of range.")
        return

    row = predictions[row_idx]
    completion_key = f"completion_{completion_idx}"
    if completion_key not in row:
        print(f"Key '{completion_key}' not found in row {row_idx}.")
        return
    if "ground_truth" not in row:
        print(f"'ground_truth' key not found in row {row_idx}.")
        return

    # Prepare examples
    completion_example = {
        "id": ID,
        "Description": f"completion",
        "Program": row[completion_key]
    }
    gt_example = {
        "id": ID,
        "Description": f"gt",
        "Program": row["ground_truth"]
    }

    save_dir = os.path.join(eval_dir, "logo_graphic")
    os.makedirs(save_dir, exist_ok=True)

    parser = ProgramParser(save_dir="logo_graphic", save_image=False, eval_mode=False)
    parser.parse_and_generate_image(completion_example)
    parser.parse_and_generate_image(gt_example)

if __name__ == "__main__":
    main()