import os
import json
from synthetic_data.__parser_pyturtle_pc import ProgramParser

# Path to the predictions.json file
predictions_path = "results/length/CodeLlama/inference/20250330_2316_imagePBEsty/predictions.json"

# Folder to store the generated images
visual_inspection_dir = os.path.join(os.path.dirname(predictions_path), "visual_inspection")
os.makedirs(visual_inspection_dir, exist_ok=True)

# Load predictions
with open(predictions_path, "r", encoding="utf-8") as f:
    predictions = json.load(f)

# IDs to process
selected_ids = [929]  # Replace with the IDs you want to process

# Initialize the ProgramParser
parser = ProgramParser(save_dir=visual_inspection_dir, save_image=False, eval_mode=False)

# Generate images for the selected IDs
for idx in selected_ids:
    if idx >= len(predictions):
        print(f"ID {idx} is out of range. Skipping...")
        continue

    prediction = predictions[idx]
    ground_truth_program = prediction.get("ground_truth")
    generated_program = prediction.get("completion", None)  # Replace with the key for generated programs if available

    # Generate and save the ground truth image
    if ground_truth_program:
        ground_truth_example = {"id": f"{idx}_gt", "Program": ground_truth_program}
        parser.parse_and_generate_image(ground_truth_example)
        print(f"Generated ground truth image for ID {idx}")

    # Generate and save the generated program image
    if generated_program:
        generated_example = {"id": f"{idx}_pred", "Program": generated_program}
        parser.parse_and_generate_image(generated_example)
        print(f"Generated generated program image for ID {idx}")

print(f"Images saved in {visual_inspection_dir}")

