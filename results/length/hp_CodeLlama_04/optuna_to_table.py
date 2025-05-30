import json
import pandas as pd

# Load the JSON file
with open("results/length/hp_CodeLlama_04/optuna_storage.json", "r") as f:
    data = json.load(f)

# Extract trials
trials = data["trials"]

# Prepare rows for the table
rows = []
for trial in trials:
    row = {"trial_id": trial["trial_id"]}
    row.update(trial["params"])
    row["comp_loss"] = round(trial["value"], 4)
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Save as CSV
df.to_csv("optuna_trials_table.csv", index=False)

# Print as Markdown table
#print(df.to_markdown(index=False))