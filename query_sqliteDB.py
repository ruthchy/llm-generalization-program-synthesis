import sqlite3
import json

def delete_trials(trial_ids, db_path="optuna_storage.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    placeholders = ", ".join("?" for _ in trial_ids)

    # Delete related entries from all relevant tables
    tables_to_clean = [
        "trial_params",
        "trial_values",
        "trial_user_attributes",
        "trial_system_attributes",
        "trials"
    ]

    for table in tables_to_clean:
        cur.execute(f"DELETE FROM {table} WHERE trial_id IN ({placeholders})", trial_ids)

    conn.commit()
    conn.close()
    print(f"Deleted trials: {trial_ids}")

# Helper function to fetch and print rows from a table
def print_table(table_name):
    conn = sqlite3.connect("optuna_storage.db")
    cur = conn.cursor()

    print(f"\n--- {table_name.upper()} ---")
    try:
        cur.execute(f"SELECT * FROM {table_name};")
        rows = cur.fetchall()
        col_names = [description[0] for description in cur.description]

        if not rows:
            print("No entries found.")
            return

        # Print column names
        print(" | ".join(col_names))
        print("-" * 80)

        # Print rows
        for row in rows:
            print(" | ".join(str(item) for item in row))
    except sqlite3.Error as e:
        print(f"Error accessing table '{table_name}': {e}")

def display_trial_params_wide(db_path="optuna_storage.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT trial_id, param_name, param_value, distribution_json
        FROM trial_params
        ORDER BY trial_id, param_name;
    """)
    rows = cur.fetchall()

    trials = {}

    for trial_id, param_name, param_value, dist_json in rows:
        dist = json.loads(dist_json)
        choices = dist.get("attributes", {}).get("choices", None)
        actual_value = param_value
        if choices is not None:
            try:
                idx = int(round(param_value))
                if 0 <= idx < len(choices):
                    actual_value = choices[idx]
            except Exception:
                pass
        if trial_id not in trials:
            trials[trial_id] = {}
        trials[trial_id][param_name] = actual_value

    column_order = [
        "learning_rate",
        "learning_rate_scheduler",
        "per_device_train_batch_size",
        "lora_rank",
        "lora_alpha"
    ]

    header = ["trial_id"] + column_order
    print(" | ".join(header))
    print("-" * 80)

    for trial_id in sorted(trials.keys()):
        row = [str(trial_id)]
        for col in column_order:
            val = trials[trial_id].get(col, "N/A")
            row.append(str(val))
        print(" | ".join(row))

    conn.close()

# === Main Execution ===

if __name__ == "__main__":
    # Optionally delete duplicate trials before displaying
    # Replace with actual duplicates
    #delete_trials([13, 14])

    # List of relevant Optuna tables
    #tables = ["studies", "trials", "trial_params"]
    tables = ["studies", "trials"]

    # Print contents of each table
    for table in tables:
        print_table(table)

    # Print trial parameters in a wide format
    print(f"\n--- TRAIL PARAMS ---")
    display_trial_params_wide()