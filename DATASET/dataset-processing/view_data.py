# Converts a JSONL dataset into a pretty-printed JSON Array for easier viewing.

import json
import os

# --- CONFIG ---
JSONL_FILE = "../DATASET/sft_training_dataset.jsonl"
PRETTY_FILE = "../DATASET/sft_training_dataset.json"

def convert_to_pretty():
    if not os.path.exists(JSONL_FILE):
        print(f"Error: Could not find {JSONL_FILE}")
        return

    print(f"Reading {JSONL_FILE}...")
    
    data_list = []
    invalid_lines = 0
    
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parse the single line back into a real object
                obj = json.loads(line)
                data_list.append(obj)
            except json.JSONDecodeError:
                invalid_lines += 1

    print(f"Successfully loaded {len(data_list)} datapoints.")
    if invalid_lines > 0:
        print(f"⚠️ Warning: Skipped {invalid_lines} corrupted lines.")

    # Save as a standard, readable JSON Array
    print(f"Saving pretty version to {PRETTY_FILE}...")
    with open(PRETTY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4) # Indent=4 makes it readable
        
    print("Done! You can now open 'readable_view.json' to check your data.")

if __name__ == "__main__":
    convert_to_pretty()