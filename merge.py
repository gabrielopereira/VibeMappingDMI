# This script merges all JSON files in the 'original data' directory into a single merged.json file in the 'merged data' directory
import json
import glob
import os

# Find all .json files in the 'original data' directory
json_files = glob.glob('fullfetcheddata/*.json')

# Initialize an empty list to hold all records from all files
merged_data = []

# Loop through each JSON file found
for file in json_files:
    # Open the current JSON file for reading
    with open(file, 'r', encoding='utf-8') as f:
        # Load the data from the file (assumes each file contains a list of records)
        data = json.load(f)
        # Add the records from this file to the merged_data list
        merged_data.extend(data)

# Ensure the output directory exists
os.makedirs('merged data', exist_ok=True)

# Write the combined data to a new file called fullmerged.json in 'merged data'
with open('fullmerged.json', 'w', encoding='utf-8') as f:
    # Dump the merged data as pretty-printed JSON
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

# Print a summary of what was done
print(f"Merged {len(json_files)} files into merged data/merged.json with {len(merged_data)} records.") 