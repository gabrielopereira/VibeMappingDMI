import json
import os

datasets = [
    'book_reviews.json',
    'merged_editorials.json',
    'merged_no_editorials.json',
    'merged_no_jats.json',
    'merged.json'
]

base_dir = 'merged data'

for dataset in datasets:
    file_path = os.path.join(base_dir, dataset)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Number of objects in {file_path}: {len(data)}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"File is not valid JSON: {file_path}") 