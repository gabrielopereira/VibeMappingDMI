import json

# Load the cleaned data
with open('merged data/merged_no_jats.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Separate editorials and non-editorials
editorials = []
non_editorials = []

for item in data:
    title = item.get('title', '').strip()
    title_lower = title.lower()
    # Check for exact 'editorial' or titles starting with 'editorial ' (case-insensitive)
    if title_lower == 'editorial' or title_lower.startswith('editorial '):
        editorials.append(item)
    else:
        non_editorials.append(item)

# Save the datasets
with open('merged data/merged_editorials.json', 'w', encoding='utf-8') as f:
    json.dump(editorials, f, ensure_ascii=False, indent=2)

with open('merged data/merged_no_editorials.json', 'w', encoding='utf-8') as f:
    json.dump(non_editorials, f, ensure_ascii=False, indent=2)

print(f"Separated {len(editorials)} editorials and {len(non_editorials)} non-editorials.") 