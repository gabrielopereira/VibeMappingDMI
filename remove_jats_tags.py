import json
import re

# Load the merged data
with open('merged data/merged.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def remove_jats_tags(obj):
    """Recursively remove all <jats:...>...</jats:...> tags and their content from all string fields in a dict or list."""
    if isinstance(obj, dict):
        return {k: remove_jats_tags(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_jats_tags(item) for item in obj]
    elif isinstance(obj, str):
        # Remove all <jats:...>...</jats:...> tags
        cleaned = re.sub(r'<jats:[^>]+>(.*?)</jats:[^>]+>', r'\1', obj, flags=re.DOTALL)
        # Remove all self-closing or empty <jats:.../> tags
        cleaned = re.sub(r'<jats:[^>]+/>', '', cleaned)
        # Remove any remaining <jats:...> or </jats:...> tags
        cleaned = re.sub(r'</?jats:[^>]+>', '', cleaned)
        return cleaned
    else:
        return obj

# Clean the data
cleaned_data = remove_jats_tags(data)

# Save the cleaned data to a new file
with open('merged data/merged_no_jats.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print("All JATS tags and their content removed. Cleaned file saved as merged_no_jats.json.")