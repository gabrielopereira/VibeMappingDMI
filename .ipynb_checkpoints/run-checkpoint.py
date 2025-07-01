## run this to install requirements: pip install -r requirements.txt

## run jupyter lab to run the notebook

## importing this to exclude stopwords
import json
from bertopic import BERTopic

# Load the JSON file
with open('2056-3051.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter out entries with empty titles
data = [item for item in data if item['title']]

# Extract abstracts and titles
abstracts = [item['abstract'] for item in data]
titles = [item['title'] for item in data]

# Extract journal title if needed later
container_titles = [item['metadata']['container_title'] for item in data]

print(f"Loaded {len(abstracts)} papers with abstracts")
print(f"Sample title: {titles[0]}")
print(f"Sample abstract: {abstracts[0][:200]}...")
print(f"Journal title: {container_titles[0]}")
print(f"Total documents: {len(titles)}")





