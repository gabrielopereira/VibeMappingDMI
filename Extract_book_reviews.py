import json
import os
import re

input_path = "merged data/merged_no_jats.json"
book_reviews_path = "merged data/book_reviews.json"
final_merged_path = "merged data/final_merged.json"

exclude_phrases = [
    "systematic review",
    "literature review",
    "meta-analysis",
    "review article",
    "review of",
    "review on"
]

def is_book_review(item):
    type_str = item.get("metadata", {}).get("type", "").lower()
    title_str = item.get("title", "").lower()
    abstract_str = item.get("abstract", "").lower()

    # Match "book review" or "book reviews" as whole words, anywhere
    is_book = (
        re.search(r"\bbook reviews?\b", type_str) or
        re.search(r"\bbook reviews?\b", title_str) or
        re.search(r"\bbook reviews?\b", abstract_str)
    )

    # Only exclude if the entry is NOT a book review by type, title, or abstract
    is_excluded = (
        not is_book and
        any(
            phrase in type_str or phrase in title_str or phrase in abstract_str
            for phrase in exclude_phrases
        )
    )

    return is_book and not is_excluded

# Load all data from merged_no_jats.json
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Find all book reviews and non-book reviews in merged_no_jats.json
new_book_reviews = [item for item in data if is_book_review(item)]
not_book_reviews = [item for item in data if not is_book_review(item)]

# Load existing book reviews if the file exists
if os.path.exists(book_reviews_path):
    try:
        with open(book_reviews_path, "r", encoding="utf-8") as f:
            existing_book_reviews = json.load(f)
    except json.JSONDecodeError:
        existing_book_reviews = []
else:
    existing_book_reviews = []

# Avoid duplicates by using IDs
existing_ids = {item.get("id") for item in existing_book_reviews}
combined_book_reviews = existing_book_reviews + [
    item for item in new_book_reviews if item.get("id") not in existing_ids
]

# Save the combined list to book_reviews.json
with open(book_reviews_path, "w", encoding="utf-8") as f:
    json.dump(combined_book_reviews, f, ensure_ascii=False, indent=2)

# Save the not-book-reviews data to final_merged.json
with open(final_merged_path, "w", encoding="utf-8") as f:
    json.dump(not_book_reviews, f, ensure_ascii=False, indent=2)

print(f"Appended {len(new_book_reviews)} new book reviews (if not already present) to {book_reviews_path}")
print(f"Saved {len(not_book_reviews)} non-book-review records to {final_merged_path}")