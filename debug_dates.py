import json
from collections import Counter

def debug_date_formats():
    """Debug the date formats in the JSON data"""
    print("Loading fullmerged.json...")
    
    with open('fullmerged.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} publications")
    
    date_formats = []
    published_dates = []
    
    for i, pub in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing publication {i}/{len(data)}")
        
        if 'metadata' in pub and 'published' in pub['metadata']:
            date_data = pub['metadata']['published']
            date_formats.append(type(date_data).__name__)
            published_dates.append(date_data)
    
    print(f"\nFound {len(published_dates)} publications with 'published' field")
    
    # Analyze date formats
    format_counts = Counter(date_formats)
    print(f"\nDate format types:")
    for format_type, count in format_counts.most_common():
        print(f"- {format_type}: {count}")
    
    # Show some examples of each format
    print(f"\nExamples of each format:")
    seen_formats = set()
    for date_data in published_dates:
        format_type = type(date_data).__name__
        if format_type not in seen_formats:
            seen_formats.add(format_type)
            print(f"\n{format_type}: {date_data}")
            if isinstance(date_data, list):
                print(f"  Length: {len(date_data)}")
                print(f"  Elements: {date_data}")
    
    # Check for array format specifically
    array_dates = [d for d in published_dates if isinstance(d, list)]
    print(f"\nArray format analysis:")
    print(f"Total array dates: {len(array_dates)}")
    
    if array_dates:
        lengths = [len(d) for d in array_dates]
        length_counts = Counter(lengths)
        print(f"Array lengths: {length_counts}")
        
        # Show examples of different lengths
        for length in sorted(length_counts.keys()):
            examples = [d for d in array_dates if len(d) == length][:3]
            print(f"\nArrays with length {length}:")
            for example in examples:
                print(f"  {example}")

if __name__ == "__main__":
    debug_date_formats() 