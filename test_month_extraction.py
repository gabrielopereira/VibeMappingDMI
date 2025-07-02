import json
from datetime import datetime
from collections import Counter

def test_month_extraction():
    """Test that months are being extracted correctly"""
    print("Loading sample data...")
    
    with open('fullmerged.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Test with first 1000 publications
    test_data = data[:1000]
    
    months = []
    
    for pub in test_data:
        if 'metadata' in pub and 'published' in pub['metadata']:
            date_data = pub['metadata']['published']
            
            if isinstance(date_data, list) and len(date_data) >= 2:
                year = int(date_data[0])
                month = int(date_data[1])
                months.append(month)
    
    print(f"Extracted months from {len(months)} publications")
    
    # Count months
    month_counts = Counter(months)
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    print(f"\nMonth distribution:")
    for month_num in sorted(month_counts.keys()):
        month_name = month_names[month_num - 1]
        count = month_counts[month_num]
        print(f"- {month_name}: {count}")
    
    # Check if all are January
    if len(month_counts) == 1 and 1 in month_counts:
        print(f"\nWARNING: All publications are showing as January!")
    else:
        print(f"\nGood! Found {len(month_counts)} different months")

if __name__ == "__main__":
    test_month_extraction() 