#!/usr/bin/env python3
"""
Check for missing timestamps in merged.json
Counts how many entries don't have a proper published timestamp in metadata.published
"""

import json
import sys

def check_missing_timestamps():
    """Check merged.json for missing timestamps"""
    
    try:
        # Load the merged.json file
        with open('merged data/merged.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total entries in merged.json: {len(data)}")
        
        missing_timestamps = 0
        empty_published = 0
        invalid_format = 0
        
        for i, entry in enumerate(data):
            # Check if metadata exists
            if 'metadata' not in entry:
                missing_timestamps += 1
                print(f"Entry {i}: Missing metadata section")
                continue
            
            metadata = entry['metadata']
            
            # Check if published field exists
            if 'published' not in metadata:
                missing_timestamps += 1
                print(f"Entry {i}: Missing 'published' field in metadata")
                continue
            
            published = metadata['published']
            
            # Check if published is empty or None
            if not published or published == [] or published is None:
                empty_published += 1
                print(f"Entry {i}: Empty published field: {published}")
                continue
            
            # Check if published has the expected format [year, month]
            if not isinstance(published, list) or len(published) < 2:
                invalid_format += 1
                print(f"Entry {i}: Invalid published format: {published}")
                continue
            
            # Check if the values are valid numbers
            if not (isinstance(published[0], int) and isinstance(published[1], int)):
                invalid_format += 1
                print(f"Entry {i}: Invalid published values (not integers): {published}")
                continue
        
        # Summary
        print("\n" + "="*50)
        print("TIMESTAMP CHECK SUMMARY")
        print("="*50)
        print(f"Total entries: {len(data)}")
        print(f"Missing timestamps: {missing_timestamps}")
        print(f"Empty published fields: {empty_published}")
        print(f"Invalid format: {invalid_format}")
        print(f"Total problematic entries: {missing_timestamps + empty_published + invalid_format}")
        print(f"Valid entries: {len(data) - (missing_timestamps + empty_published + invalid_format)}")
        
        if missing_timestamps + empty_published + invalid_format == 0:
            print("\n✅ All entries have valid timestamps!")
        else:
            print(f"\n❌ {missing_timestamps + empty_published + invalid_format} entries have timestamp issues")
        
    except FileNotFoundError:
        print("Error: merged.json not found in 'merged data/' directory")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in merged.json: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_missing_timestamps() 