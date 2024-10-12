import json
import os
import argparse
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])

def split_json_file(file_path, output_path_1, output_path_2):
    # Load the JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert dictionary to list of items (key-value pairs)
    items = list(data.items())
    
    # Calculate the split index (halfway point)
    split_index = len(items) // 2
    
    # Create two dictionaries for the split data
    data_part_1 = dict(items[:split_index])
    data_part_2 = dict(items[split_index:])
    
    # Save the first half to the first output file
    with open(output_path_1, 'w', encoding='utf-8') as f1:
        json.dump(data_part_1, f1, ensure_ascii=False, indent=4)
    
    # Save the second half to the second output file
    with open(output_path_2, 'w', encoding='utf-8') as f2:
        json.dump(data_part_2, f2, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/public_test/vimmsd-public-test.json")
    parser.add_argument("--output_file_1", type=str, default="data/public_test/vimmsd-public-test_01.json")
    parser.add_argument("--output_file_2", type=str, default="data/public_test/vimmsd-public-test_02.json")
    args = parser.parse_args()

    # Split the JSON file
    split_json_file(args.input_file, args.output_file_1, args.output_file_2)

    print(f"JSON file split into {args.output_file_1} and {args.output_file_2}.")