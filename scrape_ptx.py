import os
from typing import Dict, List
import json
import argparse
import tiktoken
import re
def data_transformations(data: str) -> str:
    # remove comments, remove all lines that start with //
    data = re.sub(r'\/\/.*\n', '', data)
    # if there is more than one space, replace with one space
    data = re.sub(r'\s+', ' ', data)
    # if there is more than one newline, replace with one newline
    data = re.sub(r'\n+', '\n', data)
    return data

# get all ptx_data from directory including subdirectories and return a list of dicts {"input": ptx}
def get_ptx_data(directory: str) -> List[Dict[str, str]]:
    ptx_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ptx'):
                with open(os.path.join(root, file), 'r') as f:
                    data = f.read()
                    ptx_data.append({"input": data_transformations(data)})
    return ptx_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PTX data from a directory")
    parser.add_argument("--directory", type=str, help="Directory to scrape PTX data from")
    parser.add_argument("--output_file", type=str, help="Output file to save PTX data to")
    args = parser.parse_args()
    ptx_data = get_ptx_data(args.directory)
    with open(args.output_file, "w") as f:
        json.dump(ptx_data, f)
    # use tiktoken to count tokens
    encoding = tiktoken.encoding_for_model("gpt-4o")
    total_tokens = 0
    for ptx in ptx_data:
        num_tokens = len(encoding.encode(ptx["input"]))
        total_tokens += num_tokens
    print(f"Total number of tokens: {total_tokens}")
