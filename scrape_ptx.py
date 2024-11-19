import os
from typing import Dict, List
import json
import argparse

# get all ptx_data from directory including subdirectories and return a list of dicts {"input": ptx}
def get_ptx_data(directory: str) -> List[Dict[str, str]]:
    ptx_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ptx'):
                with open(os.path.join(root, file), 'r') as f:
                    ptx_data.append({"input": f.read()})
    return ptx_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PTX data from a directory")
    parser.add_argument("--directory", type=str, help="Directory to scrape PTX data from")
    args = parser.parse_args()
    ptx_data = get_ptx_data(args.directory)
    print(json.dumps(ptx_data, indent=4))
