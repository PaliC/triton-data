import argparse
import os
import re

import tiktoken

# Parse the input folder path from command line arguments
parser = argparse.ArgumentParser(description="Aggregate Triton kernel logs.")
parser.add_argument(
    "input_folder", type=str, help="Path to the torch_compile_debug folder"
)
args = parser.parse_args()

# Define the folder to search
input_folder = args.input_folder
output_file = "aggregated_kernels.txt"

# Regex pattern to match lines with kernel paths
# This pattern looks for lines starting with '# kernel path: ' followed by the file path
kernel_path_pattern = re.compile(r"# kernel path: (.+)")

# List to collect all kernel texts
all_kernel_texts = []

# Walk through the directory and find all files named output_code.py
# os.walk generates the file names in the directory tree rooted at input_folder
for root, _, files in os.walk(input_folder):
    for file in files:
        # Check if the current file is 'output_code.py'
        if file == "output_code.py":
            # Construct the full path to the file
            file_path = os.path.join(root, file)
            # Open 'output_code.py' and read its lines
            with open(file_path, "r") as f:
                for line in f:
                    # Search for lines that match the kernel path pattern
                    match = kernel_path_pattern.search(line)
                    if match:
                        # Extract the kernel file path from the matched line
                        kernel_file_path = match.group(1)
                        # Check if the kernel file exists before trying to read it
                        if os.path.exists(kernel_file_path):
                            # Open the kernel file and read its content
                            with open(kernel_file_path, "r") as kernel_file:
                                all_kernel_texts.append(kernel_file.read())

# Initialize tiktoken encoding to count tokens
encoding = tiktoken.get_encoding("o200k_base")

# Count the number of tokens before writing to the output file
token_count = 0
for text in all_kernel_texts:
    tokens = encoding.encode(text)
    token_count += len(tokens)

# Write all collected kernel texts into a single output file
with open(output_file, "w") as output_f:
    # Join all kernel texts with a newline and write to the output file
    output_f.write("\n".join(all_kernel_texts))

# Print a message indicating that the aggregation is complete and the token count
print(f"Aggregated kernel content written to {output_file}")
print(f"Total number of tokens in the output file: {token_count}")
