import argparse
import os
import re
from collections import defaultdict

import tiktoken

# Parse the input folder path from command line arguments
parser = argparse.ArgumentParser(description="Aggregate Triton kernel logs.")
parser.add_argument(
    "input_folder", type=str, help="Path to the torch_compile_debug folder"
)
args = parser.parse_args()

# Define the folder to search
input_folder = args.input_folder
output_directory = "inductor_kernels"

# Regex pattern to match lines with kernel paths
# This pattern looks for lines starting with '# kernel path: ' followed by the file path
kernel_path_pattern = re.compile(r"# kernel path: (.+)")

# List to collect all kernel texts
all_kernel_texts = []

# Walk through the directory and find all files named output_code.py
# os.walk generates the file names in the directory tree rooted at input_folder

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for root, _, files in os.walk(input_folder):
    for file in files:
        # Check if the current file is 'output_code.py'
        if file == "output_code.py":

            kernel_file_paths = []

            # Construct the full path to the file
            file_path = os.path.join(root, file)
            # Open 'output_code.py' and read its lines
            model_trace_name = root.split("/")[-1]
            model_name = model_trace_name.split("__")[0]
            output_file_text = f"# {model_name}\n"
            directory_path = os.path.dirname(file_path)
            pytorch_code = os.path.join(directory_path, "fx_graph_readable.py")

            output_model_directory = os.path.join(output_directory, model_name)
            output_trace_directory = os.path.join(
                output_model_directory, model_trace_name
            )
            kernel_directory = os.path.join(output_trace_directory, "kernels")
            if not os.path.exists(output_model_directory):
                os.makedirs(output_model_directory)
            if not os.path.exists(output_trace_directory):
                os.makedirs(output_trace_directory)

            # copy over output_code.py and fx_graph_readable.py
            output_code_file = os.path.join(output_trace_directory, "output_code.py")
            fx_graph_readable_file = os.path.join(
                output_trace_directory, "fx_graph_readable.py"
            )
            with open(file_path, "r") as f:
                with open(output_code_file, "w") as output_code_file:
                    output_code_file.write(f.read())
            with open(pytorch_code, "r") as f:
                with open(fx_graph_readable_file, "w") as fx_graph_readable_file:
                    fx_graph_readable_file.write(f.read())

            # create kernel directory
            if not os.path.exists(kernel_directory):
                os.makedirs(kernel_directory)

            with open(file_path, "r") as f:
                for line in f:
                    # Search for lines that match the kernel path pattern
                    match = kernel_path_pattern.search(line)
                    if match:
                        # Extract the kernel file path from the matched line
                        kernel_file_path = match.group(1)
                        # Check if the kernel file exists before trying to read it
                        if os.path.exists(kernel_file_path):
                            # Add the kernel file path to the list
                            kernel_file_paths.append(kernel_file_path)

            # copy over all kernel files
            for kernel_file_path in kernel_file_paths:
                kernel_name = os.path.basename(kernel_file_path)
                kernel_name = kernel_name.split("/")[-1]
                with open(kernel_file_path, "r") as f:
                    with open(
                        os.path.join(kernel_directory, kernel_name), "w"
                    ) as kernel_file:
                        kernel_file.write(f.read())
                # Read the kernel file and append its contents to


# parse output directory for all kernel files and aggregate them into a single string for token counting
# walk through the directory and find all files within a kernels directory

all_kernel_texts = []
token_count = 0
encoding = tiktoken.get_encoding("o200k_base")

for root, dirname, files in os.walk(output_directory):
    for file in files:
        if root.endswith("kernels") and file.endswith(".py"):
            # Construct the full path to the file
            file_path = os.path.join(root, file)
            # Open 'output_code.py' and read its lines
            with open(file_path, "r") as f:
                for line in f:
                    # Search for lines that match the kernel
                    all_kernel_texts.append(line)
                    token_count += len(encoding.encode(line))

# Print a message indicating that the aggregation is complete and the token count
print(f"Aggregated kernel content written to {output_directory}")
print(f"Total number of tokens in the output file: {token_count}")
