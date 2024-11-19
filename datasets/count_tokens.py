import tiktoken
import json
import os

# count tokens in datasets/triton_functions.json

token_count = 0
encoding = tiktoken.get_encoding("o200k_base")

triton_functions = json.load(open("triton_functions.json", "r"))
count = 0
for function in triton_functions:
    count += 1
    token_count += len(encoding.encode(function["input"]))

print(f"Total token count: {token_count} with {count} functions")