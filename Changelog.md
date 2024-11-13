## 2024/11/13
A lot development has happened with https://github.com/PaliC/popcorn-eval

### Change in function format
Now when generating the dataset, all functions are annotated with
```python
{{function body}}
```

which is what you expect to get out of an llm as it denotes code. The dataset has been updated to reflect this.
Generation of dataset remains the same.


## 2024/29/10 and the times before
- Initial Data collection
	- Initially we were looking to figure out a decent dataset we can use in order to train an llm to produce triton kernels with.
	- There are two obvious places one can look for the data
		- We run torch inductor on a bunch of stuff and see what comes out
		- We scrape github. We did variants of both for this initial dataset.
### Inductor Kernel Scrape
- This is reasonably straightforward.
	- We just downloaded pytorch nightlies using `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118` and then in a local clone of pytorch I just ran

```
cd pytorch
export TORCH_COMPILE_DEBUG=1
python benchmarks/dynamo/**timm.py** --accuracy --inference --inductor --device cuda
		- python benchmarks/dynamo/**huggingface.py** --accuracy --inference --inductor --device cuda
```

Afterwards there should be a folder `torch_compile_debug`

I created a utility script in order to extract the cuda kernels. Therefore, I just ran
```
cd triton-data
python get_kernels_from_debug_info.py <path to torch_compile_debug folder>
```
This creates a folder called `inductor_kernels` which we will use later
- Todo: Two very useful things to do are
1. Isolate the models used in the benchmark as pytorch code such that people can program against them as part of cudabench
2. Actually get triton kernels out of torchbench

### Github Scrape
- Another option to find already written triton code is to look toward github!
- Jason Ansel (jansel) has an some infra already setup for this here: https://github.com/jansel/pytorch-jit-paritybench/tree/master
	- I basically just copy and pasted this + removed the irrelevant bits like downloading files / writing tests
	- I also modified it slightly to look for triton kernels instead of pytorch modules (look for a @triton.jit decorator)
- I made the assumptions that:
1. that we only care about repos with over 100 stars
2. the kernels in these repos will run
3. There are no helper functions or imports that are not torch or triton

In order to produce this data in triton-data I run
```
cd triton-data
python scrape_github.py # downloads files from github
cd scraper/generator
python main.py --generate-all
```
This populates `scraper/generator/generated` with the appropriate kernels.

### Make a dataset
For initial finetuning experiments. I am using torchtune specifically I'm doing full fine tuning using text completion. In general we want some dataset that looks like the one described here.
https://pytorch.org/torchtune/stable/basics/text_completion_datasets.html

I fortunately wrote a script to do this nicely, just run
```
cd triton-data
python create_json.py inductor_kernels scraper/generator/generated

```
This creates a file called `datasets/triton_functions.json` which is the dataset we are looking for

### Training the model
Now we use torchtune to train the model. Specifically I'm doing full fine tuning

To replicate this install [torchtune](https://pytorch.org/torchtune/stable/install.html)

```bash
cd triton-data
git install lfs (needed to install checkpoints)
%% the checkpoints are safetensors so we need to not ignore them %%
tune download meta-llama/Llama-3.2-3B  --output-dir /tmp/Llama-3.2-3B --hf-token <hf_token> --ignore-patterns "None"
tune run full_finetune_single_device --config tune_configs/llama_3_config.yaml epochs=1
```

We then do some generations. I created them by running some configs which point to the generated fine tuned model and the base model

```bash
tune run generate --config ./tune_configs/gen_llama3_3b_finetune_config.yaml 2> output_base.txt
tune run generate --config ./tune_configs/gen_llama3_3b_original_config.yaml 2> output_tuned.txt
```

The prompt which is used is

```
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>                                              │
                                                                                                           │
  Cutting Knowledge Date: December 2023                                                                    │
  Today Date: 29 October 2024                                                                              │
                                                                                                           │
  You are an AI assistant who helps software engineers write triton kernels which is a type of gpu kernel w│
ritten in python.t<|eot_id|><|start_header_id|>user<|end_header_id|>                                       │
                                                                                                           │
  Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, res│
ulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32;│
 ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.lo│
ad with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python ho│
st code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.m│
atmul, ensuring accuracy within 1e-4.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

From the base llama 3.2 3b model we get hot garbage
```
You are an AI assistant who helps software engineers write triton kernels which is a type of gpu kernel written in python.tuser
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with
```

However, we get something reasonable with the fine tuned variant
```
You are an AI assistant who helps software engineers write triton kernels which is a type of gpu kernel written in python.tuser
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
def matrix_mul_kernel(a_ptr, b_ptr, c_ptr, BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a = a_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    b = b_ptr + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = c_ptr + pid_m * BLOCK_SIZE_M + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_K)
    mask_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < 1024
    mask_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < 1024
    mask_k = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None] < 1024
    mask_j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :] < 1024
    a = tl.load(a, mask=mask_m[:, None])
    b = tl.load(b, mask=mask_n[None, :])
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, 1024, BLOCK_SIZE_K):
        a_k = a + k
        b_k = b + k
        a_k = tl.load(a_k, mask=mask_k, other=0)
        b_k = tl.load(b_k, mask=mask_j, other=0)
        c += tl.dot(a_k, b_k)
    c = tl.where(mask_m[:, None] & mask_n[None, :], c, 0)
    tl.store(c, c, mask=mask_m[:, None])
```

However, there is also llama3.2-3b-instruct who gives this output
```
You are an AI assistant who helps software engineers write triton kernels which is a type of gpu kernel written in python.tuser
Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.assistant
Here's a Python code snippet that implements a Triton kernel function for matrix multiplication:

```python
import numpy as np
import tritonpython as tl

# Define constants
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
M = 1024
N = 1024
K = 1024

# Define the Triton kernel function
def matrix_multiply_kernel(A, B, C):
    # Get the dimensions of the matrices
    M = A.shape[0]
    N = A.shape[1]
    K = A.shape[2]

    # Create a 3D tensor to store the partial sums
    partial_sums = tl.Tensor(shape=(M, N, K), dtype=A.dtype)

    # Create a 3D tensor to store the result
    C = tl.Tensor(shape=(M, N, K), dtype=A.dtype)

    # Define the kernel launch parameters
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K)
    block = (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # Launch the kernel
    tl.load(A, partial_sums, mask=(partial_sums.shape[0] >= BLOCK_SIZE_M and
                                    partial_sums.shape[1] >= BLOCK_SIZE_N and
                                    partial_sums.shape[2] >= BLOCK_SIZE_K))
    tl.load(B, partial_sums, mask=(partial_sums.shape[0] >= BLOCK_SIZE_M and
                                    partial_sums.shape[1] >= BLOCK_SIZE_N and
                                    partial_sums.shape[2] >= BLOCK_SIZE_K))

    # Perform the matrix multiplication
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j, k] = partial_sums[i, j, k]

    # Store the result in C
    tl.store(C, C, mask=(C.shape[0] >= BLOCK_SIZE_M and
                         C.shape[1] >= BLOCK_SIZE_N and
                         C.shape[2] >= BLOCK_SIZE_K))

# Define the host code
def main():
    # Create random matrices A and B
    A = np.random.rand(M, N, K)
    B = np.random.rand(M, N, K)

    # Create a copy of A and B for the kernel
    A_kernel = A.copy()
    B_kernel = B.copy()

    # Launch the kernel
    matrix_multiply_kernel(A_kernel, B_kernel, C)

    # Verify correctness
    C_torch = np.matmul(A, B)
    assert np.allclose(C, C_torch, atol=1e-4)

# Run the host code
if __name__ == "__main__":
    main()
```

While there is a lot that are obviously incorrect, it is more useful as it at least looks like python / triton code.
