nitial Data collection
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
