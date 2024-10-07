# triton-data

This repo contains data scraped from GitHub repositories that contain Triton code and some benchmakrs we use for indcutor.

`scrape_github.py` is a script that scrapes open source GitHub files that contain Triton code.
usage: just run `python scrape_github.py` and it should get you 1000 files of python files with triton code.
The results are stored in a folder called `github_triton_data`.

`get_kernels_from_debug_info.py` is a script that extracts data from a torch_compile_debug folder which can 
be produced when running a file which calls torch.compile with the environement variable TORCH_COMPILE_DEBUG=1

`aggregated_kernels.txt` contains the results of running `get_kernels_from_debug_info.py` on the torch_compile_debug when running pytorch's huggingface benchmark for torchinductor. Specifically from pytorch root you'd call `TORCH_COMPILE_DEBUG=1 python benchmarks/dynamo/huggingface.py --performance --inference --inductor --device cuda` to produce a similar torch_compile_debug folder to one in this repo (changes are just due to different versions of pytorch used), but shouldn't matter for initial data collection.


