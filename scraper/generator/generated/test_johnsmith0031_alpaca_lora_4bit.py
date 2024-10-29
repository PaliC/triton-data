import sys
_module = sys.modules[__name__]
del sys
Finetune4bConfig = _module
amp_wrapper = _module
arg_parser = _module
autograd_4bit = _module
custom_autotune = _module
finetune = _module
gradient_checkpointing = _module
inference = _module
matmul_utils_4bit = _module
model_attn_mlp_patch = _module
model_server = _module
server = _module
monkeypatch = _module
gptq_for_llala_lora_monkey_patch = _module
llama_attn_hijack_xformers = _module
llama_flash_attn_monkey_patch = _module
peft_tuners_lora_monkey_patch = _module
run_server = _module
custom_model_server_monkey_patch = _module
custom_monkey_patch = _module
generate_monkey_patch = _module
train_data = _module
triton_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, time, torch, torchaudio, torchvision, triton, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import logging


import torch


import torch.nn as nn


import time


import math


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from typing import Dict


import triton


import triton.language as tl

