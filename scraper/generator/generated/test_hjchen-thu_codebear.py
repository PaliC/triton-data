import sys
_module = sys.modules[__name__]
del sys
quantize = _module
serving = _module
src = _module
globals = _module
kvcache_model = _module
LlamaGPTQ = _module
QuantLinear = _module
model = _module
custom_autotune = _module
fused_base = _module
fused_llama_attn = _module
fused_llama_mlp = _module
kernels = _module
utils = _module
speculative_sampling = _module

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


from typing import Dict


from typing import Optional


from typing import Union


import torch


import torch.nn as nn


import math


import numpy as np


from logging import getLogger


from torch.cuda.amp import custom_fwd


import time


import triton


from abc import abstractmethod


import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

