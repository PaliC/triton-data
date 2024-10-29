import sys
_module = sys.modules[__name__]
del sys
conftest = _module
dot = _module
sum = _module
vec_add = _module
setup = _module
triton_viz = _module
analysis = _module
data = _module
draw = _module
interface = _module
interpreter = _module
tooltip = _module
trace = _module

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


import torch


import numpy as np


import triton


import triton.language as tl


from typing import List


from typing import Tuple


from typing import Any


import numpy.typing as npt


import inspect


from triton.runtime.interpreter import GridExecutor


from triton.runtime.interpreter import _implicit_cvt


from triton.runtime.interpreter import RESERVED_KWS


from triton.runtime.interpreter import interpreter_builder


from triton.runtime.interpreter import InterpretedFunction


from triton.runtime.interpreter import _patch_lang as triton_patch_lang


from triton.runtime import JITFunction


from typing import Optional


from functools import wraps


from triton.runtime import KernelInterface


from triton import JITFunction

