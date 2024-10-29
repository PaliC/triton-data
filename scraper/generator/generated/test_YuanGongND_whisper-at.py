import sys
_module = sys.modules[__name__]
del sys
app = _module
setup = _module
whisper_at = _module
at_post_processing = _module
audio = _module
decoding = _module
model = _module
normalizers = _module
basic = _module
english = _module
timing = _module
tokenizer = _module
transcribe = _module
triton_ops = _module
utils = _module
version = _module
whisper_transcribe_test_simple = _module
compute_wer = _module
compute_wer_cla = _module
gen_noisy_speech = _module
transcribe_esc_hubert_xl = _module
transcribe_hubert_large = _module
transcribe_wav2vec_base = _module
transcribe_wav2vec_robust = _module
transcribe_whisper = _module
baseline_sound_classification = _module
extract_as_full_whisper_all = _module
extract_esc50_hubert_xl_all_pool = _module
extract_esc50_w2v_robust_all = _module
extract_esc50_whisper_all_pool = _module
test_audio = _module
test_normalizer = _module
test_tokenizer = _module
test_transcribe = _module
whisper = _module
plot_figure1_lower = _module
plot_figure1_upper = _module
plot_figure2 = _module
plot_figure3 = _module
dataloader_feat = _module
gen_weight_file = _module
models = _module
run = _module
traintest = _module
utilities = _module
compute_flops = _module
compute_mAP = _module
rename_state_dict = _module
stats = _module
util = _module
whisper_at_as_eval = _module

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


from functools import lru_cache


import numpy as np


import torch


@triton.jit
def dtw_kernel(cost, trace, x, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: 'tl.constexpr'):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    for k in range(1, N + M + 1):
        tl.debug_barrier()
        p0 = cost + (k - 1) * cost_stride
        p1 = cost + k * cost_stride
        p2 = cost + k * cost_stride + 1
        c0 = tl.load(p0 + offsets, mask=mask)
        c1 = tl.load(p1 + offsets, mask=mask)
        c2 = tl.load(p2 + offsets, mask=mask)
        x_row = tl.load(x + (k - 1) * x_stride + offsets, mask=mask, other=0)
        cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)
        cost_ptr = cost + (k + 1) * cost_stride + 1
        tl.store(cost_ptr + offsets, cost_row, mask=mask)
        trace_ptr = trace + (k + 1) * trace_stride + 1
        tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
        tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
        tl.store(trace_ptr + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))

