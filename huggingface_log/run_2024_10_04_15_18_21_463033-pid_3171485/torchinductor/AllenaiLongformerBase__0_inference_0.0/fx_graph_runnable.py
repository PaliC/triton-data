
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.0a0+git5380214
# torch cuda version: 12.1
# torch git version: 5380214107813f63c7c59f477487d5447085b45a


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA H100 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        full = torch.ops.aten.full.default([4, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default = torch.ops.aten.full.default([4, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default_1 = torch.ops.aten.full.default([4, 1, 1, 1024], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select = torch.ops.aten.select.int(full_default_1, 1, 0);  full_default_1 = None
        select_1 = torch.ops.aten.select.int(select, 1, 0);  select = None
        ne = torch.ops.aten.ne.Scalar(arg0_1, 1)
        convert_element_type = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        cumsum = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type);  convert_element_type_1 = convert_element_type = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg5_1, add, 1);  arg5_1 = add = None
        embedding_2 = torch.ops.aten.embedding.default(arg2_1, full_default);  arg2_1 = full_default = None
        add_1 = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, embedding_2);  add_1 = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_3, arg4_1);  mul_3 = arg4_1 = None
        return (add_4, select_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 1024), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 154414080, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50265, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 12589056, device=device(type='cuda', index=0))
    reader.tensor(buf5, (4098, 768), is_leaf=True)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)