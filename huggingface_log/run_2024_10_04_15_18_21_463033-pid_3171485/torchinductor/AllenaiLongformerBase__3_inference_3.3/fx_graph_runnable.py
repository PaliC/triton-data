
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
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.debug_partitioner = True



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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
        view = torch.ops.aten.view.default(arg0_1, [4096, 768]);  arg0_1 = None
        permute = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm = torch.ops.aten.addmm.default(arg2_1, view, permute);  arg2_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [4, 1024, 768]);  addmm = None
        mul = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1 = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476);  view_1 = None
        erf = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2 = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        var_mean = torch.ops.aten.var_mean.correction(mul_2, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  mul_2 = getitem_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_4, arg4_1);  mul_4 = arg4_1 = None
        view_2 = torch.ops.aten.view.default(add_2, [4096, 768]);  add_2 = None
        permute_1 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        full_default_2 = torch.ops.aten.full.default([768, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_1, full_default_2], 1);  permute_1 = full_default_2 = None
        full_default_3 = torch.ops.aten.full.default([3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default_1 = torch.ops.aten.cat.default([arg6_1, full_default_3]);  arg6_1 = full_default_3 = None
        addmm_default = torch.ops.aten.addmm.default(cat_default_1, view_2, cat_default);  cat_default_1 = view_2 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -3);  addmm_default = None
        view_3 = torch.ops.aten.view.default(slice_tensor, [4, 1024, 50265]);  slice_tensor = None
        view_4 = torch.ops.aten.view.default(view_3, [-1, 50265])
        view_5 = torch.ops.aten.view.default(arg7_1, [-1]);  arg7_1 = None
        amax = torch.ops.aten.amax.default(view_4, [1], True)
        sub_1 = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
        exp = torch.ops.aten.exp.default(sub_1)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        ne = torch.ops.aten.ne.Scalar(view_5, -100)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, view_5, full_default);  ne = full_default = None
        unsqueeze = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_2, 1, unsqueeze);  sub_2 = unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_5, -100)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_5, -100);  view_5 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
        return (div, view_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf0, (4, 1024, 768), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 154414080, device=device(type='cuda', index=0))
    reader.tensor(buf5, (50265, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 201060, device=device(type='cuda', index=0))
    reader.tensor(buf6, (50265,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf7, (4, 1024), dtype=torch.int64, is_leaf=True)  # arg7_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)