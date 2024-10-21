
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

torch._inductor.config.fallback_random = True
torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.6.0.dev20241021+cu118
# torch cuda version: 11.8
# torch git version: 5553778a0095e7234b2cd0874c2ff4dcc0216323


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA H100 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1):
        full_default = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_1 = torch.ops.aten.view.default(iota, [1, -1]);  iota = None
        iota_1 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_2 = torch.ops.aten.view.default(iota_1, [-1, 1]);  iota_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(view_1, view_2);  view_1 = view_2 = None
        repeat = torch.ops.aten.repeat.default(sub_1, [14, 14])
        unsqueeze = torch.ops.aten.unsqueeze.default(sub_1, 1);  sub_1 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze, [14, 14, 14]);  unsqueeze = None
        clone_2 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_3 = torch.ops.aten.view.default(clone_2, [196, 14]);  clone_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_3, 2);  view_3 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_1, [196, 14, 14]);  unsqueeze_1 = None
        clone_3 = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_4 = torch.ops.aten.view.default(clone_3, [196, 196]);  clone_3 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(repeat, 2)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_4, 2)
        add_3 = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        select = torch.ops.aten.select.int(full_default, 3, 2)
        copy = torch.ops.aten.copy.default(select, unsqueeze_2);  select = unsqueeze_2 = None
        select_scatter = torch.ops.aten.select_scatter.default(full_default, copy, 3, 2);  full_default = copy = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
        select_3 = torch.ops.aten.select.int(select_scatter, 3, 1)
        copy_1 = torch.ops.aten.copy.default(select_3, unsqueeze_3);  select_3 = unsqueeze_3 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(select_scatter, copy_1, 3, 1);  select_scatter = copy_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(repeat, 0);  repeat = None
        select_6 = torch.ops.aten.select.int(select_scatter_1, 3, 0)
        copy_2 = torch.ops.aten.copy.default(select_6, unsqueeze_4);  select_6 = unsqueeze_4 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(select_scatter_1, copy_2, 3, 0);  select_scatter_1 = copy_2 = None
        device_put = torch.ops.prims.device_put.default(select_scatter_2, device(type='cuda', index=0));  select_scatter_2 = None
        full_default_1 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_2 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_27 = torch.ops.aten.view.default(iota_2, [1, -1]);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_28 = torch.ops.aten.view.default(iota_3, [-1, 1]);  iota_3 = None
        sub_7 = torch.ops.aten.sub.Tensor(view_27, view_28);  view_27 = view_28 = None
        repeat_1 = torch.ops.aten.repeat.default(sub_7, [14, 14])
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(sub_7, 1);  sub_7 = None
        expand_8 = torch.ops.aten.expand.default(unsqueeze_6, [14, 14, 14]);  unsqueeze_6 = None
        clone_16 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_29 = torch.ops.aten.view.default(clone_16, [196, 14]);  clone_16 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(view_29, 2);  view_29 = None
        expand_9 = torch.ops.aten.expand.default(unsqueeze_7, [196, 14, 14]);  unsqueeze_7 = None
        clone_17 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_30 = torch.ops.aten.view.default(clone_17, [196, 196]);  clone_17 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(repeat_1, 2)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_30, 2)
        add_13 = torch.ops.aten.add.Tensor(pow_3, pow_4);  pow_3 = pow_4 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(add_13, 0);  add_13 = None
        select_10 = torch.ops.aten.select.int(full_default_1, 3, 2)
        copy_3 = torch.ops.aten.copy.default(select_10, unsqueeze_8);  select_10 = unsqueeze_8 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(full_default_1, copy_3, 3, 2);  full_default_1 = copy_3 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
        select_13 = torch.ops.aten.select.int(select_scatter_3, 3, 1)
        copy_4 = torch.ops.aten.copy.default(select_13, unsqueeze_9);  select_13 = unsqueeze_9 = None
        select_scatter_4 = torch.ops.aten.select_scatter.default(select_scatter_3, copy_4, 3, 1);  select_scatter_3 = copy_4 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(repeat_1, 0);  repeat_1 = None
        select_16 = torch.ops.aten.select.int(select_scatter_4, 3, 0)
        copy_5 = torch.ops.aten.copy.default(select_16, unsqueeze_10);  select_16 = unsqueeze_10 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(select_scatter_4, copy_5, 3, 0);  select_scatter_4 = copy_5 = None
        device_put_1 = torch.ops.prims.device_put.default(select_scatter_5, device(type='cuda', index=0));  select_scatter_5 = None
        full_default_2 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_4 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_53 = torch.ops.aten.view.default(iota_4, [1, -1]);  iota_4 = None
        iota_5 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_54 = torch.ops.aten.view.default(iota_5, [-1, 1]);  iota_5 = None
        sub_13 = torch.ops.aten.sub.Tensor(view_53, view_54);  view_53 = view_54 = None
        repeat_2 = torch.ops.aten.repeat.default(sub_13, [14, 14])
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(sub_13, 1);  sub_13 = None
        expand_15 = torch.ops.aten.expand.default(unsqueeze_12, [14, 14, 14]);  unsqueeze_12 = None
        clone_30 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_55 = torch.ops.aten.view.default(clone_30, [196, 14]);  clone_30 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(view_55, 2);  view_55 = None
        expand_16 = torch.ops.aten.expand.default(unsqueeze_13, [196, 14, 14]);  unsqueeze_13 = None
        clone_31 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_56 = torch.ops.aten.view.default(clone_31, [196, 196]);  clone_31 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(repeat_2, 2)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_56, 2)
        add_23 = torch.ops.aten.add.Tensor(pow_5, pow_6);  pow_5 = pow_6 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(add_23, 0);  add_23 = None
        select_20 = torch.ops.aten.select.int(full_default_2, 3, 2)
        copy_6 = torch.ops.aten.copy.default(select_20, unsqueeze_14);  select_20 = unsqueeze_14 = None
        select_scatter_6 = torch.ops.aten.select_scatter.default(full_default_2, copy_6, 3, 2);  full_default_2 = copy_6 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(view_56, 0);  view_56 = None
        select_23 = torch.ops.aten.select.int(select_scatter_6, 3, 1)
        copy_7 = torch.ops.aten.copy.default(select_23, unsqueeze_15);  select_23 = unsqueeze_15 = None
        select_scatter_7 = torch.ops.aten.select_scatter.default(select_scatter_6, copy_7, 3, 1);  select_scatter_6 = copy_7 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(repeat_2, 0);  repeat_2 = None
        select_26 = torch.ops.aten.select.int(select_scatter_7, 3, 0)
        copy_8 = torch.ops.aten.copy.default(select_26, unsqueeze_16);  select_26 = unsqueeze_16 = None
        select_scatter_8 = torch.ops.aten.select_scatter.default(select_scatter_7, copy_8, 3, 0);  select_scatter_7 = copy_8 = None
        device_put_2 = torch.ops.prims.device_put.default(select_scatter_8, device(type='cuda', index=0));  select_scatter_8 = None
        full_default_3 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_6 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_79 = torch.ops.aten.view.default(iota_6, [1, -1]);  iota_6 = None
        iota_7 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_80 = torch.ops.aten.view.default(iota_7, [-1, 1]);  iota_7 = None
        sub_19 = torch.ops.aten.sub.Tensor(view_79, view_80);  view_79 = view_80 = None
        repeat_3 = torch.ops.aten.repeat.default(sub_19, [14, 14])
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(sub_19, 1);  sub_19 = None
        expand_22 = torch.ops.aten.expand.default(unsqueeze_18, [14, 14, 14]);  unsqueeze_18 = None
        clone_44 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_81 = torch.ops.aten.view.default(clone_44, [196, 14]);  clone_44 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(view_81, 2);  view_81 = None
        expand_23 = torch.ops.aten.expand.default(unsqueeze_19, [196, 14, 14]);  unsqueeze_19 = None
        clone_45 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_82 = torch.ops.aten.view.default(clone_45, [196, 196]);  clone_45 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(repeat_3, 2)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(view_82, 2)
        add_33 = torch.ops.aten.add.Tensor(pow_7, pow_8);  pow_7 = pow_8 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(add_33, 0);  add_33 = None
        select_30 = torch.ops.aten.select.int(full_default_3, 3, 2)
        copy_9 = torch.ops.aten.copy.default(select_30, unsqueeze_20);  select_30 = unsqueeze_20 = None
        select_scatter_9 = torch.ops.aten.select_scatter.default(full_default_3, copy_9, 3, 2);  full_default_3 = copy_9 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(view_82, 0);  view_82 = None
        select_33 = torch.ops.aten.select.int(select_scatter_9, 3, 1)
        copy_10 = torch.ops.aten.copy.default(select_33, unsqueeze_21);  select_33 = unsqueeze_21 = None
        select_scatter_10 = torch.ops.aten.select_scatter.default(select_scatter_9, copy_10, 3, 1);  select_scatter_9 = copy_10 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(repeat_3, 0);  repeat_3 = None
        select_36 = torch.ops.aten.select.int(select_scatter_10, 3, 0)
        copy_11 = torch.ops.aten.copy.default(select_36, unsqueeze_22);  select_36 = unsqueeze_22 = None
        select_scatter_11 = torch.ops.aten.select_scatter.default(select_scatter_10, copy_11, 3, 0);  select_scatter_10 = copy_11 = None
        device_put_3 = torch.ops.prims.device_put.default(select_scatter_11, device(type='cuda', index=0));  select_scatter_11 = None
        full_default_4 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_8 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_105 = torch.ops.aten.view.default(iota_8, [1, -1]);  iota_8 = None
        iota_9 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_106 = torch.ops.aten.view.default(iota_9, [-1, 1]);  iota_9 = None
        sub_25 = torch.ops.aten.sub.Tensor(view_105, view_106);  view_105 = view_106 = None
        repeat_4 = torch.ops.aten.repeat.default(sub_25, [14, 14])
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(sub_25, 1);  sub_25 = None
        expand_29 = torch.ops.aten.expand.default(unsqueeze_24, [14, 14, 14]);  unsqueeze_24 = None
        clone_58 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_107 = torch.ops.aten.view.default(clone_58, [196, 14]);  clone_58 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(view_107, 2);  view_107 = None
        expand_30 = torch.ops.aten.expand.default(unsqueeze_25, [196, 14, 14]);  unsqueeze_25 = None
        clone_59 = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_108 = torch.ops.aten.view.default(clone_59, [196, 196]);  clone_59 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(repeat_4, 2)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(view_108, 2)
        add_43 = torch.ops.aten.add.Tensor(pow_9, pow_10);  pow_9 = pow_10 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(add_43, 0);  add_43 = None
        select_40 = torch.ops.aten.select.int(full_default_4, 3, 2)
        copy_12 = torch.ops.aten.copy.default(select_40, unsqueeze_26);  select_40 = unsqueeze_26 = None
        select_scatter_12 = torch.ops.aten.select_scatter.default(full_default_4, copy_12, 3, 2);  full_default_4 = copy_12 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(view_108, 0);  view_108 = None
        select_43 = torch.ops.aten.select.int(select_scatter_12, 3, 1)
        copy_13 = torch.ops.aten.copy.default(select_43, unsqueeze_27);  select_43 = unsqueeze_27 = None
        select_scatter_13 = torch.ops.aten.select_scatter.default(select_scatter_12, copy_13, 3, 1);  select_scatter_12 = copy_13 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(repeat_4, 0);  repeat_4 = None
        select_46 = torch.ops.aten.select.int(select_scatter_13, 3, 0)
        copy_14 = torch.ops.aten.copy.default(select_46, unsqueeze_28);  select_46 = unsqueeze_28 = None
        select_scatter_14 = torch.ops.aten.select_scatter.default(select_scatter_13, copy_14, 3, 0);  select_scatter_13 = copy_14 = None
        device_put_4 = torch.ops.prims.device_put.default(select_scatter_14, device(type='cuda', index=0));  select_scatter_14 = None
        full_default_5 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_10 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_131 = torch.ops.aten.view.default(iota_10, [1, -1]);  iota_10 = None
        iota_11 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_132 = torch.ops.aten.view.default(iota_11, [-1, 1]);  iota_11 = None
        sub_31 = torch.ops.aten.sub.Tensor(view_131, view_132);  view_131 = view_132 = None
        repeat_5 = torch.ops.aten.repeat.default(sub_31, [14, 14])
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(sub_31, 1);  sub_31 = None
        expand_36 = torch.ops.aten.expand.default(unsqueeze_30, [14, 14, 14]);  unsqueeze_30 = None
        clone_72 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_133 = torch.ops.aten.view.default(clone_72, [196, 14]);  clone_72 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(view_133, 2);  view_133 = None
        expand_37 = torch.ops.aten.expand.default(unsqueeze_31, [196, 14, 14]);  unsqueeze_31 = None
        clone_73 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_134 = torch.ops.aten.view.default(clone_73, [196, 196]);  clone_73 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(repeat_5, 2)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_134, 2)
        add_53 = torch.ops.aten.add.Tensor(pow_11, pow_12);  pow_11 = pow_12 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(add_53, 0);  add_53 = None
        select_50 = torch.ops.aten.select.int(full_default_5, 3, 2)
        copy_15 = torch.ops.aten.copy.default(select_50, unsqueeze_32);  select_50 = unsqueeze_32 = None
        select_scatter_15 = torch.ops.aten.select_scatter.default(full_default_5, copy_15, 3, 2);  full_default_5 = copy_15 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(view_134, 0);  view_134 = None
        select_53 = torch.ops.aten.select.int(select_scatter_15, 3, 1)
        copy_16 = torch.ops.aten.copy.default(select_53, unsqueeze_33);  select_53 = unsqueeze_33 = None
        select_scatter_16 = torch.ops.aten.select_scatter.default(select_scatter_15, copy_16, 3, 1);  select_scatter_15 = copy_16 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(repeat_5, 0);  repeat_5 = None
        select_56 = torch.ops.aten.select.int(select_scatter_16, 3, 0)
        copy_17 = torch.ops.aten.copy.default(select_56, unsqueeze_34);  select_56 = unsqueeze_34 = None
        select_scatter_17 = torch.ops.aten.select_scatter.default(select_scatter_16, copy_17, 3, 0);  select_scatter_16 = copy_17 = None
        device_put_5 = torch.ops.prims.device_put.default(select_scatter_17, device(type='cuda', index=0));  select_scatter_17 = None
        full_default_6 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_12 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_157 = torch.ops.aten.view.default(iota_12, [1, -1]);  iota_12 = None
        iota_13 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_158 = torch.ops.aten.view.default(iota_13, [-1, 1]);  iota_13 = None
        sub_37 = torch.ops.aten.sub.Tensor(view_157, view_158);  view_157 = view_158 = None
        repeat_6 = torch.ops.aten.repeat.default(sub_37, [14, 14])
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(sub_37, 1);  sub_37 = None
        expand_43 = torch.ops.aten.expand.default(unsqueeze_36, [14, 14, 14]);  unsqueeze_36 = None
        clone_86 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_159 = torch.ops.aten.view.default(clone_86, [196, 14]);  clone_86 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(view_159, 2);  view_159 = None
        expand_44 = torch.ops.aten.expand.default(unsqueeze_37, [196, 14, 14]);  unsqueeze_37 = None
        clone_87 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_160 = torch.ops.aten.view.default(clone_87, [196, 196]);  clone_87 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(repeat_6, 2)
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(view_160, 2)
        add_63 = torch.ops.aten.add.Tensor(pow_13, pow_14);  pow_13 = pow_14 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(add_63, 0);  add_63 = None
        select_60 = torch.ops.aten.select.int(full_default_6, 3, 2)
        copy_18 = torch.ops.aten.copy.default(select_60, unsqueeze_38);  select_60 = unsqueeze_38 = None
        select_scatter_18 = torch.ops.aten.select_scatter.default(full_default_6, copy_18, 3, 2);  full_default_6 = copy_18 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(view_160, 0);  view_160 = None
        select_63 = torch.ops.aten.select.int(select_scatter_18, 3, 1)
        copy_19 = torch.ops.aten.copy.default(select_63, unsqueeze_39);  select_63 = unsqueeze_39 = None
        select_scatter_19 = torch.ops.aten.select_scatter.default(select_scatter_18, copy_19, 3, 1);  select_scatter_18 = copy_19 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(repeat_6, 0);  repeat_6 = None
        select_66 = torch.ops.aten.select.int(select_scatter_19, 3, 0)
        copy_20 = torch.ops.aten.copy.default(select_66, unsqueeze_40);  select_66 = unsqueeze_40 = None
        select_scatter_20 = torch.ops.aten.select_scatter.default(select_scatter_19, copy_20, 3, 0);  select_scatter_19 = copy_20 = None
        device_put_6 = torch.ops.prims.device_put.default(select_scatter_20, device(type='cuda', index=0));  select_scatter_20 = None
        full_default_7 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_14 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_183 = torch.ops.aten.view.default(iota_14, [1, -1]);  iota_14 = None
        iota_15 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_184 = torch.ops.aten.view.default(iota_15, [-1, 1]);  iota_15 = None
        sub_43 = torch.ops.aten.sub.Tensor(view_183, view_184);  view_183 = view_184 = None
        repeat_7 = torch.ops.aten.repeat.default(sub_43, [14, 14])
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(sub_43, 1);  sub_43 = None
        expand_50 = torch.ops.aten.expand.default(unsqueeze_42, [14, 14, 14]);  unsqueeze_42 = None
        clone_100 = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_185 = torch.ops.aten.view.default(clone_100, [196, 14]);  clone_100 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(view_185, 2);  view_185 = None
        expand_51 = torch.ops.aten.expand.default(unsqueeze_43, [196, 14, 14]);  unsqueeze_43 = None
        clone_101 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_186 = torch.ops.aten.view.default(clone_101, [196, 196]);  clone_101 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(repeat_7, 2)
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(view_186, 2)
        add_73 = torch.ops.aten.add.Tensor(pow_15, pow_16);  pow_15 = pow_16 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(add_73, 0);  add_73 = None
        select_70 = torch.ops.aten.select.int(full_default_7, 3, 2)
        copy_21 = torch.ops.aten.copy.default(select_70, unsqueeze_44);  select_70 = unsqueeze_44 = None
        select_scatter_21 = torch.ops.aten.select_scatter.default(full_default_7, copy_21, 3, 2);  full_default_7 = copy_21 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(view_186, 0);  view_186 = None
        select_73 = torch.ops.aten.select.int(select_scatter_21, 3, 1)
        copy_22 = torch.ops.aten.copy.default(select_73, unsqueeze_45);  select_73 = unsqueeze_45 = None
        select_scatter_22 = torch.ops.aten.select_scatter.default(select_scatter_21, copy_22, 3, 1);  select_scatter_21 = copy_22 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(repeat_7, 0);  repeat_7 = None
        select_76 = torch.ops.aten.select.int(select_scatter_22, 3, 0)
        copy_23 = torch.ops.aten.copy.default(select_76, unsqueeze_46);  select_76 = unsqueeze_46 = None
        select_scatter_23 = torch.ops.aten.select_scatter.default(select_scatter_22, copy_23, 3, 0);  select_scatter_22 = copy_23 = None
        device_put_7 = torch.ops.prims.device_put.default(select_scatter_23, device(type='cuda', index=0));  select_scatter_23 = None
        full_default_8 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_16 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_209 = torch.ops.aten.view.default(iota_16, [1, -1]);  iota_16 = None
        iota_17 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_210 = torch.ops.aten.view.default(iota_17, [-1, 1]);  iota_17 = None
        sub_49 = torch.ops.aten.sub.Tensor(view_209, view_210);  view_209 = view_210 = None
        repeat_8 = torch.ops.aten.repeat.default(sub_49, [14, 14])
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(sub_49, 1);  sub_49 = None
        expand_57 = torch.ops.aten.expand.default(unsqueeze_48, [14, 14, 14]);  unsqueeze_48 = None
        clone_114 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_211 = torch.ops.aten.view.default(clone_114, [196, 14]);  clone_114 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(view_211, 2);  view_211 = None
        expand_58 = torch.ops.aten.expand.default(unsqueeze_49, [196, 14, 14]);  unsqueeze_49 = None
        clone_115 = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_212 = torch.ops.aten.view.default(clone_115, [196, 196]);  clone_115 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(repeat_8, 2)
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(view_212, 2)
        add_83 = torch.ops.aten.add.Tensor(pow_17, pow_18);  pow_17 = pow_18 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(add_83, 0);  add_83 = None
        select_80 = torch.ops.aten.select.int(full_default_8, 3, 2)
        copy_24 = torch.ops.aten.copy.default(select_80, unsqueeze_50);  select_80 = unsqueeze_50 = None
        select_scatter_24 = torch.ops.aten.select_scatter.default(full_default_8, copy_24, 3, 2);  full_default_8 = copy_24 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(view_212, 0);  view_212 = None
        select_83 = torch.ops.aten.select.int(select_scatter_24, 3, 1)
        copy_25 = torch.ops.aten.copy.default(select_83, unsqueeze_51);  select_83 = unsqueeze_51 = None
        select_scatter_25 = torch.ops.aten.select_scatter.default(select_scatter_24, copy_25, 3, 1);  select_scatter_24 = copy_25 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(repeat_8, 0);  repeat_8 = None
        select_86 = torch.ops.aten.select.int(select_scatter_25, 3, 0)
        copy_26 = torch.ops.aten.copy.default(select_86, unsqueeze_52);  select_86 = unsqueeze_52 = None
        select_scatter_26 = torch.ops.aten.select_scatter.default(select_scatter_25, copy_26, 3, 0);  select_scatter_25 = copy_26 = None
        device_put_8 = torch.ops.prims.device_put.default(select_scatter_26, device(type='cuda', index=0));  select_scatter_26 = None
        full_default_9 = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_18 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_235 = torch.ops.aten.view.default(iota_18, [1, -1]);  iota_18 = None
        iota_19 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_236 = torch.ops.aten.view.default(iota_19, [-1, 1]);  iota_19 = None
        sub_55 = torch.ops.aten.sub.Tensor(view_235, view_236);  view_235 = view_236 = None
        repeat_9 = torch.ops.aten.repeat.default(sub_55, [14, 14])
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(sub_55, 1);  sub_55 = None
        expand_64 = torch.ops.aten.expand.default(unsqueeze_54, [14, 14, 14]);  unsqueeze_54 = None
        clone_128 = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_237 = torch.ops.aten.view.default(clone_128, [196, 14]);  clone_128 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(view_237, 2);  view_237 = None
        expand_65 = torch.ops.aten.expand.default(unsqueeze_55, [196, 14, 14]);  unsqueeze_55 = None
        clone_129 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_238 = torch.ops.aten.view.default(clone_129, [196, 196]);  clone_129 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(repeat_9, 2)
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(view_238, 2)
        add_93 = torch.ops.aten.add.Tensor(pow_19, pow_20);  pow_19 = pow_20 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(add_93, 0);  add_93 = None
        select_90 = torch.ops.aten.select.int(full_default_9, 3, 2)
        copy_27 = torch.ops.aten.copy.default(select_90, unsqueeze_56);  select_90 = unsqueeze_56 = None
        select_scatter_27 = torch.ops.aten.select_scatter.default(full_default_9, copy_27, 3, 2);  full_default_9 = copy_27 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(view_238, 0);  view_238 = None
        select_93 = torch.ops.aten.select.int(select_scatter_27, 3, 1)
        copy_28 = torch.ops.aten.copy.default(select_93, unsqueeze_57);  select_93 = unsqueeze_57 = None
        select_scatter_28 = torch.ops.aten.select_scatter.default(select_scatter_27, copy_28, 3, 1);  select_scatter_27 = copy_28 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(repeat_9, 0);  repeat_9 = None
        select_96 = torch.ops.aten.select.int(select_scatter_28, 3, 0)
        copy_29 = torch.ops.aten.copy.default(select_96, unsqueeze_58);  select_96 = unsqueeze_58 = None
        select_scatter_29 = torch.ops.aten.select_scatter.default(select_scatter_28, copy_29, 3, 0);  select_scatter_28 = copy_29 = None
        device_put_9 = torch.ops.prims.device_put.default(select_scatter_29, device(type='cuda', index=0));  select_scatter_29 = None
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_293 = torch.ops.aten.view.default(convolution_1, [8, 768, 196]);  convolution_1 = None
        permute_126 = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
        add_117 = torch.ops.aten.add.Tensor(permute_126, arg3_1);  permute_126 = arg3_1 = None
        expand_79 = torch.ops.aten.expand.default(arg4_1, [8, -1, -1]);  arg4_1 = None
        clone_159 = torch.ops.aten.clone.default(add_117, memory_format = torch.contiguous_format)
        var_mean_25 = torch.ops.aten.var_mean.correction(clone_159, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_25[0]
        getitem_57 = var_mean_25[1];  var_mean_25 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_67 = torch.ops.aten.sub.Tensor(clone_159, getitem_57);  clone_159 = getitem_57 = None
        mul_118 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_25);  sub_67 = rsqrt_25 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, arg5_1);  mul_118 = arg5_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_119, arg6_1);  mul_119 = arg6_1 = None
        permute_127 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        view_294 = torch.ops.aten.view.default(add_119, [1568, 768])
        mm_32 = torch.ops.aten.mm.default(view_294, permute_127);  view_294 = permute_127 = None
        view_295 = torch.ops.aten.view.default(mm_32, [8, 196, 1536]);  mm_32 = None
        view_296 = torch.ops.aten.view.default(view_295, [8, 196, 2, 16, 48]);  view_295 = None
        permute_128 = torch.ops.aten.permute.default(view_296, [2, 0, 3, 1, 4]);  view_296 = None
        select_101 = torch.ops.aten.select.int(permute_128, 0, 0)
        select_102 = torch.ops.aten.select.int(permute_128, 0, 1);  permute_128 = None
        expand_80 = torch.ops.aten.expand.default(device_put, [8, -1, -1, -1])
        permute_129 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        clone_160 = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_297 = torch.ops.aten.view.default(clone_160, [307328, 3]);  clone_160 = None
        mm_33 = torch.ops.aten.mm.default(view_297, permute_129);  view_297 = permute_129 = None
        view_298 = torch.ops.aten.view.default(mm_33, [8, 196, 196, 16]);  mm_33 = None
        add_120 = torch.ops.aten.add.Tensor(view_298, arg9_1);  view_298 = arg9_1 = None
        permute_130 = torch.ops.aten.permute.default(add_120, [0, 3, 1, 2]);  add_120 = None
        permute_131 = torch.ops.aten.permute.default(select_102, [0, 1, 3, 2]);  select_102 = None
        expand_81 = torch.ops.aten.expand.default(select_101, [8, 16, 196, 48]);  select_101 = None
        clone_161 = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_299 = torch.ops.aten.view.default(clone_161, [128, 196, 48]);  clone_161 = None
        expand_82 = torch.ops.aten.expand.default(permute_131, [8, 16, 48, 196]);  permute_131 = None
        clone_162 = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_300 = torch.ops.aten.view.default(clone_162, [128, 48, 196]);  clone_162 = None
        bmm_24 = torch.ops.aten.bmm.default(view_299, view_300);  view_299 = view_300 = None
        view_301 = torch.ops.aten.view.default(bmm_24, [8, 16, 196, 196]);  bmm_24 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(view_301, 1);  view_301 = None
        amax_default_11 = torch.ops.aten.amax.default(mul_tensor_22, [-1], True)
        sub_tensor_11 = torch.ops.aten.sub.Tensor(mul_tensor_22, amax_default_11);  mul_tensor_22 = amax_default_11 = None
        mul_tensor_23 = torch.ops.aten.mul.Tensor(sub_tensor_11, 0.14433756729740643);  sub_tensor_11 = None
        exp_22 = torch.ops.aten.exp.default(mul_tensor_23);  mul_tensor_23 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_32 = torch.ops.aten.div.Tensor(exp_22, sum_33);  exp_22 = sum_33 = None
        clone_163 = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        amax_23 = torch.ops.aten.amax.default(clone_163, [-1], True)
        sub_69 = torch.ops.aten.sub.Tensor(clone_163, amax_23);  clone_163 = amax_23 = None
        exp_23 = torch.ops.aten.exp.default(sub_69);  sub_69 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_23, sum_34);  exp_23 = sum_34 = None
        view_302 = torch.ops.aten.view.default(arg10_1, [1, -1, 1, 1]);  arg10_1 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(view_302)
        sub_70 = torch.ops.aten.sub.Tensor(1.0, sigmoid_20);  sigmoid_20 = None
        mul_121 = torch.ops.aten.mul.Tensor(sub_70, div_32);  sub_70 = div_32 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(view_302);  view_302 = None
        mul_122 = torch.ops.aten.mul.Tensor(sigmoid_21, div_33);  sigmoid_21 = div_33 = None
        add_121 = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(add_121, [-1])
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(sum_35, -1);  sum_35 = None
        div_34 = torch.ops.aten.div.Tensor(add_121, unsqueeze_60);  add_121 = unsqueeze_60 = None
        permute_132 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        view_303 = torch.ops.aten.view.default(add_119, [1568, 768]);  add_119 = None
        mm_34 = torch.ops.aten.mm.default(view_303, permute_132);  view_303 = permute_132 = None
        view_304 = torch.ops.aten.view.default(mm_34, [8, 196, 768]);  mm_34 = None
        view_305 = torch.ops.aten.view.default(view_304, [8, 196, 16, 48]);  view_304 = None
        permute_133 = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        expand_83 = torch.ops.aten.expand.default(div_34, [8, 16, 196, 196]);  div_34 = None
        view_306 = torch.ops.aten.view.default(expand_83, [128, 196, 196]);  expand_83 = None
        expand_84 = torch.ops.aten.expand.default(permute_133, [8, 16, 196, 48]);  permute_133 = None
        clone_165 = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_307 = torch.ops.aten.view.default(clone_165, [128, 196, 48]);  clone_165 = None
        bmm_25 = torch.ops.aten.bmm.default(view_306, view_307);  view_306 = view_307 = None
        view_308 = torch.ops.aten.view.default(bmm_25, [8, 16, 196, 48]);  bmm_25 = None
        permute_134 = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_166 = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
        view_309 = torch.ops.aten.view.default(clone_166, [8, 196, 768]);  clone_166 = None
        view_310 = torch.ops.aten.view.default(view_309, [1568, 768]);  view_309 = None
        permute_135 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg13_1, view_310, permute_135);  arg13_1 = view_310 = permute_135 = None
        view_311 = torch.ops.aten.view.default(addmm_37, [8, 196, 768]);  addmm_37 = None
        add_122 = torch.ops.aten.add.Tensor(add_117, view_311);  add_117 = view_311 = None
        clone_168 = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
        var_mean_26 = torch.ops.aten.var_mean.correction(clone_168, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_26[0]
        getitem_59 = var_mean_26[1];  var_mean_26 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_71 = torch.ops.aten.sub.Tensor(clone_168, getitem_59);  clone_168 = getitem_59 = None
        mul_123 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_26);  sub_71 = rsqrt_26 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, arg14_1);  mul_123 = arg14_1 = None
        add_124 = torch.ops.aten.add.Tensor(mul_124, arg15_1);  mul_124 = arg15_1 = None
        view_312 = torch.ops.aten.view.default(add_124, [1568, 768]);  add_124 = None
        permute_136 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg17_1, view_312, permute_136);  arg17_1 = view_312 = permute_136 = None
        view_313 = torch.ops.aten.view.default(addmm_38, [8, 196, 3072]);  addmm_38 = None
        mul_125 = torch.ops.aten.mul.Tensor(view_313, 0.5)
        mul_126 = torch.ops.aten.mul.Tensor(view_313, 0.7071067811865476);  view_313 = None
        erf_12 = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_125 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_125, add_125);  mul_125 = add_125 = None
        view_314 = torch.ops.aten.view.default(mul_127, [1568, 3072]);  mul_127 = None
        permute_137 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg19_1, view_314, permute_137);  arg19_1 = view_314 = permute_137 = None
        view_315 = torch.ops.aten.view.default(addmm_39, [8, 196, 768]);  addmm_39 = None
        add_126 = torch.ops.aten.add.Tensor(add_122, view_315);  add_122 = view_315 = None
        clone_171 = torch.ops.aten.clone.default(add_126, memory_format = torch.contiguous_format)
        var_mean_27 = torch.ops.aten.var_mean.correction(clone_171, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_27[0]
        getitem_61 = var_mean_27[1];  var_mean_27 = None
        add_127 = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_72 = torch.ops.aten.sub.Tensor(clone_171, getitem_61);  clone_171 = getitem_61 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_27);  sub_72 = rsqrt_27 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, arg20_1);  mul_128 = arg20_1 = None
        add_128 = torch.ops.aten.add.Tensor(mul_129, arg21_1);  mul_129 = arg21_1 = None
        permute_138 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        view_316 = torch.ops.aten.view.default(add_128, [1568, 768])
        mm_35 = torch.ops.aten.mm.default(view_316, permute_138);  view_316 = permute_138 = None
        view_317 = torch.ops.aten.view.default(mm_35, [8, 196, 1536]);  mm_35 = None
        view_318 = torch.ops.aten.view.default(view_317, [8, 196, 2, 16, 48]);  view_317 = None
        permute_139 = torch.ops.aten.permute.default(view_318, [2, 0, 3, 1, 4]);  view_318 = None
        select_103 = torch.ops.aten.select.int(permute_139, 0, 0)
        select_104 = torch.ops.aten.select.int(permute_139, 0, 1);  permute_139 = None
        expand_85 = torch.ops.aten.expand.default(device_put_1, [8, -1, -1, -1])
        permute_140 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        clone_172 = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_319 = torch.ops.aten.view.default(clone_172, [307328, 3]);  clone_172 = None
        mm_36 = torch.ops.aten.mm.default(view_319, permute_140);  view_319 = permute_140 = None
        view_320 = torch.ops.aten.view.default(mm_36, [8, 196, 196, 16]);  mm_36 = None
        add_129 = torch.ops.aten.add.Tensor(view_320, arg24_1);  view_320 = arg24_1 = None
        permute_141 = torch.ops.aten.permute.default(add_129, [0, 3, 1, 2]);  add_129 = None
        permute_142 = torch.ops.aten.permute.default(select_104, [0, 1, 3, 2]);  select_104 = None
        expand_86 = torch.ops.aten.expand.default(select_103, [8, 16, 196, 48]);  select_103 = None
        clone_173 = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_321 = torch.ops.aten.view.default(clone_173, [128, 196, 48]);  clone_173 = None
        expand_87 = torch.ops.aten.expand.default(permute_142, [8, 16, 48, 196]);  permute_142 = None
        clone_174 = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_322 = torch.ops.aten.view.default(clone_174, [128, 48, 196]);  clone_174 = None
        bmm_26 = torch.ops.aten.bmm.default(view_321, view_322);  view_321 = view_322 = None
        view_323 = torch.ops.aten.view.default(bmm_26, [8, 16, 196, 196]);  bmm_26 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(view_323, 1);  view_323 = None
        amax_default_10 = torch.ops.aten.amax.default(mul_tensor_20, [-1], True)
        sub_tensor_10 = torch.ops.aten.sub.Tensor(mul_tensor_20, amax_default_10);  mul_tensor_20 = amax_default_10 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.14433756729740643);  sub_tensor_10 = None
        exp_24 = torch.ops.aten.exp.default(mul_tensor_21);  mul_tensor_21 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_35 = torch.ops.aten.div.Tensor(exp_24, sum_36);  exp_24 = sum_36 = None
        clone_175 = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        amax_25 = torch.ops.aten.amax.default(clone_175, [-1], True)
        sub_74 = torch.ops.aten.sub.Tensor(clone_175, amax_25);  clone_175 = amax_25 = None
        exp_25 = torch.ops.aten.exp.default(sub_74);  sub_74 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_36 = torch.ops.aten.div.Tensor(exp_25, sum_37);  exp_25 = sum_37 = None
        view_324 = torch.ops.aten.view.default(arg25_1, [1, -1, 1, 1]);  arg25_1 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(view_324)
        sub_75 = torch.ops.aten.sub.Tensor(1.0, sigmoid_22);  sigmoid_22 = None
        mul_131 = torch.ops.aten.mul.Tensor(sub_75, div_35);  sub_75 = div_35 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(view_324);  view_324 = None
        mul_132 = torch.ops.aten.mul.Tensor(sigmoid_23, div_36);  sigmoid_23 = div_36 = None
        add_130 = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(add_130, [-1])
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(sum_38, -1);  sum_38 = None
        div_37 = torch.ops.aten.div.Tensor(add_130, unsqueeze_61);  add_130 = unsqueeze_61 = None
        permute_143 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        view_325 = torch.ops.aten.view.default(add_128, [1568, 768]);  add_128 = None
        mm_37 = torch.ops.aten.mm.default(view_325, permute_143);  view_325 = permute_143 = None
        view_326 = torch.ops.aten.view.default(mm_37, [8, 196, 768]);  mm_37 = None
        view_327 = torch.ops.aten.view.default(view_326, [8, 196, 16, 48]);  view_326 = None
        permute_144 = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        expand_88 = torch.ops.aten.expand.default(div_37, [8, 16, 196, 196]);  div_37 = None
        view_328 = torch.ops.aten.view.default(expand_88, [128, 196, 196]);  expand_88 = None
        expand_89 = torch.ops.aten.expand.default(permute_144, [8, 16, 196, 48]);  permute_144 = None
        clone_177 = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_329 = torch.ops.aten.view.default(clone_177, [128, 196, 48]);  clone_177 = None
        bmm_27 = torch.ops.aten.bmm.default(view_328, view_329);  view_328 = view_329 = None
        view_330 = torch.ops.aten.view.default(bmm_27, [8, 16, 196, 48]);  bmm_27 = None
        permute_145 = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        clone_178 = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_331 = torch.ops.aten.view.default(clone_178, [8, 196, 768]);  clone_178 = None
        view_332 = torch.ops.aten.view.default(view_331, [1568, 768]);  view_331 = None
        permute_146 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg28_1, view_332, permute_146);  arg28_1 = view_332 = permute_146 = None
        view_333 = torch.ops.aten.view.default(addmm_40, [8, 196, 768]);  addmm_40 = None
        add_131 = torch.ops.aten.add.Tensor(add_126, view_333);  add_126 = view_333 = None
        clone_180 = torch.ops.aten.clone.default(add_131, memory_format = torch.contiguous_format)
        var_mean_28 = torch.ops.aten.var_mean.correction(clone_180, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_28[0]
        getitem_63 = var_mean_28[1];  var_mean_28 = None
        add_132 = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_76 = torch.ops.aten.sub.Tensor(clone_180, getitem_63);  clone_180 = getitem_63 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_28);  sub_76 = rsqrt_28 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, arg29_1);  mul_133 = arg29_1 = None
        add_133 = torch.ops.aten.add.Tensor(mul_134, arg30_1);  mul_134 = arg30_1 = None
        view_334 = torch.ops.aten.view.default(add_133, [1568, 768]);  add_133 = None
        permute_147 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg32_1, view_334, permute_147);  arg32_1 = view_334 = permute_147 = None
        view_335 = torch.ops.aten.view.default(addmm_41, [8, 196, 3072]);  addmm_41 = None
        mul_135 = torch.ops.aten.mul.Tensor(view_335, 0.5)
        mul_136 = torch.ops.aten.mul.Tensor(view_335, 0.7071067811865476);  view_335 = None
        erf_13 = torch.ops.aten.erf.default(mul_136);  mul_136 = None
        add_134 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_135, add_134);  mul_135 = add_134 = None
        view_336 = torch.ops.aten.view.default(mul_137, [1568, 3072]);  mul_137 = None
        permute_148 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg34_1, view_336, permute_148);  arg34_1 = view_336 = permute_148 = None
        view_337 = torch.ops.aten.view.default(addmm_42, [8, 196, 768]);  addmm_42 = None
        add_135 = torch.ops.aten.add.Tensor(add_131, view_337);  add_131 = view_337 = None
        clone_183 = torch.ops.aten.clone.default(add_135, memory_format = torch.contiguous_format)
        var_mean_29 = torch.ops.aten.var_mean.correction(clone_183, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_29[0]
        getitem_65 = var_mean_29[1];  var_mean_29 = None
        add_136 = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_77 = torch.ops.aten.sub.Tensor(clone_183, getitem_65);  clone_183 = getitem_65 = None
        mul_138 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_29);  sub_77 = rsqrt_29 = None
        mul_139 = torch.ops.aten.mul.Tensor(mul_138, arg35_1);  mul_138 = arg35_1 = None
        add_137 = torch.ops.aten.add.Tensor(mul_139, arg36_1);  mul_139 = arg36_1 = None
        permute_149 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        view_338 = torch.ops.aten.view.default(add_137, [1568, 768])
        mm_38 = torch.ops.aten.mm.default(view_338, permute_149);  view_338 = permute_149 = None
        view_339 = torch.ops.aten.view.default(mm_38, [8, 196, 1536]);  mm_38 = None
        view_340 = torch.ops.aten.view.default(view_339, [8, 196, 2, 16, 48]);  view_339 = None
        permute_150 = torch.ops.aten.permute.default(view_340, [2, 0, 3, 1, 4]);  view_340 = None
        select_105 = torch.ops.aten.select.int(permute_150, 0, 0)
        select_106 = torch.ops.aten.select.int(permute_150, 0, 1);  permute_150 = None
        expand_90 = torch.ops.aten.expand.default(device_put_2, [8, -1, -1, -1])
        permute_151 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        clone_184 = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_341 = torch.ops.aten.view.default(clone_184, [307328, 3]);  clone_184 = None
        mm_39 = torch.ops.aten.mm.default(view_341, permute_151);  view_341 = permute_151 = None
        view_342 = torch.ops.aten.view.default(mm_39, [8, 196, 196, 16]);  mm_39 = None
        add_138 = torch.ops.aten.add.Tensor(view_342, arg39_1);  view_342 = arg39_1 = None
        permute_152 = torch.ops.aten.permute.default(add_138, [0, 3, 1, 2]);  add_138 = None
        permute_153 = torch.ops.aten.permute.default(select_106, [0, 1, 3, 2]);  select_106 = None
        expand_91 = torch.ops.aten.expand.default(select_105, [8, 16, 196, 48]);  select_105 = None
        clone_185 = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_343 = torch.ops.aten.view.default(clone_185, [128, 196, 48]);  clone_185 = None
        expand_92 = torch.ops.aten.expand.default(permute_153, [8, 16, 48, 196]);  permute_153 = None
        clone_186 = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_344 = torch.ops.aten.view.default(clone_186, [128, 48, 196]);  clone_186 = None
        bmm_28 = torch.ops.aten.bmm.default(view_343, view_344);  view_343 = view_344 = None
        view_345 = torch.ops.aten.view.default(bmm_28, [8, 16, 196, 196]);  bmm_28 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(view_345, 1);  view_345 = None
        amax_default_9 = torch.ops.aten.amax.default(mul_tensor_18, [-1], True)
        sub_tensor_9 = torch.ops.aten.sub.Tensor(mul_tensor_18, amax_default_9);  mul_tensor_18 = amax_default_9 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(sub_tensor_9, 0.14433756729740643);  sub_tensor_9 = None
        exp_26 = torch.ops.aten.exp.default(mul_tensor_19);  mul_tensor_19 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_38 = torch.ops.aten.div.Tensor(exp_26, sum_39);  exp_26 = sum_39 = None
        clone_187 = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        amax_27 = torch.ops.aten.amax.default(clone_187, [-1], True)
        sub_79 = torch.ops.aten.sub.Tensor(clone_187, amax_27);  clone_187 = amax_27 = None
        exp_27 = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_39 = torch.ops.aten.div.Tensor(exp_27, sum_40);  exp_27 = sum_40 = None
        view_346 = torch.ops.aten.view.default(arg40_1, [1, -1, 1, 1]);  arg40_1 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(view_346)
        sub_80 = torch.ops.aten.sub.Tensor(1.0, sigmoid_24);  sigmoid_24 = None
        mul_141 = torch.ops.aten.mul.Tensor(sub_80, div_38);  sub_80 = div_38 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(view_346);  view_346 = None
        mul_142 = torch.ops.aten.mul.Tensor(sigmoid_25, div_39);  sigmoid_25 = div_39 = None
        add_139 = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(add_139, [-1])
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(sum_41, -1);  sum_41 = None
        div_40 = torch.ops.aten.div.Tensor(add_139, unsqueeze_62);  add_139 = unsqueeze_62 = None
        permute_154 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        view_347 = torch.ops.aten.view.default(add_137, [1568, 768]);  add_137 = None
        mm_40 = torch.ops.aten.mm.default(view_347, permute_154);  view_347 = permute_154 = None
        view_348 = torch.ops.aten.view.default(mm_40, [8, 196, 768]);  mm_40 = None
        view_349 = torch.ops.aten.view.default(view_348, [8, 196, 16, 48]);  view_348 = None
        permute_155 = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
        expand_93 = torch.ops.aten.expand.default(div_40, [8, 16, 196, 196]);  div_40 = None
        view_350 = torch.ops.aten.view.default(expand_93, [128, 196, 196]);  expand_93 = None
        expand_94 = torch.ops.aten.expand.default(permute_155, [8, 16, 196, 48]);  permute_155 = None
        clone_189 = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_351 = torch.ops.aten.view.default(clone_189, [128, 196, 48]);  clone_189 = None
        bmm_29 = torch.ops.aten.bmm.default(view_350, view_351);  view_350 = view_351 = None
        view_352 = torch.ops.aten.view.default(bmm_29, [8, 16, 196, 48]);  bmm_29 = None
        permute_156 = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_190 = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_353 = torch.ops.aten.view.default(clone_190, [8, 196, 768]);  clone_190 = None
        view_354 = torch.ops.aten.view.default(view_353, [1568, 768]);  view_353 = None
        permute_157 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg43_1, view_354, permute_157);  arg43_1 = view_354 = permute_157 = None
        view_355 = torch.ops.aten.view.default(addmm_43, [8, 196, 768]);  addmm_43 = None
        add_140 = torch.ops.aten.add.Tensor(add_135, view_355);  add_135 = view_355 = None
        clone_192 = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format)
        var_mean_30 = torch.ops.aten.var_mean.correction(clone_192, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_30[0]
        getitem_67 = var_mean_30[1];  var_mean_30 = None
        add_141 = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_81 = torch.ops.aten.sub.Tensor(clone_192, getitem_67);  clone_192 = getitem_67 = None
        mul_143 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_30);  sub_81 = rsqrt_30 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, arg44_1);  mul_143 = arg44_1 = None
        add_142 = torch.ops.aten.add.Tensor(mul_144, arg45_1);  mul_144 = arg45_1 = None
        view_356 = torch.ops.aten.view.default(add_142, [1568, 768]);  add_142 = None
        permute_158 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg47_1, view_356, permute_158);  arg47_1 = view_356 = permute_158 = None
        view_357 = torch.ops.aten.view.default(addmm_44, [8, 196, 3072]);  addmm_44 = None
        mul_145 = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_146 = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_14 = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_143 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_147 = torch.ops.aten.mul.Tensor(mul_145, add_143);  mul_145 = add_143 = None
        view_358 = torch.ops.aten.view.default(mul_147, [1568, 3072]);  mul_147 = None
        permute_159 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg49_1, view_358, permute_159);  arg49_1 = view_358 = permute_159 = None
        view_359 = torch.ops.aten.view.default(addmm_45, [8, 196, 768]);  addmm_45 = None
        add_144 = torch.ops.aten.add.Tensor(add_140, view_359);  add_140 = view_359 = None
        clone_195 = torch.ops.aten.clone.default(add_144, memory_format = torch.contiguous_format)
        var_mean_31 = torch.ops.aten.var_mean.correction(clone_195, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_31[0]
        getitem_69 = var_mean_31[1];  var_mean_31 = None
        add_145 = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_82 = torch.ops.aten.sub.Tensor(clone_195, getitem_69);  clone_195 = getitem_69 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_31);  sub_82 = rsqrt_31 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, arg50_1);  mul_148 = arg50_1 = None
        add_146 = torch.ops.aten.add.Tensor(mul_149, arg51_1);  mul_149 = arg51_1 = None
        permute_160 = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        view_360 = torch.ops.aten.view.default(add_146, [1568, 768])
        mm_41 = torch.ops.aten.mm.default(view_360, permute_160);  view_360 = permute_160 = None
        view_361 = torch.ops.aten.view.default(mm_41, [8, 196, 1536]);  mm_41 = None
        view_362 = torch.ops.aten.view.default(view_361, [8, 196, 2, 16, 48]);  view_361 = None
        permute_161 = torch.ops.aten.permute.default(view_362, [2, 0, 3, 1, 4]);  view_362 = None
        select_107 = torch.ops.aten.select.int(permute_161, 0, 0)
        select_108 = torch.ops.aten.select.int(permute_161, 0, 1);  permute_161 = None
        expand_95 = torch.ops.aten.expand.default(device_put_3, [8, -1, -1, -1])
        permute_162 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        clone_196 = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_363 = torch.ops.aten.view.default(clone_196, [307328, 3]);  clone_196 = None
        mm_42 = torch.ops.aten.mm.default(view_363, permute_162);  view_363 = permute_162 = None
        view_364 = torch.ops.aten.view.default(mm_42, [8, 196, 196, 16]);  mm_42 = None
        add_147 = torch.ops.aten.add.Tensor(view_364, arg54_1);  view_364 = arg54_1 = None
        permute_163 = torch.ops.aten.permute.default(add_147, [0, 3, 1, 2]);  add_147 = None
        permute_164 = torch.ops.aten.permute.default(select_108, [0, 1, 3, 2]);  select_108 = None
        expand_96 = torch.ops.aten.expand.default(select_107, [8, 16, 196, 48]);  select_107 = None
        clone_197 = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_365 = torch.ops.aten.view.default(clone_197, [128, 196, 48]);  clone_197 = None
        expand_97 = torch.ops.aten.expand.default(permute_164, [8, 16, 48, 196]);  permute_164 = None
        clone_198 = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_366 = torch.ops.aten.view.default(clone_198, [128, 48, 196]);  clone_198 = None
        bmm_30 = torch.ops.aten.bmm.default(view_365, view_366);  view_365 = view_366 = None
        view_367 = torch.ops.aten.view.default(bmm_30, [8, 16, 196, 196]);  bmm_30 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(view_367, 1);  view_367 = None
        amax_default_8 = torch.ops.aten.amax.default(mul_tensor_16, [-1], True)
        sub_tensor_8 = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = amax_default_8 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.14433756729740643);  sub_tensor_8 = None
        exp_28 = torch.ops.aten.exp.default(mul_tensor_17);  mul_tensor_17 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_41 = torch.ops.aten.div.Tensor(exp_28, sum_42);  exp_28 = sum_42 = None
        clone_199 = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        amax_29 = torch.ops.aten.amax.default(clone_199, [-1], True)
        sub_84 = torch.ops.aten.sub.Tensor(clone_199, amax_29);  clone_199 = amax_29 = None
        exp_29 = torch.ops.aten.exp.default(sub_84);  sub_84 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_42 = torch.ops.aten.div.Tensor(exp_29, sum_43);  exp_29 = sum_43 = None
        view_368 = torch.ops.aten.view.default(arg55_1, [1, -1, 1, 1]);  arg55_1 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(view_368)
        sub_85 = torch.ops.aten.sub.Tensor(1.0, sigmoid_26);  sigmoid_26 = None
        mul_151 = torch.ops.aten.mul.Tensor(sub_85, div_41);  sub_85 = div_41 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(view_368);  view_368 = None
        mul_152 = torch.ops.aten.mul.Tensor(sigmoid_27, div_42);  sigmoid_27 = div_42 = None
        add_148 = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(add_148, [-1])
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(sum_44, -1);  sum_44 = None
        div_43 = torch.ops.aten.div.Tensor(add_148, unsqueeze_63);  add_148 = unsqueeze_63 = None
        permute_165 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        view_369 = torch.ops.aten.view.default(add_146, [1568, 768]);  add_146 = None
        mm_43 = torch.ops.aten.mm.default(view_369, permute_165);  view_369 = permute_165 = None
        view_370 = torch.ops.aten.view.default(mm_43, [8, 196, 768]);  mm_43 = None
        view_371 = torch.ops.aten.view.default(view_370, [8, 196, 16, 48]);  view_370 = None
        permute_166 = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
        expand_98 = torch.ops.aten.expand.default(div_43, [8, 16, 196, 196]);  div_43 = None
        view_372 = torch.ops.aten.view.default(expand_98, [128, 196, 196]);  expand_98 = None
        expand_99 = torch.ops.aten.expand.default(permute_166, [8, 16, 196, 48]);  permute_166 = None
        clone_201 = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_373 = torch.ops.aten.view.default(clone_201, [128, 196, 48]);  clone_201 = None
        bmm_31 = torch.ops.aten.bmm.default(view_372, view_373);  view_372 = view_373 = None
        view_374 = torch.ops.aten.view.default(bmm_31, [8, 16, 196, 48]);  bmm_31 = None
        permute_167 = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_202 = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_375 = torch.ops.aten.view.default(clone_202, [8, 196, 768]);  clone_202 = None
        view_376 = torch.ops.aten.view.default(view_375, [1568, 768]);  view_375 = None
        permute_168 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg58_1, view_376, permute_168);  arg58_1 = view_376 = permute_168 = None
        view_377 = torch.ops.aten.view.default(addmm_46, [8, 196, 768]);  addmm_46 = None
        add_149 = torch.ops.aten.add.Tensor(add_144, view_377);  add_144 = view_377 = None
        clone_204 = torch.ops.aten.clone.default(add_149, memory_format = torch.contiguous_format)
        var_mean_32 = torch.ops.aten.var_mean.correction(clone_204, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_32[0]
        getitem_71 = var_mean_32[1];  var_mean_32 = None
        add_150 = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_86 = torch.ops.aten.sub.Tensor(clone_204, getitem_71);  clone_204 = getitem_71 = None
        mul_153 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_32);  sub_86 = rsqrt_32 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_153, arg59_1);  mul_153 = arg59_1 = None
        add_151 = torch.ops.aten.add.Tensor(mul_154, arg60_1);  mul_154 = arg60_1 = None
        view_378 = torch.ops.aten.view.default(add_151, [1568, 768]);  add_151 = None
        permute_169 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg62_1, view_378, permute_169);  arg62_1 = view_378 = permute_169 = None
        view_379 = torch.ops.aten.view.default(addmm_47, [8, 196, 3072]);  addmm_47 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_379, 0.5)
        mul_156 = torch.ops.aten.mul.Tensor(view_379, 0.7071067811865476);  view_379 = None
        erf_15 = torch.ops.aten.erf.default(mul_156);  mul_156 = None
        add_152 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_155, add_152);  mul_155 = add_152 = None
        view_380 = torch.ops.aten.view.default(mul_157, [1568, 3072]);  mul_157 = None
        permute_170 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg64_1, view_380, permute_170);  arg64_1 = view_380 = permute_170 = None
        view_381 = torch.ops.aten.view.default(addmm_48, [8, 196, 768]);  addmm_48 = None
        add_153 = torch.ops.aten.add.Tensor(add_149, view_381);  add_149 = view_381 = None
        clone_207 = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
        var_mean_33 = torch.ops.aten.var_mean.correction(clone_207, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_33[0]
        getitem_73 = var_mean_33[1];  var_mean_33 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_87 = torch.ops.aten.sub.Tensor(clone_207, getitem_73);  clone_207 = getitem_73 = None
        mul_158 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_33);  sub_87 = rsqrt_33 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_158, arg65_1);  mul_158 = arg65_1 = None
        add_155 = torch.ops.aten.add.Tensor(mul_159, arg66_1);  mul_159 = arg66_1 = None
        permute_171 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        view_382 = torch.ops.aten.view.default(add_155, [1568, 768])
        mm_44 = torch.ops.aten.mm.default(view_382, permute_171);  view_382 = permute_171 = None
        view_383 = torch.ops.aten.view.default(mm_44, [8, 196, 1536]);  mm_44 = None
        view_384 = torch.ops.aten.view.default(view_383, [8, 196, 2, 16, 48]);  view_383 = None
        permute_172 = torch.ops.aten.permute.default(view_384, [2, 0, 3, 1, 4]);  view_384 = None
        select_109 = torch.ops.aten.select.int(permute_172, 0, 0)
        select_110 = torch.ops.aten.select.int(permute_172, 0, 1);  permute_172 = None
        expand_100 = torch.ops.aten.expand.default(device_put_4, [8, -1, -1, -1])
        permute_173 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        clone_208 = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_385 = torch.ops.aten.view.default(clone_208, [307328, 3]);  clone_208 = None
        mm_45 = torch.ops.aten.mm.default(view_385, permute_173);  view_385 = permute_173 = None
        view_386 = torch.ops.aten.view.default(mm_45, [8, 196, 196, 16]);  mm_45 = None
        add_156 = torch.ops.aten.add.Tensor(view_386, arg69_1);  view_386 = arg69_1 = None
        permute_174 = torch.ops.aten.permute.default(add_156, [0, 3, 1, 2]);  add_156 = None
        permute_175 = torch.ops.aten.permute.default(select_110, [0, 1, 3, 2]);  select_110 = None
        expand_101 = torch.ops.aten.expand.default(select_109, [8, 16, 196, 48]);  select_109 = None
        clone_209 = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_387 = torch.ops.aten.view.default(clone_209, [128, 196, 48]);  clone_209 = None
        expand_102 = torch.ops.aten.expand.default(permute_175, [8, 16, 48, 196]);  permute_175 = None
        clone_210 = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
        view_388 = torch.ops.aten.view.default(clone_210, [128, 48, 196]);  clone_210 = None
        bmm_32 = torch.ops.aten.bmm.default(view_387, view_388);  view_387 = view_388 = None
        view_389 = torch.ops.aten.view.default(bmm_32, [8, 16, 196, 196]);  bmm_32 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(view_389, 1);  view_389 = None
        amax_default_7 = torch.ops.aten.amax.default(mul_tensor_14, [-1], True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, 0.14433756729740643);  sub_tensor_7 = None
        exp_30 = torch.ops.aten.exp.default(mul_tensor_15);  mul_tensor_15 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_44 = torch.ops.aten.div.Tensor(exp_30, sum_45);  exp_30 = sum_45 = None
        clone_211 = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
        amax_31 = torch.ops.aten.amax.default(clone_211, [-1], True)
        sub_89 = torch.ops.aten.sub.Tensor(clone_211, amax_31);  clone_211 = amax_31 = None
        exp_31 = torch.ops.aten.exp.default(sub_89);  sub_89 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_45 = torch.ops.aten.div.Tensor(exp_31, sum_46);  exp_31 = sum_46 = None
        view_390 = torch.ops.aten.view.default(arg70_1, [1, -1, 1, 1]);  arg70_1 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(view_390)
        sub_90 = torch.ops.aten.sub.Tensor(1.0, sigmoid_28);  sigmoid_28 = None
        mul_161 = torch.ops.aten.mul.Tensor(sub_90, div_44);  sub_90 = div_44 = None
        sigmoid_29 = torch.ops.aten.sigmoid.default(view_390);  view_390 = None
        mul_162 = torch.ops.aten.mul.Tensor(sigmoid_29, div_45);  sigmoid_29 = div_45 = None
        add_157 = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(add_157, [-1])
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(sum_47, -1);  sum_47 = None
        div_46 = torch.ops.aten.div.Tensor(add_157, unsqueeze_64);  add_157 = unsqueeze_64 = None
        permute_176 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        view_391 = torch.ops.aten.view.default(add_155, [1568, 768]);  add_155 = None
        mm_46 = torch.ops.aten.mm.default(view_391, permute_176);  view_391 = permute_176 = None
        view_392 = torch.ops.aten.view.default(mm_46, [8, 196, 768]);  mm_46 = None
        view_393 = torch.ops.aten.view.default(view_392, [8, 196, 16, 48]);  view_392 = None
        permute_177 = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        expand_103 = torch.ops.aten.expand.default(div_46, [8, 16, 196, 196]);  div_46 = None
        view_394 = torch.ops.aten.view.default(expand_103, [128, 196, 196]);  expand_103 = None
        expand_104 = torch.ops.aten.expand.default(permute_177, [8, 16, 196, 48]);  permute_177 = None
        clone_213 = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_395 = torch.ops.aten.view.default(clone_213, [128, 196, 48]);  clone_213 = None
        bmm_33 = torch.ops.aten.bmm.default(view_394, view_395);  view_394 = view_395 = None
        view_396 = torch.ops.aten.view.default(bmm_33, [8, 16, 196, 48]);  bmm_33 = None
        permute_178 = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
        clone_214 = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
        view_397 = torch.ops.aten.view.default(clone_214, [8, 196, 768]);  clone_214 = None
        view_398 = torch.ops.aten.view.default(view_397, [1568, 768]);  view_397 = None
        permute_179 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg73_1, view_398, permute_179);  arg73_1 = view_398 = permute_179 = None
        view_399 = torch.ops.aten.view.default(addmm_49, [8, 196, 768]);  addmm_49 = None
        add_158 = torch.ops.aten.add.Tensor(add_153, view_399);  add_153 = view_399 = None
        clone_216 = torch.ops.aten.clone.default(add_158, memory_format = torch.contiguous_format)
        var_mean_34 = torch.ops.aten.var_mean.correction(clone_216, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_34[0]
        getitem_75 = var_mean_34[1];  var_mean_34 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_91 = torch.ops.aten.sub.Tensor(clone_216, getitem_75);  clone_216 = getitem_75 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_34);  sub_91 = rsqrt_34 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, arg74_1);  mul_163 = arg74_1 = None
        add_160 = torch.ops.aten.add.Tensor(mul_164, arg75_1);  mul_164 = arg75_1 = None
        view_400 = torch.ops.aten.view.default(add_160, [1568, 768]);  add_160 = None
        permute_180 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg77_1, view_400, permute_180);  arg77_1 = view_400 = permute_180 = None
        view_401 = torch.ops.aten.view.default(addmm_50, [8, 196, 3072]);  addmm_50 = None
        mul_165 = torch.ops.aten.mul.Tensor(view_401, 0.5)
        mul_166 = torch.ops.aten.mul.Tensor(view_401, 0.7071067811865476);  view_401 = None
        erf_16 = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_161 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_165, add_161);  mul_165 = add_161 = None
        view_402 = torch.ops.aten.view.default(mul_167, [1568, 3072]);  mul_167 = None
        permute_181 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg79_1, view_402, permute_181);  arg79_1 = view_402 = permute_181 = None
        view_403 = torch.ops.aten.view.default(addmm_51, [8, 196, 768]);  addmm_51 = None
        add_162 = torch.ops.aten.add.Tensor(add_158, view_403);  add_158 = view_403 = None
        clone_219 = torch.ops.aten.clone.default(add_162, memory_format = torch.contiguous_format)
        var_mean_35 = torch.ops.aten.var_mean.correction(clone_219, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_35[0]
        getitem_77 = var_mean_35[1];  var_mean_35 = None
        add_163 = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_92 = torch.ops.aten.sub.Tensor(clone_219, getitem_77);  clone_219 = getitem_77 = None
        mul_168 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_35);  sub_92 = rsqrt_35 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, arg80_1);  mul_168 = arg80_1 = None
        add_164 = torch.ops.aten.add.Tensor(mul_169, arg81_1);  mul_169 = arg81_1 = None
        permute_182 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        view_404 = torch.ops.aten.view.default(add_164, [1568, 768])
        mm_47 = torch.ops.aten.mm.default(view_404, permute_182);  view_404 = permute_182 = None
        view_405 = torch.ops.aten.view.default(mm_47, [8, 196, 1536]);  mm_47 = None
        view_406 = torch.ops.aten.view.default(view_405, [8, 196, 2, 16, 48]);  view_405 = None
        permute_183 = torch.ops.aten.permute.default(view_406, [2, 0, 3, 1, 4]);  view_406 = None
        select_111 = torch.ops.aten.select.int(permute_183, 0, 0)
        select_112 = torch.ops.aten.select.int(permute_183, 0, 1);  permute_183 = None
        expand_105 = torch.ops.aten.expand.default(device_put_5, [8, -1, -1, -1])
        permute_184 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        clone_220 = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_407 = torch.ops.aten.view.default(clone_220, [307328, 3]);  clone_220 = None
        mm_48 = torch.ops.aten.mm.default(view_407, permute_184);  view_407 = permute_184 = None
        view_408 = torch.ops.aten.view.default(mm_48, [8, 196, 196, 16]);  mm_48 = None
        add_165 = torch.ops.aten.add.Tensor(view_408, arg84_1);  view_408 = arg84_1 = None
        permute_185 = torch.ops.aten.permute.default(add_165, [0, 3, 1, 2]);  add_165 = None
        permute_186 = torch.ops.aten.permute.default(select_112, [0, 1, 3, 2]);  select_112 = None
        expand_106 = torch.ops.aten.expand.default(select_111, [8, 16, 196, 48]);  select_111 = None
        clone_221 = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_409 = torch.ops.aten.view.default(clone_221, [128, 196, 48]);  clone_221 = None
        expand_107 = torch.ops.aten.expand.default(permute_186, [8, 16, 48, 196]);  permute_186 = None
        clone_222 = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_410 = torch.ops.aten.view.default(clone_222, [128, 48, 196]);  clone_222 = None
        bmm_34 = torch.ops.aten.bmm.default(view_409, view_410);  view_409 = view_410 = None
        view_411 = torch.ops.aten.view.default(bmm_34, [8, 16, 196, 196]);  bmm_34 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_411, 1);  view_411 = None
        amax_default_6 = torch.ops.aten.amax.default(mul_tensor_12, [-1], True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.14433756729740643);  sub_tensor_6 = None
        exp_32 = torch.ops.aten.exp.default(mul_tensor_13);  mul_tensor_13 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_32, sum_48);  exp_32 = sum_48 = None
        clone_223 = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        amax_33 = torch.ops.aten.amax.default(clone_223, [-1], True)
        sub_94 = torch.ops.aten.sub.Tensor(clone_223, amax_33);  clone_223 = amax_33 = None
        exp_33 = torch.ops.aten.exp.default(sub_94);  sub_94 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_48 = torch.ops.aten.div.Tensor(exp_33, sum_49);  exp_33 = sum_49 = None
        view_412 = torch.ops.aten.view.default(arg85_1, [1, -1, 1, 1]);  arg85_1 = None
        sigmoid_30 = torch.ops.aten.sigmoid.default(view_412)
        sub_95 = torch.ops.aten.sub.Tensor(1.0, sigmoid_30);  sigmoid_30 = None
        mul_171 = torch.ops.aten.mul.Tensor(sub_95, div_47);  sub_95 = div_47 = None
        sigmoid_31 = torch.ops.aten.sigmoid.default(view_412);  view_412 = None
        mul_172 = torch.ops.aten.mul.Tensor(sigmoid_31, div_48);  sigmoid_31 = div_48 = None
        add_166 = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(add_166, [-1])
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(sum_50, -1);  sum_50 = None
        div_49 = torch.ops.aten.div.Tensor(add_166, unsqueeze_65);  add_166 = unsqueeze_65 = None
        permute_187 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        view_413 = torch.ops.aten.view.default(add_164, [1568, 768]);  add_164 = None
        mm_49 = torch.ops.aten.mm.default(view_413, permute_187);  view_413 = permute_187 = None
        view_414 = torch.ops.aten.view.default(mm_49, [8, 196, 768]);  mm_49 = None
        view_415 = torch.ops.aten.view.default(view_414, [8, 196, 16, 48]);  view_414 = None
        permute_188 = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
        expand_108 = torch.ops.aten.expand.default(div_49, [8, 16, 196, 196]);  div_49 = None
        view_416 = torch.ops.aten.view.default(expand_108, [128, 196, 196]);  expand_108 = None
        expand_109 = torch.ops.aten.expand.default(permute_188, [8, 16, 196, 48]);  permute_188 = None
        clone_225 = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_417 = torch.ops.aten.view.default(clone_225, [128, 196, 48]);  clone_225 = None
        bmm_35 = torch.ops.aten.bmm.default(view_416, view_417);  view_416 = view_417 = None
        view_418 = torch.ops.aten.view.default(bmm_35, [8, 16, 196, 48]);  bmm_35 = None
        permute_189 = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
        clone_226 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_419 = torch.ops.aten.view.default(clone_226, [8, 196, 768]);  clone_226 = None
        view_420 = torch.ops.aten.view.default(view_419, [1568, 768]);  view_419 = None
        permute_190 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg88_1, view_420, permute_190);  arg88_1 = view_420 = permute_190 = None
        view_421 = torch.ops.aten.view.default(addmm_52, [8, 196, 768]);  addmm_52 = None
        add_167 = torch.ops.aten.add.Tensor(add_162, view_421);  add_162 = view_421 = None
        clone_228 = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
        var_mean_36 = torch.ops.aten.var_mean.correction(clone_228, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_36[0]
        getitem_79 = var_mean_36[1];  var_mean_36 = None
        add_168 = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_96 = torch.ops.aten.sub.Tensor(clone_228, getitem_79);  clone_228 = getitem_79 = None
        mul_173 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_36);  sub_96 = rsqrt_36 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_173, arg89_1);  mul_173 = arg89_1 = None
        add_169 = torch.ops.aten.add.Tensor(mul_174, arg90_1);  mul_174 = arg90_1 = None
        view_422 = torch.ops.aten.view.default(add_169, [1568, 768]);  add_169 = None
        permute_191 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg92_1, view_422, permute_191);  arg92_1 = view_422 = permute_191 = None
        view_423 = torch.ops.aten.view.default(addmm_53, [8, 196, 3072]);  addmm_53 = None
        mul_175 = torch.ops.aten.mul.Tensor(view_423, 0.5)
        mul_176 = torch.ops.aten.mul.Tensor(view_423, 0.7071067811865476);  view_423 = None
        erf_17 = torch.ops.aten.erf.default(mul_176);  mul_176 = None
        add_170 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_175, add_170);  mul_175 = add_170 = None
        view_424 = torch.ops.aten.view.default(mul_177, [1568, 3072]);  mul_177 = None
        permute_192 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg94_1, view_424, permute_192);  arg94_1 = view_424 = permute_192 = None
        view_425 = torch.ops.aten.view.default(addmm_54, [8, 196, 768]);  addmm_54 = None
        add_171 = torch.ops.aten.add.Tensor(add_167, view_425);  add_167 = view_425 = None
        clone_231 = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format)
        var_mean_37 = torch.ops.aten.var_mean.correction(clone_231, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_37[0]
        getitem_81 = var_mean_37[1];  var_mean_37 = None
        add_172 = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_97 = torch.ops.aten.sub.Tensor(clone_231, getitem_81);  clone_231 = getitem_81 = None
        mul_178 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_37);  sub_97 = rsqrt_37 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, arg95_1);  mul_178 = arg95_1 = None
        add_173 = torch.ops.aten.add.Tensor(mul_179, arg96_1);  mul_179 = arg96_1 = None
        permute_193 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        view_426 = torch.ops.aten.view.default(add_173, [1568, 768])
        mm_50 = torch.ops.aten.mm.default(view_426, permute_193);  view_426 = permute_193 = None
        view_427 = torch.ops.aten.view.default(mm_50, [8, 196, 1536]);  mm_50 = None
        view_428 = torch.ops.aten.view.default(view_427, [8, 196, 2, 16, 48]);  view_427 = None
        permute_194 = torch.ops.aten.permute.default(view_428, [2, 0, 3, 1, 4]);  view_428 = None
        select_113 = torch.ops.aten.select.int(permute_194, 0, 0)
        select_114 = torch.ops.aten.select.int(permute_194, 0, 1);  permute_194 = None
        expand_110 = torch.ops.aten.expand.default(device_put_6, [8, -1, -1, -1])
        permute_195 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        clone_232 = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
        view_429 = torch.ops.aten.view.default(clone_232, [307328, 3]);  clone_232 = None
        mm_51 = torch.ops.aten.mm.default(view_429, permute_195);  view_429 = permute_195 = None
        view_430 = torch.ops.aten.view.default(mm_51, [8, 196, 196, 16]);  mm_51 = None
        add_174 = torch.ops.aten.add.Tensor(view_430, arg99_1);  view_430 = arg99_1 = None
        permute_196 = torch.ops.aten.permute.default(add_174, [0, 3, 1, 2]);  add_174 = None
        permute_197 = torch.ops.aten.permute.default(select_114, [0, 1, 3, 2]);  select_114 = None
        expand_111 = torch.ops.aten.expand.default(select_113, [8, 16, 196, 48]);  select_113 = None
        clone_233 = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_431 = torch.ops.aten.view.default(clone_233, [128, 196, 48]);  clone_233 = None
        expand_112 = torch.ops.aten.expand.default(permute_197, [8, 16, 48, 196]);  permute_197 = None
        clone_234 = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_432 = torch.ops.aten.view.default(clone_234, [128, 48, 196]);  clone_234 = None
        bmm_36 = torch.ops.aten.bmm.default(view_431, view_432);  view_431 = view_432 = None
        view_433 = torch.ops.aten.view.default(bmm_36, [8, 16, 196, 196]);  bmm_36 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_433, 1);  view_433 = None
        amax_default_5 = torch.ops.aten.amax.default(mul_tensor_10, [-1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, 0.14433756729740643);  sub_tensor_5 = None
        exp_34 = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_50 = torch.ops.aten.div.Tensor(exp_34, sum_51);  exp_34 = sum_51 = None
        clone_235 = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        amax_35 = torch.ops.aten.amax.default(clone_235, [-1], True)
        sub_99 = torch.ops.aten.sub.Tensor(clone_235, amax_35);  clone_235 = amax_35 = None
        exp_35 = torch.ops.aten.exp.default(sub_99);  sub_99 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_51 = torch.ops.aten.div.Tensor(exp_35, sum_52);  exp_35 = sum_52 = None
        view_434 = torch.ops.aten.view.default(arg100_1, [1, -1, 1, 1]);  arg100_1 = None
        sigmoid_32 = torch.ops.aten.sigmoid.default(view_434)
        sub_100 = torch.ops.aten.sub.Tensor(1.0, sigmoid_32);  sigmoid_32 = None
        mul_181 = torch.ops.aten.mul.Tensor(sub_100, div_50);  sub_100 = div_50 = None
        sigmoid_33 = torch.ops.aten.sigmoid.default(view_434);  view_434 = None
        mul_182 = torch.ops.aten.mul.Tensor(sigmoid_33, div_51);  sigmoid_33 = div_51 = None
        add_175 = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(add_175, [-1])
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(sum_53, -1);  sum_53 = None
        div_52 = torch.ops.aten.div.Tensor(add_175, unsqueeze_66);  add_175 = unsqueeze_66 = None
        permute_198 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        view_435 = torch.ops.aten.view.default(add_173, [1568, 768]);  add_173 = None
        mm_52 = torch.ops.aten.mm.default(view_435, permute_198);  view_435 = permute_198 = None
        view_436 = torch.ops.aten.view.default(mm_52, [8, 196, 768]);  mm_52 = None
        view_437 = torch.ops.aten.view.default(view_436, [8, 196, 16, 48]);  view_436 = None
        permute_199 = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
        expand_113 = torch.ops.aten.expand.default(div_52, [8, 16, 196, 196]);  div_52 = None
        view_438 = torch.ops.aten.view.default(expand_113, [128, 196, 196]);  expand_113 = None
        expand_114 = torch.ops.aten.expand.default(permute_199, [8, 16, 196, 48]);  permute_199 = None
        clone_237 = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_439 = torch.ops.aten.view.default(clone_237, [128, 196, 48]);  clone_237 = None
        bmm_37 = torch.ops.aten.bmm.default(view_438, view_439);  view_438 = view_439 = None
        view_440 = torch.ops.aten.view.default(bmm_37, [8, 16, 196, 48]);  bmm_37 = None
        permute_200 = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
        clone_238 = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_441 = torch.ops.aten.view.default(clone_238, [8, 196, 768]);  clone_238 = None
        view_442 = torch.ops.aten.view.default(view_441, [1568, 768]);  view_441 = None
        permute_201 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg103_1, view_442, permute_201);  arg103_1 = view_442 = permute_201 = None
        view_443 = torch.ops.aten.view.default(addmm_55, [8, 196, 768]);  addmm_55 = None
        add_176 = torch.ops.aten.add.Tensor(add_171, view_443);  add_171 = view_443 = None
        clone_240 = torch.ops.aten.clone.default(add_176, memory_format = torch.contiguous_format)
        var_mean_38 = torch.ops.aten.var_mean.correction(clone_240, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_38[0]
        getitem_83 = var_mean_38[1];  var_mean_38 = None
        add_177 = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_101 = torch.ops.aten.sub.Tensor(clone_240, getitem_83);  clone_240 = getitem_83 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_38);  sub_101 = rsqrt_38 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_183, arg104_1);  mul_183 = arg104_1 = None
        add_178 = torch.ops.aten.add.Tensor(mul_184, arg105_1);  mul_184 = arg105_1 = None
        view_444 = torch.ops.aten.view.default(add_178, [1568, 768]);  add_178 = None
        permute_202 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg107_1, view_444, permute_202);  arg107_1 = view_444 = permute_202 = None
        view_445 = torch.ops.aten.view.default(addmm_56, [8, 196, 3072]);  addmm_56 = None
        mul_185 = torch.ops.aten.mul.Tensor(view_445, 0.5)
        mul_186 = torch.ops.aten.mul.Tensor(view_445, 0.7071067811865476);  view_445 = None
        erf_18 = torch.ops.aten.erf.default(mul_186);  mul_186 = None
        add_179 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_185, add_179);  mul_185 = add_179 = None
        view_446 = torch.ops.aten.view.default(mul_187, [1568, 3072]);  mul_187 = None
        permute_203 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg109_1, view_446, permute_203);  arg109_1 = view_446 = permute_203 = None
        view_447 = torch.ops.aten.view.default(addmm_57, [8, 196, 768]);  addmm_57 = None
        add_180 = torch.ops.aten.add.Tensor(add_176, view_447);  add_176 = view_447 = None
        clone_243 = torch.ops.aten.clone.default(add_180, memory_format = torch.contiguous_format)
        var_mean_39 = torch.ops.aten.var_mean.correction(clone_243, [2], correction = 0, keepdim = True)
        getitem_84 = var_mean_39[0]
        getitem_85 = var_mean_39[1];  var_mean_39 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_102 = torch.ops.aten.sub.Tensor(clone_243, getitem_85);  clone_243 = getitem_85 = None
        mul_188 = torch.ops.aten.mul.Tensor(sub_102, rsqrt_39);  sub_102 = rsqrt_39 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, arg110_1);  mul_188 = arg110_1 = None
        add_182 = torch.ops.aten.add.Tensor(mul_189, arg111_1);  mul_189 = arg111_1 = None
        permute_204 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        view_448 = torch.ops.aten.view.default(add_182, [1568, 768])
        mm_53 = torch.ops.aten.mm.default(view_448, permute_204);  view_448 = permute_204 = None
        view_449 = torch.ops.aten.view.default(mm_53, [8, 196, 1536]);  mm_53 = None
        view_450 = torch.ops.aten.view.default(view_449, [8, 196, 2, 16, 48]);  view_449 = None
        permute_205 = torch.ops.aten.permute.default(view_450, [2, 0, 3, 1, 4]);  view_450 = None
        select_115 = torch.ops.aten.select.int(permute_205, 0, 0)
        select_116 = torch.ops.aten.select.int(permute_205, 0, 1);  permute_205 = None
        expand_115 = torch.ops.aten.expand.default(device_put_7, [8, -1, -1, -1])
        permute_206 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        clone_244 = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_451 = torch.ops.aten.view.default(clone_244, [307328, 3]);  clone_244 = None
        mm_54 = torch.ops.aten.mm.default(view_451, permute_206);  view_451 = permute_206 = None
        view_452 = torch.ops.aten.view.default(mm_54, [8, 196, 196, 16]);  mm_54 = None
        add_183 = torch.ops.aten.add.Tensor(view_452, arg114_1);  view_452 = arg114_1 = None
        permute_207 = torch.ops.aten.permute.default(add_183, [0, 3, 1, 2]);  add_183 = None
        permute_208 = torch.ops.aten.permute.default(select_116, [0, 1, 3, 2]);  select_116 = None
        expand_116 = torch.ops.aten.expand.default(select_115, [8, 16, 196, 48]);  select_115 = None
        clone_245 = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_453 = torch.ops.aten.view.default(clone_245, [128, 196, 48]);  clone_245 = None
        expand_117 = torch.ops.aten.expand.default(permute_208, [8, 16, 48, 196]);  permute_208 = None
        clone_246 = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_454 = torch.ops.aten.view.default(clone_246, [128, 48, 196]);  clone_246 = None
        bmm_38 = torch.ops.aten.bmm.default(view_453, view_454);  view_453 = view_454 = None
        view_455 = torch.ops.aten.view.default(bmm_38, [8, 16, 196, 196]);  bmm_38 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_455, 1);  view_455 = None
        amax_default_4 = torch.ops.aten.amax.default(mul_tensor_8, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.14433756729740643);  sub_tensor_4 = None
        exp_36 = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_53 = torch.ops.aten.div.Tensor(exp_36, sum_54);  exp_36 = sum_54 = None
        clone_247 = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        amax_37 = torch.ops.aten.amax.default(clone_247, [-1], True)
        sub_104 = torch.ops.aten.sub.Tensor(clone_247, amax_37);  clone_247 = amax_37 = None
        exp_37 = torch.ops.aten.exp.default(sub_104);  sub_104 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_54 = torch.ops.aten.div.Tensor(exp_37, sum_55);  exp_37 = sum_55 = None
        view_456 = torch.ops.aten.view.default(arg115_1, [1, -1, 1, 1]);  arg115_1 = None
        sigmoid_34 = torch.ops.aten.sigmoid.default(view_456)
        sub_105 = torch.ops.aten.sub.Tensor(1.0, sigmoid_34);  sigmoid_34 = None
        mul_191 = torch.ops.aten.mul.Tensor(sub_105, div_53);  sub_105 = div_53 = None
        sigmoid_35 = torch.ops.aten.sigmoid.default(view_456);  view_456 = None
        mul_192 = torch.ops.aten.mul.Tensor(sigmoid_35, div_54);  sigmoid_35 = div_54 = None
        add_184 = torch.ops.aten.add.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(add_184, [-1])
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(sum_56, -1);  sum_56 = None
        div_55 = torch.ops.aten.div.Tensor(add_184, unsqueeze_67);  add_184 = unsqueeze_67 = None
        permute_209 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        view_457 = torch.ops.aten.view.default(add_182, [1568, 768]);  add_182 = None
        mm_55 = torch.ops.aten.mm.default(view_457, permute_209);  view_457 = permute_209 = None
        view_458 = torch.ops.aten.view.default(mm_55, [8, 196, 768]);  mm_55 = None
        view_459 = torch.ops.aten.view.default(view_458, [8, 196, 16, 48]);  view_458 = None
        permute_210 = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        expand_118 = torch.ops.aten.expand.default(div_55, [8, 16, 196, 196]);  div_55 = None
        view_460 = torch.ops.aten.view.default(expand_118, [128, 196, 196]);  expand_118 = None
        expand_119 = torch.ops.aten.expand.default(permute_210, [8, 16, 196, 48]);  permute_210 = None
        clone_249 = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_461 = torch.ops.aten.view.default(clone_249, [128, 196, 48]);  clone_249 = None
        bmm_39 = torch.ops.aten.bmm.default(view_460, view_461);  view_460 = view_461 = None
        view_462 = torch.ops.aten.view.default(bmm_39, [8, 16, 196, 48]);  bmm_39 = None
        permute_211 = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
        clone_250 = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_463 = torch.ops.aten.view.default(clone_250, [8, 196, 768]);  clone_250 = None
        view_464 = torch.ops.aten.view.default(view_463, [1568, 768]);  view_463 = None
        permute_212 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg118_1, view_464, permute_212);  arg118_1 = view_464 = permute_212 = None
        view_465 = torch.ops.aten.view.default(addmm_58, [8, 196, 768]);  addmm_58 = None
        add_185 = torch.ops.aten.add.Tensor(add_180, view_465);  add_180 = view_465 = None
        clone_252 = torch.ops.aten.clone.default(add_185, memory_format = torch.contiguous_format)
        var_mean_40 = torch.ops.aten.var_mean.correction(clone_252, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_40[0]
        getitem_87 = var_mean_40[1];  var_mean_40 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_106 = torch.ops.aten.sub.Tensor(clone_252, getitem_87);  clone_252 = getitem_87 = None
        mul_193 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_40);  sub_106 = rsqrt_40 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, arg119_1);  mul_193 = arg119_1 = None
        add_187 = torch.ops.aten.add.Tensor(mul_194, arg120_1);  mul_194 = arg120_1 = None
        view_466 = torch.ops.aten.view.default(add_187, [1568, 768]);  add_187 = None
        permute_213 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg122_1, view_466, permute_213);  arg122_1 = view_466 = permute_213 = None
        view_467 = torch.ops.aten.view.default(addmm_59, [8, 196, 3072]);  addmm_59 = None
        mul_195 = torch.ops.aten.mul.Tensor(view_467, 0.5)
        mul_196 = torch.ops.aten.mul.Tensor(view_467, 0.7071067811865476);  view_467 = None
        erf_19 = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_188 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_195, add_188);  mul_195 = add_188 = None
        view_468 = torch.ops.aten.view.default(mul_197, [1568, 3072]);  mul_197 = None
        permute_214 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg124_1, view_468, permute_214);  arg124_1 = view_468 = permute_214 = None
        view_469 = torch.ops.aten.view.default(addmm_60, [8, 196, 768]);  addmm_60 = None
        add_189 = torch.ops.aten.add.Tensor(add_185, view_469);  add_185 = view_469 = None
        clone_255 = torch.ops.aten.clone.default(add_189, memory_format = torch.contiguous_format)
        var_mean_41 = torch.ops.aten.var_mean.correction(clone_255, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_41[0]
        getitem_89 = var_mean_41[1];  var_mean_41 = None
        add_190 = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_107 = torch.ops.aten.sub.Tensor(clone_255, getitem_89);  clone_255 = getitem_89 = None
        mul_198 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_41);  sub_107 = rsqrt_41 = None
        mul_199 = torch.ops.aten.mul.Tensor(mul_198, arg125_1);  mul_198 = arg125_1 = None
        add_191 = torch.ops.aten.add.Tensor(mul_199, arg126_1);  mul_199 = arg126_1 = None
        permute_215 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        view_470 = torch.ops.aten.view.default(add_191, [1568, 768])
        mm_56 = torch.ops.aten.mm.default(view_470, permute_215);  view_470 = permute_215 = None
        view_471 = torch.ops.aten.view.default(mm_56, [8, 196, 1536]);  mm_56 = None
        view_472 = torch.ops.aten.view.default(view_471, [8, 196, 2, 16, 48]);  view_471 = None
        permute_216 = torch.ops.aten.permute.default(view_472, [2, 0, 3, 1, 4]);  view_472 = None
        select_117 = torch.ops.aten.select.int(permute_216, 0, 0)
        select_118 = torch.ops.aten.select.int(permute_216, 0, 1);  permute_216 = None
        expand_120 = torch.ops.aten.expand.default(device_put_8, [8, -1, -1, -1])
        permute_217 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        clone_256 = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_473 = torch.ops.aten.view.default(clone_256, [307328, 3]);  clone_256 = None
        mm_57 = torch.ops.aten.mm.default(view_473, permute_217);  view_473 = permute_217 = None
        view_474 = torch.ops.aten.view.default(mm_57, [8, 196, 196, 16]);  mm_57 = None
        add_192 = torch.ops.aten.add.Tensor(view_474, arg129_1);  view_474 = arg129_1 = None
        permute_218 = torch.ops.aten.permute.default(add_192, [0, 3, 1, 2]);  add_192 = None
        permute_219 = torch.ops.aten.permute.default(select_118, [0, 1, 3, 2]);  select_118 = None
        expand_121 = torch.ops.aten.expand.default(select_117, [8, 16, 196, 48]);  select_117 = None
        clone_257 = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_475 = torch.ops.aten.view.default(clone_257, [128, 196, 48]);  clone_257 = None
        expand_122 = torch.ops.aten.expand.default(permute_219, [8, 16, 48, 196]);  permute_219 = None
        clone_258 = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_476 = torch.ops.aten.view.default(clone_258, [128, 48, 196]);  clone_258 = None
        bmm_40 = torch.ops.aten.bmm.default(view_475, view_476);  view_475 = view_476 = None
        view_477 = torch.ops.aten.view.default(bmm_40, [8, 16, 196, 196]);  bmm_40 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_477, 1);  view_477 = None
        amax_default_3 = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, 0.14433756729740643);  sub_tensor_3 = None
        exp_38 = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_56 = torch.ops.aten.div.Tensor(exp_38, sum_57);  exp_38 = sum_57 = None
        clone_259 = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
        amax_39 = torch.ops.aten.amax.default(clone_259, [-1], True)
        sub_109 = torch.ops.aten.sub.Tensor(clone_259, amax_39);  clone_259 = amax_39 = None
        exp_39 = torch.ops.aten.exp.default(sub_109);  sub_109 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_57 = torch.ops.aten.div.Tensor(exp_39, sum_58);  exp_39 = sum_58 = None
        view_478 = torch.ops.aten.view.default(arg130_1, [1, -1, 1, 1]);  arg130_1 = None
        sigmoid_36 = torch.ops.aten.sigmoid.default(view_478)
        sub_110 = torch.ops.aten.sub.Tensor(1.0, sigmoid_36);  sigmoid_36 = None
        mul_201 = torch.ops.aten.mul.Tensor(sub_110, div_56);  sub_110 = div_56 = None
        sigmoid_37 = torch.ops.aten.sigmoid.default(view_478);  view_478 = None
        mul_202 = torch.ops.aten.mul.Tensor(sigmoid_37, div_57);  sigmoid_37 = div_57 = None
        add_193 = torch.ops.aten.add.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(add_193, [-1])
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(sum_59, -1);  sum_59 = None
        div_58 = torch.ops.aten.div.Tensor(add_193, unsqueeze_68);  add_193 = unsqueeze_68 = None
        permute_220 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        view_479 = torch.ops.aten.view.default(add_191, [1568, 768]);  add_191 = None
        mm_58 = torch.ops.aten.mm.default(view_479, permute_220);  view_479 = permute_220 = None
        view_480 = torch.ops.aten.view.default(mm_58, [8, 196, 768]);  mm_58 = None
        view_481 = torch.ops.aten.view.default(view_480, [8, 196, 16, 48]);  view_480 = None
        permute_221 = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
        expand_123 = torch.ops.aten.expand.default(div_58, [8, 16, 196, 196]);  div_58 = None
        view_482 = torch.ops.aten.view.default(expand_123, [128, 196, 196]);  expand_123 = None
        expand_124 = torch.ops.aten.expand.default(permute_221, [8, 16, 196, 48]);  permute_221 = None
        clone_261 = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_483 = torch.ops.aten.view.default(clone_261, [128, 196, 48]);  clone_261 = None
        bmm_41 = torch.ops.aten.bmm.default(view_482, view_483);  view_482 = view_483 = None
        view_484 = torch.ops.aten.view.default(bmm_41, [8, 16, 196, 48]);  bmm_41 = None
        permute_222 = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
        clone_262 = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_485 = torch.ops.aten.view.default(clone_262, [8, 196, 768]);  clone_262 = None
        view_486 = torch.ops.aten.view.default(view_485, [1568, 768]);  view_485 = None
        permute_223 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg133_1, view_486, permute_223);  arg133_1 = view_486 = permute_223 = None
        view_487 = torch.ops.aten.view.default(addmm_61, [8, 196, 768]);  addmm_61 = None
        add_194 = torch.ops.aten.add.Tensor(add_189, view_487);  add_189 = view_487 = None
        clone_264 = torch.ops.aten.clone.default(add_194, memory_format = torch.contiguous_format)
        var_mean_42 = torch.ops.aten.var_mean.correction(clone_264, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_42[0]
        getitem_91 = var_mean_42[1];  var_mean_42 = None
        add_195 = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_111 = torch.ops.aten.sub.Tensor(clone_264, getitem_91);  clone_264 = getitem_91 = None
        mul_203 = torch.ops.aten.mul.Tensor(sub_111, rsqrt_42);  sub_111 = rsqrt_42 = None
        mul_204 = torch.ops.aten.mul.Tensor(mul_203, arg134_1);  mul_203 = arg134_1 = None
        add_196 = torch.ops.aten.add.Tensor(mul_204, arg135_1);  mul_204 = arg135_1 = None
        view_488 = torch.ops.aten.view.default(add_196, [1568, 768]);  add_196 = None
        permute_224 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg137_1, view_488, permute_224);  arg137_1 = view_488 = permute_224 = None
        view_489 = torch.ops.aten.view.default(addmm_62, [8, 196, 3072]);  addmm_62 = None
        mul_205 = torch.ops.aten.mul.Tensor(view_489, 0.5)
        mul_206 = torch.ops.aten.mul.Tensor(view_489, 0.7071067811865476);  view_489 = None
        erf_20 = torch.ops.aten.erf.default(mul_206);  mul_206 = None
        add_197 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_205, add_197);  mul_205 = add_197 = None
        view_490 = torch.ops.aten.view.default(mul_207, [1568, 3072]);  mul_207 = None
        permute_225 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg139_1, view_490, permute_225);  arg139_1 = view_490 = permute_225 = None
        view_491 = torch.ops.aten.view.default(addmm_63, [8, 196, 768]);  addmm_63 = None
        add_198 = torch.ops.aten.add.Tensor(add_194, view_491);  add_194 = view_491 = None
        clone_267 = torch.ops.aten.clone.default(add_198, memory_format = torch.contiguous_format)
        var_mean_43 = torch.ops.aten.var_mean.correction(clone_267, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_43[0]
        getitem_93 = var_mean_43[1];  var_mean_43 = None
        add_199 = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_112 = torch.ops.aten.sub.Tensor(clone_267, getitem_93);  clone_267 = getitem_93 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_43);  sub_112 = rsqrt_43 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, arg140_1);  mul_208 = arg140_1 = None
        add_200 = torch.ops.aten.add.Tensor(mul_209, arg141_1);  mul_209 = arg141_1 = None
        permute_226 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        view_492 = torch.ops.aten.view.default(add_200, [1568, 768])
        mm_59 = torch.ops.aten.mm.default(view_492, permute_226);  view_492 = permute_226 = None
        view_493 = torch.ops.aten.view.default(mm_59, [8, 196, 1536]);  mm_59 = None
        view_494 = torch.ops.aten.view.default(view_493, [8, 196, 2, 16, 48]);  view_493 = None
        permute_227 = torch.ops.aten.permute.default(view_494, [2, 0, 3, 1, 4]);  view_494 = None
        select_119 = torch.ops.aten.select.int(permute_227, 0, 0)
        select_120 = torch.ops.aten.select.int(permute_227, 0, 1);  permute_227 = None
        expand_125 = torch.ops.aten.expand.default(device_put_9, [8, -1, -1, -1])
        permute_228 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        clone_268 = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_495 = torch.ops.aten.view.default(clone_268, [307328, 3]);  clone_268 = None
        mm_60 = torch.ops.aten.mm.default(view_495, permute_228);  view_495 = permute_228 = None
        view_496 = torch.ops.aten.view.default(mm_60, [8, 196, 196, 16]);  mm_60 = None
        add_201 = torch.ops.aten.add.Tensor(view_496, arg144_1);  view_496 = arg144_1 = None
        permute_229 = torch.ops.aten.permute.default(add_201, [0, 3, 1, 2]);  add_201 = None
        permute_230 = torch.ops.aten.permute.default(select_120, [0, 1, 3, 2]);  select_120 = None
        expand_126 = torch.ops.aten.expand.default(select_119, [8, 16, 196, 48]);  select_119 = None
        clone_269 = torch.ops.aten.clone.default(expand_126, memory_format = torch.contiguous_format);  expand_126 = None
        view_497 = torch.ops.aten.view.default(clone_269, [128, 196, 48]);  clone_269 = None
        expand_127 = torch.ops.aten.expand.default(permute_230, [8, 16, 48, 196]);  permute_230 = None
        clone_270 = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_498 = torch.ops.aten.view.default(clone_270, [128, 48, 196]);  clone_270 = None
        bmm_42 = torch.ops.aten.bmm.default(view_497, view_498);  view_497 = view_498 = None
        view_499 = torch.ops.aten.view.default(bmm_42, [8, 16, 196, 196]);  bmm_42 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_499, 1);  view_499 = None
        amax_default_2 = torch.ops.aten.amax.default(mul_tensor_4, [-1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.14433756729740643);  sub_tensor_2 = None
        exp_40 = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_59 = torch.ops.aten.div.Tensor(exp_40, sum_60);  exp_40 = sum_60 = None
        clone_271 = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        amax_41 = torch.ops.aten.amax.default(clone_271, [-1], True)
        sub_114 = torch.ops.aten.sub.Tensor(clone_271, amax_41);  clone_271 = amax_41 = None
        exp_41 = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_60 = torch.ops.aten.div.Tensor(exp_41, sum_61);  exp_41 = sum_61 = None
        view_500 = torch.ops.aten.view.default(arg145_1, [1, -1, 1, 1]);  arg145_1 = None
        sigmoid_38 = torch.ops.aten.sigmoid.default(view_500)
        sub_115 = torch.ops.aten.sub.Tensor(1.0, sigmoid_38);  sigmoid_38 = None
        mul_211 = torch.ops.aten.mul.Tensor(sub_115, div_59);  sub_115 = div_59 = None
        sigmoid_39 = torch.ops.aten.sigmoid.default(view_500);  view_500 = None
        mul_212 = torch.ops.aten.mul.Tensor(sigmoid_39, div_60);  sigmoid_39 = div_60 = None
        add_202 = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(add_202, [-1])
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(sum_62, -1);  sum_62 = None
        div_61 = torch.ops.aten.div.Tensor(add_202, unsqueeze_69);  add_202 = unsqueeze_69 = None
        permute_231 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        view_501 = torch.ops.aten.view.default(add_200, [1568, 768]);  add_200 = None
        mm_61 = torch.ops.aten.mm.default(view_501, permute_231);  view_501 = permute_231 = None
        view_502 = torch.ops.aten.view.default(mm_61, [8, 196, 768]);  mm_61 = None
        view_503 = torch.ops.aten.view.default(view_502, [8, 196, 16, 48]);  view_502 = None
        permute_232 = torch.ops.aten.permute.default(view_503, [0, 2, 1, 3]);  view_503 = None
        expand_128 = torch.ops.aten.expand.default(div_61, [8, 16, 196, 196]);  div_61 = None
        view_504 = torch.ops.aten.view.default(expand_128, [128, 196, 196]);  expand_128 = None
        expand_129 = torch.ops.aten.expand.default(permute_232, [8, 16, 196, 48]);  permute_232 = None
        clone_273 = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_505 = torch.ops.aten.view.default(clone_273, [128, 196, 48]);  clone_273 = None
        bmm_43 = torch.ops.aten.bmm.default(view_504, view_505);  view_504 = view_505 = None
        view_506 = torch.ops.aten.view.default(bmm_43, [8, 16, 196, 48]);  bmm_43 = None
        permute_233 = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None
        clone_274 = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
        view_507 = torch.ops.aten.view.default(clone_274, [8, 196, 768]);  clone_274 = None
        view_508 = torch.ops.aten.view.default(view_507, [1568, 768]);  view_507 = None
        permute_234 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg148_1, view_508, permute_234);  arg148_1 = view_508 = permute_234 = None
        view_509 = torch.ops.aten.view.default(addmm_64, [8, 196, 768]);  addmm_64 = None
        add_203 = torch.ops.aten.add.Tensor(add_198, view_509);  add_198 = view_509 = None
        clone_276 = torch.ops.aten.clone.default(add_203, memory_format = torch.contiguous_format)
        var_mean_44 = torch.ops.aten.var_mean.correction(clone_276, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_44[0]
        getitem_95 = var_mean_44[1];  var_mean_44 = None
        add_204 = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        sub_116 = torch.ops.aten.sub.Tensor(clone_276, getitem_95);  clone_276 = getitem_95 = None
        mul_213 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_44);  sub_116 = rsqrt_44 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, arg149_1);  mul_213 = arg149_1 = None
        add_205 = torch.ops.aten.add.Tensor(mul_214, arg150_1);  mul_214 = arg150_1 = None
        view_510 = torch.ops.aten.view.default(add_205, [1568, 768]);  add_205 = None
        permute_235 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg152_1, view_510, permute_235);  arg152_1 = view_510 = permute_235 = None
        view_511 = torch.ops.aten.view.default(addmm_65, [8, 196, 3072]);  addmm_65 = None
        mul_215 = torch.ops.aten.mul.Tensor(view_511, 0.5)
        mul_216 = torch.ops.aten.mul.Tensor(view_511, 0.7071067811865476);  view_511 = None
        erf_21 = torch.ops.aten.erf.default(mul_216);  mul_216 = None
        add_206 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_215, add_206);  mul_215 = add_206 = None
        view_512 = torch.ops.aten.view.default(mul_217, [1568, 3072]);  mul_217 = None
        permute_236 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg154_1, view_512, permute_236);  arg154_1 = view_512 = permute_236 = None
        view_513 = torch.ops.aten.view.default(addmm_66, [8, 196, 768]);  addmm_66 = None
        add_207 = torch.ops.aten.add.Tensor(add_203, view_513);  add_203 = view_513 = None
        cat_1 = torch.ops.aten.cat.default([expand_79, add_207], 1);  expand_79 = add_207 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_45[0]
        getitem_97 = var_mean_45[1];  var_mean_45 = None
        add_208 = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        sub_117 = torch.ops.aten.sub.Tensor(cat_1, getitem_97);  getitem_97 = None
        mul_218 = torch.ops.aten.mul.Tensor(sub_117, rsqrt_45);  sub_117 = rsqrt_45 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, arg155_1);  mul_218 = arg155_1 = None
        add_209 = torch.ops.aten.add.Tensor(mul_219, arg156_1);  mul_219 = arg156_1 = None
        permute_237 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        view_514 = torch.ops.aten.view.default(add_209, [1576, 768]);  add_209 = None
        mm_62 = torch.ops.aten.mm.default(view_514, permute_237);  view_514 = permute_237 = None
        view_515 = torch.ops.aten.view.default(mm_62, [8, 197, 2304]);  mm_62 = None
        view_516 = torch.ops.aten.view.default(view_515, [8, 197, 3, 16, 48]);  view_515 = None
        permute_238 = torch.ops.aten.permute.default(view_516, [2, 0, 3, 1, 4]);  view_516 = None
        unbind_2 = torch.ops.aten.unbind.int(permute_238);  permute_238 = None
        getitem_98 = unbind_2[0]
        getitem_99 = unbind_2[1]
        getitem_100 = unbind_2[2];  unbind_2 = None
        permute_239 = torch.ops.aten.permute.default(getitem_99, [0, 1, 3, 2]);  getitem_99 = None
        expand_130 = torch.ops.aten.expand.default(getitem_98, [8, 16, 197, 48]);  getitem_98 = None
        clone_279 = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_517 = torch.ops.aten.view.default(clone_279, [128, 197, 48]);  clone_279 = None
        expand_131 = torch.ops.aten.expand.default(permute_239, [8, 16, 48, 197]);  permute_239 = None
        clone_280 = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_518 = torch.ops.aten.view.default(clone_280, [128, 48, 197]);  clone_280 = None
        bmm_44 = torch.ops.aten.bmm.default(view_517, view_518);  view_517 = view_518 = None
        view_519 = torch.ops.aten.view.default(bmm_44, [8, 16, 197, 197]);  bmm_44 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_519, 1);  view_519 = None
        amax_default_1 = torch.ops.aten.amax.default(mul_tensor_2, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, 0.14433756729740643);  sub_tensor_1 = None
        exp_42 = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_62 = torch.ops.aten.div.Tensor(exp_42, sum_63);  exp_42 = sum_63 = None
        expand_132 = torch.ops.aten.expand.default(div_62, [8, 16, 197, 197]);  div_62 = None
        view_520 = torch.ops.aten.view.default(expand_132, [128, 197, 197]);  expand_132 = None
        expand_133 = torch.ops.aten.expand.default(getitem_100, [8, 16, 197, 48]);  getitem_100 = None
        clone_282 = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_521 = torch.ops.aten.view.default(clone_282, [128, 197, 48]);  clone_282 = None
        bmm_45 = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
        view_522 = torch.ops.aten.view.default(bmm_45, [8, 16, 197, 48]);  bmm_45 = None
        permute_240 = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
        clone_283 = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_523 = torch.ops.aten.view.default(clone_283, [8, 197, 768]);  clone_283 = None
        view_524 = torch.ops.aten.view.default(view_523, [1576, 768]);  view_523 = None
        permute_241 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg159_1, view_524, permute_241);  arg159_1 = view_524 = permute_241 = None
        view_525 = torch.ops.aten.view.default(addmm_67, [8, 197, 768]);  addmm_67 = None
        add_210 = torch.ops.aten.add.Tensor(cat_1, view_525);  cat_1 = view_525 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_210, [2], correction = 0, keepdim = True)
        getitem_101 = var_mean_46[0]
        getitem_102 = var_mean_46[1];  var_mean_46 = None
        add_211 = torch.ops.aten.add.Tensor(getitem_101, 1e-06);  getitem_101 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        sub_119 = torch.ops.aten.sub.Tensor(add_210, getitem_102);  getitem_102 = None
        mul_221 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_46);  sub_119 = rsqrt_46 = None
        mul_222 = torch.ops.aten.mul.Tensor(mul_221, arg160_1);  mul_221 = arg160_1 = None
        add_212 = torch.ops.aten.add.Tensor(mul_222, arg161_1);  mul_222 = arg161_1 = None
        view_526 = torch.ops.aten.view.default(add_212, [1576, 768]);  add_212 = None
        permute_242 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg163_1, view_526, permute_242);  arg163_1 = view_526 = permute_242 = None
        view_527 = torch.ops.aten.view.default(addmm_68, [8, 197, 3072]);  addmm_68 = None
        mul_223 = torch.ops.aten.mul.Tensor(view_527, 0.5)
        mul_224 = torch.ops.aten.mul.Tensor(view_527, 0.7071067811865476);  view_527 = None
        erf_22 = torch.ops.aten.erf.default(mul_224);  mul_224 = None
        add_213 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_223, add_213);  mul_223 = add_213 = None
        view_528 = torch.ops.aten.view.default(mul_225, [1576, 3072]);  mul_225 = None
        permute_243 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg165_1, view_528, permute_243);  arg165_1 = view_528 = permute_243 = None
        view_529 = torch.ops.aten.view.default(addmm_69, [8, 197, 768]);  addmm_69 = None
        add_214 = torch.ops.aten.add.Tensor(add_210, view_529);  add_210 = view_529 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_214, [2], correction = 0, keepdim = True)
        getitem_103 = var_mean_47[0]
        getitem_104 = var_mean_47[1];  var_mean_47 = None
        add_215 = torch.ops.aten.add.Tensor(getitem_103, 1e-06);  getitem_103 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_120 = torch.ops.aten.sub.Tensor(add_214, getitem_104);  getitem_104 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_120, rsqrt_47);  sub_120 = rsqrt_47 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, arg166_1);  mul_226 = arg166_1 = None
        add_216 = torch.ops.aten.add.Tensor(mul_227, arg167_1);  mul_227 = arg167_1 = None
        permute_244 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        view_530 = torch.ops.aten.view.default(add_216, [1576, 768]);  add_216 = None
        mm_63 = torch.ops.aten.mm.default(view_530, permute_244);  view_530 = permute_244 = None
        view_531 = torch.ops.aten.view.default(mm_63, [8, 197, 2304]);  mm_63 = None
        view_532 = torch.ops.aten.view.default(view_531, [8, 197, 3, 16, 48]);  view_531 = None
        permute_245 = torch.ops.aten.permute.default(view_532, [2, 0, 3, 1, 4]);  view_532 = None
        unbind_3 = torch.ops.aten.unbind.int(permute_245);  permute_245 = None
        getitem_105 = unbind_3[0]
        getitem_106 = unbind_3[1]
        getitem_107 = unbind_3[2];  unbind_3 = None
        permute_246 = torch.ops.aten.permute.default(getitem_106, [0, 1, 3, 2]);  getitem_106 = None
        expand_134 = torch.ops.aten.expand.default(getitem_105, [8, 16, 197, 48]);  getitem_105 = None
        clone_287 = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
        view_533 = torch.ops.aten.view.default(clone_287, [128, 197, 48]);  clone_287 = None
        expand_135 = torch.ops.aten.expand.default(permute_246, [8, 16, 48, 197]);  permute_246 = None
        clone_288 = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_534 = torch.ops.aten.view.default(clone_288, [128, 48, 197]);  clone_288 = None
        bmm_46 = torch.ops.aten.bmm.default(view_533, view_534);  view_533 = view_534 = None
        view_535 = torch.ops.aten.view.default(bmm_46, [8, 16, 197, 197]);  bmm_46 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_535, 1);  view_535 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, 0.14433756729740643);  sub_tensor = None
        exp_43 = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_63 = torch.ops.aten.div.Tensor(exp_43, sum_64);  exp_43 = sum_64 = None
        expand_136 = torch.ops.aten.expand.default(div_63, [8, 16, 197, 197]);  div_63 = None
        view_536 = torch.ops.aten.view.default(expand_136, [128, 197, 197]);  expand_136 = None
        expand_137 = torch.ops.aten.expand.default(getitem_107, [8, 16, 197, 48]);  getitem_107 = None
        clone_290 = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_537 = torch.ops.aten.view.default(clone_290, [128, 197, 48]);  clone_290 = None
        bmm_47 = torch.ops.aten.bmm.default(view_536, view_537);  view_536 = view_537 = None
        view_538 = torch.ops.aten.view.default(bmm_47, [8, 16, 197, 48]);  bmm_47 = None
        permute_247 = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
        clone_291 = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_539 = torch.ops.aten.view.default(clone_291, [8, 197, 768]);  clone_291 = None
        view_540 = torch.ops.aten.view.default(view_539, [1576, 768]);  view_539 = None
        permute_248 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg170_1, view_540, permute_248);  arg170_1 = view_540 = permute_248 = None
        view_541 = torch.ops.aten.view.default(addmm_70, [8, 197, 768]);  addmm_70 = None
        add_217 = torch.ops.aten.add.Tensor(add_214, view_541);  add_214 = view_541 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_217, [2], correction = 0, keepdim = True)
        getitem_108 = var_mean_48[0]
        getitem_109 = var_mean_48[1];  var_mean_48 = None
        add_218 = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_122 = torch.ops.aten.sub.Tensor(add_217, getitem_109);  getitem_109 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_122, rsqrt_48);  sub_122 = rsqrt_48 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, arg171_1);  mul_229 = arg171_1 = None
        add_219 = torch.ops.aten.add.Tensor(mul_230, arg172_1);  mul_230 = arg172_1 = None
        view_542 = torch.ops.aten.view.default(add_219, [1576, 768]);  add_219 = None
        permute_249 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg174_1, view_542, permute_249);  arg174_1 = view_542 = permute_249 = None
        view_543 = torch.ops.aten.view.default(addmm_71, [8, 197, 3072]);  addmm_71 = None
        mul_231 = torch.ops.aten.mul.Tensor(view_543, 0.5)
        mul_232 = torch.ops.aten.mul.Tensor(view_543, 0.7071067811865476);  view_543 = None
        erf_23 = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_220 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_231, add_220);  mul_231 = add_220 = None
        view_544 = torch.ops.aten.view.default(mul_233, [1576, 3072]);  mul_233 = None
        permute_250 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg176_1, view_544, permute_250);  arg176_1 = view_544 = permute_250 = None
        view_545 = torch.ops.aten.view.default(addmm_72, [8, 197, 768]);  addmm_72 = None
        add_221 = torch.ops.aten.add.Tensor(add_217, view_545);  add_217 = view_545 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(add_221, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_49[0]
        getitem_111 = var_mean_49[1];  var_mean_49 = None
        add_222 = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_123 = torch.ops.aten.sub.Tensor(add_221, getitem_111);  add_221 = getitem_111 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_123, rsqrt_49);  sub_123 = rsqrt_49 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, arg177_1);  mul_234 = arg177_1 = None
        add_223 = torch.ops.aten.add.Tensor(mul_235, arg178_1);  mul_235 = arg178_1 = None
        select_121 = torch.ops.aten.select.int(add_223, 1, 0);  add_223 = None
        clone_295 = torch.ops.aten.clone.default(select_121);  select_121 = None
        permute_251 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg180_1, clone_295, permute_251);  arg180_1 = clone_295 = permute_251 = None
        return (addmm_73, device_put, device_put_1, device_put_2, device_put_3, device_put_4, device_put_5, device_put_6, device_put_7, device_put_8, device_put_9)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768, 3, 16, 16), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 602112, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 196, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1, 1, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1536, 768), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf8, (16, 3), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf9, (16,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf10, (16,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768, 768), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf16, (3072, 768), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf17, (3072,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768, 3072), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1536, 768), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf23, (16, 3), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf24, (16,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf25, (16,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768, 768), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768, 768), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf31, (3072, 768), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf32, (3072,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768, 3072), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1536, 768), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf38, (16, 3), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf39, (16,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf40, (16,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768, 768), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf46, (3072, 768), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3072,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768, 3072), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1536, 768), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf53, (16, 3), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf54, (16,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf55, (16,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768, 768), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf61, (3072, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf62, (3072,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768, 3072), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf67, (1536, 768), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf68, (16, 3), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf69, (16,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf70, (16,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768, 768), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768, 768), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf76, (3072, 768), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf77, (3072,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768, 3072), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf82, (1536, 768), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf83, (16, 3), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf84, (16,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf85, (16,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 768), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768, 768), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf91, (3072, 768), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf92, (3072,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768, 3072), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf97, (1536, 768), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf98, (16, 3), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf99, (16,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf100, (16,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768, 768), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 768), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf106, (3072, 768), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf107, (3072,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 3072), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf112, (1536, 768), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf113, (16, 3), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf114, (16,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf115, (16,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768, 768), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768, 768), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf121, (3072, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf122, (3072,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768, 3072), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf127, (1536, 768), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf128, (16, 3), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf129, (16,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf130, (16,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768, 768), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768, 768), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf136, (3072, 768), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf137, (3072,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768, 3072), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1536, 768), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf143, (16, 3), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf144, (16,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf145, (16,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768, 768), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768, 768), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf151, (3072, 768), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf152, (3072,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768, 3072), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf157, (2304, 768), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768, 768), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf162, (3072, 768), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf163, (3072,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768, 3072), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf168, (2304, 768), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768, 768), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf173, (3072, 768), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf174, (3072,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768, 3072), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1000, 768), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1000,), is_leaf=True)  # arg180_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)