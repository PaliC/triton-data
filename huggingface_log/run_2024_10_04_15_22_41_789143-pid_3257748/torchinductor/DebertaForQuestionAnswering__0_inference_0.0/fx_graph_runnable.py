
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
        self.register_buffer('_tensor_constant0', tensor(64.))
        self.register_buffer('_tensor_constant1', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant2', tensor(64.))
        self.register_buffer('_tensor_constant3', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant4', tensor(64.))
        self.register_buffer('_tensor_constant5', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant6', tensor(64.))
        self.register_buffer('_tensor_constant7', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant8', tensor(64.))
        self.register_buffer('_tensor_constant9', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant10', tensor(64.))
        self.register_buffer('_tensor_constant11', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant12', tensor(64.))
        self.register_buffer('_tensor_constant13', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant14', tensor(64.))
        self.register_buffer('_tensor_constant15', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant16', tensor(64.))
        self.register_buffer('_tensor_constant17', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant18', tensor(64.))
        self.register_buffer('_tensor_constant19', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant20', tensor(64.))
        self.register_buffer('_tensor_constant21', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant22', tensor(64.))
        self.register_buffer('_tensor_constant23', tensor(-3.4028e+38))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1):
        full = torch.ops.aten.full.default([16, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        embedding = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, arg1_1);  arg3_1 = arg1_1 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        mean = torch.ops.aten.mean.dim(add, [-1], True)
        sub = torch.ops.aten.sub.Tensor(add, mean)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        mean_1 = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(add, mean);  add = mean = None
        add_1 = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
        sqrt = torch.ops.aten.sqrt.default(add_1);  add_1 = None
        div = torch.ops.aten.div.Tensor(sub_1, sqrt);  sub_1 = sqrt = None
        mul = torch.ops.aten.mul.Tensor(arg4_1, div);  arg4_1 = div = None
        add_2 = torch.ops.aten.add.Tensor(mul, arg5_1);  mul = arg5_1 = None
        full_default = torch.ops.aten.full.default([16, 512, 1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        squeeze = torch.ops.aten.squeeze.dim(unsqueeze_2, -2)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
        mul_2 = torch.ops.aten.mul.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        view = torch.ops.aten.view.default(add_2, [8192, 768])
        mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1 = torch.ops.aten.view.default(mm, [16, 512, 2304]);  mm = None
        view_2 = torch.ops.aten.view.default(view_1, [16, 512, 12, -1]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        split = torch.ops.aten.split.Tensor(permute_1, 64, -1);  permute_1 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2];  split = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg7_1, 0);  arg7_1 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        view_3 = torch.ops.aten.view.default(unsqueeze_5, [1, 1, 12, -1]);  unsqueeze_5 = None
        permute_2 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        add_3 = torch.ops.aten.add.Tensor(getitem, permute_2);  getitem = permute_2 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        view_4 = torch.ops.aten.view.default(unsqueeze_7, [1, 1, 12, -1]);  unsqueeze_7 = None
        permute_3 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, permute_3);  getitem_2 = permute_3 = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        mul_3 = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = mul_3 = None
        full_default_1 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_1 = torch.ops.aten.div.Tensor(add_3, full_default_1);  add_3 = full_default_1 = None
        permute_4 = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand = torch.ops.aten.expand.default(div_1, [16, 12, 512, 64]);  div_1 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_5 = torch.ops.aten.view.default(clone, [192, 512, 64]);  clone = None
        expand_1 = torch.ops.aten.expand.default(permute_4, [16, 12, 64, 512]);  permute_4 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_6 = torch.ops.aten.view.default(clone_1, [192, 64, 512]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7 = torch.ops.aten.view.default(bmm, [16, 12, 512, 512]);  bmm = None
        convert_element_type = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type = None
        full_default_2 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant1 = self._tensor_constant1;  _tensor_constant1 = None
        full_default_3 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(full_default_2, full_default_3, view_7);  full_default_3 = view_7 = None
        amax = torch.ops.aten.amax.default(where, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(full_default_2, full_default_4, div_2);  full_default_2 = full_default_4 = div_2 = None
        expand_3 = torch.ops.aten.expand.default(add_4, [16, 12, 512, 64]);  add_4 = None
        clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_9 = torch.ops.aten.view.default(clone_2, [192, 512, 64]);  clone_2 = None
        expand_4 = torch.ops.aten.expand.default(where_1, [16, 12, 512, 512]);  where_1 = None
        view_10 = torch.ops.aten.view.default(expand_4, [192, 512, 512]);  expand_4 = None
        bmm_1 = torch.ops.aten.bmm.default(view_10, view_9);  view_10 = view_9 = None
        view_11 = torch.ops.aten.view.default(bmm_1, [16, 12, 512, 64]);  bmm_1 = None
        permute_5 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_12 = torch.ops.aten.view.default(clone_3, [16, 512, -1]);  clone_3 = None
        view_13 = torch.ops.aten.view.default(view_12, [8192, 768]);  view_12 = None
        permute_6 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm = torch.ops.aten.addmm.default(arg10_1, view_13, permute_6);  arg10_1 = view_13 = permute_6 = None
        view_14 = torch.ops.aten.view.default(addmm, [16, 512, 768]);  addmm = None
        add_5 = torch.ops.aten.add.Tensor(view_14, add_2);  view_14 = add_2 = None
        mean_2 = torch.ops.aten.mean.dim(add_5, [-1], True)
        sub_3 = torch.ops.aten.sub.Tensor(add_5, mean_2)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sub_3, 2);  sub_3 = None
        mean_3 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_5, mean_2);  add_5 = mean_2 = None
        add_6 = torch.ops.aten.add.Tensor(mean_3, 1e-07);  mean_3 = None
        sqrt_2 = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        div_3 = torch.ops.aten.div.Tensor(sub_4, sqrt_2);  sub_4 = sqrt_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(arg11_1, div_3);  arg11_1 = div_3 = None
        add_7 = torch.ops.aten.add.Tensor(mul_4, arg12_1);  mul_4 = arg12_1 = None
        view_15 = torch.ops.aten.view.default(add_7, [8192, 768])
        permute_7 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg14_1, view_15, permute_7);  arg14_1 = view_15 = permute_7 = None
        view_16 = torch.ops.aten.view.default(addmm_1, [16, 512, 3072]);  addmm_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_16, 0.5)
        mul_6 = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476);  view_16 = None
        erf = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
        view_17 = torch.ops.aten.view.default(mul_7, [8192, 3072]);  mul_7 = None
        permute_8 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg16_1, view_17, permute_8);  arg16_1 = view_17 = permute_8 = None
        view_18 = torch.ops.aten.view.default(addmm_2, [16, 512, 768]);  addmm_2 = None
        add_9 = torch.ops.aten.add.Tensor(view_18, add_7);  view_18 = add_7 = None
        mean_4 = torch.ops.aten.mean.dim(add_9, [-1], True)
        sub_5 = torch.ops.aten.sub.Tensor(add_9, mean_4)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sub_5, 2);  sub_5 = None
        mean_5 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_9, mean_4);  add_9 = mean_4 = None
        add_10 = torch.ops.aten.add.Tensor(mean_5, 1e-07);  mean_5 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        div_4 = torch.ops.aten.div.Tensor(sub_6, sqrt_3);  sub_6 = sqrt_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(arg17_1, div_4);  arg17_1 = div_4 = None
        add_11 = torch.ops.aten.add.Tensor(mul_8, arg18_1);  mul_8 = arg18_1 = None
        permute_9 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        view_19 = torch.ops.aten.view.default(add_11, [8192, 768])
        mm_1 = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20 = torch.ops.aten.view.default(mm_1, [16, 512, 2304]);  mm_1 = None
        view_21 = torch.ops.aten.view.default(view_20, [16, 512, 12, -1]);  view_20 = None
        permute_10 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        split_1 = torch.ops.aten.split.Tensor(permute_10, 64, -1);  permute_10 = None
        getitem_3 = split_1[0]
        getitem_4 = split_1[1]
        getitem_5 = split_1[2];  split_1 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg20_1, 0);  arg20_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
        view_22 = torch.ops.aten.view.default(unsqueeze_9, [1, 1, 12, -1]);  unsqueeze_9 = None
        permute_11 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_3, permute_11);  getitem_3 = permute_11 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(arg21_1, 0);  arg21_1 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
        view_23 = torch.ops.aten.view.default(unsqueeze_11, [1, 1, 12, -1]);  unsqueeze_11 = None
        permute_12 = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_5, permute_12);  getitem_5 = permute_12 = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        mul_9 = torch.ops.aten.mul.Tensor(lift_fresh_copy_2, 1);  lift_fresh_copy_2 = mul_9 = None
        full_default_5 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_5 = torch.ops.aten.div.Tensor(add_12, full_default_5);  add_12 = full_default_5 = None
        permute_13 = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
        expand_5 = torch.ops.aten.expand.default(div_5, [16, 12, 512, 64]);  div_5 = None
        clone_4 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_24 = torch.ops.aten.view.default(clone_4, [192, 512, 64]);  clone_4 = None
        expand_6 = torch.ops.aten.expand.default(permute_13, [16, 12, 64, 512]);  permute_13 = None
        clone_5 = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_25 = torch.ops.aten.view.default(clone_5, [192, 64, 512]);  clone_5 = None
        bmm_2 = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = view_25 = None
        view_26 = torch.ops.aten.view.default(bmm_2, [16, 12, 512, 512]);  bmm_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_1 = None
        full_default_6 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant3 = self._tensor_constant3;  _tensor_constant3 = None
        full_default_7 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_2 = torch.ops.aten.where.self(full_default_6, full_default_7, view_26);  full_default_7 = view_26 = None
        amax_1 = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(where_2, amax_1);  where_2 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        full_default_8 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(full_default_6, full_default_8, div_6);  full_default_6 = full_default_8 = div_6 = None
        expand_8 = torch.ops.aten.expand.default(add_13, [16, 12, 512, 64]);  add_13 = None
        clone_6 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_28 = torch.ops.aten.view.default(clone_6, [192, 512, 64]);  clone_6 = None
        expand_9 = torch.ops.aten.expand.default(where_3, [16, 12, 512, 512]);  where_3 = None
        view_29 = torch.ops.aten.view.default(expand_9, [192, 512, 512]);  expand_9 = None
        bmm_3 = torch.ops.aten.bmm.default(view_29, view_28);  view_29 = view_28 = None
        view_30 = torch.ops.aten.view.default(bmm_3, [16, 12, 512, 64]);  bmm_3 = None
        permute_14 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_7 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_31 = torch.ops.aten.view.default(clone_7, [16, 512, -1]);  clone_7 = None
        view_32 = torch.ops.aten.view.default(view_31, [8192, 768]);  view_31 = None
        permute_15 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg23_1, view_32, permute_15);  arg23_1 = view_32 = permute_15 = None
        view_33 = torch.ops.aten.view.default(addmm_3, [16, 512, 768]);  addmm_3 = None
        add_14 = torch.ops.aten.add.Tensor(view_33, add_11);  view_33 = add_11 = None
        mean_6 = torch.ops.aten.mean.dim(add_14, [-1], True)
        sub_8 = torch.ops.aten.sub.Tensor(add_14, mean_6)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sub_8, 2);  sub_8 = None
        mean_7 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_14, mean_6);  add_14 = mean_6 = None
        add_15 = torch.ops.aten.add.Tensor(mean_7, 1e-07);  mean_7 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_15);  add_15 = None
        div_7 = torch.ops.aten.div.Tensor(sub_9, sqrt_5);  sub_9 = sqrt_5 = None
        mul_10 = torch.ops.aten.mul.Tensor(arg24_1, div_7);  arg24_1 = div_7 = None
        add_16 = torch.ops.aten.add.Tensor(mul_10, arg25_1);  mul_10 = arg25_1 = None
        view_34 = torch.ops.aten.view.default(add_16, [8192, 768])
        permute_16 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg27_1, view_34, permute_16);  arg27_1 = view_34 = permute_16 = None
        view_35 = torch.ops.aten.view.default(addmm_4, [16, 512, 3072]);  addmm_4 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_35, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_17 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_17);  mul_11 = add_17 = None
        view_36 = torch.ops.aten.view.default(mul_13, [8192, 3072]);  mul_13 = None
        permute_17 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg29_1, view_36, permute_17);  arg29_1 = view_36 = permute_17 = None
        view_37 = torch.ops.aten.view.default(addmm_5, [16, 512, 768]);  addmm_5 = None
        add_18 = torch.ops.aten.add.Tensor(view_37, add_16);  view_37 = add_16 = None
        mean_8 = torch.ops.aten.mean.dim(add_18, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(add_18, mean_8)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(sub_10, 2);  sub_10 = None
        mean_9 = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_18, mean_8);  add_18 = mean_8 = None
        add_19 = torch.ops.aten.add.Tensor(mean_9, 1e-07);  mean_9 = None
        sqrt_6 = torch.ops.aten.sqrt.default(add_19);  add_19 = None
        div_8 = torch.ops.aten.div.Tensor(sub_11, sqrt_6);  sub_11 = sqrt_6 = None
        mul_14 = torch.ops.aten.mul.Tensor(arg30_1, div_8);  arg30_1 = div_8 = None
        add_20 = torch.ops.aten.add.Tensor(mul_14, arg31_1);  mul_14 = arg31_1 = None
        permute_18 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        view_38 = torch.ops.aten.view.default(add_20, [8192, 768])
        mm_2 = torch.ops.aten.mm.default(view_38, permute_18);  view_38 = permute_18 = None
        view_39 = torch.ops.aten.view.default(mm_2, [16, 512, 2304]);  mm_2 = None
        view_40 = torch.ops.aten.view.default(view_39, [16, 512, 12, -1]);  view_39 = None
        permute_19 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        split_2 = torch.ops.aten.split.Tensor(permute_19, 64, -1);  permute_19 = None
        getitem_6 = split_2[0]
        getitem_7 = split_2[1]
        getitem_8 = split_2[2];  split_2 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg33_1, 0);  arg33_1 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        view_41 = torch.ops.aten.view.default(unsqueeze_13, [1, 1, 12, -1]);  unsqueeze_13 = None
        permute_20 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_6, permute_20);  getitem_6 = permute_20 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(arg34_1, 0);  arg34_1 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
        view_42 = torch.ops.aten.view.default(unsqueeze_15, [1, 1, 12, -1]);  unsqueeze_15 = None
        permute_21 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_8, permute_21);  getitem_8 = permute_21 = None
        _tensor_constant4 = self._tensor_constant4
        lift_fresh_copy_4 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        mul_15 = torch.ops.aten.mul.Tensor(lift_fresh_copy_4, 1);  lift_fresh_copy_4 = mul_15 = None
        full_default_9 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_9 = torch.ops.aten.div.Tensor(add_21, full_default_9);  add_21 = full_default_9 = None
        permute_22 = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
        expand_10 = torch.ops.aten.expand.default(div_9, [16, 12, 512, 64]);  div_9 = None
        clone_8 = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_43 = torch.ops.aten.view.default(clone_8, [192, 512, 64]);  clone_8 = None
        expand_11 = torch.ops.aten.expand.default(permute_22, [16, 12, 64, 512]);  permute_22 = None
        clone_9 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_44 = torch.ops.aten.view.default(clone_9, [192, 64, 512]);  clone_9 = None
        bmm_4 = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
        view_45 = torch.ops.aten.view.default(bmm_4, [16, 12, 512, 512]);  bmm_4 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_2 = None
        full_default_10 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant5 = self._tensor_constant5;  _tensor_constant5 = None
        full_default_11 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_4 = torch.ops.aten.where.self(full_default_10, full_default_11, view_45);  full_default_11 = view_45 = None
        amax_2 = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_12 = torch.ops.aten.sub.Tensor(where_4, amax_2);  where_4 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        full_default_12 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5 = torch.ops.aten.where.self(full_default_10, full_default_12, div_10);  full_default_10 = full_default_12 = div_10 = None
        expand_13 = torch.ops.aten.expand.default(add_22, [16, 12, 512, 64]);  add_22 = None
        clone_10 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_47 = torch.ops.aten.view.default(clone_10, [192, 512, 64]);  clone_10 = None
        expand_14 = torch.ops.aten.expand.default(where_5, [16, 12, 512, 512]);  where_5 = None
        view_48 = torch.ops.aten.view.default(expand_14, [192, 512, 512]);  expand_14 = None
        bmm_5 = torch.ops.aten.bmm.default(view_48, view_47);  view_48 = view_47 = None
        view_49 = torch.ops.aten.view.default(bmm_5, [16, 12, 512, 64]);  bmm_5 = None
        permute_23 = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        clone_11 = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_50 = torch.ops.aten.view.default(clone_11, [16, 512, -1]);  clone_11 = None
        view_51 = torch.ops.aten.view.default(view_50, [8192, 768]);  view_50 = None
        permute_24 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg36_1, view_51, permute_24);  arg36_1 = view_51 = permute_24 = None
        view_52 = torch.ops.aten.view.default(addmm_6, [16, 512, 768]);  addmm_6 = None
        add_23 = torch.ops.aten.add.Tensor(view_52, add_20);  view_52 = add_20 = None
        mean_10 = torch.ops.aten.mean.dim(add_23, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(add_23, mean_10)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(sub_13, 2);  sub_13 = None
        mean_11 = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_23, mean_10);  add_23 = mean_10 = None
        add_24 = torch.ops.aten.add.Tensor(mean_11, 1e-07);  mean_11 = None
        sqrt_8 = torch.ops.aten.sqrt.default(add_24);  add_24 = None
        div_11 = torch.ops.aten.div.Tensor(sub_14, sqrt_8);  sub_14 = sqrt_8 = None
        mul_16 = torch.ops.aten.mul.Tensor(arg37_1, div_11);  arg37_1 = div_11 = None
        add_25 = torch.ops.aten.add.Tensor(mul_16, arg38_1);  mul_16 = arg38_1 = None
        view_53 = torch.ops.aten.view.default(add_25, [8192, 768])
        permute_25 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg40_1, view_53, permute_25);  arg40_1 = view_53 = permute_25 = None
        view_54 = torch.ops.aten.view.default(addmm_7, [16, 512, 3072]);  addmm_7 = None
        mul_17 = torch.ops.aten.mul.Tensor(view_54, 0.5)
        mul_18 = torch.ops.aten.mul.Tensor(view_54, 0.7071067811865476);  view_54 = None
        erf_2 = torch.ops.aten.erf.default(mul_18);  mul_18 = None
        add_26 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_17, add_26);  mul_17 = add_26 = None
        view_55 = torch.ops.aten.view.default(mul_19, [8192, 3072]);  mul_19 = None
        permute_26 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg42_1, view_55, permute_26);  arg42_1 = view_55 = permute_26 = None
        view_56 = torch.ops.aten.view.default(addmm_8, [16, 512, 768]);  addmm_8 = None
        add_27 = torch.ops.aten.add.Tensor(view_56, add_25);  view_56 = add_25 = None
        mean_12 = torch.ops.aten.mean.dim(add_27, [-1], True)
        sub_15 = torch.ops.aten.sub.Tensor(add_27, mean_12)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(sub_15, 2);  sub_15 = None
        mean_13 = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_27, mean_12);  add_27 = mean_12 = None
        add_28 = torch.ops.aten.add.Tensor(mean_13, 1e-07);  mean_13 = None
        sqrt_9 = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        div_12 = torch.ops.aten.div.Tensor(sub_16, sqrt_9);  sub_16 = sqrt_9 = None
        mul_20 = torch.ops.aten.mul.Tensor(arg43_1, div_12);  arg43_1 = div_12 = None
        add_29 = torch.ops.aten.add.Tensor(mul_20, arg44_1);  mul_20 = arg44_1 = None
        permute_27 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        view_57 = torch.ops.aten.view.default(add_29, [8192, 768])
        mm_3 = torch.ops.aten.mm.default(view_57, permute_27);  view_57 = permute_27 = None
        view_58 = torch.ops.aten.view.default(mm_3, [16, 512, 2304]);  mm_3 = None
        view_59 = torch.ops.aten.view.default(view_58, [16, 512, 12, -1]);  view_58 = None
        permute_28 = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        split_3 = torch.ops.aten.split.Tensor(permute_28, 64, -1);  permute_28 = None
        getitem_9 = split_3[0]
        getitem_10 = split_3[1]
        getitem_11 = split_3[2];  split_3 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg46_1, 0);  arg46_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        view_60 = torch.ops.aten.view.default(unsqueeze_17, [1, 1, 12, -1]);  unsqueeze_17 = None
        permute_29 = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_9, permute_29);  getitem_9 = permute_29 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(arg47_1, 0);  arg47_1 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
        view_61 = torch.ops.aten.view.default(unsqueeze_19, [1, 1, 12, -1]);  unsqueeze_19 = None
        permute_30 = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        add_31 = torch.ops.aten.add.Tensor(getitem_11, permute_30);  getitem_11 = permute_30 = None
        _tensor_constant6 = self._tensor_constant6
        lift_fresh_copy_6 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        mul_21 = torch.ops.aten.mul.Tensor(lift_fresh_copy_6, 1);  lift_fresh_copy_6 = mul_21 = None
        full_default_13 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_13 = torch.ops.aten.div.Tensor(add_30, full_default_13);  add_30 = full_default_13 = None
        permute_31 = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
        expand_15 = torch.ops.aten.expand.default(div_13, [16, 12, 512, 64]);  div_13 = None
        clone_12 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_62 = torch.ops.aten.view.default(clone_12, [192, 512, 64]);  clone_12 = None
        expand_16 = torch.ops.aten.expand.default(permute_31, [16, 12, 64, 512]);  permute_31 = None
        clone_13 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_63 = torch.ops.aten.view.default(clone_13, [192, 64, 512]);  clone_13 = None
        bmm_6 = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
        view_64 = torch.ops.aten.view.default(bmm_6, [16, 12, 512, 512]);  bmm_6 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_3 = None
        full_default_14 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant7 = self._tensor_constant7;  _tensor_constant7 = None
        full_default_15 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_6 = torch.ops.aten.where.self(full_default_14, full_default_15, view_64);  full_default_15 = view_64 = None
        amax_3 = torch.ops.aten.amax.default(where_6, [-1], True)
        sub_17 = torch.ops.aten.sub.Tensor(where_6, amax_3);  where_6 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        full_default_16 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(full_default_14, full_default_16, div_14);  full_default_14 = full_default_16 = div_14 = None
        expand_18 = torch.ops.aten.expand.default(add_31, [16, 12, 512, 64]);  add_31 = None
        clone_14 = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_66 = torch.ops.aten.view.default(clone_14, [192, 512, 64]);  clone_14 = None
        expand_19 = torch.ops.aten.expand.default(where_7, [16, 12, 512, 512]);  where_7 = None
        view_67 = torch.ops.aten.view.default(expand_19, [192, 512, 512]);  expand_19 = None
        bmm_7 = torch.ops.aten.bmm.default(view_67, view_66);  view_67 = view_66 = None
        view_68 = torch.ops.aten.view.default(bmm_7, [16, 12, 512, 64]);  bmm_7 = None
        permute_32 = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_15 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_69 = torch.ops.aten.view.default(clone_15, [16, 512, -1]);  clone_15 = None
        view_70 = torch.ops.aten.view.default(view_69, [8192, 768]);  view_69 = None
        permute_33 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg49_1, view_70, permute_33);  arg49_1 = view_70 = permute_33 = None
        view_71 = torch.ops.aten.view.default(addmm_9, [16, 512, 768]);  addmm_9 = None
        add_32 = torch.ops.aten.add.Tensor(view_71, add_29);  view_71 = add_29 = None
        mean_14 = torch.ops.aten.mean.dim(add_32, [-1], True)
        sub_18 = torch.ops.aten.sub.Tensor(add_32, mean_14)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(sub_18, 2);  sub_18 = None
        mean_15 = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_32, mean_14);  add_32 = mean_14 = None
        add_33 = torch.ops.aten.add.Tensor(mean_15, 1e-07);  mean_15 = None
        sqrt_11 = torch.ops.aten.sqrt.default(add_33);  add_33 = None
        div_15 = torch.ops.aten.div.Tensor(sub_19, sqrt_11);  sub_19 = sqrt_11 = None
        mul_22 = torch.ops.aten.mul.Tensor(arg50_1, div_15);  arg50_1 = div_15 = None
        add_34 = torch.ops.aten.add.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
        view_72 = torch.ops.aten.view.default(add_34, [8192, 768])
        permute_34 = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg53_1, view_72, permute_34);  arg53_1 = view_72 = permute_34 = None
        view_73 = torch.ops.aten.view.default(addmm_10, [16, 512, 3072]);  addmm_10 = None
        mul_23 = torch.ops.aten.mul.Tensor(view_73, 0.5)
        mul_24 = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
        erf_3 = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_35 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_23, add_35);  mul_23 = add_35 = None
        view_74 = torch.ops.aten.view.default(mul_25, [8192, 3072]);  mul_25 = None
        permute_35 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg55_1, view_74, permute_35);  arg55_1 = view_74 = permute_35 = None
        view_75 = torch.ops.aten.view.default(addmm_11, [16, 512, 768]);  addmm_11 = None
        add_36 = torch.ops.aten.add.Tensor(view_75, add_34);  view_75 = add_34 = None
        mean_16 = torch.ops.aten.mean.dim(add_36, [-1], True)
        sub_20 = torch.ops.aten.sub.Tensor(add_36, mean_16)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(sub_20, 2);  sub_20 = None
        mean_17 = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_36, mean_16);  add_36 = mean_16 = None
        add_37 = torch.ops.aten.add.Tensor(mean_17, 1e-07);  mean_17 = None
        sqrt_12 = torch.ops.aten.sqrt.default(add_37);  add_37 = None
        div_16 = torch.ops.aten.div.Tensor(sub_21, sqrt_12);  sub_21 = sqrt_12 = None
        mul_26 = torch.ops.aten.mul.Tensor(arg56_1, div_16);  arg56_1 = div_16 = None
        add_38 = torch.ops.aten.add.Tensor(mul_26, arg57_1);  mul_26 = arg57_1 = None
        permute_36 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        view_76 = torch.ops.aten.view.default(add_38, [8192, 768])
        mm_4 = torch.ops.aten.mm.default(view_76, permute_36);  view_76 = permute_36 = None
        view_77 = torch.ops.aten.view.default(mm_4, [16, 512, 2304]);  mm_4 = None
        view_78 = torch.ops.aten.view.default(view_77, [16, 512, 12, -1]);  view_77 = None
        permute_37 = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        split_4 = torch.ops.aten.split.Tensor(permute_37, 64, -1);  permute_37 = None
        getitem_12 = split_4[0]
        getitem_13 = split_4[1]
        getitem_14 = split_4[2];  split_4 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(arg59_1, 0);  arg59_1 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None
        view_79 = torch.ops.aten.view.default(unsqueeze_21, [1, 1, 12, -1]);  unsqueeze_21 = None
        permute_38 = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_12, permute_38);  getitem_12 = permute_38 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(arg60_1, 0);  arg60_1 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
        view_80 = torch.ops.aten.view.default(unsqueeze_23, [1, 1, 12, -1]);  unsqueeze_23 = None
        permute_39 = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_14, permute_39);  getitem_14 = permute_39 = None
        _tensor_constant8 = self._tensor_constant8
        lift_fresh_copy_8 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        mul_27 = torch.ops.aten.mul.Tensor(lift_fresh_copy_8, 1);  lift_fresh_copy_8 = mul_27 = None
        full_default_17 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_17 = torch.ops.aten.div.Tensor(add_39, full_default_17);  add_39 = full_default_17 = None
        permute_40 = torch.ops.aten.permute.default(getitem_13, [0, 1, 3, 2]);  getitem_13 = None
        expand_20 = torch.ops.aten.expand.default(div_17, [16, 12, 512, 64]);  div_17 = None
        clone_16 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_81 = torch.ops.aten.view.default(clone_16, [192, 512, 64]);  clone_16 = None
        expand_21 = torch.ops.aten.expand.default(permute_40, [16, 12, 64, 512]);  permute_40 = None
        clone_17 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_82 = torch.ops.aten.view.default(clone_17, [192, 64, 512]);  clone_17 = None
        bmm_8 = torch.ops.aten.bmm.default(view_81, view_82);  view_81 = view_82 = None
        view_83 = torch.ops.aten.view.default(bmm_8, [16, 12, 512, 512]);  bmm_8 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_4 = None
        full_default_18 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant9 = self._tensor_constant9;  _tensor_constant9 = None
        full_default_19 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_8 = torch.ops.aten.where.self(full_default_18, full_default_19, view_83);  full_default_19 = view_83 = None
        amax_4 = torch.ops.aten.amax.default(where_8, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(where_8, amax_4);  where_8 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        full_default_20 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9 = torch.ops.aten.where.self(full_default_18, full_default_20, div_18);  full_default_18 = full_default_20 = div_18 = None
        expand_23 = torch.ops.aten.expand.default(add_40, [16, 12, 512, 64]);  add_40 = None
        clone_18 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_85 = torch.ops.aten.view.default(clone_18, [192, 512, 64]);  clone_18 = None
        expand_24 = torch.ops.aten.expand.default(where_9, [16, 12, 512, 512]);  where_9 = None
        view_86 = torch.ops.aten.view.default(expand_24, [192, 512, 512]);  expand_24 = None
        bmm_9 = torch.ops.aten.bmm.default(view_86, view_85);  view_86 = view_85 = None
        view_87 = torch.ops.aten.view.default(bmm_9, [16, 12, 512, 64]);  bmm_9 = None
        permute_41 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        clone_19 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_88 = torch.ops.aten.view.default(clone_19, [16, 512, -1]);  clone_19 = None
        view_89 = torch.ops.aten.view.default(view_88, [8192, 768]);  view_88 = None
        permute_42 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg62_1, view_89, permute_42);  arg62_1 = view_89 = permute_42 = None
        view_90 = torch.ops.aten.view.default(addmm_12, [16, 512, 768]);  addmm_12 = None
        add_41 = torch.ops.aten.add.Tensor(view_90, add_38);  view_90 = add_38 = None
        mean_18 = torch.ops.aten.mean.dim(add_41, [-1], True)
        sub_23 = torch.ops.aten.sub.Tensor(add_41, mean_18)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(sub_23, 2);  sub_23 = None
        mean_19 = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_41, mean_18);  add_41 = mean_18 = None
        add_42 = torch.ops.aten.add.Tensor(mean_19, 1e-07);  mean_19 = None
        sqrt_14 = torch.ops.aten.sqrt.default(add_42);  add_42 = None
        div_19 = torch.ops.aten.div.Tensor(sub_24, sqrt_14);  sub_24 = sqrt_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(arg63_1, div_19);  arg63_1 = div_19 = None
        add_43 = torch.ops.aten.add.Tensor(mul_28, arg64_1);  mul_28 = arg64_1 = None
        view_91 = torch.ops.aten.view.default(add_43, [8192, 768])
        permute_43 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg66_1, view_91, permute_43);  arg66_1 = view_91 = permute_43 = None
        view_92 = torch.ops.aten.view.default(addmm_13, [16, 512, 3072]);  addmm_13 = None
        mul_29 = torch.ops.aten.mul.Tensor(view_92, 0.5)
        mul_30 = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476);  view_92 = None
        erf_4 = torch.ops.aten.erf.default(mul_30);  mul_30 = None
        add_44 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_29, add_44);  mul_29 = add_44 = None
        view_93 = torch.ops.aten.view.default(mul_31, [8192, 3072]);  mul_31 = None
        permute_44 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg68_1, view_93, permute_44);  arg68_1 = view_93 = permute_44 = None
        view_94 = torch.ops.aten.view.default(addmm_14, [16, 512, 768]);  addmm_14 = None
        add_45 = torch.ops.aten.add.Tensor(view_94, add_43);  view_94 = add_43 = None
        mean_20 = torch.ops.aten.mean.dim(add_45, [-1], True)
        sub_25 = torch.ops.aten.sub.Tensor(add_45, mean_20)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(sub_25, 2);  sub_25 = None
        mean_21 = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_45, mean_20);  add_45 = mean_20 = None
        add_46 = torch.ops.aten.add.Tensor(mean_21, 1e-07);  mean_21 = None
        sqrt_15 = torch.ops.aten.sqrt.default(add_46);  add_46 = None
        div_20 = torch.ops.aten.div.Tensor(sub_26, sqrt_15);  sub_26 = sqrt_15 = None
        mul_32 = torch.ops.aten.mul.Tensor(arg69_1, div_20);  arg69_1 = div_20 = None
        add_47 = torch.ops.aten.add.Tensor(mul_32, arg70_1);  mul_32 = arg70_1 = None
        permute_45 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        view_95 = torch.ops.aten.view.default(add_47, [8192, 768])
        mm_5 = torch.ops.aten.mm.default(view_95, permute_45);  view_95 = permute_45 = None
        view_96 = torch.ops.aten.view.default(mm_5, [16, 512, 2304]);  mm_5 = None
        view_97 = torch.ops.aten.view.default(view_96, [16, 512, 12, -1]);  view_96 = None
        permute_46 = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        split_5 = torch.ops.aten.split.Tensor(permute_46, 64, -1);  permute_46 = None
        getitem_15 = split_5[0]
        getitem_16 = split_5[1]
        getitem_17 = split_5[2];  split_5 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(arg72_1, 0);  arg72_1 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, 1);  unsqueeze_24 = None
        view_98 = torch.ops.aten.view.default(unsqueeze_25, [1, 1, 12, -1]);  unsqueeze_25 = None
        permute_47 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_15, permute_47);  getitem_15 = permute_47 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(arg73_1, 0);  arg73_1 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, 1);  unsqueeze_26 = None
        view_99 = torch.ops.aten.view.default(unsqueeze_27, [1, 1, 12, -1]);  unsqueeze_27 = None
        permute_48 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_17, permute_48);  getitem_17 = permute_48 = None
        _tensor_constant10 = self._tensor_constant10
        lift_fresh_copy_10 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        mul_33 = torch.ops.aten.mul.Tensor(lift_fresh_copy_10, 1);  lift_fresh_copy_10 = mul_33 = None
        full_default_21 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_21 = torch.ops.aten.div.Tensor(add_48, full_default_21);  add_48 = full_default_21 = None
        permute_49 = torch.ops.aten.permute.default(getitem_16, [0, 1, 3, 2]);  getitem_16 = None
        expand_25 = torch.ops.aten.expand.default(div_21, [16, 12, 512, 64]);  div_21 = None
        clone_20 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_100 = torch.ops.aten.view.default(clone_20, [192, 512, 64]);  clone_20 = None
        expand_26 = torch.ops.aten.expand.default(permute_49, [16, 12, 64, 512]);  permute_49 = None
        clone_21 = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_101 = torch.ops.aten.view.default(clone_21, [192, 64, 512]);  clone_21 = None
        bmm_10 = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
        view_102 = torch.ops.aten.view.default(bmm_10, [16, 12, 512, 512]);  bmm_10 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_5 = None
        full_default_22 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant11 = self._tensor_constant11;  _tensor_constant11 = None
        full_default_23 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_10 = torch.ops.aten.where.self(full_default_22, full_default_23, view_102);  full_default_23 = view_102 = None
        amax_5 = torch.ops.aten.amax.default(where_10, [-1], True)
        sub_27 = torch.ops.aten.sub.Tensor(where_10, amax_5);  where_10 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_22 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        full_default_24 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11 = torch.ops.aten.where.self(full_default_22, full_default_24, div_22);  full_default_22 = full_default_24 = div_22 = None
        expand_28 = torch.ops.aten.expand.default(add_49, [16, 12, 512, 64]);  add_49 = None
        clone_22 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_104 = torch.ops.aten.view.default(clone_22, [192, 512, 64]);  clone_22 = None
        expand_29 = torch.ops.aten.expand.default(where_11, [16, 12, 512, 512]);  where_11 = None
        view_105 = torch.ops.aten.view.default(expand_29, [192, 512, 512]);  expand_29 = None
        bmm_11 = torch.ops.aten.bmm.default(view_105, view_104);  view_105 = view_104 = None
        view_106 = torch.ops.aten.view.default(bmm_11, [16, 12, 512, 64]);  bmm_11 = None
        permute_50 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_23 = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_107 = torch.ops.aten.view.default(clone_23, [16, 512, -1]);  clone_23 = None
        view_108 = torch.ops.aten.view.default(view_107, [8192, 768]);  view_107 = None
        permute_51 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg75_1, view_108, permute_51);  arg75_1 = view_108 = permute_51 = None
        view_109 = torch.ops.aten.view.default(addmm_15, [16, 512, 768]);  addmm_15 = None
        add_50 = torch.ops.aten.add.Tensor(view_109, add_47);  view_109 = add_47 = None
        mean_22 = torch.ops.aten.mean.dim(add_50, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(add_50, mean_22)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(sub_28, 2);  sub_28 = None
        mean_23 = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_50, mean_22);  add_50 = mean_22 = None
        add_51 = torch.ops.aten.add.Tensor(mean_23, 1e-07);  mean_23 = None
        sqrt_17 = torch.ops.aten.sqrt.default(add_51);  add_51 = None
        div_23 = torch.ops.aten.div.Tensor(sub_29, sqrt_17);  sub_29 = sqrt_17 = None
        mul_34 = torch.ops.aten.mul.Tensor(arg76_1, div_23);  arg76_1 = div_23 = None
        add_52 = torch.ops.aten.add.Tensor(mul_34, arg77_1);  mul_34 = arg77_1 = None
        view_110 = torch.ops.aten.view.default(add_52, [8192, 768])
        permute_52 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg79_1, view_110, permute_52);  arg79_1 = view_110 = permute_52 = None
        view_111 = torch.ops.aten.view.default(addmm_16, [16, 512, 3072]);  addmm_16 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_111, 0.5)
        mul_36 = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
        erf_5 = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_53 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_35, add_53);  mul_35 = add_53 = None
        view_112 = torch.ops.aten.view.default(mul_37, [8192, 3072]);  mul_37 = None
        permute_53 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg81_1, view_112, permute_53);  arg81_1 = view_112 = permute_53 = None
        view_113 = torch.ops.aten.view.default(addmm_17, [16, 512, 768]);  addmm_17 = None
        add_54 = torch.ops.aten.add.Tensor(view_113, add_52);  view_113 = add_52 = None
        mean_24 = torch.ops.aten.mean.dim(add_54, [-1], True)
        sub_30 = torch.ops.aten.sub.Tensor(add_54, mean_24)
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(sub_30, 2);  sub_30 = None
        mean_25 = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_54, mean_24);  add_54 = mean_24 = None
        add_55 = torch.ops.aten.add.Tensor(mean_25, 1e-07);  mean_25 = None
        sqrt_18 = torch.ops.aten.sqrt.default(add_55);  add_55 = None
        div_24 = torch.ops.aten.div.Tensor(sub_31, sqrt_18);  sub_31 = sqrt_18 = None
        mul_38 = torch.ops.aten.mul.Tensor(arg82_1, div_24);  arg82_1 = div_24 = None
        add_56 = torch.ops.aten.add.Tensor(mul_38, arg83_1);  mul_38 = arg83_1 = None
        permute_54 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        view_114 = torch.ops.aten.view.default(add_56, [8192, 768])
        mm_6 = torch.ops.aten.mm.default(view_114, permute_54);  view_114 = permute_54 = None
        view_115 = torch.ops.aten.view.default(mm_6, [16, 512, 2304]);  mm_6 = None
        view_116 = torch.ops.aten.view.default(view_115, [16, 512, 12, -1]);  view_115 = None
        permute_55 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        split_6 = torch.ops.aten.split.Tensor(permute_55, 64, -1);  permute_55 = None
        getitem_18 = split_6[0]
        getitem_19 = split_6[1]
        getitem_20 = split_6[2];  split_6 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(arg85_1, 0);  arg85_1 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, 1);  unsqueeze_28 = None
        view_117 = torch.ops.aten.view.default(unsqueeze_29, [1, 1, 12, -1]);  unsqueeze_29 = None
        permute_56 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_18, permute_56);  getitem_18 = permute_56 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(arg86_1, 0);  arg86_1 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, 1);  unsqueeze_30 = None
        view_118 = torch.ops.aten.view.default(unsqueeze_31, [1, 1, 12, -1]);  unsqueeze_31 = None
        permute_57 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_20, permute_57);  getitem_20 = permute_57 = None
        _tensor_constant12 = self._tensor_constant12
        lift_fresh_copy_12 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        mul_39 = torch.ops.aten.mul.Tensor(lift_fresh_copy_12, 1);  lift_fresh_copy_12 = mul_39 = None
        full_default_25 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_25 = torch.ops.aten.div.Tensor(add_57, full_default_25);  add_57 = full_default_25 = None
        permute_58 = torch.ops.aten.permute.default(getitem_19, [0, 1, 3, 2]);  getitem_19 = None
        expand_30 = torch.ops.aten.expand.default(div_25, [16, 12, 512, 64]);  div_25 = None
        clone_24 = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_119 = torch.ops.aten.view.default(clone_24, [192, 512, 64]);  clone_24 = None
        expand_31 = torch.ops.aten.expand.default(permute_58, [16, 12, 64, 512]);  permute_58 = None
        clone_25 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_120 = torch.ops.aten.view.default(clone_25, [192, 64, 512]);  clone_25 = None
        bmm_12 = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121 = torch.ops.aten.view.default(bmm_12, [16, 12, 512, 512]);  bmm_12 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_6 = None
        full_default_26 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant13 = self._tensor_constant13;  _tensor_constant13 = None
        full_default_27 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_12 = torch.ops.aten.where.self(full_default_26, full_default_27, view_121);  full_default_27 = view_121 = None
        amax_6 = torch.ops.aten.amax.default(where_12, [-1], True)
        sub_32 = torch.ops.aten.sub.Tensor(where_12, amax_6);  where_12 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_26 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        full_default_28 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13 = torch.ops.aten.where.self(full_default_26, full_default_28, div_26);  full_default_26 = full_default_28 = div_26 = None
        expand_33 = torch.ops.aten.expand.default(add_58, [16, 12, 512, 64]);  add_58 = None
        clone_26 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_123 = torch.ops.aten.view.default(clone_26, [192, 512, 64]);  clone_26 = None
        expand_34 = torch.ops.aten.expand.default(where_13, [16, 12, 512, 512]);  where_13 = None
        view_124 = torch.ops.aten.view.default(expand_34, [192, 512, 512]);  expand_34 = None
        bmm_13 = torch.ops.aten.bmm.default(view_124, view_123);  view_124 = view_123 = None
        view_125 = torch.ops.aten.view.default(bmm_13, [16, 12, 512, 64]);  bmm_13 = None
        permute_59 = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        clone_27 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_126 = torch.ops.aten.view.default(clone_27, [16, 512, -1]);  clone_27 = None
        view_127 = torch.ops.aten.view.default(view_126, [8192, 768]);  view_126 = None
        permute_60 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg88_1, view_127, permute_60);  arg88_1 = view_127 = permute_60 = None
        view_128 = torch.ops.aten.view.default(addmm_18, [16, 512, 768]);  addmm_18 = None
        add_59 = torch.ops.aten.add.Tensor(view_128, add_56);  view_128 = add_56 = None
        mean_26 = torch.ops.aten.mean.dim(add_59, [-1], True)
        sub_33 = torch.ops.aten.sub.Tensor(add_59, mean_26)
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(sub_33, 2);  sub_33 = None
        mean_27 = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_59, mean_26);  add_59 = mean_26 = None
        add_60 = torch.ops.aten.add.Tensor(mean_27, 1e-07);  mean_27 = None
        sqrt_20 = torch.ops.aten.sqrt.default(add_60);  add_60 = None
        div_27 = torch.ops.aten.div.Tensor(sub_34, sqrt_20);  sub_34 = sqrt_20 = None
        mul_40 = torch.ops.aten.mul.Tensor(arg89_1, div_27);  arg89_1 = div_27 = None
        add_61 = torch.ops.aten.add.Tensor(mul_40, arg90_1);  mul_40 = arg90_1 = None
        view_129 = torch.ops.aten.view.default(add_61, [8192, 768])
        permute_61 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg92_1, view_129, permute_61);  arg92_1 = view_129 = permute_61 = None
        view_130 = torch.ops.aten.view.default(addmm_19, [16, 512, 3072]);  addmm_19 = None
        mul_41 = torch.ops.aten.mul.Tensor(view_130, 0.5)
        mul_42 = torch.ops.aten.mul.Tensor(view_130, 0.7071067811865476);  view_130 = None
        erf_6 = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_62 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_41, add_62);  mul_41 = add_62 = None
        view_131 = torch.ops.aten.view.default(mul_43, [8192, 3072]);  mul_43 = None
        permute_62 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg94_1, view_131, permute_62);  arg94_1 = view_131 = permute_62 = None
        view_132 = torch.ops.aten.view.default(addmm_20, [16, 512, 768]);  addmm_20 = None
        add_63 = torch.ops.aten.add.Tensor(view_132, add_61);  view_132 = add_61 = None
        mean_28 = torch.ops.aten.mean.dim(add_63, [-1], True)
        sub_35 = torch.ops.aten.sub.Tensor(add_63, mean_28)
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(sub_35, 2);  sub_35 = None
        mean_29 = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_63, mean_28);  add_63 = mean_28 = None
        add_64 = torch.ops.aten.add.Tensor(mean_29, 1e-07);  mean_29 = None
        sqrt_21 = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        div_28 = torch.ops.aten.div.Tensor(sub_36, sqrt_21);  sub_36 = sqrt_21 = None
        mul_44 = torch.ops.aten.mul.Tensor(arg95_1, div_28);  arg95_1 = div_28 = None
        add_65 = torch.ops.aten.add.Tensor(mul_44, arg96_1);  mul_44 = arg96_1 = None
        permute_63 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        view_133 = torch.ops.aten.view.default(add_65, [8192, 768])
        mm_7 = torch.ops.aten.mm.default(view_133, permute_63);  view_133 = permute_63 = None
        view_134 = torch.ops.aten.view.default(mm_7, [16, 512, 2304]);  mm_7 = None
        view_135 = torch.ops.aten.view.default(view_134, [16, 512, 12, -1]);  view_134 = None
        permute_64 = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        split_7 = torch.ops.aten.split.Tensor(permute_64, 64, -1);  permute_64 = None
        getitem_21 = split_7[0]
        getitem_22 = split_7[1]
        getitem_23 = split_7[2];  split_7 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(arg98_1, 0);  arg98_1 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, 1);  unsqueeze_32 = None
        view_136 = torch.ops.aten.view.default(unsqueeze_33, [1, 1, 12, -1]);  unsqueeze_33 = None
        permute_65 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_21, permute_65);  getitem_21 = permute_65 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(arg99_1, 0);  arg99_1 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, 1);  unsqueeze_34 = None
        view_137 = torch.ops.aten.view.default(unsqueeze_35, [1, 1, 12, -1]);  unsqueeze_35 = None
        permute_66 = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        add_67 = torch.ops.aten.add.Tensor(getitem_23, permute_66);  getitem_23 = permute_66 = None
        _tensor_constant14 = self._tensor_constant14
        lift_fresh_copy_14 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        mul_45 = torch.ops.aten.mul.Tensor(lift_fresh_copy_14, 1);  lift_fresh_copy_14 = mul_45 = None
        full_default_29 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_29 = torch.ops.aten.div.Tensor(add_66, full_default_29);  add_66 = full_default_29 = None
        permute_67 = torch.ops.aten.permute.default(getitem_22, [0, 1, 3, 2]);  getitem_22 = None
        expand_35 = torch.ops.aten.expand.default(div_29, [16, 12, 512, 64]);  div_29 = None
        clone_28 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_138 = torch.ops.aten.view.default(clone_28, [192, 512, 64]);  clone_28 = None
        expand_36 = torch.ops.aten.expand.default(permute_67, [16, 12, 64, 512]);  permute_67 = None
        clone_29 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_139 = torch.ops.aten.view.default(clone_29, [192, 64, 512]);  clone_29 = None
        bmm_14 = torch.ops.aten.bmm.default(view_138, view_139);  view_138 = view_139 = None
        view_140 = torch.ops.aten.view.default(bmm_14, [16, 12, 512, 512]);  bmm_14 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_7 = None
        full_default_30 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant15 = self._tensor_constant15;  _tensor_constant15 = None
        full_default_31 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_14 = torch.ops.aten.where.self(full_default_30, full_default_31, view_140);  full_default_31 = view_140 = None
        amax_7 = torch.ops.aten.amax.default(where_14, [-1], True)
        sub_37 = torch.ops.aten.sub.Tensor(where_14, amax_7);  where_14 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_30 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        full_default_32 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15 = torch.ops.aten.where.self(full_default_30, full_default_32, div_30);  full_default_30 = full_default_32 = div_30 = None
        expand_38 = torch.ops.aten.expand.default(add_67, [16, 12, 512, 64]);  add_67 = None
        clone_30 = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_142 = torch.ops.aten.view.default(clone_30, [192, 512, 64]);  clone_30 = None
        expand_39 = torch.ops.aten.expand.default(where_15, [16, 12, 512, 512]);  where_15 = None
        view_143 = torch.ops.aten.view.default(expand_39, [192, 512, 512]);  expand_39 = None
        bmm_15 = torch.ops.aten.bmm.default(view_143, view_142);  view_143 = view_142 = None
        view_144 = torch.ops.aten.view.default(bmm_15, [16, 12, 512, 64]);  bmm_15 = None
        permute_68 = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        clone_31 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_145 = torch.ops.aten.view.default(clone_31, [16, 512, -1]);  clone_31 = None
        view_146 = torch.ops.aten.view.default(view_145, [8192, 768]);  view_145 = None
        permute_69 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg101_1, view_146, permute_69);  arg101_1 = view_146 = permute_69 = None
        view_147 = torch.ops.aten.view.default(addmm_21, [16, 512, 768]);  addmm_21 = None
        add_68 = torch.ops.aten.add.Tensor(view_147, add_65);  view_147 = add_65 = None
        mean_30 = torch.ops.aten.mean.dim(add_68, [-1], True)
        sub_38 = torch.ops.aten.sub.Tensor(add_68, mean_30)
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(sub_38, 2);  sub_38 = None
        mean_31 = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_68, mean_30);  add_68 = mean_30 = None
        add_69 = torch.ops.aten.add.Tensor(mean_31, 1e-07);  mean_31 = None
        sqrt_23 = torch.ops.aten.sqrt.default(add_69);  add_69 = None
        div_31 = torch.ops.aten.div.Tensor(sub_39, sqrt_23);  sub_39 = sqrt_23 = None
        mul_46 = torch.ops.aten.mul.Tensor(arg102_1, div_31);  arg102_1 = div_31 = None
        add_70 = torch.ops.aten.add.Tensor(mul_46, arg103_1);  mul_46 = arg103_1 = None
        view_148 = torch.ops.aten.view.default(add_70, [8192, 768])
        permute_70 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg105_1, view_148, permute_70);  arg105_1 = view_148 = permute_70 = None
        view_149 = torch.ops.aten.view.default(addmm_22, [16, 512, 3072]);  addmm_22 = None
        mul_47 = torch.ops.aten.mul.Tensor(view_149, 0.5)
        mul_48 = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
        erf_7 = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_71 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_47, add_71);  mul_47 = add_71 = None
        view_150 = torch.ops.aten.view.default(mul_49, [8192, 3072]);  mul_49 = None
        permute_71 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg107_1, view_150, permute_71);  arg107_1 = view_150 = permute_71 = None
        view_151 = torch.ops.aten.view.default(addmm_23, [16, 512, 768]);  addmm_23 = None
        add_72 = torch.ops.aten.add.Tensor(view_151, add_70);  view_151 = add_70 = None
        mean_32 = torch.ops.aten.mean.dim(add_72, [-1], True)
        sub_40 = torch.ops.aten.sub.Tensor(add_72, mean_32)
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(sub_40, 2);  sub_40 = None
        mean_33 = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_72, mean_32);  add_72 = mean_32 = None
        add_73 = torch.ops.aten.add.Tensor(mean_33, 1e-07);  mean_33 = None
        sqrt_24 = torch.ops.aten.sqrt.default(add_73);  add_73 = None
        div_32 = torch.ops.aten.div.Tensor(sub_41, sqrt_24);  sub_41 = sqrt_24 = None
        mul_50 = torch.ops.aten.mul.Tensor(arg108_1, div_32);  arg108_1 = div_32 = None
        add_74 = torch.ops.aten.add.Tensor(mul_50, arg109_1);  mul_50 = arg109_1 = None
        permute_72 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        view_152 = torch.ops.aten.view.default(add_74, [8192, 768])
        mm_8 = torch.ops.aten.mm.default(view_152, permute_72);  view_152 = permute_72 = None
        view_153 = torch.ops.aten.view.default(mm_8, [16, 512, 2304]);  mm_8 = None
        view_154 = torch.ops.aten.view.default(view_153, [16, 512, 12, -1]);  view_153 = None
        permute_73 = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        split_8 = torch.ops.aten.split.Tensor(permute_73, 64, -1);  permute_73 = None
        getitem_24 = split_8[0]
        getitem_25 = split_8[1]
        getitem_26 = split_8[2];  split_8 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(arg111_1, 0);  arg111_1 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, 1);  unsqueeze_36 = None
        view_155 = torch.ops.aten.view.default(unsqueeze_37, [1, 1, 12, -1]);  unsqueeze_37 = None
        permute_74 = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_24, permute_74);  getitem_24 = permute_74 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg112_1, 0);  arg112_1 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, 1);  unsqueeze_38 = None
        view_156 = torch.ops.aten.view.default(unsqueeze_39, [1, 1, 12, -1]);  unsqueeze_39 = None
        permute_75 = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_26, permute_75);  getitem_26 = permute_75 = None
        _tensor_constant16 = self._tensor_constant16
        lift_fresh_copy_16 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        mul_51 = torch.ops.aten.mul.Tensor(lift_fresh_copy_16, 1);  lift_fresh_copy_16 = mul_51 = None
        full_default_33 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_33 = torch.ops.aten.div.Tensor(add_75, full_default_33);  add_75 = full_default_33 = None
        permute_76 = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
        expand_40 = torch.ops.aten.expand.default(div_33, [16, 12, 512, 64]);  div_33 = None
        clone_32 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_157 = torch.ops.aten.view.default(clone_32, [192, 512, 64]);  clone_32 = None
        expand_41 = torch.ops.aten.expand.default(permute_76, [16, 12, 64, 512]);  permute_76 = None
        clone_33 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_158 = torch.ops.aten.view.default(clone_33, [192, 64, 512]);  clone_33 = None
        bmm_16 = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
        view_159 = torch.ops.aten.view.default(bmm_16, [16, 12, 512, 512]);  bmm_16 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_8 = None
        full_default_34 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant17 = self._tensor_constant17;  _tensor_constant17 = None
        full_default_35 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_16 = torch.ops.aten.where.self(full_default_34, full_default_35, view_159);  full_default_35 = view_159 = None
        amax_8 = torch.ops.aten.amax.default(where_16, [-1], True)
        sub_42 = torch.ops.aten.sub.Tensor(where_16, amax_8);  where_16 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_34 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        full_default_36 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17 = torch.ops.aten.where.self(full_default_34, full_default_36, div_34);  full_default_34 = full_default_36 = div_34 = None
        expand_43 = torch.ops.aten.expand.default(add_76, [16, 12, 512, 64]);  add_76 = None
        clone_34 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_161 = torch.ops.aten.view.default(clone_34, [192, 512, 64]);  clone_34 = None
        expand_44 = torch.ops.aten.expand.default(where_17, [16, 12, 512, 512]);  where_17 = None
        view_162 = torch.ops.aten.view.default(expand_44, [192, 512, 512]);  expand_44 = None
        bmm_17 = torch.ops.aten.bmm.default(view_162, view_161);  view_162 = view_161 = None
        view_163 = torch.ops.aten.view.default(bmm_17, [16, 12, 512, 64]);  bmm_17 = None
        permute_77 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        clone_35 = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_164 = torch.ops.aten.view.default(clone_35, [16, 512, -1]);  clone_35 = None
        view_165 = torch.ops.aten.view.default(view_164, [8192, 768]);  view_164 = None
        permute_78 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg114_1, view_165, permute_78);  arg114_1 = view_165 = permute_78 = None
        view_166 = torch.ops.aten.view.default(addmm_24, [16, 512, 768]);  addmm_24 = None
        add_77 = torch.ops.aten.add.Tensor(view_166, add_74);  view_166 = add_74 = None
        mean_34 = torch.ops.aten.mean.dim(add_77, [-1], True)
        sub_43 = torch.ops.aten.sub.Tensor(add_77, mean_34)
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(sub_43, 2);  sub_43 = None
        mean_35 = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_77, mean_34);  add_77 = mean_34 = None
        add_78 = torch.ops.aten.add.Tensor(mean_35, 1e-07);  mean_35 = None
        sqrt_26 = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        div_35 = torch.ops.aten.div.Tensor(sub_44, sqrt_26);  sub_44 = sqrt_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(arg115_1, div_35);  arg115_1 = div_35 = None
        add_79 = torch.ops.aten.add.Tensor(mul_52, arg116_1);  mul_52 = arg116_1 = None
        view_167 = torch.ops.aten.view.default(add_79, [8192, 768])
        permute_79 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg118_1, view_167, permute_79);  arg118_1 = view_167 = permute_79 = None
        view_168 = torch.ops.aten.view.default(addmm_25, [16, 512, 3072]);  addmm_25 = None
        mul_53 = torch.ops.aten.mul.Tensor(view_168, 0.5)
        mul_54 = torch.ops.aten.mul.Tensor(view_168, 0.7071067811865476);  view_168 = None
        erf_8 = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_80 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_53, add_80);  mul_53 = add_80 = None
        view_169 = torch.ops.aten.view.default(mul_55, [8192, 3072]);  mul_55 = None
        permute_80 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg120_1, view_169, permute_80);  arg120_1 = view_169 = permute_80 = None
        view_170 = torch.ops.aten.view.default(addmm_26, [16, 512, 768]);  addmm_26 = None
        add_81 = torch.ops.aten.add.Tensor(view_170, add_79);  view_170 = add_79 = None
        mean_36 = torch.ops.aten.mean.dim(add_81, [-1], True)
        sub_45 = torch.ops.aten.sub.Tensor(add_81, mean_36)
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(sub_45, 2);  sub_45 = None
        mean_37 = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_81, mean_36);  add_81 = mean_36 = None
        add_82 = torch.ops.aten.add.Tensor(mean_37, 1e-07);  mean_37 = None
        sqrt_27 = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        div_36 = torch.ops.aten.div.Tensor(sub_46, sqrt_27);  sub_46 = sqrt_27 = None
        mul_56 = torch.ops.aten.mul.Tensor(arg121_1, div_36);  arg121_1 = div_36 = None
        add_83 = torch.ops.aten.add.Tensor(mul_56, arg122_1);  mul_56 = arg122_1 = None
        permute_81 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        view_171 = torch.ops.aten.view.default(add_83, [8192, 768])
        mm_9 = torch.ops.aten.mm.default(view_171, permute_81);  view_171 = permute_81 = None
        view_172 = torch.ops.aten.view.default(mm_9, [16, 512, 2304]);  mm_9 = None
        view_173 = torch.ops.aten.view.default(view_172, [16, 512, 12, -1]);  view_172 = None
        permute_82 = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        split_9 = torch.ops.aten.split.Tensor(permute_82, 64, -1);  permute_82 = None
        getitem_27 = split_9[0]
        getitem_28 = split_9[1]
        getitem_29 = split_9[2];  split_9 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(arg124_1, 0);  arg124_1 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, 1);  unsqueeze_40 = None
        view_174 = torch.ops.aten.view.default(unsqueeze_41, [1, 1, 12, -1]);  unsqueeze_41 = None
        permute_83 = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_27, permute_83);  getitem_27 = permute_83 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(arg125_1, 0);  arg125_1 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, 1);  unsqueeze_42 = None
        view_175 = torch.ops.aten.view.default(unsqueeze_43, [1, 1, 12, -1]);  unsqueeze_43 = None
        permute_84 = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_29, permute_84);  getitem_29 = permute_84 = None
        _tensor_constant18 = self._tensor_constant18
        lift_fresh_copy_18 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        mul_57 = torch.ops.aten.mul.Tensor(lift_fresh_copy_18, 1);  lift_fresh_copy_18 = mul_57 = None
        full_default_37 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_37 = torch.ops.aten.div.Tensor(add_84, full_default_37);  add_84 = full_default_37 = None
        permute_85 = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_45 = torch.ops.aten.expand.default(div_37, [16, 12, 512, 64]);  div_37 = None
        clone_36 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_176 = torch.ops.aten.view.default(clone_36, [192, 512, 64]);  clone_36 = None
        expand_46 = torch.ops.aten.expand.default(permute_85, [16, 12, 64, 512]);  permute_85 = None
        clone_37 = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_177 = torch.ops.aten.view.default(clone_37, [192, 64, 512]);  clone_37 = None
        bmm_18 = torch.ops.aten.bmm.default(view_176, view_177);  view_176 = view_177 = None
        view_178 = torch.ops.aten.view.default(bmm_18, [16, 12, 512, 512]);  bmm_18 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_9 = None
        full_default_38 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant19 = self._tensor_constant19;  _tensor_constant19 = None
        full_default_39 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_18 = torch.ops.aten.where.self(full_default_38, full_default_39, view_178);  full_default_39 = view_178 = None
        amax_9 = torch.ops.aten.amax.default(where_18, [-1], True)
        sub_47 = torch.ops.aten.sub.Tensor(where_18, amax_9);  where_18 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_47);  sub_47 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_38 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        full_default_40 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19 = torch.ops.aten.where.self(full_default_38, full_default_40, div_38);  full_default_38 = full_default_40 = div_38 = None
        expand_48 = torch.ops.aten.expand.default(add_85, [16, 12, 512, 64]);  add_85 = None
        clone_38 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_180 = torch.ops.aten.view.default(clone_38, [192, 512, 64]);  clone_38 = None
        expand_49 = torch.ops.aten.expand.default(where_19, [16, 12, 512, 512]);  where_19 = None
        view_181 = torch.ops.aten.view.default(expand_49, [192, 512, 512]);  expand_49 = None
        bmm_19 = torch.ops.aten.bmm.default(view_181, view_180);  view_181 = view_180 = None
        view_182 = torch.ops.aten.view.default(bmm_19, [16, 12, 512, 64]);  bmm_19 = None
        permute_86 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_39 = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_183 = torch.ops.aten.view.default(clone_39, [16, 512, -1]);  clone_39 = None
        view_184 = torch.ops.aten.view.default(view_183, [8192, 768]);  view_183 = None
        permute_87 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg127_1, view_184, permute_87);  arg127_1 = view_184 = permute_87 = None
        view_185 = torch.ops.aten.view.default(addmm_27, [16, 512, 768]);  addmm_27 = None
        add_86 = torch.ops.aten.add.Tensor(view_185, add_83);  view_185 = add_83 = None
        mean_38 = torch.ops.aten.mean.dim(add_86, [-1], True)
        sub_48 = torch.ops.aten.sub.Tensor(add_86, mean_38)
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(sub_48, 2);  sub_48 = None
        mean_39 = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_86, mean_38);  add_86 = mean_38 = None
        add_87 = torch.ops.aten.add.Tensor(mean_39, 1e-07);  mean_39 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_87);  add_87 = None
        div_39 = torch.ops.aten.div.Tensor(sub_49, sqrt_29);  sub_49 = sqrt_29 = None
        mul_58 = torch.ops.aten.mul.Tensor(arg128_1, div_39);  arg128_1 = div_39 = None
        add_88 = torch.ops.aten.add.Tensor(mul_58, arg129_1);  mul_58 = arg129_1 = None
        view_186 = torch.ops.aten.view.default(add_88, [8192, 768])
        permute_88 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg131_1, view_186, permute_88);  arg131_1 = view_186 = permute_88 = None
        view_187 = torch.ops.aten.view.default(addmm_28, [16, 512, 3072]);  addmm_28 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_187, 0.5)
        mul_60 = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476);  view_187 = None
        erf_9 = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_89 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_59, add_89);  mul_59 = add_89 = None
        view_188 = torch.ops.aten.view.default(mul_61, [8192, 3072]);  mul_61 = None
        permute_89 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg133_1, view_188, permute_89);  arg133_1 = view_188 = permute_89 = None
        view_189 = torch.ops.aten.view.default(addmm_29, [16, 512, 768]);  addmm_29 = None
        add_90 = torch.ops.aten.add.Tensor(view_189, add_88);  view_189 = add_88 = None
        mean_40 = torch.ops.aten.mean.dim(add_90, [-1], True)
        sub_50 = torch.ops.aten.sub.Tensor(add_90, mean_40)
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(sub_50, 2);  sub_50 = None
        mean_41 = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_90, mean_40);  add_90 = mean_40 = None
        add_91 = torch.ops.aten.add.Tensor(mean_41, 1e-07);  mean_41 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_91);  add_91 = None
        div_40 = torch.ops.aten.div.Tensor(sub_51, sqrt_30);  sub_51 = sqrt_30 = None
        mul_62 = torch.ops.aten.mul.Tensor(arg134_1, div_40);  arg134_1 = div_40 = None
        add_92 = torch.ops.aten.add.Tensor(mul_62, arg135_1);  mul_62 = arg135_1 = None
        permute_90 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        view_190 = torch.ops.aten.view.default(add_92, [8192, 768])
        mm_10 = torch.ops.aten.mm.default(view_190, permute_90);  view_190 = permute_90 = None
        view_191 = torch.ops.aten.view.default(mm_10, [16, 512, 2304]);  mm_10 = None
        view_192 = torch.ops.aten.view.default(view_191, [16, 512, 12, -1]);  view_191 = None
        permute_91 = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        split_10 = torch.ops.aten.split.Tensor(permute_91, 64, -1);  permute_91 = None
        getitem_30 = split_10[0]
        getitem_31 = split_10[1]
        getitem_32 = split_10[2];  split_10 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(arg137_1, 0);  arg137_1 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, 1);  unsqueeze_44 = None
        view_193 = torch.ops.aten.view.default(unsqueeze_45, [1, 1, 12, -1]);  unsqueeze_45 = None
        permute_92 = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_30, permute_92);  getitem_30 = permute_92 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(arg138_1, 0);  arg138_1 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, 1);  unsqueeze_46 = None
        view_194 = torch.ops.aten.view.default(unsqueeze_47, [1, 1, 12, -1]);  unsqueeze_47 = None
        permute_93 = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_32, permute_93);  getitem_32 = permute_93 = None
        _tensor_constant20 = self._tensor_constant20
        lift_fresh_copy_20 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
        mul_63 = torch.ops.aten.mul.Tensor(lift_fresh_copy_20, 1);  lift_fresh_copy_20 = mul_63 = None
        full_default_41 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_41 = torch.ops.aten.div.Tensor(add_93, full_default_41);  add_93 = full_default_41 = None
        permute_94 = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
        expand_50 = torch.ops.aten.expand.default(div_41, [16, 12, 512, 64]);  div_41 = None
        clone_40 = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_195 = torch.ops.aten.view.default(clone_40, [192, 512, 64]);  clone_40 = None
        expand_51 = torch.ops.aten.expand.default(permute_94, [16, 12, 64, 512]);  permute_94 = None
        clone_41 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_196 = torch.ops.aten.view.default(clone_41, [192, 64, 512]);  clone_41 = None
        bmm_20 = torch.ops.aten.bmm.default(view_195, view_196);  view_195 = view_196 = None
        view_197 = torch.ops.aten.view.default(bmm_20, [16, 12, 512, 512]);  bmm_20 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  convert_element_type_10 = None
        full_default_42 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant21 = self._tensor_constant21;  _tensor_constant21 = None
        full_default_43 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_20 = torch.ops.aten.where.self(full_default_42, full_default_43, view_197);  full_default_43 = view_197 = None
        amax_10 = torch.ops.aten.amax.default(where_20, [-1], True)
        sub_52 = torch.ops.aten.sub.Tensor(where_20, amax_10);  where_20 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_42 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        full_default_44 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21 = torch.ops.aten.where.self(full_default_42, full_default_44, div_42);  full_default_42 = full_default_44 = div_42 = None
        expand_53 = torch.ops.aten.expand.default(add_94, [16, 12, 512, 64]);  add_94 = None
        clone_42 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_199 = torch.ops.aten.view.default(clone_42, [192, 512, 64]);  clone_42 = None
        expand_54 = torch.ops.aten.expand.default(where_21, [16, 12, 512, 512]);  where_21 = None
        view_200 = torch.ops.aten.view.default(expand_54, [192, 512, 512]);  expand_54 = None
        bmm_21 = torch.ops.aten.bmm.default(view_200, view_199);  view_200 = view_199 = None
        view_201 = torch.ops.aten.view.default(bmm_21, [16, 12, 512, 64]);  bmm_21 = None
        permute_95 = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        clone_43 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_202 = torch.ops.aten.view.default(clone_43, [16, 512, -1]);  clone_43 = None
        view_203 = torch.ops.aten.view.default(view_202, [8192, 768]);  view_202 = None
        permute_96 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg140_1, view_203, permute_96);  arg140_1 = view_203 = permute_96 = None
        view_204 = torch.ops.aten.view.default(addmm_30, [16, 512, 768]);  addmm_30 = None
        add_95 = torch.ops.aten.add.Tensor(view_204, add_92);  view_204 = add_92 = None
        mean_42 = torch.ops.aten.mean.dim(add_95, [-1], True)
        sub_53 = torch.ops.aten.sub.Tensor(add_95, mean_42)
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(sub_53, 2);  sub_53 = None
        mean_43 = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_95, mean_42);  add_95 = mean_42 = None
        add_96 = torch.ops.aten.add.Tensor(mean_43, 1e-07);  mean_43 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        div_43 = torch.ops.aten.div.Tensor(sub_54, sqrt_32);  sub_54 = sqrt_32 = None
        mul_64 = torch.ops.aten.mul.Tensor(arg141_1, div_43);  arg141_1 = div_43 = None
        add_97 = torch.ops.aten.add.Tensor(mul_64, arg142_1);  mul_64 = arg142_1 = None
        view_205 = torch.ops.aten.view.default(add_97, [8192, 768])
        permute_97 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg144_1, view_205, permute_97);  arg144_1 = view_205 = permute_97 = None
        view_206 = torch.ops.aten.view.default(addmm_31, [16, 512, 3072]);  addmm_31 = None
        mul_65 = torch.ops.aten.mul.Tensor(view_206, 0.5)
        mul_66 = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
        erf_10 = torch.ops.aten.erf.default(mul_66);  mul_66 = None
        add_98 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_65, add_98);  mul_65 = add_98 = None
        view_207 = torch.ops.aten.view.default(mul_67, [8192, 3072]);  mul_67 = None
        permute_98 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg146_1, view_207, permute_98);  arg146_1 = view_207 = permute_98 = None
        view_208 = torch.ops.aten.view.default(addmm_32, [16, 512, 768]);  addmm_32 = None
        add_99 = torch.ops.aten.add.Tensor(view_208, add_97);  view_208 = add_97 = None
        mean_44 = torch.ops.aten.mean.dim(add_99, [-1], True)
        sub_55 = torch.ops.aten.sub.Tensor(add_99, mean_44)
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(sub_55, 2);  sub_55 = None
        mean_45 = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        sub_56 = torch.ops.aten.sub.Tensor(add_99, mean_44);  add_99 = mean_44 = None
        add_100 = torch.ops.aten.add.Tensor(mean_45, 1e-07);  mean_45 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        div_44 = torch.ops.aten.div.Tensor(sub_56, sqrt_33);  sub_56 = sqrt_33 = None
        mul_68 = torch.ops.aten.mul.Tensor(arg147_1, div_44);  arg147_1 = div_44 = None
        add_101 = torch.ops.aten.add.Tensor(mul_68, arg148_1);  mul_68 = arg148_1 = None
        permute_99 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        view_209 = torch.ops.aten.view.default(add_101, [8192, 768])
        mm_11 = torch.ops.aten.mm.default(view_209, permute_99);  view_209 = permute_99 = None
        view_210 = torch.ops.aten.view.default(mm_11, [16, 512, 2304]);  mm_11 = None
        view_211 = torch.ops.aten.view.default(view_210, [16, 512, 12, -1]);  view_210 = None
        permute_100 = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        split_11 = torch.ops.aten.split.Tensor(permute_100, 64, -1);  permute_100 = None
        getitem_33 = split_11[0]
        getitem_34 = split_11[1]
        getitem_35 = split_11[2];  split_11 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(arg150_1, 0);  arg150_1 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, 1);  unsqueeze_48 = None
        view_212 = torch.ops.aten.view.default(unsqueeze_49, [1, 1, 12, -1]);  unsqueeze_49 = None
        permute_101 = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_33, permute_101);  getitem_33 = permute_101 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(arg151_1, 0);  arg151_1 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, 1);  unsqueeze_50 = None
        view_213 = torch.ops.aten.view.default(unsqueeze_51, [1, 1, 12, -1]);  unsqueeze_51 = None
        permute_102 = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        add_103 = torch.ops.aten.add.Tensor(getitem_35, permute_102);  getitem_35 = permute_102 = None
        _tensor_constant22 = self._tensor_constant22
        lift_fresh_copy_22 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
        mul_69 = torch.ops.aten.mul.Tensor(lift_fresh_copy_22, 1);  lift_fresh_copy_22 = mul_69 = None
        full_default_45 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_45 = torch.ops.aten.div.Tensor(add_102, full_default_45);  add_102 = full_default_45 = None
        permute_103 = torch.ops.aten.permute.default(getitem_34, [0, 1, 3, 2]);  getitem_34 = None
        expand_55 = torch.ops.aten.expand.default(div_45, [16, 12, 512, 64]);  div_45 = None
        clone_44 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_214 = torch.ops.aten.view.default(clone_44, [192, 512, 64]);  clone_44 = None
        expand_56 = torch.ops.aten.expand.default(permute_103, [16, 12, 64, 512]);  permute_103 = None
        clone_45 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_215 = torch.ops.aten.view.default(clone_45, [192, 64, 512]);  clone_45 = None
        bmm_22 = torch.ops.aten.bmm.default(view_214, view_215);  view_214 = view_215 = None
        view_216 = torch.ops.aten.view.default(bmm_22, [16, 12, 512, 512]);  bmm_22 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  mul_2 = convert_element_type_11 = None
        full_default_46 = torch.ops.aten.full.default([16, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant23 = self._tensor_constant23;  _tensor_constant23 = None
        full_default_47 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_22 = torch.ops.aten.where.self(full_default_46, full_default_47, view_216);  full_default_47 = view_216 = None
        amax_11 = torch.ops.aten.amax.default(where_22, [-1], True)
        sub_57 = torch.ops.aten.sub.Tensor(where_22, amax_11);  where_22 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_46 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        full_default_48 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23 = torch.ops.aten.where.self(full_default_46, full_default_48, div_46);  full_default_46 = full_default_48 = div_46 = None
        expand_58 = torch.ops.aten.expand.default(add_103, [16, 12, 512, 64]);  add_103 = None
        clone_46 = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_218 = torch.ops.aten.view.default(clone_46, [192, 512, 64]);  clone_46 = None
        expand_59 = torch.ops.aten.expand.default(where_23, [16, 12, 512, 512]);  where_23 = None
        view_219 = torch.ops.aten.view.default(expand_59, [192, 512, 512]);  expand_59 = None
        bmm_23 = torch.ops.aten.bmm.default(view_219, view_218);  view_219 = view_218 = None
        view_220 = torch.ops.aten.view.default(bmm_23, [16, 12, 512, 64]);  bmm_23 = None
        permute_104 = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        clone_47 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_221 = torch.ops.aten.view.default(clone_47, [16, 512, -1]);  clone_47 = None
        view_222 = torch.ops.aten.view.default(view_221, [8192, 768]);  view_221 = None
        permute_105 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg153_1, view_222, permute_105);  arg153_1 = view_222 = permute_105 = None
        view_223 = torch.ops.aten.view.default(addmm_33, [16, 512, 768]);  addmm_33 = None
        add_104 = torch.ops.aten.add.Tensor(view_223, add_101);  view_223 = add_101 = None
        mean_46 = torch.ops.aten.mean.dim(add_104, [-1], True)
        sub_58 = torch.ops.aten.sub.Tensor(add_104, mean_46)
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(sub_58, 2);  sub_58 = None
        mean_47 = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_104, mean_46);  add_104 = mean_46 = None
        add_105 = torch.ops.aten.add.Tensor(mean_47, 1e-07);  mean_47 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_105);  add_105 = None
        div_47 = torch.ops.aten.div.Tensor(sub_59, sqrt_35);  sub_59 = sqrt_35 = None
        mul_70 = torch.ops.aten.mul.Tensor(arg154_1, div_47);  arg154_1 = div_47 = None
        add_106 = torch.ops.aten.add.Tensor(mul_70, arg155_1);  mul_70 = arg155_1 = None
        view_224 = torch.ops.aten.view.default(add_106, [8192, 768])
        permute_106 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg157_1, view_224, permute_106);  arg157_1 = view_224 = permute_106 = None
        view_225 = torch.ops.aten.view.default(addmm_34, [16, 512, 3072]);  addmm_34 = None
        mul_71 = torch.ops.aten.mul.Tensor(view_225, 0.5)
        mul_72 = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476);  view_225 = None
        erf_11 = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_107 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_71, add_107);  mul_71 = add_107 = None
        view_226 = torch.ops.aten.view.default(mul_73, [8192, 3072]);  mul_73 = None
        permute_107 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg159_1, view_226, permute_107);  arg159_1 = view_226 = permute_107 = None
        view_227 = torch.ops.aten.view.default(addmm_35, [16, 512, 768]);  addmm_35 = None
        add_108 = torch.ops.aten.add.Tensor(view_227, add_106);  view_227 = add_106 = None
        mean_48 = torch.ops.aten.mean.dim(add_108, [-1], True)
        sub_60 = torch.ops.aten.sub.Tensor(add_108, mean_48)
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(sub_60, 2);  sub_60 = None
        mean_49 = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        sub_61 = torch.ops.aten.sub.Tensor(add_108, mean_48);  add_108 = mean_48 = None
        add_109 = torch.ops.aten.add.Tensor(mean_49, 1e-07);  mean_49 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        div_48 = torch.ops.aten.div.Tensor(sub_61, sqrt_36);  sub_61 = sqrt_36 = None
        mul_74 = torch.ops.aten.mul.Tensor(arg160_1, div_48);  arg160_1 = div_48 = None
        add_110 = torch.ops.aten.add.Tensor(mul_74, arg161_1);  mul_74 = arg161_1 = None
        view_228 = torch.ops.aten.view.default(add_110, [8192, 768]);  add_110 = None
        permute_108 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg163_1, view_228, permute_108);  arg163_1 = view_228 = permute_108 = None
        view_229 = torch.ops.aten.view.default(addmm_36, [16, 512, 2]);  addmm_36 = None
        split_12 = torch.ops.aten.split.Tensor(view_229, 1, -1);  view_229 = None
        getitem_36 = split_12[0]
        getitem_37 = split_12[1];  split_12 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(getitem_36, -1);  getitem_36 = None
        clone_48 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(getitem_37, -1);  getitem_37 = None
        clone_49 = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        clamp_min = torch.ops.aten.clamp_min.default(arg164_1, 0);  arg164_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(arg165_1, 0);  arg165_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
        amax_12 = torch.ops.aten.amax.default(clone_48, [1], True)
        sub_62 = torch.ops.aten.sub.Tensor(clone_48, amax_12);  amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_62)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, log);  sub_62 = log = None
        ne = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_49 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24 = torch.ops.aten.where.self(ne, clamp_max, full_default_49);  ne = full_default_49 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather = torch.ops.aten.gather.default(sub_63, 1, unsqueeze_52);  sub_63 = unsqueeze_52 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        ne_1 = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_50 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25 = torch.ops.aten.where.self(ne_1, neg, full_default_50);  ne_1 = neg = full_default_50 = None
        ne_2 = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
        sum_14 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15 = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_49 = torch.ops.aten.div.Tensor(sum_15, convert_element_type_12);  sum_15 = convert_element_type_12 = None
        amax_13 = torch.ops.aten.amax.default(clone_49, [1], True)
        sub_64 = torch.ops.aten.sub.Tensor(clone_49, amax_13);  amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_64)
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
        log_1 = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_65 = torch.ops.aten.sub.Tensor(sub_64, log_1);  sub_64 = log_1 = None
        ne_3 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_51 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26 = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_51);  ne_3 = full_default_51 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(where_26, 1);  where_26 = None
        gather_1 = torch.ops.aten.gather.default(sub_65, 1, unsqueeze_53);  sub_65 = unsqueeze_53 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        ne_4 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_52 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27 = torch.ops.aten.where.self(ne_4, neg_1, full_default_52);  ne_4 = neg_1 = full_default_52 = None
        ne_5 = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
        sum_17 = torch.ops.aten.sum.default(ne_5);  ne_5 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
        sum_18 = torch.ops.aten.sum.default(where_27);  where_27 = None
        div_50 = torch.ops.aten.div.Tensor(sum_18, convert_element_type_13);  sum_18 = convert_element_type_13 = None
        add_111 = torch.ops.aten.add.Tensor(div_49, div_50);  div_49 = div_50 = None
        div_51 = torch.ops.aten.div.Tensor(add_111, 2);  add_111 = None
        return (div_51, clone_48, clone_49)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (16, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 154414080, device=device(type='cuda', index=0))
    reader.tensor(buf2, (50265, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2304, 768), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768, 768), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf13, (3072, 768), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf14, (3072,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768, 3072), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf19, (2304, 768), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768, 768), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (3072, 768), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf27, (3072,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768, 3072), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf32, (2304, 768), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768, 768), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf39, (3072, 768), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf40, (3072,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 3072), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2304, 768), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768, 768), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf52, (3072, 768), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf53, (3072,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 3072), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf58, (2304, 768), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf65, (3072, 768), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf66, (3072,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768, 3072), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf71, (2304, 768), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 768), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf78, (3072, 768), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf79, (3072,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768, 3072), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf84, (2304, 768), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
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
    buf97 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf97, (2304, 768), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768, 768), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf104, (3072, 768), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf105, (3072,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 3072), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf110, (2304, 768), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768, 768), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf117, (3072, 768), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf118, (3072,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768, 3072), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf123, (2304, 768), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768, 768), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf130, (3072, 768), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf131, (3072,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768, 3072), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf136, (2304, 768), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768, 768), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf143, (3072, 768), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf144, (3072,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768, 3072), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf149, (2304, 768), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768, 768), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf156, (3072, 768), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf157, (3072,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768, 3072), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf162, (2, 768), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf163, (2,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf164, (16,), dtype=torch.int64, is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf165, (16,), dtype=torch.int64, is_leaf=True)  # arg165_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)