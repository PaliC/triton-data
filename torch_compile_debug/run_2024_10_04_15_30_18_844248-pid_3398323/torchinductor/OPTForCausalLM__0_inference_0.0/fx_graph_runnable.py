
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
        self.register_buffer('_tensor_constant0', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant1', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant2', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant3', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant4', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant5', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant6', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant7', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant8', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant9', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant10', tensor(-3.4028e+38, device='cuda:0').cuda())
        self.register_buffer('_tensor_constant11', tensor(-3.4028e+38, device='cuda:0').cuda())

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 2048]);  arg0_1 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, view, 1);  view = None
        full = torch.ops.aten.full.default([2, 2048], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default = torch.ops.aten.full.default([2048, 2048], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota, 1)
        view_1 = torch.ops.aten.view.default(add, [2048, 1]);  add = None
        lt = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [2, 1, 2048, 2048]);  unsqueeze_3 = None
        sub = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sub, torch.bool)
        scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1 = torch.ops.aten.where.self(convert_element_type, scalar_tensor_1, sub);  convert_element_type = scalar_tensor_1 = sub = where_1 = None
        full_default_2 = torch.ops.aten.full.default([2, 1, 2048, 2048], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_5, [2, 1, 2048, 2048]);  unsqueeze_5 = None
        full_default_3 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(full_default_2, full_default_3, expand_2);  full_default_2 = full_default_3 = expand_2 = None
        full_default_4 = torch.ops.aten.full.default([2, 2048], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cumsum = torch.ops.aten.cumsum.default(full_default_4, 1);  full_default_4 = None
        sub_1 = torch.ops.aten.sub.Tensor(cumsum, 1);  cumsum = None
        add_1 = torch.ops.aten.add.Tensor(sub_1, 2);  sub_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, add_1);  arg2_1 = add_1 = None
        add_2 = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_2, getitem_1);  getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        view_2 = torch.ops.aten.view.default(add_4, [4096, 768])
        permute = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm = torch.ops.aten.addmm.default(arg6_1, view_2, permute);  arg6_1 = view_2 = permute = None
        view_3 = torch.ops.aten.view.default(addmm, [2, 2048, 768]);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
        view_4 = torch.ops.aten.view.default(add_4, [4096, 768])
        permute_1 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, view_4, permute_1);  arg8_1 = view_4 = permute_1 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [2, 2048, 768]);  addmm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [2, -1, 12, 64]);  view_5 = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7 = torch.ops.aten.view.default(add_4, [4096, 768]);  add_4 = None
        permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg10_1, view_7, permute_3);  arg10_1 = view_7 = permute_3 = None
        view_8 = torch.ops.aten.view.default(addmm_2, [2, 2048, 768]);  addmm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [2, -1, 12, 64]);  view_8 = None
        permute_4 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_1 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_10 = torch.ops.aten.view.default(mul_3, [2, 2048, 12, 64]);  mul_3 = None
        permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_2 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11 = torch.ops.aten.view.default(clone_2, [24, -1, 64]);  clone_2 = None
        view_12 = torch.ops.aten.view.default(clone, [24, -1, 64])
        view_13 = torch.ops.aten.view.default(clone_1, [24, -1, 64])
        permute_6 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
        view_14 = torch.ops.aten.view.default(bmm, [2, 12, 2048, 2048]);  bmm = None
        add_5 = torch.ops.aten.add.Tensor(view_14, where_2);  view_14 = None
        _tensor_constant0 = self._tensor_constant0;  _tensor_constant0 = None
        full_default_5 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum = torch.ops.aten.maximum.default(add_5, full_default_5);  add_5 = full_default_5 = None
        view_15 = torch.ops.aten.view.default(maximum, [24, 2048, 2048]);  maximum = None
        amax = torch.ops.aten.amax.default(view_15, [-1], True)
        sub_3 = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
        exp = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        bmm_1 = torch.ops.aten.bmm.default(div, view_13);  div = view_13 = None
        view_16 = torch.ops.aten.view.default(bmm_1, [2, 12, 2048, 64]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_4 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17 = torch.ops.aten.view.default(clone_4, [2, 2048, 768]);  clone_4 = None
        view_18 = torch.ops.aten.view.default(view_17, [4096, 768]);  view_17 = None
        permute_8 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg12_1, view_18, permute_8);  arg12_1 = view_18 = permute_8 = None
        view_19 = torch.ops.aten.view.default(addmm_3, [2, 2048, 768]);  addmm_3 = None
        add_6 = torch.ops.aten.add.Tensor(add_2, view_19);  add_2 = view_19 = None
        view_20 = torch.ops.aten.view.default(add_6, [-1, 768]);  add_6 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(view_20, [1], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_7 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_4 = torch.ops.aten.sub.Tensor(view_20, getitem_3);  getitem_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_8 = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        permute_9 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg16_1, add_8, permute_9);  arg16_1 = add_8 = permute_9 = None
        relu = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
        permute_10 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg18_1, relu, permute_10);  arg18_1 = relu = permute_10 = None
        add_9 = torch.ops.aten.add.Tensor(view_20, addmm_5);  view_20 = addmm_5 = None
        view_21 = torch.ops.aten.view.default(add_9, [2, 2048, 768]);  add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(view_21, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_5 = torch.ops.aten.sub.Tensor(view_21, getitem_5);  getitem_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, arg19_1);  mul_6 = arg19_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_7, arg20_1);  mul_7 = arg20_1 = None
        view_22 = torch.ops.aten.view.default(add_11, [4096, 768])
        permute_11 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg22_1, view_22, permute_11);  arg22_1 = view_22 = permute_11 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [2, 2048, 768]);  addmm_6 = None
        mul_8 = torch.ops.aten.mul.Tensor(view_23, 0.125);  view_23 = None
        view_24 = torch.ops.aten.view.default(add_11, [4096, 768])
        permute_12 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg24_1, view_24, permute_12);  arg24_1 = view_24 = permute_12 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [2, 2048, 768]);  addmm_7 = None
        view_26 = torch.ops.aten.view.default(view_25, [2, -1, 12, 64]);  view_25 = None
        permute_13 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_7 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_27 = torch.ops.aten.view.default(add_11, [4096, 768]);  add_11 = None
        permute_14 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg26_1, view_27, permute_14);  arg26_1 = view_27 = permute_14 = None
        view_28 = torch.ops.aten.view.default(addmm_8, [2, 2048, 768]);  addmm_8 = None
        view_29 = torch.ops.aten.view.default(view_28, [2, -1, 12, 64]);  view_28 = None
        permute_15 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        clone_8 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_30 = torch.ops.aten.view.default(mul_8, [2, 2048, 12, 64]);  mul_8 = None
        permute_16 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_9 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_31 = torch.ops.aten.view.default(clone_9, [24, -1, 64]);  clone_9 = None
        view_32 = torch.ops.aten.view.default(clone_7, [24, -1, 64])
        view_33 = torch.ops.aten.view.default(clone_8, [24, -1, 64])
        permute_17 = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        bmm_2 = torch.ops.aten.bmm.default(view_31, permute_17);  view_31 = permute_17 = None
        view_34 = torch.ops.aten.view.default(bmm_2, [2, 12, 2048, 2048]);  bmm_2 = None
        add_12 = torch.ops.aten.add.Tensor(view_34, where_2);  view_34 = None
        _tensor_constant1 = self._tensor_constant1;  _tensor_constant1 = None
        full_default_6 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_1 = torch.ops.aten.maximum.default(add_12, full_default_6);  add_12 = full_default_6 = None
        view_35 = torch.ops.aten.view.default(maximum_1, [24, 2048, 2048]);  maximum_1 = None
        amax_1 = torch.ops.aten.amax.default(view_35, [-1], True)
        sub_6 = torch.ops.aten.sub.Tensor(view_35, amax_1);  view_35 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        bmm_3 = torch.ops.aten.bmm.default(div_1, view_33);  div_1 = view_33 = None
        view_36 = torch.ops.aten.view.default(bmm_3, [2, 12, 2048, 64]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_11 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37 = torch.ops.aten.view.default(clone_11, [2, 2048, 768]);  clone_11 = None
        view_38 = torch.ops.aten.view.default(view_37, [4096, 768]);  view_37 = None
        permute_19 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg28_1, view_38, permute_19);  arg28_1 = view_38 = permute_19 = None
        view_39 = torch.ops.aten.view.default(addmm_9, [2, 2048, 768]);  addmm_9 = None
        add_13 = torch.ops.aten.add.Tensor(view_21, view_39);  view_21 = view_39 = None
        view_40 = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(view_40, [1], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_7 = torch.ops.aten.sub.Tensor(view_40, getitem_7);  getitem_7 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_3);  sub_7 = rsqrt_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg29_1);  mul_9 = arg29_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_10, arg30_1);  mul_10 = arg30_1 = None
        permute_20 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg32_1, add_15, permute_20);  arg32_1 = add_15 = permute_20 = None
        relu_1 = torch.ops.aten.relu.default(addmm_10);  addmm_10 = None
        permute_21 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg34_1, relu_1, permute_21);  arg34_1 = relu_1 = permute_21 = None
        add_16 = torch.ops.aten.add.Tensor(view_40, addmm_11);  view_40 = addmm_11 = None
        view_41 = torch.ops.aten.view.default(add_16, [2, 2048, 768]);  add_16 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(view_41, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_8 = torch.ops.aten.sub.Tensor(view_41, getitem_9);  getitem_9 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = rsqrt_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, arg35_1);  mul_11 = arg35_1 = None
        add_18 = torch.ops.aten.add.Tensor(mul_12, arg36_1);  mul_12 = arg36_1 = None
        view_42 = torch.ops.aten.view.default(add_18, [4096, 768])
        permute_22 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg38_1, view_42, permute_22);  arg38_1 = view_42 = permute_22 = None
        view_43 = torch.ops.aten.view.default(addmm_12, [2, 2048, 768]);  addmm_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(view_43, 0.125);  view_43 = None
        view_44 = torch.ops.aten.view.default(add_18, [4096, 768])
        permute_23 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg40_1, view_44, permute_23);  arg40_1 = view_44 = permute_23 = None
        view_45 = torch.ops.aten.view.default(addmm_13, [2, 2048, 768]);  addmm_13 = None
        view_46 = torch.ops.aten.view.default(view_45, [2, -1, 12, 64]);  view_45 = None
        permute_24 = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        clone_14 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_47 = torch.ops.aten.view.default(add_18, [4096, 768]);  add_18 = None
        permute_25 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg42_1, view_47, permute_25);  arg42_1 = view_47 = permute_25 = None
        view_48 = torch.ops.aten.view.default(addmm_14, [2, 2048, 768]);  addmm_14 = None
        view_49 = torch.ops.aten.view.default(view_48, [2, -1, 12, 64]);  view_48 = None
        permute_26 = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        clone_15 = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_50 = torch.ops.aten.view.default(mul_13, [2, 2048, 12, 64]);  mul_13 = None
        permute_27 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_16 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_51 = torch.ops.aten.view.default(clone_16, [24, -1, 64]);  clone_16 = None
        view_52 = torch.ops.aten.view.default(clone_14, [24, -1, 64])
        view_53 = torch.ops.aten.view.default(clone_15, [24, -1, 64])
        permute_28 = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
        bmm_4 = torch.ops.aten.bmm.default(view_51, permute_28);  view_51 = permute_28 = None
        view_54 = torch.ops.aten.view.default(bmm_4, [2, 12, 2048, 2048]);  bmm_4 = None
        add_19 = torch.ops.aten.add.Tensor(view_54, where_2);  view_54 = None
        _tensor_constant2 = self._tensor_constant2;  _tensor_constant2 = None
        full_default_7 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_2 = torch.ops.aten.maximum.default(add_19, full_default_7);  add_19 = full_default_7 = None
        view_55 = torch.ops.aten.view.default(maximum_2, [24, 2048, 2048]);  maximum_2 = None
        amax_2 = torch.ops.aten.amax.default(view_55, [-1], True)
        sub_9 = torch.ops.aten.sub.Tensor(view_55, amax_2);  view_55 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        bmm_5 = torch.ops.aten.bmm.default(div_2, view_53);  div_2 = view_53 = None
        view_56 = torch.ops.aten.view.default(bmm_5, [2, 12, 2048, 64]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        clone_18 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_57 = torch.ops.aten.view.default(clone_18, [2, 2048, 768]);  clone_18 = None
        view_58 = torch.ops.aten.view.default(view_57, [4096, 768]);  view_57 = None
        permute_30 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg44_1, view_58, permute_30);  arg44_1 = view_58 = permute_30 = None
        view_59 = torch.ops.aten.view.default(addmm_15, [2, 2048, 768]);  addmm_15 = None
        add_20 = torch.ops.aten.add.Tensor(view_41, view_59);  view_41 = view_59 = None
        view_60 = torch.ops.aten.view.default(add_20, [-1, 768]);  add_20 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(view_60, [1], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_10 = torch.ops.aten.sub.Tensor(view_60, getitem_11);  getitem_11 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = rsqrt_5 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg45_1);  mul_14 = arg45_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_15, arg46_1);  mul_15 = arg46_1 = None
        permute_31 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg48_1, add_22, permute_31);  arg48_1 = add_22 = permute_31 = None
        relu_2 = torch.ops.aten.relu.default(addmm_16);  addmm_16 = None
        permute_32 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg50_1, relu_2, permute_32);  arg50_1 = relu_2 = permute_32 = None
        add_23 = torch.ops.aten.add.Tensor(view_60, addmm_17);  view_60 = addmm_17 = None
        view_61 = torch.ops.aten.view.default(add_23, [2, 2048, 768]);  add_23 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(view_61, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_11 = torch.ops.aten.sub.Tensor(view_61, getitem_13);  getitem_13 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_6);  sub_11 = rsqrt_6 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg51_1);  mul_16 = arg51_1 = None
        add_25 = torch.ops.aten.add.Tensor(mul_17, arg52_1);  mul_17 = arg52_1 = None
        view_62 = torch.ops.aten.view.default(add_25, [4096, 768])
        permute_33 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg54_1, view_62, permute_33);  arg54_1 = view_62 = permute_33 = None
        view_63 = torch.ops.aten.view.default(addmm_18, [2, 2048, 768]);  addmm_18 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_63, 0.125);  view_63 = None
        view_64 = torch.ops.aten.view.default(add_25, [4096, 768])
        permute_34 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg56_1, view_64, permute_34);  arg56_1 = view_64 = permute_34 = None
        view_65 = torch.ops.aten.view.default(addmm_19, [2, 2048, 768]);  addmm_19 = None
        view_66 = torch.ops.aten.view.default(view_65, [2, -1, 12, 64]);  view_65 = None
        permute_35 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        clone_21 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_67 = torch.ops.aten.view.default(add_25, [4096, 768]);  add_25 = None
        permute_36 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg58_1, view_67, permute_36);  arg58_1 = view_67 = permute_36 = None
        view_68 = torch.ops.aten.view.default(addmm_20, [2, 2048, 768]);  addmm_20 = None
        view_69 = torch.ops.aten.view.default(view_68, [2, -1, 12, 64]);  view_68 = None
        permute_37 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        clone_22 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_70 = torch.ops.aten.view.default(mul_18, [2, 2048, 12, 64]);  mul_18 = None
        permute_38 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_23 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_71 = torch.ops.aten.view.default(clone_23, [24, -1, 64]);  clone_23 = None
        view_72 = torch.ops.aten.view.default(clone_21, [24, -1, 64])
        view_73 = torch.ops.aten.view.default(clone_22, [24, -1, 64])
        permute_39 = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
        bmm_6 = torch.ops.aten.bmm.default(view_71, permute_39);  view_71 = permute_39 = None
        view_74 = torch.ops.aten.view.default(bmm_6, [2, 12, 2048, 2048]);  bmm_6 = None
        add_26 = torch.ops.aten.add.Tensor(view_74, where_2);  view_74 = None
        _tensor_constant3 = self._tensor_constant3;  _tensor_constant3 = None
        full_default_8 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_3 = torch.ops.aten.maximum.default(add_26, full_default_8);  add_26 = full_default_8 = None
        view_75 = torch.ops.aten.view.default(maximum_3, [24, 2048, 2048]);  maximum_3 = None
        amax_3 = torch.ops.aten.amax.default(view_75, [-1], True)
        sub_12 = torch.ops.aten.sub.Tensor(view_75, amax_3);  view_75 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        bmm_7 = torch.ops.aten.bmm.default(div_3, view_73);  div_3 = view_73 = None
        view_76 = torch.ops.aten.view.default(bmm_7, [2, 12, 2048, 64]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_25 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_77 = torch.ops.aten.view.default(clone_25, [2, 2048, 768]);  clone_25 = None
        view_78 = torch.ops.aten.view.default(view_77, [4096, 768]);  view_77 = None
        permute_41 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg60_1, view_78, permute_41);  arg60_1 = view_78 = permute_41 = None
        view_79 = torch.ops.aten.view.default(addmm_21, [2, 2048, 768]);  addmm_21 = None
        add_27 = torch.ops.aten.add.Tensor(view_61, view_79);  view_61 = view_79 = None
        view_80 = torch.ops.aten.view.default(add_27, [-1, 768]);  add_27 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(view_80, [1], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_13 = torch.ops.aten.sub.Tensor(view_80, getitem_15);  getitem_15 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_7);  sub_13 = rsqrt_7 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, arg61_1);  mul_19 = arg61_1 = None
        add_29 = torch.ops.aten.add.Tensor(mul_20, arg62_1);  mul_20 = arg62_1 = None
        permute_42 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg64_1, add_29, permute_42);  arg64_1 = add_29 = permute_42 = None
        relu_3 = torch.ops.aten.relu.default(addmm_22);  addmm_22 = None
        permute_43 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg66_1, relu_3, permute_43);  arg66_1 = relu_3 = permute_43 = None
        add_30 = torch.ops.aten.add.Tensor(view_80, addmm_23);  view_80 = addmm_23 = None
        view_81 = torch.ops.aten.view.default(add_30, [2, 2048, 768]);  add_30 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(view_81, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_31 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_14 = torch.ops.aten.sub.Tensor(view_81, getitem_17);  getitem_17 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = rsqrt_8 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, arg67_1);  mul_21 = arg67_1 = None
        add_32 = torch.ops.aten.add.Tensor(mul_22, arg68_1);  mul_22 = arg68_1 = None
        view_82 = torch.ops.aten.view.default(add_32, [4096, 768])
        permute_44 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg70_1, view_82, permute_44);  arg70_1 = view_82 = permute_44 = None
        view_83 = torch.ops.aten.view.default(addmm_24, [2, 2048, 768]);  addmm_24 = None
        mul_23 = torch.ops.aten.mul.Tensor(view_83, 0.125);  view_83 = None
        view_84 = torch.ops.aten.view.default(add_32, [4096, 768])
        permute_45 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg72_1, view_84, permute_45);  arg72_1 = view_84 = permute_45 = None
        view_85 = torch.ops.aten.view.default(addmm_25, [2, 2048, 768]);  addmm_25 = None
        view_86 = torch.ops.aten.view.default(view_85, [2, -1, 12, 64]);  view_85 = None
        permute_46 = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        clone_28 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_87 = torch.ops.aten.view.default(add_32, [4096, 768]);  add_32 = None
        permute_47 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg74_1, view_87, permute_47);  arg74_1 = view_87 = permute_47 = None
        view_88 = torch.ops.aten.view.default(addmm_26, [2, 2048, 768]);  addmm_26 = None
        view_89 = torch.ops.aten.view.default(view_88, [2, -1, 12, 64]);  view_88 = None
        permute_48 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_29 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_90 = torch.ops.aten.view.default(mul_23, [2, 2048, 12, 64]);  mul_23 = None
        permute_49 = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_30 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_91 = torch.ops.aten.view.default(clone_30, [24, -1, 64]);  clone_30 = None
        view_92 = torch.ops.aten.view.default(clone_28, [24, -1, 64])
        view_93 = torch.ops.aten.view.default(clone_29, [24, -1, 64])
        permute_50 = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
        bmm_8 = torch.ops.aten.bmm.default(view_91, permute_50);  view_91 = permute_50 = None
        view_94 = torch.ops.aten.view.default(bmm_8, [2, 12, 2048, 2048]);  bmm_8 = None
        add_33 = torch.ops.aten.add.Tensor(view_94, where_2);  view_94 = None
        _tensor_constant4 = self._tensor_constant4;  _tensor_constant4 = None
        full_default_9 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_4 = torch.ops.aten.maximum.default(add_33, full_default_9);  add_33 = full_default_9 = None
        view_95 = torch.ops.aten.view.default(maximum_4, [24, 2048, 2048]);  maximum_4 = None
        amax_4 = torch.ops.aten.amax.default(view_95, [-1], True)
        sub_15 = torch.ops.aten.sub.Tensor(view_95, amax_4);  view_95 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        bmm_9 = torch.ops.aten.bmm.default(div_4, view_93);  div_4 = view_93 = None
        view_96 = torch.ops.aten.view.default(bmm_9, [2, 12, 2048, 64]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        clone_32 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_97 = torch.ops.aten.view.default(clone_32, [2, 2048, 768]);  clone_32 = None
        view_98 = torch.ops.aten.view.default(view_97, [4096, 768]);  view_97 = None
        permute_52 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg76_1, view_98, permute_52);  arg76_1 = view_98 = permute_52 = None
        view_99 = torch.ops.aten.view.default(addmm_27, [2, 2048, 768]);  addmm_27 = None
        add_34 = torch.ops.aten.add.Tensor(view_81, view_99);  view_81 = view_99 = None
        view_100 = torch.ops.aten.view.default(add_34, [-1, 768]);  add_34 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(view_100, [1], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_35 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_16 = torch.ops.aten.sub.Tensor(view_100, getitem_19);  getitem_19 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_9);  sub_16 = rsqrt_9 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg77_1);  mul_24 = arg77_1 = None
        add_36 = torch.ops.aten.add.Tensor(mul_25, arg78_1);  mul_25 = arg78_1 = None
        permute_53 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg80_1, add_36, permute_53);  arg80_1 = add_36 = permute_53 = None
        relu_4 = torch.ops.aten.relu.default(addmm_28);  addmm_28 = None
        permute_54 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg82_1, relu_4, permute_54);  arg82_1 = relu_4 = permute_54 = None
        add_37 = torch.ops.aten.add.Tensor(view_100, addmm_29);  view_100 = addmm_29 = None
        view_101 = torch.ops.aten.view.default(add_37, [2, 2048, 768]);  add_37 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(view_101, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_17 = torch.ops.aten.sub.Tensor(view_101, getitem_21);  getitem_21 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_10);  sub_17 = rsqrt_10 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg83_1);  mul_26 = arg83_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_27, arg84_1);  mul_27 = arg84_1 = None
        view_102 = torch.ops.aten.view.default(add_39, [4096, 768])
        permute_55 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg86_1, view_102, permute_55);  arg86_1 = view_102 = permute_55 = None
        view_103 = torch.ops.aten.view.default(addmm_30, [2, 2048, 768]);  addmm_30 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_103, 0.125);  view_103 = None
        view_104 = torch.ops.aten.view.default(add_39, [4096, 768])
        permute_56 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg88_1, view_104, permute_56);  arg88_1 = view_104 = permute_56 = None
        view_105 = torch.ops.aten.view.default(addmm_31, [2, 2048, 768]);  addmm_31 = None
        view_106 = torch.ops.aten.view.default(view_105, [2, -1, 12, 64]);  view_105 = None
        permute_57 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_35 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_107 = torch.ops.aten.view.default(add_39, [4096, 768]);  add_39 = None
        permute_58 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg90_1, view_107, permute_58);  arg90_1 = view_107 = permute_58 = None
        view_108 = torch.ops.aten.view.default(addmm_32, [2, 2048, 768]);  addmm_32 = None
        view_109 = torch.ops.aten.view.default(view_108, [2, -1, 12, 64]);  view_108 = None
        permute_59 = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        clone_36 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_110 = torch.ops.aten.view.default(mul_28, [2, 2048, 12, 64]);  mul_28 = None
        permute_60 = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        clone_37 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_111 = torch.ops.aten.view.default(clone_37, [24, -1, 64]);  clone_37 = None
        view_112 = torch.ops.aten.view.default(clone_35, [24, -1, 64])
        view_113 = torch.ops.aten.view.default(clone_36, [24, -1, 64])
        permute_61 = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
        bmm_10 = torch.ops.aten.bmm.default(view_111, permute_61);  view_111 = permute_61 = None
        view_114 = torch.ops.aten.view.default(bmm_10, [2, 12, 2048, 2048]);  bmm_10 = None
        add_40 = torch.ops.aten.add.Tensor(view_114, where_2);  view_114 = None
        _tensor_constant5 = self._tensor_constant5;  _tensor_constant5 = None
        full_default_10 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_5 = torch.ops.aten.maximum.default(add_40, full_default_10);  add_40 = full_default_10 = None
        view_115 = torch.ops.aten.view.default(maximum_5, [24, 2048, 2048]);  maximum_5 = None
        amax_5 = torch.ops.aten.amax.default(view_115, [-1], True)
        sub_18 = torch.ops.aten.sub.Tensor(view_115, amax_5);  view_115 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        bmm_11 = torch.ops.aten.bmm.default(div_5, view_113);  div_5 = view_113 = None
        view_116 = torch.ops.aten.view.default(bmm_11, [2, 12, 2048, 64]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_39 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_117 = torch.ops.aten.view.default(clone_39, [2, 2048, 768]);  clone_39 = None
        view_118 = torch.ops.aten.view.default(view_117, [4096, 768]);  view_117 = None
        permute_63 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg92_1, view_118, permute_63);  arg92_1 = view_118 = permute_63 = None
        view_119 = torch.ops.aten.view.default(addmm_33, [2, 2048, 768]);  addmm_33 = None
        add_41 = torch.ops.aten.add.Tensor(view_101, view_119);  view_101 = view_119 = None
        view_120 = torch.ops.aten.view.default(add_41, [-1, 768]);  add_41 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(view_120, [1], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_19 = torch.ops.aten.sub.Tensor(view_120, getitem_23);  getitem_23 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_11);  sub_19 = rsqrt_11 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, arg93_1);  mul_29 = arg93_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_30, arg94_1);  mul_30 = arg94_1 = None
        permute_64 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg96_1, add_43, permute_64);  arg96_1 = add_43 = permute_64 = None
        relu_5 = torch.ops.aten.relu.default(addmm_34);  addmm_34 = None
        permute_65 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg98_1, relu_5, permute_65);  arg98_1 = relu_5 = permute_65 = None
        add_44 = torch.ops.aten.add.Tensor(view_120, addmm_35);  view_120 = addmm_35 = None
        view_121 = torch.ops.aten.view.default(add_44, [2, 2048, 768]);  add_44 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(view_121, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_20 = torch.ops.aten.sub.Tensor(view_121, getitem_25);  getitem_25 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_12);  sub_20 = rsqrt_12 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, arg99_1);  mul_31 = arg99_1 = None
        add_46 = torch.ops.aten.add.Tensor(mul_32, arg100_1);  mul_32 = arg100_1 = None
        view_122 = torch.ops.aten.view.default(add_46, [4096, 768])
        permute_66 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg102_1, view_122, permute_66);  arg102_1 = view_122 = permute_66 = None
        view_123 = torch.ops.aten.view.default(addmm_36, [2, 2048, 768]);  addmm_36 = None
        mul_33 = torch.ops.aten.mul.Tensor(view_123, 0.125);  view_123 = None
        view_124 = torch.ops.aten.view.default(add_46, [4096, 768])
        permute_67 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg104_1, view_124, permute_67);  arg104_1 = view_124 = permute_67 = None
        view_125 = torch.ops.aten.view.default(addmm_37, [2, 2048, 768]);  addmm_37 = None
        view_126 = torch.ops.aten.view.default(view_125, [2, -1, 12, 64]);  view_125 = None
        permute_68 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_42 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_127 = torch.ops.aten.view.default(add_46, [4096, 768]);  add_46 = None
        permute_69 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg106_1, view_127, permute_69);  arg106_1 = view_127 = permute_69 = None
        view_128 = torch.ops.aten.view.default(addmm_38, [2, 2048, 768]);  addmm_38 = None
        view_129 = torch.ops.aten.view.default(view_128, [2, -1, 12, 64]);  view_128 = None
        permute_70 = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_43 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_130 = torch.ops.aten.view.default(mul_33, [2, 2048, 12, 64]);  mul_33 = None
        permute_71 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        clone_44 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_131 = torch.ops.aten.view.default(clone_44, [24, -1, 64]);  clone_44 = None
        view_132 = torch.ops.aten.view.default(clone_42, [24, -1, 64])
        view_133 = torch.ops.aten.view.default(clone_43, [24, -1, 64])
        permute_72 = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
        bmm_12 = torch.ops.aten.bmm.default(view_131, permute_72);  view_131 = permute_72 = None
        view_134 = torch.ops.aten.view.default(bmm_12, [2, 12, 2048, 2048]);  bmm_12 = None
        add_47 = torch.ops.aten.add.Tensor(view_134, where_2);  view_134 = None
        _tensor_constant6 = self._tensor_constant6;  _tensor_constant6 = None
        full_default_11 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_6 = torch.ops.aten.maximum.default(add_47, full_default_11);  add_47 = full_default_11 = None
        view_135 = torch.ops.aten.view.default(maximum_6, [24, 2048, 2048]);  maximum_6 = None
        amax_6 = torch.ops.aten.amax.default(view_135, [-1], True)
        sub_21 = torch.ops.aten.sub.Tensor(view_135, amax_6);  view_135 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        bmm_13 = torch.ops.aten.bmm.default(div_6, view_133);  div_6 = view_133 = None
        view_136 = torch.ops.aten.view.default(bmm_13, [2, 12, 2048, 64]);  bmm_13 = None
        permute_73 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        clone_46 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_137 = torch.ops.aten.view.default(clone_46, [2, 2048, 768]);  clone_46 = None
        view_138 = torch.ops.aten.view.default(view_137, [4096, 768]);  view_137 = None
        permute_74 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg108_1, view_138, permute_74);  arg108_1 = view_138 = permute_74 = None
        view_139 = torch.ops.aten.view.default(addmm_39, [2, 2048, 768]);  addmm_39 = None
        add_48 = torch.ops.aten.add.Tensor(view_121, view_139);  view_121 = view_139 = None
        view_140 = torch.ops.aten.view.default(add_48, [-1, 768]);  add_48 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(view_140, [1], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_22 = torch.ops.aten.sub.Tensor(view_140, getitem_27);  getitem_27 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_13);  sub_22 = rsqrt_13 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg109_1);  mul_34 = arg109_1 = None
        add_50 = torch.ops.aten.add.Tensor(mul_35, arg110_1);  mul_35 = arg110_1 = None
        permute_75 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg112_1, add_50, permute_75);  arg112_1 = add_50 = permute_75 = None
        relu_6 = torch.ops.aten.relu.default(addmm_40);  addmm_40 = None
        permute_76 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg114_1, relu_6, permute_76);  arg114_1 = relu_6 = permute_76 = None
        add_51 = torch.ops.aten.add.Tensor(view_140, addmm_41);  view_140 = addmm_41 = None
        view_141 = torch.ops.aten.view.default(add_51, [2, 2048, 768]);  add_51 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(view_141, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_52 = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_23 = torch.ops.aten.sub.Tensor(view_141, getitem_29);  getitem_29 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_14);  sub_23 = rsqrt_14 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg115_1);  mul_36 = arg115_1 = None
        add_53 = torch.ops.aten.add.Tensor(mul_37, arg116_1);  mul_37 = arg116_1 = None
        view_142 = torch.ops.aten.view.default(add_53, [4096, 768])
        permute_77 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg118_1, view_142, permute_77);  arg118_1 = view_142 = permute_77 = None
        view_143 = torch.ops.aten.view.default(addmm_42, [2, 2048, 768]);  addmm_42 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_143, 0.125);  view_143 = None
        view_144 = torch.ops.aten.view.default(add_53, [4096, 768])
        permute_78 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg120_1, view_144, permute_78);  arg120_1 = view_144 = permute_78 = None
        view_145 = torch.ops.aten.view.default(addmm_43, [2, 2048, 768]);  addmm_43 = None
        view_146 = torch.ops.aten.view.default(view_145, [2, -1, 12, 64]);  view_145 = None
        permute_79 = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_49 = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_147 = torch.ops.aten.view.default(add_53, [4096, 768]);  add_53 = None
        permute_80 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg122_1, view_147, permute_80);  arg122_1 = view_147 = permute_80 = None
        view_148 = torch.ops.aten.view.default(addmm_44, [2, 2048, 768]);  addmm_44 = None
        view_149 = torch.ops.aten.view.default(view_148, [2, -1, 12, 64]);  view_148 = None
        permute_81 = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        clone_50 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_150 = torch.ops.aten.view.default(mul_38, [2, 2048, 12, 64]);  mul_38 = None
        permute_82 = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        clone_51 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_151 = torch.ops.aten.view.default(clone_51, [24, -1, 64]);  clone_51 = None
        view_152 = torch.ops.aten.view.default(clone_49, [24, -1, 64])
        view_153 = torch.ops.aten.view.default(clone_50, [24, -1, 64])
        permute_83 = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
        bmm_14 = torch.ops.aten.bmm.default(view_151, permute_83);  view_151 = permute_83 = None
        view_154 = torch.ops.aten.view.default(bmm_14, [2, 12, 2048, 2048]);  bmm_14 = None
        add_54 = torch.ops.aten.add.Tensor(view_154, where_2);  view_154 = None
        _tensor_constant7 = self._tensor_constant7;  _tensor_constant7 = None
        full_default_12 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_7 = torch.ops.aten.maximum.default(add_54, full_default_12);  add_54 = full_default_12 = None
        view_155 = torch.ops.aten.view.default(maximum_7, [24, 2048, 2048]);  maximum_7 = None
        amax_7 = torch.ops.aten.amax.default(view_155, [-1], True)
        sub_24 = torch.ops.aten.sub.Tensor(view_155, amax_7);  view_155 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        bmm_15 = torch.ops.aten.bmm.default(div_7, view_153);  div_7 = view_153 = None
        view_156 = torch.ops.aten.view.default(bmm_15, [2, 12, 2048, 64]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        clone_53 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_157 = torch.ops.aten.view.default(clone_53, [2, 2048, 768]);  clone_53 = None
        view_158 = torch.ops.aten.view.default(view_157, [4096, 768]);  view_157 = None
        permute_85 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg124_1, view_158, permute_85);  arg124_1 = view_158 = permute_85 = None
        view_159 = torch.ops.aten.view.default(addmm_45, [2, 2048, 768]);  addmm_45 = None
        add_55 = torch.ops.aten.add.Tensor(view_141, view_159);  view_141 = view_159 = None
        view_160 = torch.ops.aten.view.default(add_55, [-1, 768]);  add_55 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(view_160, [1], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_25 = torch.ops.aten.sub.Tensor(view_160, getitem_31);  getitem_31 = None
        mul_39 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_15);  sub_25 = rsqrt_15 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_39, arg125_1);  mul_39 = arg125_1 = None
        add_57 = torch.ops.aten.add.Tensor(mul_40, arg126_1);  mul_40 = arg126_1 = None
        permute_86 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg128_1, add_57, permute_86);  arg128_1 = add_57 = permute_86 = None
        relu_7 = torch.ops.aten.relu.default(addmm_46);  addmm_46 = None
        permute_87 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg130_1, relu_7, permute_87);  arg130_1 = relu_7 = permute_87 = None
        add_58 = torch.ops.aten.add.Tensor(view_160, addmm_47);  view_160 = addmm_47 = None
        view_161 = torch.ops.aten.view.default(add_58, [2, 2048, 768]);  add_58 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(view_161, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_59 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_26 = torch.ops.aten.sub.Tensor(view_161, getitem_33);  getitem_33 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_16);  sub_26 = rsqrt_16 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, arg131_1);  mul_41 = arg131_1 = None
        add_60 = torch.ops.aten.add.Tensor(mul_42, arg132_1);  mul_42 = arg132_1 = None
        view_162 = torch.ops.aten.view.default(add_60, [4096, 768])
        permute_88 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg134_1, view_162, permute_88);  arg134_1 = view_162 = permute_88 = None
        view_163 = torch.ops.aten.view.default(addmm_48, [2, 2048, 768]);  addmm_48 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_163, 0.125);  view_163 = None
        view_164 = torch.ops.aten.view.default(add_60, [4096, 768])
        permute_89 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg136_1, view_164, permute_89);  arg136_1 = view_164 = permute_89 = None
        view_165 = torch.ops.aten.view.default(addmm_49, [2, 2048, 768]);  addmm_49 = None
        view_166 = torch.ops.aten.view.default(view_165, [2, -1, 12, 64]);  view_165 = None
        permute_90 = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_56 = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_167 = torch.ops.aten.view.default(add_60, [4096, 768]);  add_60 = None
        permute_91 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg138_1, view_167, permute_91);  arg138_1 = view_167 = permute_91 = None
        view_168 = torch.ops.aten.view.default(addmm_50, [2, 2048, 768]);  addmm_50 = None
        view_169 = torch.ops.aten.view.default(view_168, [2, -1, 12, 64]);  view_168 = None
        permute_92 = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        clone_57 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_170 = torch.ops.aten.view.default(mul_43, [2, 2048, 12, 64]);  mul_43 = None
        permute_93 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_58 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_171 = torch.ops.aten.view.default(clone_58, [24, -1, 64]);  clone_58 = None
        view_172 = torch.ops.aten.view.default(clone_56, [24, -1, 64])
        view_173 = torch.ops.aten.view.default(clone_57, [24, -1, 64])
        permute_94 = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        bmm_16 = torch.ops.aten.bmm.default(view_171, permute_94);  view_171 = permute_94 = None
        view_174 = torch.ops.aten.view.default(bmm_16, [2, 12, 2048, 2048]);  bmm_16 = None
        add_61 = torch.ops.aten.add.Tensor(view_174, where_2);  view_174 = None
        _tensor_constant8 = self._tensor_constant8;  _tensor_constant8 = None
        full_default_13 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_8 = torch.ops.aten.maximum.default(add_61, full_default_13);  add_61 = full_default_13 = None
        view_175 = torch.ops.aten.view.default(maximum_8, [24, 2048, 2048]);  maximum_8 = None
        amax_8 = torch.ops.aten.amax.default(view_175, [-1], True)
        sub_27 = torch.ops.aten.sub.Tensor(view_175, amax_8);  view_175 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        bmm_17 = torch.ops.aten.bmm.default(div_8, view_173);  div_8 = view_173 = None
        view_176 = torch.ops.aten.view.default(bmm_17, [2, 12, 2048, 64]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
        clone_60 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_177 = torch.ops.aten.view.default(clone_60, [2, 2048, 768]);  clone_60 = None
        view_178 = torch.ops.aten.view.default(view_177, [4096, 768]);  view_177 = None
        permute_96 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg140_1, view_178, permute_96);  arg140_1 = view_178 = permute_96 = None
        view_179 = torch.ops.aten.view.default(addmm_51, [2, 2048, 768]);  addmm_51 = None
        add_62 = torch.ops.aten.add.Tensor(view_161, view_179);  view_161 = view_179 = None
        view_180 = torch.ops.aten.view.default(add_62, [-1, 768]);  add_62 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(view_180, [1], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_63 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_28 = torch.ops.aten.sub.Tensor(view_180, getitem_35);  getitem_35 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_17);  sub_28 = rsqrt_17 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg141_1);  mul_44 = arg141_1 = None
        add_64 = torch.ops.aten.add.Tensor(mul_45, arg142_1);  mul_45 = arg142_1 = None
        permute_97 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg144_1, add_64, permute_97);  arg144_1 = add_64 = permute_97 = None
        relu_8 = torch.ops.aten.relu.default(addmm_52);  addmm_52 = None
        permute_98 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg146_1, relu_8, permute_98);  arg146_1 = relu_8 = permute_98 = None
        add_65 = torch.ops.aten.add.Tensor(view_180, addmm_53);  view_180 = addmm_53 = None
        view_181 = torch.ops.aten.view.default(add_65, [2, 2048, 768]);  add_65 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(view_181, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_29 = torch.ops.aten.sub.Tensor(view_181, getitem_37);  getitem_37 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = rsqrt_18 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, arg147_1);  mul_46 = arg147_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_47, arg148_1);  mul_47 = arg148_1 = None
        view_182 = torch.ops.aten.view.default(add_67, [4096, 768])
        permute_99 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg150_1, view_182, permute_99);  arg150_1 = view_182 = permute_99 = None
        view_183 = torch.ops.aten.view.default(addmm_54, [2, 2048, 768]);  addmm_54 = None
        mul_48 = torch.ops.aten.mul.Tensor(view_183, 0.125);  view_183 = None
        view_184 = torch.ops.aten.view.default(add_67, [4096, 768])
        permute_100 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg152_1, view_184, permute_100);  arg152_1 = view_184 = permute_100 = None
        view_185 = torch.ops.aten.view.default(addmm_55, [2, 2048, 768]);  addmm_55 = None
        view_186 = torch.ops.aten.view.default(view_185, [2, -1, 12, 64]);  view_185 = None
        permute_101 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_63 = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_187 = torch.ops.aten.view.default(add_67, [4096, 768]);  add_67 = None
        permute_102 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg154_1, view_187, permute_102);  arg154_1 = view_187 = permute_102 = None
        view_188 = torch.ops.aten.view.default(addmm_56, [2, 2048, 768]);  addmm_56 = None
        view_189 = torch.ops.aten.view.default(view_188, [2, -1, 12, 64]);  view_188 = None
        permute_103 = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        clone_64 = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_190 = torch.ops.aten.view.default(mul_48, [2, 2048, 12, 64]);  mul_48 = None
        permute_104 = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_65 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_191 = torch.ops.aten.view.default(clone_65, [24, -1, 64]);  clone_65 = None
        view_192 = torch.ops.aten.view.default(clone_63, [24, -1, 64])
        view_193 = torch.ops.aten.view.default(clone_64, [24, -1, 64])
        permute_105 = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
        bmm_18 = torch.ops.aten.bmm.default(view_191, permute_105);  view_191 = permute_105 = None
        view_194 = torch.ops.aten.view.default(bmm_18, [2, 12, 2048, 2048]);  bmm_18 = None
        add_68 = torch.ops.aten.add.Tensor(view_194, where_2);  view_194 = None
        _tensor_constant9 = self._tensor_constant9;  _tensor_constant9 = None
        full_default_14 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_9 = torch.ops.aten.maximum.default(add_68, full_default_14);  add_68 = full_default_14 = None
        view_195 = torch.ops.aten.view.default(maximum_9, [24, 2048, 2048]);  maximum_9 = None
        amax_9 = torch.ops.aten.amax.default(view_195, [-1], True)
        sub_30 = torch.ops.aten.sub.Tensor(view_195, amax_9);  view_195 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        bmm_19 = torch.ops.aten.bmm.default(div_9, view_193);  div_9 = view_193 = None
        view_196 = torch.ops.aten.view.default(bmm_19, [2, 12, 2048, 64]);  bmm_19 = None
        permute_106 = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        clone_67 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_197 = torch.ops.aten.view.default(clone_67, [2, 2048, 768]);  clone_67 = None
        view_198 = torch.ops.aten.view.default(view_197, [4096, 768]);  view_197 = None
        permute_107 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg156_1, view_198, permute_107);  arg156_1 = view_198 = permute_107 = None
        view_199 = torch.ops.aten.view.default(addmm_57, [2, 2048, 768]);  addmm_57 = None
        add_69 = torch.ops.aten.add.Tensor(view_181, view_199);  view_181 = view_199 = None
        view_200 = torch.ops.aten.view.default(add_69, [-1, 768]);  add_69 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(view_200, [1], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_31 = torch.ops.aten.sub.Tensor(view_200, getitem_39);  getitem_39 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_19);  sub_31 = rsqrt_19 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg157_1);  mul_49 = arg157_1 = None
        add_71 = torch.ops.aten.add.Tensor(mul_50, arg158_1);  mul_50 = arg158_1 = None
        permute_108 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg160_1, add_71, permute_108);  arg160_1 = add_71 = permute_108 = None
        relu_9 = torch.ops.aten.relu.default(addmm_58);  addmm_58 = None
        permute_109 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg162_1, relu_9, permute_109);  arg162_1 = relu_9 = permute_109 = None
        add_72 = torch.ops.aten.add.Tensor(view_200, addmm_59);  view_200 = addmm_59 = None
        view_201 = torch.ops.aten.view.default(add_72, [2, 2048, 768]);  add_72 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(view_201, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_73 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_32 = torch.ops.aten.sub.Tensor(view_201, getitem_41);  getitem_41 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = rsqrt_20 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, arg163_1);  mul_51 = arg163_1 = None
        add_74 = torch.ops.aten.add.Tensor(mul_52, arg164_1);  mul_52 = arg164_1 = None
        view_202 = torch.ops.aten.view.default(add_74, [4096, 768])
        permute_110 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg166_1, view_202, permute_110);  arg166_1 = view_202 = permute_110 = None
        view_203 = torch.ops.aten.view.default(addmm_60, [2, 2048, 768]);  addmm_60 = None
        mul_53 = torch.ops.aten.mul.Tensor(view_203, 0.125);  view_203 = None
        view_204 = torch.ops.aten.view.default(add_74, [4096, 768])
        permute_111 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg168_1, view_204, permute_111);  arg168_1 = view_204 = permute_111 = None
        view_205 = torch.ops.aten.view.default(addmm_61, [2, 2048, 768]);  addmm_61 = None
        view_206 = torch.ops.aten.view.default(view_205, [2, -1, 12, 64]);  view_205 = None
        permute_112 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_70 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_207 = torch.ops.aten.view.default(add_74, [4096, 768]);  add_74 = None
        permute_113 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg170_1, view_207, permute_113);  arg170_1 = view_207 = permute_113 = None
        view_208 = torch.ops.aten.view.default(addmm_62, [2, 2048, 768]);  addmm_62 = None
        view_209 = torch.ops.aten.view.default(view_208, [2, -1, 12, 64]);  view_208 = None
        permute_114 = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_71 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_210 = torch.ops.aten.view.default(mul_53, [2, 2048, 12, 64]);  mul_53 = None
        permute_115 = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        clone_72 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_211 = torch.ops.aten.view.default(clone_72, [24, -1, 64]);  clone_72 = None
        view_212 = torch.ops.aten.view.default(clone_70, [24, -1, 64])
        view_213 = torch.ops.aten.view.default(clone_71, [24, -1, 64])
        permute_116 = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        bmm_20 = torch.ops.aten.bmm.default(view_211, permute_116);  view_211 = permute_116 = None
        view_214 = torch.ops.aten.view.default(bmm_20, [2, 12, 2048, 2048]);  bmm_20 = None
        add_75 = torch.ops.aten.add.Tensor(view_214, where_2);  view_214 = None
        _tensor_constant10 = self._tensor_constant10;  _tensor_constant10 = None
        full_default_15 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_10 = torch.ops.aten.maximum.default(add_75, full_default_15);  add_75 = full_default_15 = None
        view_215 = torch.ops.aten.view.default(maximum_10, [24, 2048, 2048]);  maximum_10 = None
        amax_10 = torch.ops.aten.amax.default(view_215, [-1], True)
        sub_33 = torch.ops.aten.sub.Tensor(view_215, amax_10);  view_215 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        bmm_21 = torch.ops.aten.bmm.default(div_10, view_213);  div_10 = view_213 = None
        view_216 = torch.ops.aten.view.default(bmm_21, [2, 12, 2048, 64]);  bmm_21 = None
        permute_117 = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        clone_74 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_217 = torch.ops.aten.view.default(clone_74, [2, 2048, 768]);  clone_74 = None
        view_218 = torch.ops.aten.view.default(view_217, [4096, 768]);  view_217 = None
        permute_118 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg172_1, view_218, permute_118);  arg172_1 = view_218 = permute_118 = None
        view_219 = torch.ops.aten.view.default(addmm_63, [2, 2048, 768]);  addmm_63 = None
        add_76 = torch.ops.aten.add.Tensor(view_201, view_219);  view_201 = view_219 = None
        view_220 = torch.ops.aten.view.default(add_76, [-1, 768]);  add_76 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(view_220, [1], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_77 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_34 = torch.ops.aten.sub.Tensor(view_220, getitem_43);  getitem_43 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = rsqrt_21 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, arg173_1);  mul_54 = arg173_1 = None
        add_78 = torch.ops.aten.add.Tensor(mul_55, arg174_1);  mul_55 = arg174_1 = None
        permute_119 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg176_1, add_78, permute_119);  arg176_1 = add_78 = permute_119 = None
        relu_10 = torch.ops.aten.relu.default(addmm_64);  addmm_64 = None
        permute_120 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg178_1, relu_10, permute_120);  arg178_1 = relu_10 = permute_120 = None
        add_79 = torch.ops.aten.add.Tensor(view_220, addmm_65);  view_220 = addmm_65 = None
        view_221 = torch.ops.aten.view.default(add_79, [2, 2048, 768]);  add_79 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(view_221, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_35 = torch.ops.aten.sub.Tensor(view_221, getitem_45);  getitem_45 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = rsqrt_22 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg179_1);  mul_56 = arg179_1 = None
        add_81 = torch.ops.aten.add.Tensor(mul_57, arg180_1);  mul_57 = arg180_1 = None
        view_222 = torch.ops.aten.view.default(add_81, [4096, 768])
        permute_121 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg182_1, view_222, permute_121);  arg182_1 = view_222 = permute_121 = None
        view_223 = torch.ops.aten.view.default(addmm_66, [2, 2048, 768]);  addmm_66 = None
        mul_58 = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
        view_224 = torch.ops.aten.view.default(add_81, [4096, 768])
        permute_122 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg184_1, view_224, permute_122);  arg184_1 = view_224 = permute_122 = None
        view_225 = torch.ops.aten.view.default(addmm_67, [2, 2048, 768]);  addmm_67 = None
        view_226 = torch.ops.aten.view.default(view_225, [2, -1, 12, 64]);  view_225 = None
        permute_123 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_77 = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
        view_227 = torch.ops.aten.view.default(add_81, [4096, 768]);  add_81 = None
        permute_124 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg186_1, view_227, permute_124);  arg186_1 = view_227 = permute_124 = None
        view_228 = torch.ops.aten.view.default(addmm_68, [2, 2048, 768]);  addmm_68 = None
        view_229 = torch.ops.aten.view.default(view_228, [2, -1, 12, 64]);  view_228 = None
        permute_125 = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        clone_78 = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_230 = torch.ops.aten.view.default(mul_58, [2, 2048, 12, 64]);  mul_58 = None
        permute_126 = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        clone_79 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_231 = torch.ops.aten.view.default(clone_79, [24, -1, 64]);  clone_79 = None
        view_232 = torch.ops.aten.view.default(clone_77, [24, -1, 64])
        view_233 = torch.ops.aten.view.default(clone_78, [24, -1, 64])
        permute_127 = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        bmm_22 = torch.ops.aten.bmm.default(view_231, permute_127);  view_231 = permute_127 = None
        view_234 = torch.ops.aten.view.default(bmm_22, [2, 12, 2048, 2048]);  bmm_22 = None
        add_82 = torch.ops.aten.add.Tensor(view_234, where_2);  view_234 = where_2 = None
        _tensor_constant11 = self._tensor_constant11;  _tensor_constant11 = None
        full_default_16 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_11 = torch.ops.aten.maximum.default(add_82, full_default_16);  add_82 = full_default_16 = None
        view_235 = torch.ops.aten.view.default(maximum_11, [24, 2048, 2048]);  maximum_11 = None
        amax_11 = torch.ops.aten.amax.default(view_235, [-1], True)
        sub_36 = torch.ops.aten.sub.Tensor(view_235, amax_11);  view_235 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        bmm_23 = torch.ops.aten.bmm.default(div_11, view_233);  div_11 = view_233 = None
        view_236 = torch.ops.aten.view.default(bmm_23, [2, 12, 2048, 64]);  bmm_23 = None
        permute_128 = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_81 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_237 = torch.ops.aten.view.default(clone_81, [2, 2048, 768]);  clone_81 = None
        view_238 = torch.ops.aten.view.default(view_237, [4096, 768]);  view_237 = None
        permute_129 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg188_1, view_238, permute_129);  arg188_1 = view_238 = permute_129 = None
        view_239 = torch.ops.aten.view.default(addmm_69, [2, 2048, 768]);  addmm_69 = None
        add_83 = torch.ops.aten.add.Tensor(view_221, view_239);  view_221 = view_239 = None
        view_240 = torch.ops.aten.view.default(add_83, [-1, 768]);  add_83 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(view_240, [1], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_37 = torch.ops.aten.sub.Tensor(view_240, getitem_47);  getitem_47 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = rsqrt_23 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, arg189_1);  mul_59 = arg189_1 = None
        add_85 = torch.ops.aten.add.Tensor(mul_60, arg190_1);  mul_60 = arg190_1 = None
        permute_130 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg192_1, add_85, permute_130);  arg192_1 = add_85 = permute_130 = None
        relu_11 = torch.ops.aten.relu.default(addmm_70);  addmm_70 = None
        permute_131 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg194_1, relu_11, permute_131);  arg194_1 = relu_11 = permute_131 = None
        add_86 = torch.ops.aten.add.Tensor(view_240, addmm_71);  view_240 = addmm_71 = None
        view_241 = torch.ops.aten.view.default(add_86, [2, 2048, 768]);  add_86 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(view_241, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_87 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_38 = torch.ops.aten.sub.Tensor(view_241, getitem_49);  view_241 = getitem_49 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = rsqrt_24 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, arg195_1);  mul_61 = arg195_1 = None
        add_88 = torch.ops.aten.add.Tensor(mul_62, arg196_1);  mul_62 = arg196_1 = None
        permute_132 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_242 = torch.ops.aten.view.default(add_88, [4096, 768]);  add_88 = None
        mm = torch.ops.aten.mm.default(view_242, permute_132);  view_242 = permute_132 = None
        view_243 = torch.ops.aten.view.default(mm, [2, 2048, 50272]);  mm = None
        slice_9 = torch.ops.aten.slice.Tensor(view_243, 1, 0, -1)
        clone_84 = torch.ops.aten.clone.default(slice_9, memory_format = torch.contiguous_format);  slice_9 = None
        slice_11 = torch.ops.aten.slice.Tensor(arg197_1, 1, 1, 9223372036854775807);  arg197_1 = None
        clone_85 = torch.ops.aten.clone.default(slice_11, memory_format = torch.contiguous_format);  slice_11 = None
        view_244 = torch.ops.aten.view.default(clone_84, [-1, 50272]);  clone_84 = None
        view_245 = torch.ops.aten.view.default(clone_85, [-1]);  clone_85 = None
        amax_12 = torch.ops.aten.amax.default(view_244, [1], True)
        sub_39 = torch.ops.aten.sub.Tensor(view_244, amax_12);  view_244 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_39)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_40 = torch.ops.aten.sub.Tensor(sub_39, log);  sub_39 = log = None
        ne = torch.ops.aten.ne.Scalar(view_245, -100)
        full_default_17 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne, view_245, full_default_17);  ne = full_default_17 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather = torch.ops.aten.gather.default(sub_40, 1, unsqueeze_6);  sub_40 = unsqueeze_6 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_245, -100)
        full_default_18 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4 = torch.ops.aten.where.self(ne_1, neg, full_default_18);  ne_1 = neg = full_default_18 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_245, -100);  view_245 = None
        sum_14 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15 = torch.ops.aten.sum.default(where_4);  where_4 = None
        div_12 = torch.ops.aten.div.Tensor(sum_15, convert_element_type_3);  sum_15 = convert_element_type_3 = None
        return (div_12, view_243, clone, clone_1, clone_7, clone_8, clone_14, clone_15, clone_21, clone_22, clone_28, clone_29, clone_35, clone_36, clone_42, clone_43, clone_49, clone_50, clone_56, clone_57, clone_63, clone_64, clone_70, clone_71, clone_77, clone_78)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 2048), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 154435584, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50272, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 6297600, device=device(type='cuda', index=0))
    reader.tensor(buf2, (2050, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768, 768), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768, 768), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf15, (3072, 768), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf16, (3072,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768, 3072), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768, 768), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768, 768), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768, 768), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
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
    buf37 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768, 768), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768, 768), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768, 768), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3072, 768), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (3072,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768, 3072), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768, 768), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768, 768), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768, 768), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768, 768), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf63, (3072, 768), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf64, (3072,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768, 3072), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768, 768), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768, 768), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768, 768), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768, 768), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf79, (3072, 768), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf80, (3072,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768, 3072), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 768), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768, 768), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768, 768), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768, 768), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf95, (3072, 768), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf96, (3072,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768, 3072), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768, 768), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768, 768), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768, 768), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768, 768), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf111, (3072, 768), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf112, (3072,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768, 3072), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768, 768), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768, 768), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768, 768), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf127, (3072, 768), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf128, (3072,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768, 3072), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768, 768), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768, 768), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768, 768), is_leaf=True)  # arg137_1
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
    buf149 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768, 768), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768, 768), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768, 768), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768, 768), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf159, (3072, 768), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf160, (3072,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768, 3072), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 768), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768, 768), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768, 768), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768, 768), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf174, (768,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf175, (3072, 768), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf176, (3072,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768, 3072), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf179, (768,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf180, (768,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768, 768), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768, 768), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768, 768), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768, 768), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf189, (768,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf191, (3072, 768), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf192, (3072,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768, 3072), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf195, (768,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf196, (768,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf197, (2, 2048), dtype=torch.int64, is_leaf=True)  # arg197_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)