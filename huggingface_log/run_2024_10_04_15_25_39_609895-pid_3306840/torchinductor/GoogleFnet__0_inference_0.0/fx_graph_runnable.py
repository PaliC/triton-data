
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1):
        expand = torch.ops.aten.expand.default(arg1_1, [16, 512]);  arg1_1 = None
        embedding = torch.ops.aten.embedding.default(arg3_1, arg0_1, 3);  arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg5_1, arg2_1);  arg5_1 = arg2_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        view = torch.ops.aten.view.default(add_3, [8192, 768]);  add_3 = None
        permute = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [16, 512, 768]);  addmm = None
        convert_element_type = torch.ops.prims.convert_element_type.default(view_1, torch.complex64)
        _fft_c2c = torch.ops.aten._fft_c2c.default(convert_element_type, [1, 2], 0, True);  convert_element_type = None
        view_as_real = torch.ops.aten.view_as_real.default(_fft_c2c);  _fft_c2c = None
        select = torch.ops.aten.select.int(view_as_real, 3, 0);  view_as_real = None
        add_4 = torch.ops.aten.add.Tensor(view_1, select);  view_1 = select = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        view_2 = torch.ops.aten.view.default(add_6, [8192, 768])
        permute_1 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg13_1, view_2, permute_1);  arg13_1 = view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [16, 512, 3072]);  addmm_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_3, 0.5)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_3, 3.0)
        mul_5 = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7 = torch.ops.aten.add.Tensor(view_3, mul_5);  view_3 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        view_4 = torch.ops.aten.view.default(mul_7, [8192, 3072]);  mul_7 = None
        permute_2 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg15_1, view_4, permute_2);  arg15_1 = view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(addmm_2, [16, 512, 768]);  addmm_2 = None
        add_9 = torch.ops.aten.add.Tensor(view_5, add_6);  view_5 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg16_1);  mul_8 = arg16_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, arg17_1);  mul_9 = arg17_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_11, torch.complex64)
        _fft_c2c_1 = torch.ops.aten._fft_c2c.default(convert_element_type_1, [1, 2], 0, True);  convert_element_type_1 = None
        view_as_real_1 = torch.ops.aten.view_as_real.default(_fft_c2c_1);  _fft_c2c_1 = None
        select_1 = torch.ops.aten.select.int(view_as_real_1, 3, 0);  view_as_real_1 = None
        add_12 = torch.ops.aten.add.Tensor(add_11, select_1);  add_11 = select_1 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_12, getitem_7);  add_12 = getitem_7 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg18_1);  mul_10 = arg18_1 = None
        add_14 = torch.ops.aten.add.Tensor(mul_11, arg19_1);  mul_11 = arg19_1 = None
        view_6 = torch.ops.aten.view.default(add_14, [8192, 768])
        permute_3 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg21_1, view_6, permute_3);  arg21_1 = view_6 = permute_3 = None
        view_7 = torch.ops.aten.view.default(addmm_3, [16, 512, 3072]);  addmm_3 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_7, 0.5)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_7, 3.0)
        mul_13 = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15 = torch.ops.aten.add.Tensor(view_7, mul_13);  view_7 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16 = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_8 = torch.ops.aten.view.default(mul_15, [8192, 3072]);  mul_15 = None
        permute_4 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg23_1, view_8, permute_4);  arg23_1 = view_8 = permute_4 = None
        view_9 = torch.ops.aten.view.default(addmm_4, [16, 512, 768]);  addmm_4 = None
        add_17 = torch.ops.aten.add.Tensor(view_9, add_14);  view_9 = add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg24_1);  mul_16 = arg24_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_17, arg25_1);  mul_17 = arg25_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_19, torch.complex64)
        _fft_c2c_2 = torch.ops.aten._fft_c2c.default(convert_element_type_2, [1, 2], 0, True);  convert_element_type_2 = None
        view_as_real_2 = torch.ops.aten.view_as_real.default(_fft_c2c_2);  _fft_c2c_2 = None
        select_2 = torch.ops.aten.select.int(view_as_real_2, 3, 0);  view_as_real_2 = None
        add_20 = torch.ops.aten.add.Tensor(add_19, select_2);  add_19 = select_2 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_20, getitem_11);  add_20 = getitem_11 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg26_1);  mul_18 = arg26_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_19, arg27_1);  mul_19 = arg27_1 = None
        view_10 = torch.ops.aten.view.default(add_22, [8192, 768])
        permute_5 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg29_1, view_10, permute_5);  arg29_1 = view_10 = permute_5 = None
        view_11 = torch.ops.aten.view.default(addmm_5, [16, 512, 3072]);  addmm_5 = None
        mul_20 = torch.ops.aten.mul.Tensor(view_11, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
        mul_21 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23 = torch.ops.aten.add.Tensor(view_11, mul_21);  view_11 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24 = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        view_12 = torch.ops.aten.view.default(mul_23, [8192, 3072]);  mul_23 = None
        permute_6 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg31_1, view_12, permute_6);  arg31_1 = view_12 = permute_6 = None
        view_13 = torch.ops.aten.view.default(addmm_6, [16, 512, 768]);  addmm_6 = None
        add_25 = torch.ops.aten.add.Tensor(view_13, add_22);  view_13 = add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg32_1);  mul_24 = arg32_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_25, arg33_1);  mul_25 = arg33_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_27, torch.complex64)
        _fft_c2c_3 = torch.ops.aten._fft_c2c.default(convert_element_type_3, [1, 2], 0, True);  convert_element_type_3 = None
        view_as_real_3 = torch.ops.aten.view_as_real.default(_fft_c2c_3);  _fft_c2c_3 = None
        select_3 = torch.ops.aten.select.int(view_as_real_3, 3, 0);  view_as_real_3 = None
        add_28 = torch.ops.aten.add.Tensor(add_27, select_3);  add_27 = select_3 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_28, getitem_15);  add_28 = getitem_15 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg34_1);  mul_26 = arg34_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_27, arg35_1);  mul_27 = arg35_1 = None
        view_14 = torch.ops.aten.view.default(add_30, [8192, 768])
        permute_7 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg37_1, view_14, permute_7);  arg37_1 = view_14 = permute_7 = None
        view_15 = torch.ops.aten.view.default(addmm_7, [16, 512, 3072]);  addmm_7 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_15, 0.5)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
        mul_29 = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31 = torch.ops.aten.add.Tensor(view_15, mul_29);  view_15 = mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32 = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        view_16 = torch.ops.aten.view.default(mul_31, [8192, 3072]);  mul_31 = None
        permute_8 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg39_1, view_16, permute_8);  arg39_1 = view_16 = permute_8 = None
        view_17 = torch.ops.aten.view.default(addmm_8, [16, 512, 768]);  addmm_8 = None
        add_33 = torch.ops.aten.add.Tensor(view_17, add_30);  view_17 = add_30 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg40_1);  mul_32 = arg40_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_33, arg41_1);  mul_33 = arg41_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(add_35, torch.complex64)
        _fft_c2c_4 = torch.ops.aten._fft_c2c.default(convert_element_type_4, [1, 2], 0, True);  convert_element_type_4 = None
        view_as_real_4 = torch.ops.aten.view_as_real.default(_fft_c2c_4);  _fft_c2c_4 = None
        select_4 = torch.ops.aten.select.int(view_as_real_4, 3, 0);  view_as_real_4 = None
        add_36 = torch.ops.aten.add.Tensor(add_35, select_4);  add_35 = select_4 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_36, getitem_19);  add_36 = getitem_19 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg42_1);  mul_34 = arg42_1 = None
        add_38 = torch.ops.aten.add.Tensor(mul_35, arg43_1);  mul_35 = arg43_1 = None
        view_18 = torch.ops.aten.view.default(add_38, [8192, 768])
        permute_9 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg45_1, view_18, permute_9);  arg45_1 = view_18 = permute_9 = None
        view_19 = torch.ops.aten.view.default(addmm_9, [16, 512, 3072]);  addmm_9 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_19, 0.5)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
        mul_37 = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39 = torch.ops.aten.add.Tensor(view_19, mul_37);  view_19 = mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40 = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        view_20 = torch.ops.aten.view.default(mul_39, [8192, 3072]);  mul_39 = None
        permute_10 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg47_1, view_20, permute_10);  arg47_1 = view_20 = permute_10 = None
        view_21 = torch.ops.aten.view.default(addmm_10, [16, 512, 768]);  addmm_10 = None
        add_41 = torch.ops.aten.add.Tensor(view_21, add_38);  view_21 = add_38 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg48_1);  mul_40 = arg48_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_41, arg49_1);  mul_41 = arg49_1 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(add_43, torch.complex64)
        _fft_c2c_5 = torch.ops.aten._fft_c2c.default(convert_element_type_5, [1, 2], 0, True);  convert_element_type_5 = None
        view_as_real_5 = torch.ops.aten.view_as_real.default(_fft_c2c_5);  _fft_c2c_5 = None
        select_5 = torch.ops.aten.select.int(view_as_real_5, 3, 0);  view_as_real_5 = None
        add_44 = torch.ops.aten.add.Tensor(add_43, select_5);  add_43 = select_5 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_44, getitem_23);  add_44 = getitem_23 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg50_1);  mul_42 = arg50_1 = None
        add_46 = torch.ops.aten.add.Tensor(mul_43, arg51_1);  mul_43 = arg51_1 = None
        view_22 = torch.ops.aten.view.default(add_46, [8192, 768])
        permute_11 = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg53_1, view_22, permute_11);  arg53_1 = view_22 = permute_11 = None
        view_23 = torch.ops.aten.view.default(addmm_11, [16, 512, 3072]);  addmm_11 = None
        mul_44 = torch.ops.aten.mul.Tensor(view_23, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
        mul_45 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47 = torch.ops.aten.add.Tensor(view_23, mul_45);  view_23 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48 = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        view_24 = torch.ops.aten.view.default(mul_47, [8192, 3072]);  mul_47 = None
        permute_12 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg55_1, view_24, permute_12);  arg55_1 = view_24 = permute_12 = None
        view_25 = torch.ops.aten.view.default(addmm_12, [16, 512, 768]);  addmm_12 = None
        add_49 = torch.ops.aten.add.Tensor(view_25, add_46);  view_25 = add_46 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg56_1);  mul_48 = arg56_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_49, arg57_1);  mul_49 = arg57_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(add_51, torch.complex64)
        _fft_c2c_6 = torch.ops.aten._fft_c2c.default(convert_element_type_6, [1, 2], 0, True);  convert_element_type_6 = None
        view_as_real_6 = torch.ops.aten.view_as_real.default(_fft_c2c_6);  _fft_c2c_6 = None
        select_6 = torch.ops.aten.select.int(view_as_real_6, 3, 0);  view_as_real_6 = None
        add_52 = torch.ops.aten.add.Tensor(add_51, select_6);  add_51 = select_6 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_52, getitem_27);  add_52 = getitem_27 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg58_1);  mul_50 = arg58_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_51, arg59_1);  mul_51 = arg59_1 = None
        view_26 = torch.ops.aten.view.default(add_54, [8192, 768])
        permute_13 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg61_1, view_26, permute_13);  arg61_1 = view_26 = permute_13 = None
        view_27 = torch.ops.aten.view.default(addmm_13, [16, 512, 3072]);  addmm_13 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_27, 0.5)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
        mul_53 = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55 = torch.ops.aten.add.Tensor(view_27, mul_53);  view_27 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56 = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        view_28 = torch.ops.aten.view.default(mul_55, [8192, 3072]);  mul_55 = None
        permute_14 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg63_1, view_28, permute_14);  arg63_1 = view_28 = permute_14 = None
        view_29 = torch.ops.aten.view.default(addmm_14, [16, 512, 768]);  addmm_14 = None
        add_57 = torch.ops.aten.add.Tensor(view_29, add_54);  view_29 = add_54 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg64_1);  mul_56 = arg64_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_57, arg65_1);  mul_57 = arg65_1 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_59, torch.complex64)
        _fft_c2c_7 = torch.ops.aten._fft_c2c.default(convert_element_type_7, [1, 2], 0, True);  convert_element_type_7 = None
        view_as_real_7 = torch.ops.aten.view_as_real.default(_fft_c2c_7);  _fft_c2c_7 = None
        select_7 = torch.ops.aten.select.int(view_as_real_7, 3, 0);  view_as_real_7 = None
        add_60 = torch.ops.aten.add.Tensor(add_59, select_7);  add_59 = select_7 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_60, getitem_31);  add_60 = getitem_31 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg66_1);  mul_58 = arg66_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_59, arg67_1);  mul_59 = arg67_1 = None
        view_30 = torch.ops.aten.view.default(add_62, [8192, 768])
        permute_15 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg69_1, view_30, permute_15);  arg69_1 = view_30 = permute_15 = None
        view_31 = torch.ops.aten.view.default(addmm_15, [16, 512, 3072]);  addmm_15 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_31, 0.5)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
        mul_61 = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63 = torch.ops.aten.add.Tensor(view_31, mul_61);  view_31 = mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64 = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        view_32 = torch.ops.aten.view.default(mul_63, [8192, 3072]);  mul_63 = None
        permute_16 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg71_1, view_32, permute_16);  arg71_1 = view_32 = permute_16 = None
        view_33 = torch.ops.aten.view.default(addmm_16, [16, 512, 768]);  addmm_16 = None
        add_65 = torch.ops.aten.add.Tensor(view_33, add_62);  view_33 = add_62 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg72_1);  mul_64 = arg72_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_65, arg73_1);  mul_65 = arg73_1 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_67, torch.complex64)
        _fft_c2c_8 = torch.ops.aten._fft_c2c.default(convert_element_type_8, [1, 2], 0, True);  convert_element_type_8 = None
        view_as_real_8 = torch.ops.aten.view_as_real.default(_fft_c2c_8);  _fft_c2c_8 = None
        select_8 = torch.ops.aten.select.int(view_as_real_8, 3, 0);  view_as_real_8 = None
        add_68 = torch.ops.aten.add.Tensor(add_67, select_8);  add_67 = select_8 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_68, getitem_35);  add_68 = getitem_35 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg74_1);  mul_66 = arg74_1 = None
        add_70 = torch.ops.aten.add.Tensor(mul_67, arg75_1);  mul_67 = arg75_1 = None
        view_34 = torch.ops.aten.view.default(add_70, [8192, 768])
        permute_17 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg77_1, view_34, permute_17);  arg77_1 = view_34 = permute_17 = None
        view_35 = torch.ops.aten.view.default(addmm_17, [16, 512, 3072]);  addmm_17 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_35, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
        mul_69 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71 = torch.ops.aten.add.Tensor(view_35, mul_69);  view_35 = mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72 = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        view_36 = torch.ops.aten.view.default(mul_71, [8192, 3072]);  mul_71 = None
        permute_18 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg79_1, view_36, permute_18);  arg79_1 = view_36 = permute_18 = None
        view_37 = torch.ops.aten.view.default(addmm_18, [16, 512, 768]);  addmm_18 = None
        add_73 = torch.ops.aten.add.Tensor(view_37, add_70);  view_37 = add_70 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg80_1);  mul_72 = arg80_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_73, arg81_1);  mul_73 = arg81_1 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_75, torch.complex64)
        _fft_c2c_9 = torch.ops.aten._fft_c2c.default(convert_element_type_9, [1, 2], 0, True);  convert_element_type_9 = None
        view_as_real_9 = torch.ops.aten.view_as_real.default(_fft_c2c_9);  _fft_c2c_9 = None
        select_9 = torch.ops.aten.select.int(view_as_real_9, 3, 0);  view_as_real_9 = None
        add_76 = torch.ops.aten.add.Tensor(add_75, select_9);  add_75 = select_9 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_77 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_76, getitem_39);  add_76 = getitem_39 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg82_1);  mul_74 = arg82_1 = None
        add_78 = torch.ops.aten.add.Tensor(mul_75, arg83_1);  mul_75 = arg83_1 = None
        view_38 = torch.ops.aten.view.default(add_78, [8192, 768])
        permute_19 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg85_1, view_38, permute_19);  arg85_1 = view_38 = permute_19 = None
        view_39 = torch.ops.aten.view.default(addmm_19, [16, 512, 3072]);  addmm_19 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_39, 0.5)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(view_39, 3.0)
        mul_77 = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79 = torch.ops.aten.add.Tensor(view_39, mul_77);  view_39 = mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80 = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        view_40 = torch.ops.aten.view.default(mul_79, [8192, 3072]);  mul_79 = None
        permute_20 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg87_1, view_40, permute_20);  arg87_1 = view_40 = permute_20 = None
        view_41 = torch.ops.aten.view.default(addmm_20, [16, 512, 768]);  addmm_20 = None
        add_81 = torch.ops.aten.add.Tensor(view_41, add_78);  view_41 = add_78 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg88_1);  mul_80 = arg88_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_81, arg89_1);  mul_81 = arg89_1 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_83, torch.complex64)
        _fft_c2c_10 = torch.ops.aten._fft_c2c.default(convert_element_type_10, [1, 2], 0, True);  convert_element_type_10 = None
        view_as_real_10 = torch.ops.aten.view_as_real.default(_fft_c2c_10);  _fft_c2c_10 = None
        select_10 = torch.ops.aten.select.int(view_as_real_10, 3, 0);  view_as_real_10 = None
        add_84 = torch.ops.aten.add.Tensor(add_83, select_10);  add_83 = select_10 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_84, getitem_43);  add_84 = getitem_43 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg90_1);  mul_82 = arg90_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_83, arg91_1);  mul_83 = arg91_1 = None
        view_42 = torch.ops.aten.view.default(add_86, [8192, 768])
        permute_21 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg93_1, view_42, permute_21);  arg93_1 = view_42 = permute_21 = None
        view_43 = torch.ops.aten.view.default(addmm_21, [16, 512, 3072]);  addmm_21 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_43, 0.5)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
        mul_85 = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87 = torch.ops.aten.add.Tensor(view_43, mul_85);  view_43 = mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88 = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        view_44 = torch.ops.aten.view.default(mul_87, [8192, 3072]);  mul_87 = None
        permute_22 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg95_1, view_44, permute_22);  arg95_1 = view_44 = permute_22 = None
        view_45 = torch.ops.aten.view.default(addmm_22, [16, 512, 768]);  addmm_22 = None
        add_89 = torch.ops.aten.add.Tensor(view_45, add_86);  view_45 = add_86 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg96_1);  mul_88 = arg96_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_89, arg97_1);  mul_89 = arg97_1 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_91, torch.complex64)
        _fft_c2c_11 = torch.ops.aten._fft_c2c.default(convert_element_type_11, [1, 2], 0, True);  convert_element_type_11 = None
        view_as_real_11 = torch.ops.aten.view_as_real.default(_fft_c2c_11);  _fft_c2c_11 = None
        select_11 = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None
        add_92 = torch.ops.aten.add.Tensor(add_91, select_11);  add_91 = select_11 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_92, getitem_47);  add_92 = getitem_47 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg98_1);  mul_90 = arg98_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_91, arg99_1);  mul_91 = arg99_1 = None
        view_46 = torch.ops.aten.view.default(add_94, [8192, 768])
        permute_23 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg101_1, view_46, permute_23);  arg101_1 = view_46 = permute_23 = None
        view_47 = torch.ops.aten.view.default(addmm_23, [16, 512, 3072]);  addmm_23 = None
        mul_92 = torch.ops.aten.mul.Tensor(view_47, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_93 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95 = torch.ops.aten.add.Tensor(view_47, mul_93);  view_47 = mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96 = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        view_48 = torch.ops.aten.view.default(mul_95, [8192, 3072]);  mul_95 = None
        permute_24 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg103_1, view_48, permute_24);  arg103_1 = view_48 = permute_24 = None
        view_49 = torch.ops.aten.view.default(addmm_24, [16, 512, 768]);  addmm_24 = None
        add_97 = torch.ops.aten.add.Tensor(view_49, add_94);  view_49 = add_94 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg104_1);  mul_96 = arg104_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_97, arg105_1);  mul_97 = arg105_1 = None
        view_50 = torch.ops.aten.view.default(add_99, [8192, 768]);  add_99 = None
        permute_26 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg109_1, view_50, permute_26);  arg109_1 = view_50 = permute_26 = None
        view_51 = torch.ops.aten.view.default(addmm_26, [16, 512, 768]);  addmm_26 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_51, 0.5)
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(view_51, 3.0)
        mul_99 = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_100 = torch.ops.aten.add.Tensor(view_51, mul_99);  view_51 = mul_99 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_13 = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
        add_101 = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_98, add_101);  mul_98 = add_101 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_25 = torch.ops.aten.sub.Tensor(mul_101, getitem_51);  mul_101 = getitem_51 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg110_1);  mul_102 = arg110_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_103, arg111_1);  mul_103 = arg111_1 = None
        view_52 = torch.ops.aten.view.default(add_103, [8192, 768]);  add_103 = None
        permute_27 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg112_1, view_52, permute_27);  arg112_1 = view_52 = permute_27 = None
        view_53 = torch.ops.aten.view.default(addmm_27, [16, 512, 32000]);  addmm_27 = None
        view_54 = torch.ops.aten.view.default(view_53, [-1, 32000])
        view_55 = torch.ops.aten.view.default(arg113_1, [-1]);  arg113_1 = None
        amax = torch.ops.aten.amax.default(view_54, [1], True)
        sub_26 = torch.ops.aten.sub.Tensor(view_54, amax);  view_54 = amax = None
        exp = torch.ops.aten.exp.default(sub_26)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27 = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne = torch.ops.aten.ne.Scalar(view_55, -100)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, view_55, full_default);  ne = full_default = None
        unsqueeze = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_27, 1, unsqueeze);  sub_27 = unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_55, -100)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_55, -100);  view_55 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type_12);  sum_3 = convert_element_type_12 = None
        return (div, view_53)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (16, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf3, (32000, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf4, (4, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768, 768), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf12, (3072, 768), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf13, (3072,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768, 3072), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf20, (3072, 768), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf21, (3072,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768, 3072), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf28, (3072, 768), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf29, (3072,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768, 3072), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (3072, 768), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf37, (3072,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768, 3072), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf44, (3072, 768), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf45, (3072,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768, 3072), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
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
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf60, (3072, 768), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf61, (3072,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768, 3072), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf68, (3072, 768), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf69, (3072,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768, 3072), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
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
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf84, (3072, 768), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf85, (3072,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 3072), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf92, (3072, 768), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf93, (3072,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768, 3072), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf100, (3072, 768), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf101, (3072,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 3072), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 768), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 768), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 128000, device=device(type='cuda', index=0))
    reader.tensor(buf112, (32000,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf113, (16, 512), dtype=torch.int64, is_leaf=True)  # arg113_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)