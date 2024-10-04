
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1):
        view = torch.ops.aten.view.default(arg1_1, [-1, 128])
        embedding = torch.ops.aten.embedding.default(arg2_1, view, 0);  view = None
        mul = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, iota);  arg3_1 = iota = None
        add = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        view_1 = torch.ops.aten.view.default(add_2, [8192, 512])
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view_1, permute);  arg7_1 = view_1 = permute = None
        view_2 = torch.ops.aten.view.default(addmm, [64, 128, 512]);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(view_2, 0.1767766952966369);  view_2 = None
        view_3 = torch.ops.aten.view.default(add_2, [8192, 512])
        permute_1 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_3, permute_1);  arg9_1 = view_3 = permute_1 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [64, 128, 512]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [64, -1, 16, 32]);  view_4 = None
        permute_2 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_6 = torch.ops.aten.view.default(add_2, [8192, 512])
        permute_3 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_6, permute_3);  arg11_1 = view_6 = permute_3 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [64, 128, 512]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [64, -1, 16, 32]);  view_7 = None
        permute_4 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_9 = torch.ops.aten.view.default(mul_3, [64, 128, 16, 32]);  mul_3 = None
        permute_5 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_10 = torch.ops.aten.view.default(clone_3, [1024, -1, 32]);  clone_3 = None
        view_11 = torch.ops.aten.view.default(clone_1, [1024, -1, 32]);  clone_1 = None
        view_12 = torch.ops.aten.view.default(clone_2, [1024, -1, 32]);  clone_2 = None
        unsqueeze_default_45 = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
        unsqueeze_default_46 = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
        unsqueeze_default_47 = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
        _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_45, unsqueeze_default_46, unsqueeze_default_47, None, False, scale = 1.0);  unsqueeze_default_45 = unsqueeze_default_46 = unsqueeze_default_47 = None
        getitem_99 = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
        squeeze_dim_15 = torch.ops.aten.squeeze.dim(getitem_99, 0);  getitem_99 = None
        view_13 = torch.ops.aten.view.default(squeeze_dim_15, [64, 16, 128, 32]);  squeeze_dim_15 = None
        permute_7 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_14 = torch.ops.aten.view.default(clone_5, [64, 128, 512]);  clone_5 = None
        view_15 = torch.ops.aten.view.default(view_14, [8192, 512]);  view_14 = None
        permute_8 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_15, permute_8);  arg13_1 = view_15 = permute_8 = None
        view_16 = torch.ops.aten.view.default(addmm_3, [64, 128, 512]);  addmm_3 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, view_16);  add_2 = view_16 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_5, arg15_1);  mul_5 = arg15_1 = None
        view_17 = torch.ops.aten.view.default(add_5, [8192, 512])
        permute_9 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_17, permute_9);  arg17_1 = view_17 = permute_9 = None
        view_18 = torch.ops.aten.view.default(addmm_4, [64, 128, 2048]);  addmm_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(view_18, 0.5)
        mul_7 = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
        erf = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
        view_19 = torch.ops.aten.view.default(mul_8, [8192, 2048]);  mul_8 = None
        permute_10 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_19, permute_10);  arg19_1 = view_19 = permute_10 = None
        view_20 = torch.ops.aten.view.default(addmm_5, [64, 128, 512]);  addmm_5 = None
        add_7 = torch.ops.aten.add.Tensor(add_5, view_20);  add_5 = view_20 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        add_9 = torch.ops.aten.add.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        view_21 = torch.ops.aten.view.default(add_9, [8192, 512])
        permute_11 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_21, permute_11);  arg23_1 = view_21 = permute_11 = None
        view_22 = torch.ops.aten.view.default(addmm_6, [64, 128, 512]);  addmm_6 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_22, 0.1767766952966369);  view_22 = None
        view_23 = torch.ops.aten.view.default(add_9, [8192, 512])
        permute_12 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_23, permute_12);  arg25_1 = view_23 = permute_12 = None
        view_24 = torch.ops.aten.view.default(addmm_7, [64, 128, 512]);  addmm_7 = None
        view_25 = torch.ops.aten.view.default(view_24, [64, -1, 16, 32]);  view_24 = None
        permute_13 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_9 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_26 = torch.ops.aten.view.default(add_9, [8192, 512])
        permute_14 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg27_1, view_26, permute_14);  arg27_1 = view_26 = permute_14 = None
        view_27 = torch.ops.aten.view.default(addmm_8, [64, 128, 512]);  addmm_8 = None
        view_28 = torch.ops.aten.view.default(view_27, [64, -1, 16, 32]);  view_27 = None
        permute_15 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_10 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_29 = torch.ops.aten.view.default(mul_11, [64, 128, 16, 32]);  mul_11 = None
        permute_16 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        clone_11 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_30 = torch.ops.aten.view.default(clone_11, [1024, -1, 32]);  clone_11 = None
        view_31 = torch.ops.aten.view.default(clone_9, [1024, -1, 32]);  clone_9 = None
        view_32 = torch.ops.aten.view.default(clone_10, [1024, -1, 32]);  clone_10 = None
        unsqueeze_default_42 = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
        unsqueeze_default_43 = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
        unsqueeze_default_44 = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
        _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_42, unsqueeze_default_43, unsqueeze_default_44, None, False, scale = 1.0);  unsqueeze_default_42 = unsqueeze_default_43 = unsqueeze_default_44 = None
        getitem_98 = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
        squeeze_dim_14 = torch.ops.aten.squeeze.dim(getitem_98, 0);  getitem_98 = None
        view_33 = torch.ops.aten.view.default(squeeze_dim_14, [64, 16, 128, 32]);  squeeze_dim_14 = None
        permute_18 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_13 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_34 = torch.ops.aten.view.default(clone_13, [64, 128, 512]);  clone_13 = None
        view_35 = torch.ops.aten.view.default(view_34, [8192, 512]);  view_34 = None
        permute_19 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg29_1, view_35, permute_19);  arg29_1 = view_35 = permute_19 = None
        view_36 = torch.ops.aten.view.default(addmm_9, [64, 128, 512]);  addmm_9 = None
        add_10 = torch.ops.aten.add.Tensor(add_9, view_36);  add_9 = view_36 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg30_1);  mul_12 = arg30_1 = None
        add_12 = torch.ops.aten.add.Tensor(mul_13, arg31_1);  mul_13 = arg31_1 = None
        view_37 = torch.ops.aten.view.default(add_12, [8192, 512])
        permute_20 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg33_1, view_37, permute_20);  arg33_1 = view_37 = permute_20 = None
        view_38 = torch.ops.aten.view.default(addmm_10, [64, 128, 2048]);  addmm_10 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_38, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
        erf_1 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_13 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_13);  mul_14 = add_13 = None
        view_39 = torch.ops.aten.view.default(mul_16, [8192, 2048]);  mul_16 = None
        permute_21 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg35_1, view_39, permute_21);  arg35_1 = view_39 = permute_21 = None
        view_40 = torch.ops.aten.view.default(addmm_11, [64, 128, 512]);  addmm_11 = None
        add_14 = torch.ops.aten.add.Tensor(add_12, view_40);  add_12 = view_40 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg36_1);  mul_17 = arg36_1 = None
        add_16 = torch.ops.aten.add.Tensor(mul_18, arg37_1);  mul_18 = arg37_1 = None
        view_41 = torch.ops.aten.view.default(add_16, [8192, 512])
        permute_22 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg39_1, view_41, permute_22);  arg39_1 = view_41 = permute_22 = None
        view_42 = torch.ops.aten.view.default(addmm_12, [64, 128, 512]);  addmm_12 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_42, 0.1767766952966369);  view_42 = None
        view_43 = torch.ops.aten.view.default(add_16, [8192, 512])
        permute_23 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg41_1, view_43, permute_23);  arg41_1 = view_43 = permute_23 = None
        view_44 = torch.ops.aten.view.default(addmm_13, [64, 128, 512]);  addmm_13 = None
        view_45 = torch.ops.aten.view.default(view_44, [64, -1, 16, 32]);  view_44 = None
        permute_24 = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_17 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_46 = torch.ops.aten.view.default(add_16, [8192, 512])
        permute_25 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg43_1, view_46, permute_25);  arg43_1 = view_46 = permute_25 = None
        view_47 = torch.ops.aten.view.default(addmm_14, [64, 128, 512]);  addmm_14 = None
        view_48 = torch.ops.aten.view.default(view_47, [64, -1, 16, 32]);  view_47 = None
        permute_26 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_18 = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_49 = torch.ops.aten.view.default(mul_19, [64, 128, 16, 32]);  mul_19 = None
        permute_27 = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        clone_19 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_50 = torch.ops.aten.view.default(clone_19, [1024, -1, 32]);  clone_19 = None
        view_51 = torch.ops.aten.view.default(clone_17, [1024, -1, 32]);  clone_17 = None
        view_52 = torch.ops.aten.view.default(clone_18, [1024, -1, 32]);  clone_18 = None
        unsqueeze_default_39 = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
        unsqueeze_default_40 = torch.ops.aten.unsqueeze.default(view_51, 0);  view_51 = None
        unsqueeze_default_41 = torch.ops.aten.unsqueeze.default(view_52, 0);  view_52 = None
        _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_39, unsqueeze_default_40, unsqueeze_default_41, None, False, scale = 1.0);  unsqueeze_default_39 = unsqueeze_default_40 = unsqueeze_default_41 = None
        getitem_97 = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
        squeeze_dim_13 = torch.ops.aten.squeeze.dim(getitem_97, 0);  getitem_97 = None
        view_53 = torch.ops.aten.view.default(squeeze_dim_13, [64, 16, 128, 32]);  squeeze_dim_13 = None
        permute_29 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_21 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_54 = torch.ops.aten.view.default(clone_21, [64, 128, 512]);  clone_21 = None
        view_55 = torch.ops.aten.view.default(view_54, [8192, 512]);  view_54 = None
        permute_30 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg45_1, view_55, permute_30);  arg45_1 = view_55 = permute_30 = None
        view_56 = torch.ops.aten.view.default(addmm_15, [64, 128, 512]);  addmm_15 = None
        add_17 = torch.ops.aten.add.Tensor(add_16, view_56);  add_16 = view_56 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg46_1);  mul_20 = arg46_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_21, arg47_1);  mul_21 = arg47_1 = None
        view_57 = torch.ops.aten.view.default(add_19, [8192, 512])
        permute_31 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg49_1, view_57, permute_31);  arg49_1 = view_57 = permute_31 = None
        view_58 = torch.ops.aten.view.default(addmm_16, [64, 128, 2048]);  addmm_16 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_20 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_20);  mul_22 = add_20 = None
        view_59 = torch.ops.aten.view.default(mul_24, [8192, 2048]);  mul_24 = None
        permute_32 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg51_1, view_59, permute_32);  arg51_1 = view_59 = permute_32 = None
        view_60 = torch.ops.aten.view.default(addmm_17, [64, 128, 512]);  addmm_17 = None
        add_21 = torch.ops.aten.add.Tensor(add_19, view_60);  add_19 = view_60 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, arg52_1);  mul_25 = arg52_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_26, arg53_1);  mul_26 = arg53_1 = None
        view_61 = torch.ops.aten.view.default(add_23, [8192, 512])
        permute_33 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg55_1, view_61, permute_33);  arg55_1 = view_61 = permute_33 = None
        view_62 = torch.ops.aten.view.default(addmm_18, [64, 128, 512]);  addmm_18 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_62, 0.1767766952966369);  view_62 = None
        view_63 = torch.ops.aten.view.default(add_23, [8192, 512])
        permute_34 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg57_1, view_63, permute_34);  arg57_1 = view_63 = permute_34 = None
        view_64 = torch.ops.aten.view.default(addmm_19, [64, 128, 512]);  addmm_19 = None
        view_65 = torch.ops.aten.view.default(view_64, [64, -1, 16, 32]);  view_64 = None
        permute_35 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_25 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_66 = torch.ops.aten.view.default(add_23, [8192, 512])
        permute_36 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg59_1, view_66, permute_36);  arg59_1 = view_66 = permute_36 = None
        view_67 = torch.ops.aten.view.default(addmm_20, [64, 128, 512]);  addmm_20 = None
        view_68 = torch.ops.aten.view.default(view_67, [64, -1, 16, 32]);  view_67 = None
        permute_37 = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_26 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_69 = torch.ops.aten.view.default(mul_27, [64, 128, 16, 32]);  mul_27 = None
        permute_38 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        clone_27 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_70 = torch.ops.aten.view.default(clone_27, [1024, -1, 32]);  clone_27 = None
        view_71 = torch.ops.aten.view.default(clone_25, [1024, -1, 32]);  clone_25 = None
        view_72 = torch.ops.aten.view.default(clone_26, [1024, -1, 32]);  clone_26 = None
        unsqueeze_default_36 = torch.ops.aten.unsqueeze.default(view_70, 0);  view_70 = None
        unsqueeze_default_37 = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
        unsqueeze_default_38 = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
        _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_36, unsqueeze_default_37, unsqueeze_default_38, None, False, scale = 1.0);  unsqueeze_default_36 = unsqueeze_default_37 = unsqueeze_default_38 = None
        getitem_96 = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
        squeeze_dim_12 = torch.ops.aten.squeeze.dim(getitem_96, 0);  getitem_96 = None
        view_73 = torch.ops.aten.view.default(squeeze_dim_12, [64, 16, 128, 32]);  squeeze_dim_12 = None
        permute_40 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        clone_29 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_74 = torch.ops.aten.view.default(clone_29, [64, 128, 512]);  clone_29 = None
        view_75 = torch.ops.aten.view.default(view_74, [8192, 512]);  view_74 = None
        permute_41 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg61_1, view_75, permute_41);  arg61_1 = view_75 = permute_41 = None
        view_76 = torch.ops.aten.view.default(addmm_21, [64, 128, 512]);  addmm_21 = None
        add_24 = torch.ops.aten.add.Tensor(add_23, view_76);  add_23 = view_76 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg62_1);  mul_28 = arg62_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_29, arg63_1);  mul_29 = arg63_1 = None
        view_77 = torch.ops.aten.view.default(add_26, [8192, 512])
        permute_42 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg65_1, view_77, permute_42);  arg65_1 = view_77 = permute_42 = None
        view_78 = torch.ops.aten.view.default(addmm_22, [64, 128, 2048]);  addmm_22 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_78, 0.5)
        mul_31 = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
        erf_3 = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_27 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_30, add_27);  mul_30 = add_27 = None
        view_79 = torch.ops.aten.view.default(mul_32, [8192, 2048]);  mul_32 = None
        permute_43 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg67_1, view_79, permute_43);  arg67_1 = view_79 = permute_43 = None
        view_80 = torch.ops.aten.view.default(addmm_23, [64, 128, 512]);  addmm_23 = None
        add_28 = torch.ops.aten.add.Tensor(add_26, view_80);  add_26 = view_80 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg68_1);  mul_33 = arg68_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_34, arg69_1);  mul_34 = arg69_1 = None
        view_81 = torch.ops.aten.view.default(add_30, [8192, 512])
        permute_44 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg71_1, view_81, permute_44);  arg71_1 = view_81 = permute_44 = None
        view_82 = torch.ops.aten.view.default(addmm_24, [64, 128, 512]);  addmm_24 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_82, 0.1767766952966369);  view_82 = None
        view_83 = torch.ops.aten.view.default(add_30, [8192, 512])
        permute_45 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg73_1, view_83, permute_45);  arg73_1 = view_83 = permute_45 = None
        view_84 = torch.ops.aten.view.default(addmm_25, [64, 128, 512]);  addmm_25 = None
        view_85 = torch.ops.aten.view.default(view_84, [64, -1, 16, 32]);  view_84 = None
        permute_46 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        clone_33 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_86 = torch.ops.aten.view.default(add_30, [8192, 512])
        permute_47 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg75_1, view_86, permute_47);  arg75_1 = view_86 = permute_47 = None
        view_87 = torch.ops.aten.view.default(addmm_26, [64, 128, 512]);  addmm_26 = None
        view_88 = torch.ops.aten.view.default(view_87, [64, -1, 16, 32]);  view_87 = None
        permute_48 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_34 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_89 = torch.ops.aten.view.default(mul_35, [64, 128, 16, 32]);  mul_35 = None
        permute_49 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_35 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_90 = torch.ops.aten.view.default(clone_35, [1024, -1, 32]);  clone_35 = None
        view_91 = torch.ops.aten.view.default(clone_33, [1024, -1, 32]);  clone_33 = None
        view_92 = torch.ops.aten.view.default(clone_34, [1024, -1, 32]);  clone_34 = None
        unsqueeze_default_33 = torch.ops.aten.unsqueeze.default(view_90, 0);  view_90 = None
        unsqueeze_default_34 = torch.ops.aten.unsqueeze.default(view_91, 0);  view_91 = None
        unsqueeze_default_35 = torch.ops.aten.unsqueeze.default(view_92, 0);  view_92 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, None, False, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
        getitem_95 = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        squeeze_dim_11 = torch.ops.aten.squeeze.dim(getitem_95, 0);  getitem_95 = None
        view_93 = torch.ops.aten.view.default(squeeze_dim_11, [64, 16, 128, 32]);  squeeze_dim_11 = None
        permute_51 = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        clone_37 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_94 = torch.ops.aten.view.default(clone_37, [64, 128, 512]);  clone_37 = None
        view_95 = torch.ops.aten.view.default(view_94, [8192, 512]);  view_94 = None
        permute_52 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg77_1, view_95, permute_52);  arg77_1 = view_95 = permute_52 = None
        view_96 = torch.ops.aten.view.default(addmm_27, [64, 128, 512]);  addmm_27 = None
        add_31 = torch.ops.aten.add.Tensor(add_30, view_96);  add_30 = view_96 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg78_1);  mul_36 = arg78_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_37, arg79_1);  mul_37 = arg79_1 = None
        view_97 = torch.ops.aten.view.default(add_33, [8192, 512])
        permute_53 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg81_1, view_97, permute_53);  arg81_1 = view_97 = permute_53 = None
        view_98 = torch.ops.aten.view.default(addmm_28, [64, 128, 2048]);  addmm_28 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_98, 0.5)
        mul_39 = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
        erf_4 = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_34 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, add_34);  mul_38 = add_34 = None
        view_99 = torch.ops.aten.view.default(mul_40, [8192, 2048]);  mul_40 = None
        permute_54 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg83_1, view_99, permute_54);  arg83_1 = view_99 = permute_54 = None
        view_100 = torch.ops.aten.view.default(addmm_29, [64, 128, 512]);  addmm_29 = None
        add_35 = torch.ops.aten.add.Tensor(add_33, view_100);  add_33 = view_100 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, arg84_1);  mul_41 = arg84_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_42, arg85_1);  mul_42 = arg85_1 = None
        view_101 = torch.ops.aten.view.default(add_37, [8192, 512])
        permute_55 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg87_1, view_101, permute_55);  arg87_1 = view_101 = permute_55 = None
        view_102 = torch.ops.aten.view.default(addmm_30, [64, 128, 512]);  addmm_30 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_102, 0.1767766952966369);  view_102 = None
        view_103 = torch.ops.aten.view.default(add_37, [8192, 512])
        permute_56 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg89_1, view_103, permute_56);  arg89_1 = view_103 = permute_56 = None
        view_104 = torch.ops.aten.view.default(addmm_31, [64, 128, 512]);  addmm_31 = None
        view_105 = torch.ops.aten.view.default(view_104, [64, -1, 16, 32]);  view_104 = None
        permute_57 = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        clone_41 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_106 = torch.ops.aten.view.default(add_37, [8192, 512])
        permute_58 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg91_1, view_106, permute_58);  arg91_1 = view_106 = permute_58 = None
        view_107 = torch.ops.aten.view.default(addmm_32, [64, 128, 512]);  addmm_32 = None
        view_108 = torch.ops.aten.view.default(view_107, [64, -1, 16, 32]);  view_107 = None
        permute_59 = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_42 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_109 = torch.ops.aten.view.default(mul_43, [64, 128, 16, 32]);  mul_43 = None
        permute_60 = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        clone_43 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_110 = torch.ops.aten.view.default(clone_43, [1024, -1, 32]);  clone_43 = None
        view_111 = torch.ops.aten.view.default(clone_41, [1024, -1, 32]);  clone_41 = None
        view_112 = torch.ops.aten.view.default(clone_42, [1024, -1, 32]);  clone_42 = None
        unsqueeze_default_30 = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
        unsqueeze_default_31 = torch.ops.aten.unsqueeze.default(view_111, 0);  view_111 = None
        unsqueeze_default_32 = torch.ops.aten.unsqueeze.default(view_112, 0);  view_112 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, None, False, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
        getitem_94 = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        squeeze_dim_10 = torch.ops.aten.squeeze.dim(getitem_94, 0);  getitem_94 = None
        view_113 = torch.ops.aten.view.default(squeeze_dim_10, [64, 16, 128, 32]);  squeeze_dim_10 = None
        permute_62 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_45 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_114 = torch.ops.aten.view.default(clone_45, [64, 128, 512]);  clone_45 = None
        view_115 = torch.ops.aten.view.default(view_114, [8192, 512]);  view_114 = None
        permute_63 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg93_1, view_115, permute_63);  arg93_1 = view_115 = permute_63 = None
        view_116 = torch.ops.aten.view.default(addmm_33, [64, 128, 512]);  addmm_33 = None
        add_38 = torch.ops.aten.add.Tensor(add_37, view_116);  add_37 = view_116 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg94_1);  mul_44 = arg94_1 = None
        add_40 = torch.ops.aten.add.Tensor(mul_45, arg95_1);  mul_45 = arg95_1 = None
        view_117 = torch.ops.aten.view.default(add_40, [8192, 512])
        permute_64 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg97_1, view_117, permute_64);  arg97_1 = view_117 = permute_64 = None
        view_118 = torch.ops.aten.view.default(addmm_34, [64, 128, 2048]);  addmm_34 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_118, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
        erf_5 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_41 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_41);  mul_46 = add_41 = None
        view_119 = torch.ops.aten.view.default(mul_48, [8192, 2048]);  mul_48 = None
        permute_65 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg99_1, view_119, permute_65);  arg99_1 = view_119 = permute_65 = None
        view_120 = torch.ops.aten.view.default(addmm_35, [64, 128, 512]);  addmm_35 = None
        add_42 = torch.ops.aten.add.Tensor(add_40, view_120);  add_40 = view_120 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg100_1);  mul_49 = arg100_1 = None
        add_44 = torch.ops.aten.add.Tensor(mul_50, arg101_1);  mul_50 = arg101_1 = None
        view_121 = torch.ops.aten.view.default(add_44, [8192, 512])
        permute_66 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg103_1, view_121, permute_66);  arg103_1 = view_121 = permute_66 = None
        view_122 = torch.ops.aten.view.default(addmm_36, [64, 128, 512]);  addmm_36 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_122, 0.1767766952966369);  view_122 = None
        view_123 = torch.ops.aten.view.default(add_44, [8192, 512])
        permute_67 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg105_1, view_123, permute_67);  arg105_1 = view_123 = permute_67 = None
        view_124 = torch.ops.aten.view.default(addmm_37, [64, 128, 512]);  addmm_37 = None
        view_125 = torch.ops.aten.view.default(view_124, [64, -1, 16, 32]);  view_124 = None
        permute_68 = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        clone_49 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_126 = torch.ops.aten.view.default(add_44, [8192, 512])
        permute_69 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg107_1, view_126, permute_69);  arg107_1 = view_126 = permute_69 = None
        view_127 = torch.ops.aten.view.default(addmm_38, [64, 128, 512]);  addmm_38 = None
        view_128 = torch.ops.aten.view.default(view_127, [64, -1, 16, 32]);  view_127 = None
        permute_70 = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        clone_50 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_129 = torch.ops.aten.view.default(mul_51, [64, 128, 16, 32]);  mul_51 = None
        permute_71 = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_51 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_130 = torch.ops.aten.view.default(clone_51, [1024, -1, 32]);  clone_51 = None
        view_131 = torch.ops.aten.view.default(clone_49, [1024, -1, 32]);  clone_49 = None
        view_132 = torch.ops.aten.view.default(clone_50, [1024, -1, 32]);  clone_50 = None
        unsqueeze_default_27 = torch.ops.aten.unsqueeze.default(view_130, 0);  view_130 = None
        unsqueeze_default_28 = torch.ops.aten.unsqueeze.default(view_131, 0);  view_131 = None
        unsqueeze_default_29 = torch.ops.aten.unsqueeze.default(view_132, 0);  view_132 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, None, False, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
        getitem_93 = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        squeeze_dim_9 = torch.ops.aten.squeeze.dim(getitem_93, 0);  getitem_93 = None
        view_133 = torch.ops.aten.view.default(squeeze_dim_9, [64, 16, 128, 32]);  squeeze_dim_9 = None
        permute_73 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        clone_53 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_134 = torch.ops.aten.view.default(clone_53, [64, 128, 512]);  clone_53 = None
        view_135 = torch.ops.aten.view.default(view_134, [8192, 512]);  view_134 = None
        permute_74 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg109_1, view_135, permute_74);  arg109_1 = view_135 = permute_74 = None
        view_136 = torch.ops.aten.view.default(addmm_39, [64, 128, 512]);  addmm_39 = None
        add_45 = torch.ops.aten.add.Tensor(add_44, view_136);  add_44 = view_136 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_45, getitem_27);  add_45 = getitem_27 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg110_1);  mul_52 = arg110_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_53, arg111_1);  mul_53 = arg111_1 = None
        view_137 = torch.ops.aten.view.default(add_47, [8192, 512])
        permute_75 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg113_1, view_137, permute_75);  arg113_1 = view_137 = permute_75 = None
        view_138 = torch.ops.aten.view.default(addmm_40, [64, 128, 2048]);  addmm_40 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_138, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
        erf_6 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_48 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_48);  mul_54 = add_48 = None
        view_139 = torch.ops.aten.view.default(mul_56, [8192, 2048]);  mul_56 = None
        permute_76 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg115_1, view_139, permute_76);  arg115_1 = view_139 = permute_76 = None
        view_140 = torch.ops.aten.view.default(addmm_41, [64, 128, 512]);  addmm_41 = None
        add_49 = torch.ops.aten.add.Tensor(add_47, view_140);  add_47 = view_140 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_49, getitem_29);  add_49 = getitem_29 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg116_1);  mul_57 = arg116_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_58, arg117_1);  mul_58 = arg117_1 = None
        view_141 = torch.ops.aten.view.default(add_51, [8192, 512])
        permute_77 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg119_1, view_141, permute_77);  arg119_1 = view_141 = permute_77 = None
        view_142 = torch.ops.aten.view.default(addmm_42, [64, 128, 512]);  addmm_42 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_142, 0.1767766952966369);  view_142 = None
        view_143 = torch.ops.aten.view.default(add_51, [8192, 512])
        permute_78 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg121_1, view_143, permute_78);  arg121_1 = view_143 = permute_78 = None
        view_144 = torch.ops.aten.view.default(addmm_43, [64, 128, 512]);  addmm_43 = None
        view_145 = torch.ops.aten.view.default(view_144, [64, -1, 16, 32]);  view_144 = None
        permute_79 = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        clone_57 = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_146 = torch.ops.aten.view.default(add_51, [8192, 512])
        permute_80 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg123_1, view_146, permute_80);  arg123_1 = view_146 = permute_80 = None
        view_147 = torch.ops.aten.view.default(addmm_44, [64, 128, 512]);  addmm_44 = None
        view_148 = torch.ops.aten.view.default(view_147, [64, -1, 16, 32]);  view_147 = None
        permute_81 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_58 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_149 = torch.ops.aten.view.default(mul_59, [64, 128, 16, 32]);  mul_59 = None
        permute_82 = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        clone_59 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_150 = torch.ops.aten.view.default(clone_59, [1024, -1, 32]);  clone_59 = None
        view_151 = torch.ops.aten.view.default(clone_57, [1024, -1, 32]);  clone_57 = None
        view_152 = torch.ops.aten.view.default(clone_58, [1024, -1, 32]);  clone_58 = None
        unsqueeze_default_24 = torch.ops.aten.unsqueeze.default(view_150, 0);  view_150 = None
        unsqueeze_default_25 = torch.ops.aten.unsqueeze.default(view_151, 0);  view_151 = None
        unsqueeze_default_26 = torch.ops.aten.unsqueeze.default(view_152, 0);  view_152 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, None, False, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
        getitem_92 = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        squeeze_dim_8 = torch.ops.aten.squeeze.dim(getitem_92, 0);  getitem_92 = None
        view_153 = torch.ops.aten.view.default(squeeze_dim_8, [64, 16, 128, 32]);  squeeze_dim_8 = None
        permute_84 = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        clone_61 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_154 = torch.ops.aten.view.default(clone_61, [64, 128, 512]);  clone_61 = None
        view_155 = torch.ops.aten.view.default(view_154, [8192, 512]);  view_154 = None
        permute_85 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg125_1, view_155, permute_85);  arg125_1 = view_155 = permute_85 = None
        view_156 = torch.ops.aten.view.default(addmm_45, [64, 128, 512]);  addmm_45 = None
        add_52 = torch.ops.aten.add.Tensor(add_51, view_156);  add_51 = view_156 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_52, getitem_31);  add_52 = getitem_31 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg126_1);  mul_60 = arg126_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_61, arg127_1);  mul_61 = arg127_1 = None
        view_157 = torch.ops.aten.view.default(add_54, [8192, 512])
        permute_86 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg129_1, view_157, permute_86);  arg129_1 = view_157 = permute_86 = None
        view_158 = torch.ops.aten.view.default(addmm_46, [64, 128, 2048]);  addmm_46 = None
        mul_62 = torch.ops.aten.mul.Tensor(view_158, 0.5)
        mul_63 = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
        erf_7 = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_55 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_62, add_55);  mul_62 = add_55 = None
        view_159 = torch.ops.aten.view.default(mul_64, [8192, 2048]);  mul_64 = None
        permute_87 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg131_1, view_159, permute_87);  arg131_1 = view_159 = permute_87 = None
        view_160 = torch.ops.aten.view.default(addmm_47, [64, 128, 512]);  addmm_47 = None
        add_56 = torch.ops.aten.add.Tensor(add_54, view_160);  add_54 = view_160 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_56, getitem_33);  add_56 = getitem_33 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg132_1);  mul_65 = arg132_1 = None
        add_58 = torch.ops.aten.add.Tensor(mul_66, arg133_1);  mul_66 = arg133_1 = None
        view_161 = torch.ops.aten.view.default(arg1_1, [-1, 128]);  arg1_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg2_1, view_161, 0);  view_161 = None
        mul_67 = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
        full_default = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_1 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_59 = torch.ops.aten.add.Tensor(iota_1, 1)
        view_162 = torch.ops.aten.view.default(add_59, [128, 1]);  add_59 = None
        lt = torch.ops.aten.lt.Tensor(iota_1, view_162);  iota_1 = view_162 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        iota_2 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding_3 = torch.ops.aten.embedding.default(arg134_1, iota_2);  arg134_1 = iota_2 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(mul_67, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_25 = torch.ops.aten.sub.Tensor(mul_67, getitem_35);  mul_67 = getitem_35 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_17);  sub_25 = rsqrt_17 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, arg135_1);  mul_68 = arg135_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_69, arg136_1);  mul_69 = arg136_1 = None
        add_62 = torch.ops.aten.add.Tensor(add_61, embedding_3);  add_61 = embedding_3 = None
        view_163 = torch.ops.aten.view.default(add_62, [8192, 512])
        permute_88 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg138_1, view_163, permute_88);  arg138_1 = view_163 = permute_88 = None
        view_164 = torch.ops.aten.view.default(addmm_48, [64, 128, 512]);  addmm_48 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_164, 0.1767766952966369);  view_164 = None
        view_165 = torch.ops.aten.view.default(add_62, [8192, 512])
        permute_89 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg140_1, view_165, permute_89);  arg140_1 = view_165 = permute_89 = None
        view_166 = torch.ops.aten.view.default(addmm_49, [64, 128, 512]);  addmm_49 = None
        view_167 = torch.ops.aten.view.default(view_166, [64, -1, 16, 32]);  view_166 = None
        permute_90 = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        clone_66 = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_168 = torch.ops.aten.view.default(add_62, [8192, 512])
        permute_91 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg142_1, view_168, permute_91);  arg142_1 = view_168 = permute_91 = None
        view_169 = torch.ops.aten.view.default(addmm_50, [64, 128, 512]);  addmm_50 = None
        view_170 = torch.ops.aten.view.default(view_169, [64, -1, 16, 32]);  view_169 = None
        permute_92 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_67 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_171 = torch.ops.aten.view.default(mul_70, [64, 128, 16, 32]);  mul_70 = None
        permute_93 = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        clone_68 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_172 = torch.ops.aten.view.default(clone_68, [1024, -1, 32]);  clone_68 = None
        view_173 = torch.ops.aten.view.default(clone_66, [1024, -1, 32]);  clone_66 = None
        view_174 = torch.ops.aten.view.default(clone_67, [1024, -1, 32]);  clone_67 = None
        permute_94 = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
        bmm_16 = torch.ops.aten.bmm.default(view_172, permute_94);  view_172 = permute_94 = None
        view_175 = torch.ops.aten.view.default(bmm_16, [64, 16, 128, 128]);  bmm_16 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [64, 1, 128, 128]);  unsqueeze_3 = None
        add_63 = torch.ops.aten.add.Tensor(view_175, expand_1);  view_175 = None
        view_176 = torch.ops.aten.view.default(add_63, [1024, 128, 128]);  add_63 = None
        amax_8 = torch.ops.aten.amax.default(view_176, [-1], True)
        sub_26 = torch.ops.aten.sub.Tensor(view_176, amax_8);  view_176 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        bmm_17 = torch.ops.aten.bmm.default(div_8, view_174);  div_8 = view_174 = None
        view_177 = torch.ops.aten.view.default(bmm_17, [64, 16, 128, 32]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        clone_70 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_178 = torch.ops.aten.view.default(clone_70, [64, 128, 512]);  clone_70 = None
        view_179 = torch.ops.aten.view.default(view_178, [8192, 512]);  view_178 = None
        permute_96 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg144_1, view_179, permute_96);  arg144_1 = view_179 = permute_96 = None
        view_180 = torch.ops.aten.view.default(addmm_51, [64, 128, 512]);  addmm_51 = None
        add_64 = torch.ops.aten.add.Tensor(add_62, view_180);  add_62 = view_180 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_64, getitem_37);  add_64 = getitem_37 = None
        mul_71 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_71, arg145_1);  mul_71 = arg145_1 = None
        add_66 = torch.ops.aten.add.Tensor(mul_72, arg146_1);  mul_72 = arg146_1 = None
        view_181 = torch.ops.aten.view.default(add_66, [8192, 512])
        permute_97 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg148_1, view_181, permute_97);  arg148_1 = view_181 = permute_97 = None
        view_182 = torch.ops.aten.view.default(addmm_52, [64, 128, 512]);  addmm_52 = None
        mul_73 = torch.ops.aten.mul.Tensor(view_182, 0.1767766952966369);  view_182 = None
        view_183 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_98 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg150_1, view_183, permute_98);  arg150_1 = view_183 = permute_98 = None
        view_184 = torch.ops.aten.view.default(addmm_53, [64, 128, 512]);  addmm_53 = None
        view_185 = torch.ops.aten.view.default(view_184, [64, -1, 16, 32]);  view_184 = None
        permute_99 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_72 = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_186 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_100 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg152_1, view_186, permute_100);  arg152_1 = view_186 = permute_100 = None
        view_187 = torch.ops.aten.view.default(addmm_54, [64, 128, 512]);  addmm_54 = None
        view_188 = torch.ops.aten.view.default(view_187, [64, -1, 16, 32]);  view_187 = None
        permute_101 = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        clone_73 = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_189 = torch.ops.aten.view.default(mul_73, [64, 128, 16, 32]);  mul_73 = None
        permute_102 = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        clone_74 = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_190 = torch.ops.aten.view.default(clone_74, [1024, -1, 32]);  clone_74 = None
        view_191 = torch.ops.aten.view.default(clone_72, [1024, -1, 32]);  clone_72 = None
        view_192 = torch.ops.aten.view.default(clone_73, [1024, -1, 32]);  clone_73 = None
        unsqueeze_default_21 = torch.ops.aten.unsqueeze.default(view_190, 0);  view_190 = None
        unsqueeze_default_22 = torch.ops.aten.unsqueeze.default(view_191, 0);  view_191 = None
        unsqueeze_default_23 = torch.ops.aten.unsqueeze.default(view_192, 0);  view_192 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, None, False, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
        getitem_91 = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        squeeze_dim_7 = torch.ops.aten.squeeze.dim(getitem_91, 0);  getitem_91 = None
        view_193 = torch.ops.aten.view.default(squeeze_dim_7, [64, 16, 128, 32]);  squeeze_dim_7 = None
        permute_104 = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        clone_76 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_194 = torch.ops.aten.view.default(clone_76, [64, 128, 512]);  clone_76 = None
        view_195 = torch.ops.aten.view.default(view_194, [8192, 512]);  view_194 = None
        permute_105 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg154_1, view_195, permute_105);  arg154_1 = view_195 = permute_105 = None
        view_196 = torch.ops.aten.view.default(addmm_55, [64, 128, 512]);  addmm_55 = None
        add_67 = torch.ops.aten.add.Tensor(add_66, view_196);  add_66 = view_196 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_67, getitem_39);  add_67 = getitem_39 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg155_1);  mul_74 = arg155_1 = None
        add_69 = torch.ops.aten.add.Tensor(mul_75, arg156_1);  mul_75 = arg156_1 = None
        view_197 = torch.ops.aten.view.default(add_69, [8192, 512])
        permute_106 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg158_1, view_197, permute_106);  arg158_1 = view_197 = permute_106 = None
        view_198 = torch.ops.aten.view.default(addmm_56, [64, 128, 2048]);  addmm_56 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_198, 0.5)
        mul_77 = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476);  view_198 = None
        erf_8 = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_70 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_76, add_70);  mul_76 = add_70 = None
        view_199 = torch.ops.aten.view.default(mul_78, [8192, 2048]);  mul_78 = None
        permute_107 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg160_1, view_199, permute_107);  arg160_1 = view_199 = permute_107 = None
        view_200 = torch.ops.aten.view.default(addmm_57, [64, 128, 512]);  addmm_57 = None
        add_71 = torch.ops.aten.add.Tensor(add_69, view_200);  add_69 = view_200 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_71, getitem_41);  add_71 = getitem_41 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, arg161_1);  mul_79 = arg161_1 = None
        add_73 = torch.ops.aten.add.Tensor(mul_80, arg162_1);  mul_80 = arg162_1 = None
        view_201 = torch.ops.aten.view.default(add_73, [8192, 512])
        permute_108 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg164_1, view_201, permute_108);  arg164_1 = view_201 = permute_108 = None
        view_202 = torch.ops.aten.view.default(addmm_58, [64, 128, 512]);  addmm_58 = None
        mul_81 = torch.ops.aten.mul.Tensor(view_202, 0.1767766952966369);  view_202 = None
        view_203 = torch.ops.aten.view.default(add_73, [8192, 512])
        permute_109 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg166_1, view_203, permute_109);  arg166_1 = view_203 = permute_109 = None
        view_204 = torch.ops.aten.view.default(addmm_59, [64, 128, 512]);  addmm_59 = None
        view_205 = torch.ops.aten.view.default(view_204, [64, -1, 16, 32]);  view_204 = None
        permute_110 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_80 = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_206 = torch.ops.aten.view.default(add_73, [8192, 512])
        permute_111 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg168_1, view_206, permute_111);  arg168_1 = view_206 = permute_111 = None
        view_207 = torch.ops.aten.view.default(addmm_60, [64, 128, 512]);  addmm_60 = None
        view_208 = torch.ops.aten.view.default(view_207, [64, -1, 16, 32]);  view_207 = None
        permute_112 = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_81 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_209 = torch.ops.aten.view.default(mul_81, [64, 128, 16, 32]);  mul_81 = None
        permute_113 = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_82 = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_210 = torch.ops.aten.view.default(clone_82, [1024, -1, 32]);  clone_82 = None
        view_211 = torch.ops.aten.view.default(clone_80, [1024, -1, 32]);  clone_80 = None
        view_212 = torch.ops.aten.view.default(clone_81, [1024, -1, 32]);  clone_81 = None
        permute_114 = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        bmm_20 = torch.ops.aten.bmm.default(view_210, permute_114);  view_210 = permute_114 = None
        view_213 = torch.ops.aten.view.default(bmm_20, [64, 16, 128, 128]);  bmm_20 = None
        add_74 = torch.ops.aten.add.Tensor(view_213, expand_1);  view_213 = None
        view_214 = torch.ops.aten.view.default(add_74, [1024, 128, 128]);  add_74 = None
        amax_10 = torch.ops.aten.amax.default(view_214, [-1], True)
        sub_31 = torch.ops.aten.sub.Tensor(view_214, amax_10);  view_214 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        bmm_21 = torch.ops.aten.bmm.default(div_10, view_212);  div_10 = view_212 = None
        view_215 = torch.ops.aten.view.default(bmm_21, [64, 16, 128, 32]);  bmm_21 = None
        permute_115 = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        clone_84 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_216 = torch.ops.aten.view.default(clone_84, [64, 128, 512]);  clone_84 = None
        view_217 = torch.ops.aten.view.default(view_216, [8192, 512]);  view_216 = None
        permute_116 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg170_1, view_217, permute_116);  arg170_1 = view_217 = permute_116 = None
        view_218 = torch.ops.aten.view.default(addmm_61, [64, 128, 512]);  addmm_61 = None
        add_75 = torch.ops.aten.add.Tensor(add_73, view_218);  add_73 = view_218 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_75, getitem_43);  add_75 = getitem_43 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg171_1);  mul_82 = arg171_1 = None
        add_77 = torch.ops.aten.add.Tensor(mul_83, arg172_1);  mul_83 = arg172_1 = None
        view_219 = torch.ops.aten.view.default(add_77, [8192, 512])
        permute_117 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg174_1, view_219, permute_117);  arg174_1 = view_219 = permute_117 = None
        view_220 = torch.ops.aten.view.default(addmm_62, [64, 128, 512]);  addmm_62 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_220, 0.1767766952966369);  view_220 = None
        view_221 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_118 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg176_1, view_221, permute_118);  arg176_1 = view_221 = permute_118 = None
        view_222 = torch.ops.aten.view.default(addmm_63, [64, 128, 512]);  addmm_63 = None
        view_223 = torch.ops.aten.view.default(view_222, [64, -1, 16, 32]);  view_222 = None
        permute_119 = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
        clone_86 = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
        view_224 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_120 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg178_1, view_224, permute_120);  arg178_1 = view_224 = permute_120 = None
        view_225 = torch.ops.aten.view.default(addmm_64, [64, 128, 512]);  addmm_64 = None
        view_226 = torch.ops.aten.view.default(view_225, [64, -1, 16, 32]);  view_225 = None
        permute_121 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_87 = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_227 = torch.ops.aten.view.default(mul_84, [64, 128, 16, 32]);  mul_84 = None
        permute_122 = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        clone_88 = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_228 = torch.ops.aten.view.default(clone_88, [1024, -1, 32]);  clone_88 = None
        view_229 = torch.ops.aten.view.default(clone_86, [1024, -1, 32]);  clone_86 = None
        view_230 = torch.ops.aten.view.default(clone_87, [1024, -1, 32]);  clone_87 = None
        unsqueeze_default_18 = torch.ops.aten.unsqueeze.default(view_228, 0);  view_228 = None
        unsqueeze_default_19 = torch.ops.aten.unsqueeze.default(view_229, 0);  view_229 = None
        unsqueeze_default_20 = torch.ops.aten.unsqueeze.default(view_230, 0);  view_230 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, None, False, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
        getitem_90 = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        squeeze_dim_6 = torch.ops.aten.squeeze.dim(getitem_90, 0);  getitem_90 = None
        view_231 = torch.ops.aten.view.default(squeeze_dim_6, [64, 16, 128, 32]);  squeeze_dim_6 = None
        permute_124 = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        clone_90 = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_232 = torch.ops.aten.view.default(clone_90, [64, 128, 512]);  clone_90 = None
        view_233 = torch.ops.aten.view.default(view_232, [8192, 512]);  view_232 = None
        permute_125 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg180_1, view_233, permute_125);  arg180_1 = view_233 = permute_125 = None
        view_234 = torch.ops.aten.view.default(addmm_65, [64, 128, 512]);  addmm_65 = None
        add_78 = torch.ops.aten.add.Tensor(add_77, view_234);  add_77 = view_234 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_79 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_78, getitem_45);  add_78 = getitem_45 = None
        mul_85 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, arg181_1);  mul_85 = arg181_1 = None
        add_80 = torch.ops.aten.add.Tensor(mul_86, arg182_1);  mul_86 = arg182_1 = None
        view_235 = torch.ops.aten.view.default(add_80, [8192, 512])
        permute_126 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg184_1, view_235, permute_126);  arg184_1 = view_235 = permute_126 = None
        view_236 = torch.ops.aten.view.default(addmm_66, [64, 128, 2048]);  addmm_66 = None
        mul_87 = torch.ops.aten.mul.Tensor(view_236, 0.5)
        mul_88 = torch.ops.aten.mul.Tensor(view_236, 0.7071067811865476);  view_236 = None
        erf_9 = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_81 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_87, add_81);  mul_87 = add_81 = None
        view_237 = torch.ops.aten.view.default(mul_89, [8192, 2048]);  mul_89 = None
        permute_127 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg186_1, view_237, permute_127);  arg186_1 = view_237 = permute_127 = None
        view_238 = torch.ops.aten.view.default(addmm_67, [64, 128, 512]);  addmm_67 = None
        add_82 = torch.ops.aten.add.Tensor(add_80, view_238);  add_80 = view_238 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_83 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_82, getitem_47);  add_82 = getitem_47 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg187_1);  mul_90 = arg187_1 = None
        add_84 = torch.ops.aten.add.Tensor(mul_91, arg188_1);  mul_91 = arg188_1 = None
        view_239 = torch.ops.aten.view.default(add_84, [8192, 512])
        permute_128 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg190_1, view_239, permute_128);  arg190_1 = view_239 = permute_128 = None
        view_240 = torch.ops.aten.view.default(addmm_68, [64, 128, 512]);  addmm_68 = None
        mul_92 = torch.ops.aten.mul.Tensor(view_240, 0.1767766952966369);  view_240 = None
        view_241 = torch.ops.aten.view.default(add_84, [8192, 512])
        permute_129 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg192_1, view_241, permute_129);  arg192_1 = view_241 = permute_129 = None
        view_242 = torch.ops.aten.view.default(addmm_69, [64, 128, 512]);  addmm_69 = None
        view_243 = torch.ops.aten.view.default(view_242, [64, -1, 16, 32]);  view_242 = None
        permute_130 = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        clone_94 = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_244 = torch.ops.aten.view.default(add_84, [8192, 512])
        permute_131 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg194_1, view_244, permute_131);  arg194_1 = view_244 = permute_131 = None
        view_245 = torch.ops.aten.view.default(addmm_70, [64, 128, 512]);  addmm_70 = None
        view_246 = torch.ops.aten.view.default(view_245, [64, -1, 16, 32]);  view_245 = None
        permute_132 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_95 = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_247 = torch.ops.aten.view.default(mul_92, [64, 128, 16, 32]);  mul_92 = None
        permute_133 = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        clone_96 = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_248 = torch.ops.aten.view.default(clone_96, [1024, -1, 32]);  clone_96 = None
        view_249 = torch.ops.aten.view.default(clone_94, [1024, -1, 32]);  clone_94 = None
        view_250 = torch.ops.aten.view.default(clone_95, [1024, -1, 32]);  clone_95 = None
        permute_134 = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
        bmm_24 = torch.ops.aten.bmm.default(view_248, permute_134);  view_248 = permute_134 = None
        view_251 = torch.ops.aten.view.default(bmm_24, [64, 16, 128, 128]);  bmm_24 = None
        add_85 = torch.ops.aten.add.Tensor(view_251, expand_1);  view_251 = None
        view_252 = torch.ops.aten.view.default(add_85, [1024, 128, 128]);  add_85 = None
        amax_12 = torch.ops.aten.amax.default(view_252, [-1], True)
        sub_36 = torch.ops.aten.sub.Tensor(view_252, amax_12);  view_252 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        bmm_25 = torch.ops.aten.bmm.default(div_12, view_250);  div_12 = view_250 = None
        view_253 = torch.ops.aten.view.default(bmm_25, [64, 16, 128, 32]);  bmm_25 = None
        permute_135 = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        clone_98 = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_254 = torch.ops.aten.view.default(clone_98, [64, 128, 512]);  clone_98 = None
        view_255 = torch.ops.aten.view.default(view_254, [8192, 512]);  view_254 = None
        permute_136 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg196_1, view_255, permute_136);  arg196_1 = view_255 = permute_136 = None
        view_256 = torch.ops.aten.view.default(addmm_71, [64, 128, 512]);  addmm_71 = None
        add_86 = torch.ops.aten.add.Tensor(add_84, view_256);  add_84 = view_256 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_87 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_86, getitem_49);  add_86 = getitem_49 = None
        mul_93 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_93, arg197_1);  mul_93 = arg197_1 = None
        add_88 = torch.ops.aten.add.Tensor(mul_94, arg198_1);  mul_94 = arg198_1 = None
        view_257 = torch.ops.aten.view.default(add_88, [8192, 512])
        permute_137 = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg200_1, view_257, permute_137);  arg200_1 = view_257 = permute_137 = None
        view_258 = torch.ops.aten.view.default(addmm_72, [64, 128, 512]);  addmm_72 = None
        mul_95 = torch.ops.aten.mul.Tensor(view_258, 0.1767766952966369);  view_258 = None
        view_259 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_138 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg202_1, view_259, permute_138);  arg202_1 = view_259 = permute_138 = None
        view_260 = torch.ops.aten.view.default(addmm_73, [64, 128, 512]);  addmm_73 = None
        view_261 = torch.ops.aten.view.default(view_260, [64, -1, 16, 32]);  view_260 = None
        permute_139 = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        clone_100 = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_262 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_140 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg204_1, view_262, permute_140);  arg204_1 = view_262 = permute_140 = None
        view_263 = torch.ops.aten.view.default(addmm_74, [64, 128, 512]);  addmm_74 = None
        view_264 = torch.ops.aten.view.default(view_263, [64, -1, 16, 32]);  view_263 = None
        permute_141 = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        clone_101 = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_265 = torch.ops.aten.view.default(mul_95, [64, 128, 16, 32]);  mul_95 = None
        permute_142 = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_102 = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_266 = torch.ops.aten.view.default(clone_102, [1024, -1, 32]);  clone_102 = None
        view_267 = torch.ops.aten.view.default(clone_100, [1024, -1, 32]);  clone_100 = None
        view_268 = torch.ops.aten.view.default(clone_101, [1024, -1, 32]);  clone_101 = None
        unsqueeze_default_15 = torch.ops.aten.unsqueeze.default(view_266, 0);  view_266 = None
        unsqueeze_default_16 = torch.ops.aten.unsqueeze.default(view_267, 0);  view_267 = None
        unsqueeze_default_17 = torch.ops.aten.unsqueeze.default(view_268, 0);  view_268 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, None, False, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
        getitem_89 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        squeeze_dim_5 = torch.ops.aten.squeeze.dim(getitem_89, 0);  getitem_89 = None
        view_269 = torch.ops.aten.view.default(squeeze_dim_5, [64, 16, 128, 32]);  squeeze_dim_5 = None
        permute_144 = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        clone_104 = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_270 = torch.ops.aten.view.default(clone_104, [64, 128, 512]);  clone_104 = None
        view_271 = torch.ops.aten.view.default(view_270, [8192, 512]);  view_270 = None
        permute_145 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg206_1, view_271, permute_145);  arg206_1 = view_271 = permute_145 = None
        view_272 = torch.ops.aten.view.default(addmm_75, [64, 128, 512]);  addmm_75 = None
        add_89 = torch.ops.aten.add.Tensor(add_88, view_272);  add_88 = view_272 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_89, getitem_51);  add_89 = getitem_51 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = rsqrt_25 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg207_1);  mul_96 = arg207_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_97, arg208_1);  mul_97 = arg208_1 = None
        view_273 = torch.ops.aten.view.default(add_91, [8192, 512])
        permute_146 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg210_1, view_273, permute_146);  arg210_1 = view_273 = permute_146 = None
        view_274 = torch.ops.aten.view.default(addmm_76, [64, 128, 2048]);  addmm_76 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_274, 0.5)
        mul_99 = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
        erf_10 = torch.ops.aten.erf.default(mul_99);  mul_99 = None
        add_92 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_98, add_92);  mul_98 = add_92 = None
        view_275 = torch.ops.aten.view.default(mul_100, [8192, 2048]);  mul_100 = None
        permute_147 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg212_1, view_275, permute_147);  arg212_1 = view_275 = permute_147 = None
        view_276 = torch.ops.aten.view.default(addmm_77, [64, 128, 512]);  addmm_77 = None
        add_93 = torch.ops.aten.add.Tensor(add_91, view_276);  add_91 = view_276 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_93, getitem_53);  add_93 = getitem_53 = None
        mul_101 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, arg213_1);  mul_101 = arg213_1 = None
        add_95 = torch.ops.aten.add.Tensor(mul_102, arg214_1);  mul_102 = arg214_1 = None
        view_277 = torch.ops.aten.view.default(add_95, [8192, 512])
        permute_148 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg216_1, view_277, permute_148);  arg216_1 = view_277 = permute_148 = None
        view_278 = torch.ops.aten.view.default(addmm_78, [64, 128, 512]);  addmm_78 = None
        mul_103 = torch.ops.aten.mul.Tensor(view_278, 0.1767766952966369);  view_278 = None
        view_279 = torch.ops.aten.view.default(add_95, [8192, 512])
        permute_149 = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg218_1, view_279, permute_149);  arg218_1 = view_279 = permute_149 = None
        view_280 = torch.ops.aten.view.default(addmm_79, [64, 128, 512]);  addmm_79 = None
        view_281 = torch.ops.aten.view.default(view_280, [64, -1, 16, 32]);  view_280 = None
        permute_150 = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
        clone_108 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_282 = torch.ops.aten.view.default(add_95, [8192, 512])
        permute_151 = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg220_1, view_282, permute_151);  arg220_1 = view_282 = permute_151 = None
        view_283 = torch.ops.aten.view.default(addmm_80, [64, 128, 512]);  addmm_80 = None
        view_284 = torch.ops.aten.view.default(view_283, [64, -1, 16, 32]);  view_283 = None
        permute_152 = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
        clone_109 = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_285 = torch.ops.aten.view.default(mul_103, [64, 128, 16, 32]);  mul_103 = None
        permute_153 = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        clone_110 = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_286 = torch.ops.aten.view.default(clone_110, [1024, -1, 32]);  clone_110 = None
        view_287 = torch.ops.aten.view.default(clone_108, [1024, -1, 32]);  clone_108 = None
        view_288 = torch.ops.aten.view.default(clone_109, [1024, -1, 32]);  clone_109 = None
        permute_154 = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
        bmm_28 = torch.ops.aten.bmm.default(view_286, permute_154);  view_286 = permute_154 = None
        view_289 = torch.ops.aten.view.default(bmm_28, [64, 16, 128, 128]);  bmm_28 = None
        add_96 = torch.ops.aten.add.Tensor(view_289, expand_1);  view_289 = None
        view_290 = torch.ops.aten.view.default(add_96, [1024, 128, 128]);  add_96 = None
        amax_14 = torch.ops.aten.amax.default(view_290, [-1], True)
        sub_41 = torch.ops.aten.sub.Tensor(view_290, amax_14);  view_290 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        bmm_29 = torch.ops.aten.bmm.default(div_14, view_288);  div_14 = view_288 = None
        view_291 = torch.ops.aten.view.default(bmm_29, [64, 16, 128, 32]);  bmm_29 = None
        permute_155 = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_112 = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_292 = torch.ops.aten.view.default(clone_112, [64, 128, 512]);  clone_112 = None
        view_293 = torch.ops.aten.view.default(view_292, [8192, 512]);  view_292 = None
        permute_156 = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg222_1, view_293, permute_156);  arg222_1 = view_293 = permute_156 = None
        view_294 = torch.ops.aten.view.default(addmm_81, [64, 128, 512]);  addmm_81 = None
        add_97 = torch.ops.aten.add.Tensor(add_95, view_294);  add_95 = view_294 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_97, getitem_55);  add_97 = getitem_55 = None
        mul_104 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = rsqrt_27 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, arg223_1);  mul_104 = arg223_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_105, arg224_1);  mul_105 = arg224_1 = None
        view_295 = torch.ops.aten.view.default(add_99, [8192, 512])
        permute_157 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg226_1, view_295, permute_157);  arg226_1 = view_295 = permute_157 = None
        view_296 = torch.ops.aten.view.default(addmm_82, [64, 128, 512]);  addmm_82 = None
        mul_106 = torch.ops.aten.mul.Tensor(view_296, 0.1767766952966369);  view_296 = None
        view_297 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_158 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg228_1, view_297, permute_158);  arg228_1 = view_297 = permute_158 = None
        view_298 = torch.ops.aten.view.default(addmm_83, [64, 128, 512]);  addmm_83 = None
        view_299 = torch.ops.aten.view.default(view_298, [64, -1, 16, 32]);  view_298 = None
        permute_159 = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
        clone_114 = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_300 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_160 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg230_1, view_300, permute_160);  arg230_1 = view_300 = permute_160 = None
        view_301 = torch.ops.aten.view.default(addmm_84, [64, 128, 512]);  addmm_84 = None
        view_302 = torch.ops.aten.view.default(view_301, [64, -1, 16, 32]);  view_301 = None
        permute_161 = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_115 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_303 = torch.ops.aten.view.default(mul_106, [64, 128, 16, 32]);  mul_106 = None
        permute_162 = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        clone_116 = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_304 = torch.ops.aten.view.default(clone_116, [1024, -1, 32]);  clone_116 = None
        view_305 = torch.ops.aten.view.default(clone_114, [1024, -1, 32]);  clone_114 = None
        view_306 = torch.ops.aten.view.default(clone_115, [1024, -1, 32]);  clone_115 = None
        unsqueeze_default_12 = torch.ops.aten.unsqueeze.default(view_304, 0);  view_304 = None
        unsqueeze_default_13 = torch.ops.aten.unsqueeze.default(view_305, 0);  view_305 = None
        unsqueeze_default_14 = torch.ops.aten.unsqueeze.default(view_306, 0);  view_306 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, None, False, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
        getitem_88 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        squeeze_dim_4 = torch.ops.aten.squeeze.dim(getitem_88, 0);  getitem_88 = None
        view_307 = torch.ops.aten.view.default(squeeze_dim_4, [64, 16, 128, 32]);  squeeze_dim_4 = None
        permute_164 = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        clone_118 = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_308 = torch.ops.aten.view.default(clone_118, [64, 128, 512]);  clone_118 = None
        view_309 = torch.ops.aten.view.default(view_308, [8192, 512]);  view_308 = None
        permute_165 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg232_1, view_309, permute_165);  arg232_1 = view_309 = permute_165 = None
        view_310 = torch.ops.aten.view.default(addmm_85, [64, 128, 512]);  addmm_85 = None
        add_100 = torch.ops.aten.add.Tensor(add_99, view_310);  add_99 = view_310 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_101 = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_100, getitem_57);  add_100 = getitem_57 = None
        mul_107 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_28);  sub_44 = rsqrt_28 = None
        mul_108 = torch.ops.aten.mul.Tensor(mul_107, arg233_1);  mul_107 = arg233_1 = None
        add_102 = torch.ops.aten.add.Tensor(mul_108, arg234_1);  mul_108 = arg234_1 = None
        view_311 = torch.ops.aten.view.default(add_102, [8192, 512])
        permute_166 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg236_1, view_311, permute_166);  arg236_1 = view_311 = permute_166 = None
        view_312 = torch.ops.aten.view.default(addmm_86, [64, 128, 2048]);  addmm_86 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_312, 0.5)
        mul_110 = torch.ops.aten.mul.Tensor(view_312, 0.7071067811865476);  view_312 = None
        erf_11 = torch.ops.aten.erf.default(mul_110);  mul_110 = None
        add_103 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_109, add_103);  mul_109 = add_103 = None
        view_313 = torch.ops.aten.view.default(mul_111, [8192, 2048]);  mul_111 = None
        permute_167 = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg238_1, view_313, permute_167);  arg238_1 = view_313 = permute_167 = None
        view_314 = torch.ops.aten.view.default(addmm_87, [64, 128, 512]);  addmm_87 = None
        add_104 = torch.ops.aten.add.Tensor(add_102, view_314);  add_102 = view_314 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_105 = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_104, getitem_59);  add_104 = getitem_59 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = rsqrt_29 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, arg239_1);  mul_112 = arg239_1 = None
        add_106 = torch.ops.aten.add.Tensor(mul_113, arg240_1);  mul_113 = arg240_1 = None
        view_315 = torch.ops.aten.view.default(add_106, [8192, 512])
        permute_168 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg242_1, view_315, permute_168);  arg242_1 = view_315 = permute_168 = None
        view_316 = torch.ops.aten.view.default(addmm_88, [64, 128, 512]);  addmm_88 = None
        mul_114 = torch.ops.aten.mul.Tensor(view_316, 0.1767766952966369);  view_316 = None
        view_317 = torch.ops.aten.view.default(add_106, [8192, 512])
        permute_169 = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg244_1, view_317, permute_169);  arg244_1 = view_317 = permute_169 = None
        view_318 = torch.ops.aten.view.default(addmm_89, [64, 128, 512]);  addmm_89 = None
        view_319 = torch.ops.aten.view.default(view_318, [64, -1, 16, 32]);  view_318 = None
        permute_170 = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_122 = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_320 = torch.ops.aten.view.default(add_106, [8192, 512])
        permute_171 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg246_1, view_320, permute_171);  arg246_1 = view_320 = permute_171 = None
        view_321 = torch.ops.aten.view.default(addmm_90, [64, 128, 512]);  addmm_90 = None
        view_322 = torch.ops.aten.view.default(view_321, [64, -1, 16, 32]);  view_321 = None
        permute_172 = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_123 = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_323 = torch.ops.aten.view.default(mul_114, [64, 128, 16, 32]);  mul_114 = None
        permute_173 = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        clone_124 = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
        view_324 = torch.ops.aten.view.default(clone_124, [1024, -1, 32]);  clone_124 = None
        view_325 = torch.ops.aten.view.default(clone_122, [1024, -1, 32]);  clone_122 = None
        view_326 = torch.ops.aten.view.default(clone_123, [1024, -1, 32]);  clone_123 = None
        permute_174 = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
        bmm_32 = torch.ops.aten.bmm.default(view_324, permute_174);  view_324 = permute_174 = None
        view_327 = torch.ops.aten.view.default(bmm_32, [64, 16, 128, 128]);  bmm_32 = None
        add_107 = torch.ops.aten.add.Tensor(view_327, expand_1);  view_327 = None
        view_328 = torch.ops.aten.view.default(add_107, [1024, 128, 128]);  add_107 = None
        amax_16 = torch.ops.aten.amax.default(view_328, [-1], True)
        sub_46 = torch.ops.aten.sub.Tensor(view_328, amax_16);  view_328 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        bmm_33 = torch.ops.aten.bmm.default(div_16, view_326);  div_16 = view_326 = None
        view_329 = torch.ops.aten.view.default(bmm_33, [64, 16, 128, 32]);  bmm_33 = None
        permute_175 = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        clone_126 = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_330 = torch.ops.aten.view.default(clone_126, [64, 128, 512]);  clone_126 = None
        view_331 = torch.ops.aten.view.default(view_330, [8192, 512]);  view_330 = None
        permute_176 = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg248_1, view_331, permute_176);  arg248_1 = view_331 = permute_176 = None
        view_332 = torch.ops.aten.view.default(addmm_91, [64, 128, 512]);  addmm_91 = None
        add_108 = torch.ops.aten.add.Tensor(add_106, view_332);  add_106 = view_332 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_109 = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_108, getitem_61);  add_108 = getitem_61 = None
        mul_115 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_30);  sub_47 = rsqrt_30 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, arg249_1);  mul_115 = arg249_1 = None
        add_110 = torch.ops.aten.add.Tensor(mul_116, arg250_1);  mul_116 = arg250_1 = None
        view_333 = torch.ops.aten.view.default(add_110, [8192, 512])
        permute_177 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg252_1, view_333, permute_177);  arg252_1 = view_333 = permute_177 = None
        view_334 = torch.ops.aten.view.default(addmm_92, [64, 128, 512]);  addmm_92 = None
        mul_117 = torch.ops.aten.mul.Tensor(view_334, 0.1767766952966369);  view_334 = None
        view_335 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_178 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg254_1, view_335, permute_178);  arg254_1 = view_335 = permute_178 = None
        view_336 = torch.ops.aten.view.default(addmm_93, [64, 128, 512]);  addmm_93 = None
        view_337 = torch.ops.aten.view.default(view_336, [64, -1, 16, 32]);  view_336 = None
        permute_179 = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
        clone_128 = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_338 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_180 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg256_1, view_338, permute_180);  arg256_1 = view_338 = permute_180 = None
        view_339 = torch.ops.aten.view.default(addmm_94, [64, 128, 512]);  addmm_94 = None
        view_340 = torch.ops.aten.view.default(view_339, [64, -1, 16, 32]);  view_339 = None
        permute_181 = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_129 = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_341 = torch.ops.aten.view.default(mul_117, [64, 128, 16, 32]);  mul_117 = None
        permute_182 = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
        clone_130 = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_342 = torch.ops.aten.view.default(clone_130, [1024, -1, 32]);  clone_130 = None
        view_343 = torch.ops.aten.view.default(clone_128, [1024, -1, 32]);  clone_128 = None
        view_344 = torch.ops.aten.view.default(clone_129, [1024, -1, 32]);  clone_129 = None
        unsqueeze_default_9 = torch.ops.aten.unsqueeze.default(view_342, 0);  view_342 = None
        unsqueeze_default_10 = torch.ops.aten.unsqueeze.default(view_343, 0);  view_343 = None
        unsqueeze_default_11 = torch.ops.aten.unsqueeze.default(view_344, 0);  view_344 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, None, False, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
        getitem_87 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(getitem_87, 0);  getitem_87 = None
        view_345 = torch.ops.aten.view.default(squeeze_dim_3, [64, 16, 128, 32]);  squeeze_dim_3 = None
        permute_184 = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_132 = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_346 = torch.ops.aten.view.default(clone_132, [64, 128, 512]);  clone_132 = None
        view_347 = torch.ops.aten.view.default(view_346, [8192, 512]);  view_346 = None
        permute_185 = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg258_1, view_347, permute_185);  arg258_1 = view_347 = permute_185 = None
        view_348 = torch.ops.aten.view.default(addmm_95, [64, 128, 512]);  addmm_95 = None
        add_111 = torch.ops.aten.add.Tensor(add_110, view_348);  add_110 = view_348 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_112 = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_111, getitem_63);  add_111 = getitem_63 = None
        mul_118 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_31);  sub_49 = rsqrt_31 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, arg259_1);  mul_118 = arg259_1 = None
        add_113 = torch.ops.aten.add.Tensor(mul_119, arg260_1);  mul_119 = arg260_1 = None
        view_349 = torch.ops.aten.view.default(add_113, [8192, 512])
        permute_186 = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg262_1, view_349, permute_186);  arg262_1 = view_349 = permute_186 = None
        view_350 = torch.ops.aten.view.default(addmm_96, [64, 128, 2048]);  addmm_96 = None
        mul_120 = torch.ops.aten.mul.Tensor(view_350, 0.5)
        mul_121 = torch.ops.aten.mul.Tensor(view_350, 0.7071067811865476);  view_350 = None
        erf_12 = torch.ops.aten.erf.default(mul_121);  mul_121 = None
        add_114 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_120, add_114);  mul_120 = add_114 = None
        view_351 = torch.ops.aten.view.default(mul_122, [8192, 2048]);  mul_122 = None
        permute_187 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg264_1, view_351, permute_187);  arg264_1 = view_351 = permute_187 = None
        view_352 = torch.ops.aten.view.default(addmm_97, [64, 128, 512]);  addmm_97 = None
        add_115 = torch.ops.aten.add.Tensor(add_113, view_352);  add_113 = view_352 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_116 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_50 = torch.ops.aten.sub.Tensor(add_115, getitem_65);  add_115 = getitem_65 = None
        mul_123 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_32);  sub_50 = rsqrt_32 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, arg265_1);  mul_123 = arg265_1 = None
        add_117 = torch.ops.aten.add.Tensor(mul_124, arg266_1);  mul_124 = arg266_1 = None
        view_353 = torch.ops.aten.view.default(add_117, [8192, 512])
        permute_188 = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg268_1, view_353, permute_188);  arg268_1 = view_353 = permute_188 = None
        view_354 = torch.ops.aten.view.default(addmm_98, [64, 128, 512]);  addmm_98 = None
        mul_125 = torch.ops.aten.mul.Tensor(view_354, 0.1767766952966369);  view_354 = None
        view_355 = torch.ops.aten.view.default(add_117, [8192, 512])
        permute_189 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg270_1, view_355, permute_189);  arg270_1 = view_355 = permute_189 = None
        view_356 = torch.ops.aten.view.default(addmm_99, [64, 128, 512]);  addmm_99 = None
        view_357 = torch.ops.aten.view.default(view_356, [64, -1, 16, 32]);  view_356 = None
        permute_190 = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        clone_136 = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_358 = torch.ops.aten.view.default(add_117, [8192, 512])
        permute_191 = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg272_1, view_358, permute_191);  arg272_1 = view_358 = permute_191 = None
        view_359 = torch.ops.aten.view.default(addmm_100, [64, 128, 512]);  addmm_100 = None
        view_360 = torch.ops.aten.view.default(view_359, [64, -1, 16, 32]);  view_359 = None
        permute_192 = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        clone_137 = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_361 = torch.ops.aten.view.default(mul_125, [64, 128, 16, 32]);  mul_125 = None
        permute_193 = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
        clone_138 = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_362 = torch.ops.aten.view.default(clone_138, [1024, -1, 32]);  clone_138 = None
        view_363 = torch.ops.aten.view.default(clone_136, [1024, -1, 32]);  clone_136 = None
        view_364 = torch.ops.aten.view.default(clone_137, [1024, -1, 32]);  clone_137 = None
        permute_194 = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
        bmm_36 = torch.ops.aten.bmm.default(view_362, permute_194);  view_362 = permute_194 = None
        view_365 = torch.ops.aten.view.default(bmm_36, [64, 16, 128, 128]);  bmm_36 = None
        add_118 = torch.ops.aten.add.Tensor(view_365, expand_1);  view_365 = None
        view_366 = torch.ops.aten.view.default(add_118, [1024, 128, 128]);  add_118 = None
        amax_18 = torch.ops.aten.amax.default(view_366, [-1], True)
        sub_51 = torch.ops.aten.sub.Tensor(view_366, amax_18);  view_366 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        bmm_37 = torch.ops.aten.bmm.default(div_18, view_364);  div_18 = view_364 = None
        view_367 = torch.ops.aten.view.default(bmm_37, [64, 16, 128, 32]);  bmm_37 = None
        permute_195 = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
        clone_140 = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_368 = torch.ops.aten.view.default(clone_140, [64, 128, 512]);  clone_140 = None
        view_369 = torch.ops.aten.view.default(view_368, [8192, 512]);  view_368 = None
        permute_196 = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg274_1, view_369, permute_196);  arg274_1 = view_369 = permute_196 = None
        view_370 = torch.ops.aten.view.default(addmm_101, [64, 128, 512]);  addmm_101 = None
        add_119 = torch.ops.aten.add.Tensor(add_117, view_370);  add_117 = view_370 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_120 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_52 = torch.ops.aten.sub.Tensor(add_119, getitem_67);  add_119 = getitem_67 = None
        mul_126 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_33);  sub_52 = rsqrt_33 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_126, arg275_1);  mul_126 = arg275_1 = None
        add_121 = torch.ops.aten.add.Tensor(mul_127, arg276_1);  mul_127 = arg276_1 = None
        view_371 = torch.ops.aten.view.default(add_121, [8192, 512])
        permute_197 = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg278_1, view_371, permute_197);  arg278_1 = view_371 = permute_197 = None
        view_372 = torch.ops.aten.view.default(addmm_102, [64, 128, 512]);  addmm_102 = None
        mul_128 = torch.ops.aten.mul.Tensor(view_372, 0.1767766952966369);  view_372 = None
        view_373 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_198 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg280_1, view_373, permute_198);  arg280_1 = view_373 = permute_198 = None
        view_374 = torch.ops.aten.view.default(addmm_103, [64, 128, 512]);  addmm_103 = None
        view_375 = torch.ops.aten.view.default(view_374, [64, -1, 16, 32]);  view_374 = None
        permute_199 = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        clone_142 = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_376 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_200 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg282_1, view_376, permute_200);  arg282_1 = view_376 = permute_200 = None
        view_377 = torch.ops.aten.view.default(addmm_104, [64, 128, 512]);  addmm_104 = None
        view_378 = torch.ops.aten.view.default(view_377, [64, -1, 16, 32]);  view_377 = None
        permute_201 = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_143 = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_379 = torch.ops.aten.view.default(mul_128, [64, 128, 16, 32]);  mul_128 = None
        permute_202 = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        clone_144 = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_380 = torch.ops.aten.view.default(clone_144, [1024, -1, 32]);  clone_144 = None
        view_381 = torch.ops.aten.view.default(clone_142, [1024, -1, 32]);  clone_142 = None
        view_382 = torch.ops.aten.view.default(clone_143, [1024, -1, 32]);  clone_143 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(view_380, 0);  view_380 = None
        unsqueeze_default_7 = torch.ops.aten.unsqueeze.default(view_381, 0);  view_381 = None
        unsqueeze_default_8 = torch.ops.aten.unsqueeze.default(view_382, 0);  view_382 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, None, False, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
        getitem_86 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(getitem_86, 0);  getitem_86 = None
        view_383 = torch.ops.aten.view.default(squeeze_dim_2, [64, 16, 128, 32]);  squeeze_dim_2 = None
        permute_204 = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_146 = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_384 = torch.ops.aten.view.default(clone_146, [64, 128, 512]);  clone_146 = None
        view_385 = torch.ops.aten.view.default(view_384, [8192, 512]);  view_384 = None
        permute_205 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg284_1, view_385, permute_205);  arg284_1 = view_385 = permute_205 = None
        view_386 = torch.ops.aten.view.default(addmm_105, [64, 128, 512]);  addmm_105 = None
        add_122 = torch.ops.aten.add.Tensor(add_121, view_386);  add_121 = view_386 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_122, getitem_69);  add_122 = getitem_69 = None
        mul_129 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_34);  sub_54 = rsqrt_34 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, arg285_1);  mul_129 = arg285_1 = None
        add_124 = torch.ops.aten.add.Tensor(mul_130, arg286_1);  mul_130 = arg286_1 = None
        view_387 = torch.ops.aten.view.default(add_124, [8192, 512])
        permute_206 = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg288_1, view_387, permute_206);  arg288_1 = view_387 = permute_206 = None
        view_388 = torch.ops.aten.view.default(addmm_106, [64, 128, 2048]);  addmm_106 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_388, 0.5)
        mul_132 = torch.ops.aten.mul.Tensor(view_388, 0.7071067811865476);  view_388 = None
        erf_13 = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_125 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_131, add_125);  mul_131 = add_125 = None
        view_389 = torch.ops.aten.view.default(mul_133, [8192, 2048]);  mul_133 = None
        permute_207 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg290_1, view_389, permute_207);  arg290_1 = view_389 = permute_207 = None
        view_390 = torch.ops.aten.view.default(addmm_107, [64, 128, 512]);  addmm_107 = None
        add_126 = torch.ops.aten.add.Tensor(add_124, view_390);  add_124 = view_390 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_127 = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_55 = torch.ops.aten.sub.Tensor(add_126, getitem_71);  add_126 = getitem_71 = None
        mul_134 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_35);  sub_55 = rsqrt_35 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, arg291_1);  mul_134 = arg291_1 = None
        add_128 = torch.ops.aten.add.Tensor(mul_135, arg292_1);  mul_135 = arg292_1 = None
        view_391 = torch.ops.aten.view.default(add_128, [8192, 512])
        permute_208 = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg294_1, view_391, permute_208);  arg294_1 = view_391 = permute_208 = None
        view_392 = torch.ops.aten.view.default(addmm_108, [64, 128, 512]);  addmm_108 = None
        mul_136 = torch.ops.aten.mul.Tensor(view_392, 0.1767766952966369);  view_392 = None
        view_393 = torch.ops.aten.view.default(add_128, [8192, 512])
        permute_209 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg296_1, view_393, permute_209);  arg296_1 = view_393 = permute_209 = None
        view_394 = torch.ops.aten.view.default(addmm_109, [64, 128, 512]);  addmm_109 = None
        view_395 = torch.ops.aten.view.default(view_394, [64, -1, 16, 32]);  view_394 = None
        permute_210 = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
        clone_150 = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_396 = torch.ops.aten.view.default(add_128, [8192, 512])
        permute_211 = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg298_1, view_396, permute_211);  arg298_1 = view_396 = permute_211 = None
        view_397 = torch.ops.aten.view.default(addmm_110, [64, 128, 512]);  addmm_110 = None
        view_398 = torch.ops.aten.view.default(view_397, [64, -1, 16, 32]);  view_397 = None
        permute_212 = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        clone_151 = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_399 = torch.ops.aten.view.default(mul_136, [64, 128, 16, 32]);  mul_136 = None
        permute_213 = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_152 = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_400 = torch.ops.aten.view.default(clone_152, [1024, -1, 32]);  clone_152 = None
        view_401 = torch.ops.aten.view.default(clone_150, [1024, -1, 32]);  clone_150 = None
        view_402 = torch.ops.aten.view.default(clone_151, [1024, -1, 32]);  clone_151 = None
        permute_214 = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None
        bmm_40 = torch.ops.aten.bmm.default(view_400, permute_214);  view_400 = permute_214 = None
        view_403 = torch.ops.aten.view.default(bmm_40, [64, 16, 128, 128]);  bmm_40 = None
        add_129 = torch.ops.aten.add.Tensor(view_403, expand_1);  view_403 = None
        view_404 = torch.ops.aten.view.default(add_129, [1024, 128, 128]);  add_129 = None
        amax_20 = torch.ops.aten.amax.default(view_404, [-1], True)
        sub_56 = torch.ops.aten.sub.Tensor(view_404, amax_20);  view_404 = amax_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_56);  sub_56 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        bmm_41 = torch.ops.aten.bmm.default(div_20, view_402);  div_20 = view_402 = None
        view_405 = torch.ops.aten.view.default(bmm_41, [64, 16, 128, 32]);  bmm_41 = None
        permute_215 = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
        clone_154 = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_406 = torch.ops.aten.view.default(clone_154, [64, 128, 512]);  clone_154 = None
        view_407 = torch.ops.aten.view.default(view_406, [8192, 512]);  view_406 = None
        permute_216 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg300_1, view_407, permute_216);  arg300_1 = view_407 = permute_216 = None
        view_408 = torch.ops.aten.view.default(addmm_111, [64, 128, 512]);  addmm_111 = None
        add_130 = torch.ops.aten.add.Tensor(add_128, view_408);  add_128 = view_408 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_131 = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_130, getitem_73);  add_130 = getitem_73 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_36);  sub_57 = rsqrt_36 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, arg301_1);  mul_137 = arg301_1 = None
        add_132 = torch.ops.aten.add.Tensor(mul_138, arg302_1);  mul_138 = arg302_1 = None
        view_409 = torch.ops.aten.view.default(add_132, [8192, 512])
        permute_217 = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg304_1, view_409, permute_217);  arg304_1 = view_409 = permute_217 = None
        view_410 = torch.ops.aten.view.default(addmm_112, [64, 128, 512]);  addmm_112 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_410, 0.1767766952966369);  view_410 = None
        view_411 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_218 = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg306_1, view_411, permute_218);  arg306_1 = view_411 = permute_218 = None
        view_412 = torch.ops.aten.view.default(addmm_113, [64, 128, 512]);  addmm_113 = None
        view_413 = torch.ops.aten.view.default(view_412, [64, -1, 16, 32]);  view_412 = None
        permute_219 = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
        clone_156 = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_414 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_220 = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg308_1, view_414, permute_220);  arg308_1 = view_414 = permute_220 = None
        view_415 = torch.ops.aten.view.default(addmm_114, [64, 128, 512]);  addmm_114 = None
        view_416 = torch.ops.aten.view.default(view_415, [64, -1, 16, 32]);  view_415 = None
        permute_221 = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        clone_157 = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_417 = torch.ops.aten.view.default(mul_139, [64, 128, 16, 32]);  mul_139 = None
        permute_222 = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
        clone_158 = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_418 = torch.ops.aten.view.default(clone_158, [1024, -1, 32]);  clone_158 = None
        view_419 = torch.ops.aten.view.default(clone_156, [1024, -1, 32]);  clone_156 = None
        view_420 = torch.ops.aten.view.default(clone_157, [1024, -1, 32]);  clone_157 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(view_418, 0);  view_418 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(view_419, 0);  view_419 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(view_420, 0);  view_420 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, None, False, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
        getitem_85 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(getitem_85, 0);  getitem_85 = None
        view_421 = torch.ops.aten.view.default(squeeze_dim_1, [64, 16, 128, 32]);  squeeze_dim_1 = None
        permute_224 = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
        clone_160 = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_422 = torch.ops.aten.view.default(clone_160, [64, 128, 512]);  clone_160 = None
        view_423 = torch.ops.aten.view.default(view_422, [8192, 512]);  view_422 = None
        permute_225 = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg310_1, view_423, permute_225);  arg310_1 = view_423 = permute_225 = None
        view_424 = torch.ops.aten.view.default(addmm_115, [64, 128, 512]);  addmm_115 = None
        add_133 = torch.ops.aten.add.Tensor(add_132, view_424);  add_132 = view_424 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_133, getitem_75);  add_133 = getitem_75 = None
        mul_140 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_37);  sub_59 = rsqrt_37 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, arg311_1);  mul_140 = arg311_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_141, arg312_1);  mul_141 = arg312_1 = None
        view_425 = torch.ops.aten.view.default(add_135, [8192, 512])
        permute_226 = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg314_1, view_425, permute_226);  arg314_1 = view_425 = permute_226 = None
        view_426 = torch.ops.aten.view.default(addmm_116, [64, 128, 2048]);  addmm_116 = None
        mul_142 = torch.ops.aten.mul.Tensor(view_426, 0.5)
        mul_143 = torch.ops.aten.mul.Tensor(view_426, 0.7071067811865476);  view_426 = None
        erf_14 = torch.ops.aten.erf.default(mul_143);  mul_143 = None
        add_136 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_142, add_136);  mul_142 = add_136 = None
        view_427 = torch.ops.aten.view.default(mul_144, [8192, 2048]);  mul_144 = None
        permute_227 = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg316_1, view_427, permute_227);  arg316_1 = view_427 = permute_227 = None
        view_428 = torch.ops.aten.view.default(addmm_117, [64, 128, 512]);  addmm_117 = None
        add_137 = torch.ops.aten.add.Tensor(add_135, view_428);  add_135 = view_428 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_60 = torch.ops.aten.sub.Tensor(add_137, getitem_77);  add_137 = getitem_77 = None
        mul_145 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_38);  sub_60 = rsqrt_38 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, arg317_1);  mul_145 = arg317_1 = None
        add_139 = torch.ops.aten.add.Tensor(mul_146, arg318_1);  mul_146 = arg318_1 = None
        view_429 = torch.ops.aten.view.default(add_139, [8192, 512])
        permute_228 = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg320_1, view_429, permute_228);  arg320_1 = view_429 = permute_228 = None
        view_430 = torch.ops.aten.view.default(addmm_118, [64, 128, 512]);  addmm_118 = None
        mul_147 = torch.ops.aten.mul.Tensor(view_430, 0.1767766952966369);  view_430 = None
        view_431 = torch.ops.aten.view.default(add_139, [8192, 512])
        permute_229 = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg322_1, view_431, permute_229);  arg322_1 = view_431 = permute_229 = None
        view_432 = torch.ops.aten.view.default(addmm_119, [64, 128, 512]);  addmm_119 = None
        view_433 = torch.ops.aten.view.default(view_432, [64, -1, 16, 32]);  view_432 = None
        permute_230 = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None
        clone_164 = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_434 = torch.ops.aten.view.default(add_139, [8192, 512])
        permute_231 = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg324_1, view_434, permute_231);  arg324_1 = view_434 = permute_231 = None
        view_435 = torch.ops.aten.view.default(addmm_120, [64, 128, 512]);  addmm_120 = None
        view_436 = torch.ops.aten.view.default(view_435, [64, -1, 16, 32]);  view_435 = None
        permute_232 = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
        clone_165 = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_437 = torch.ops.aten.view.default(mul_147, [64, 128, 16, 32]);  mul_147 = None
        permute_233 = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
        clone_166 = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
        view_438 = torch.ops.aten.view.default(clone_166, [1024, -1, 32]);  clone_166 = None
        view_439 = torch.ops.aten.view.default(clone_164, [1024, -1, 32]);  clone_164 = None
        view_440 = torch.ops.aten.view.default(clone_165, [1024, -1, 32]);  clone_165 = None
        permute_234 = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
        bmm_44 = torch.ops.aten.bmm.default(view_438, permute_234);  view_438 = permute_234 = None
        view_441 = torch.ops.aten.view.default(bmm_44, [64, 16, 128, 128]);  bmm_44 = None
        add_140 = torch.ops.aten.add.Tensor(view_441, expand_1);  view_441 = expand_1 = None
        view_442 = torch.ops.aten.view.default(add_140, [1024, 128, 128]);  add_140 = None
        amax_22 = torch.ops.aten.amax.default(view_442, [-1], True)
        sub_61 = torch.ops.aten.sub.Tensor(view_442, amax_22);  view_442 = amax_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        bmm_45 = torch.ops.aten.bmm.default(div_22, view_440);  div_22 = view_440 = None
        view_443 = torch.ops.aten.view.default(bmm_45, [64, 16, 128, 32]);  bmm_45 = None
        permute_235 = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
        clone_168 = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_444 = torch.ops.aten.view.default(clone_168, [64, 128, 512]);  clone_168 = None
        view_445 = torch.ops.aten.view.default(view_444, [8192, 512]);  view_444 = None
        permute_236 = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg326_1, view_445, permute_236);  arg326_1 = view_445 = permute_236 = None
        view_446 = torch.ops.aten.view.default(addmm_121, [64, 128, 512]);  addmm_121 = None
        add_141 = torch.ops.aten.add.Tensor(add_139, view_446);  add_139 = view_446 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_62 = torch.ops.aten.sub.Tensor(add_141, getitem_79);  add_141 = getitem_79 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_39);  sub_62 = rsqrt_39 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, arg327_1);  mul_148 = arg327_1 = None
        add_143 = torch.ops.aten.add.Tensor(mul_149, arg328_1);  mul_149 = arg328_1 = None
        view_447 = torch.ops.aten.view.default(add_143, [8192, 512])
        permute_237 = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg330_1, view_447, permute_237);  arg330_1 = view_447 = permute_237 = None
        view_448 = torch.ops.aten.view.default(addmm_122, [64, 128, 512]);  addmm_122 = None
        mul_150 = torch.ops.aten.mul.Tensor(view_448, 0.1767766952966369);  view_448 = None
        view_449 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_238 = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg332_1, view_449, permute_238);  arg332_1 = view_449 = permute_238 = None
        view_450 = torch.ops.aten.view.default(addmm_123, [64, 128, 512]);  addmm_123 = None
        view_451 = torch.ops.aten.view.default(view_450, [64, -1, 16, 32]);  view_450 = None
        permute_239 = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
        clone_170 = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_452 = torch.ops.aten.view.default(add_58, [8192, 512])
        permute_240 = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg334_1, view_452, permute_240);  arg334_1 = view_452 = permute_240 = None
        view_453 = torch.ops.aten.view.default(addmm_124, [64, 128, 512]);  addmm_124 = None
        view_454 = torch.ops.aten.view.default(view_453, [64, -1, 16, 32]);  view_453 = None
        permute_241 = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        clone_171 = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        view_455 = torch.ops.aten.view.default(mul_150, [64, 128, 16, 32]);  mul_150 = None
        permute_242 = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
        clone_172 = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_456 = torch.ops.aten.view.default(clone_172, [1024, -1, 32]);  clone_172 = None
        view_457 = torch.ops.aten.view.default(clone_170, [1024, -1, 32]);  clone_170 = None
        view_458 = torch.ops.aten.view.default(clone_171, [1024, -1, 32]);  clone_171 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(view_456, 0);  view_456 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(view_457, 0);  view_457 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(view_458, 0);  view_458 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, False, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
        getitem_84 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(getitem_84, 0);  getitem_84 = None
        view_459 = torch.ops.aten.view.default(squeeze_dim, [64, 16, 128, 32]);  squeeze_dim = None
        permute_244 = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        clone_174 = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_460 = torch.ops.aten.view.default(clone_174, [64, 128, 512]);  clone_174 = None
        view_461 = torch.ops.aten.view.default(view_460, [8192, 512]);  view_460 = None
        permute_245 = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg336_1, view_461, permute_245);  arg336_1 = view_461 = permute_245 = None
        view_462 = torch.ops.aten.view.default(addmm_125, [64, 128, 512]);  addmm_125 = None
        add_144 = torch.ops.aten.add.Tensor(add_143, view_462);  add_143 = view_462 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_145 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_64 = torch.ops.aten.sub.Tensor(add_144, getitem_81);  add_144 = getitem_81 = None
        mul_151 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_40);  sub_64 = rsqrt_40 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_151, arg337_1);  mul_151 = arg337_1 = None
        add_146 = torch.ops.aten.add.Tensor(mul_152, arg338_1);  mul_152 = arg338_1 = None
        view_463 = torch.ops.aten.view.default(add_146, [8192, 512])
        permute_246 = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg340_1, view_463, permute_246);  arg340_1 = view_463 = permute_246 = None
        view_464 = torch.ops.aten.view.default(addmm_126, [64, 128, 2048]);  addmm_126 = None
        mul_153 = torch.ops.aten.mul.Tensor(view_464, 0.5)
        mul_154 = torch.ops.aten.mul.Tensor(view_464, 0.7071067811865476);  view_464 = None
        erf_15 = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_147 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_153, add_147);  mul_153 = add_147 = None
        view_465 = torch.ops.aten.view.default(mul_155, [8192, 2048]);  mul_155 = None
        permute_247 = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg342_1, view_465, permute_247);  arg342_1 = view_465 = permute_247 = None
        view_466 = torch.ops.aten.view.default(addmm_127, [64, 128, 512]);  addmm_127 = None
        add_148 = torch.ops.aten.add.Tensor(add_146, view_466);  add_146 = view_466 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_149 = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_65 = torch.ops.aten.sub.Tensor(add_148, getitem_83);  add_148 = getitem_83 = None
        mul_156 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_41);  sub_65 = rsqrt_41 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, arg343_1);  mul_156 = arg343_1 = None
        add_150 = torch.ops.aten.add.Tensor(mul_157, arg344_1);  mul_157 = arg344_1 = None
        permute_248 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view_467 = torch.ops.aten.view.default(add_150, [8192, 512]);  add_150 = None
        full_default_4 = torch.ops.aten.full.default([512, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_248, full_default_4], 1);  permute_248 = full_default_4 = None
        mm_default = torch.ops.aten.mm.default(view_467, cat_default);  view_467 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_468 = torch.ops.aten.view.default(slice_tensor, [64, 128, 50265]);  slice_tensor = None
        add_151 = torch.ops.aten.add.Tensor(view_468, arg345_1);  view_468 = arg345_1 = None
        view_469 = torch.ops.aten.view.default(add_151, [-1, 50265])
        view_470 = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        amax_24 = torch.ops.aten.amax.default(view_469, [1], True)
        sub_66 = torch.ops.aten.sub.Tensor(view_469, amax_24);  view_469 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_66)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_67 = torch.ops.aten.sub.Tensor(sub_66, log);  sub_66 = log = None
        ne = torch.ops.aten.ne.Scalar(view_470, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_470, full_default_2);  ne = full_default_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_67, 1, unsqueeze_4);  sub_67 = unsqueeze_4 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_470, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_470, -100);  view_470 = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_24 = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
        return (div_24, add_151, add_58)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (64, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (64, 128), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 102942720, device=device(type='cuda', index=0))
    reader.tensor(buf2, (50265, 512), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512, 512), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf4, (512,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512, 512), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf8, (512, 512), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf9, (512,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf10, (512, 512), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf11, (512,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512, 512), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf14, (512,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf16, (2048, 512), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf17, (2048,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512, 2048), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf21, (512,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512, 512), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512, 512), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512, 512), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf28, (512, 512), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf29, (512,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf31, (512,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf32, (2048, 512), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf33, (2048,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512, 2048), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf36, (512,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512, 512), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf40, (512, 512), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf41, (512,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512, 512), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf44, (512, 512), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf45, (512,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf48, (2048, 512), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf49, (2048,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512, 2048), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf52, (512,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf53, (512,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf54, (512, 512), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512, 512), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512, 512), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf59, (512,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf60, (512, 512), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf64, (2048, 512), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf65, (2048,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512, 2048), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512, 512), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512, 512), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512, 512), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512, 512), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf77, (512,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf78, (512,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf80, (2048, 512), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf81, (2048,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512, 2048), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf84, (512,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512, 512), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512, 512), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512, 512), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512, 512), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf94, (512,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf95, (512,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf96, (2048, 512), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf97, (2048,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512, 2048), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf102, (512, 512), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf103, (512,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512, 512), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512, 512), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf107, (512,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf108, (512, 512), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf112, (2048, 512), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf113, (2048,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512, 2048), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512, 512), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512, 512), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf121, (512,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf122, (512, 512), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf123, (512,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf124, (512, 512), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf125, (512,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf126, (512,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf127, (512,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf128, (2048, 512), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf129, (2048,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf130, (512, 2048), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf131, (512,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf133, (512,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf134, (512, 512), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf135, (512,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf136, (512,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512, 512), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512, 512), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf141, (512, 512), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf142, (512,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf143, (512, 512), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf144, (512,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf145, (512,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf146, (512,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf147, (512, 512), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf148, (512,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf149, (512, 512), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf150, (512,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf151, (512, 512), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf152, (512,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf153, (512, 512), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf154, (512,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf155, (512,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf156, (512,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf157, (2048, 512), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf158, (2048,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf159, (512, 2048), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf160, (512,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf161, (512,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512, 512), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf164, (512,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf165, (512, 512), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf166, (512,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf167, (512, 512), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf168, (512,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf169, (512, 512), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf170, (512,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf171, (512,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf172, (512,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf173, (512, 512), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512, 512), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf176, (512,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf177, (512, 512), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf178, (512,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf179, (512, 512), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf183, (2048, 512), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf184, (2048,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf185, (512, 2048), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf187, (512,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf188, (512,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf189, (512, 512), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf190, (512,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf191, (512, 512), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512, 512), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf194, (512,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf195, (512, 512), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf196, (512,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf197, (512,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf198, (512,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf199, (512, 512), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf200, (512,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf201, (512, 512), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf202, (512,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf203, (512, 512), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf205, (512, 512), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf206, (512,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf207, (512,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf208, (512,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf209, (2048, 512), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf210, (2048,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf211, (512, 2048), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf212, (512,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf213, (512,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf214, (512,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf215, (512, 512), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf216, (512,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf217, (512, 512), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf218, (512,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf219, (512, 512), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf220, (512,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512, 512), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf222, (512,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf224, (512,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf225, (512, 512), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf226, (512,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf227, (512, 512), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf228, (512,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf229, (512, 512), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf230, (512,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf231, (512, 512), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf232, (512,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf233, (512,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf234, (512,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf235, (2048, 512), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf236, (2048,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf237, (512, 2048), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf238, (512,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf239, (512,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf240, (512,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf241, (512, 512), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf242, (512,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf243, (512, 512), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf244, (512,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf245, (512, 512), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf246, (512,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf247, (512, 512), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf248, (512,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf249, (512,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf250, (512,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf251, (512, 512), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf252, (512,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf253, (512, 512), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf254, (512,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf255, (512, 512), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf256, (512,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf257, (512, 512), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf258, (512,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf259, (512,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf260, (512,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf261, (2048, 512), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf262, (2048,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf263, (512, 2048), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf264, (512,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf265, (512,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf266, (512,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf267, (512, 512), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf268, (512,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf269, (512, 512), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf270, (512,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf271, (512, 512), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf272, (512,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf273, (512, 512), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf274, (512,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf275, (512,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf276, (512,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf277, (512, 512), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf278, (512,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf279, (512, 512), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf280, (512,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf281, (512, 512), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf282, (512,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf283, (512, 512), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf284, (512,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf285, (512,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf286, (512,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf287, (2048, 512), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf288, (2048,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf289, (512, 2048), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf290, (512,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf291, (512,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf292, (512,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf293, (512, 512), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf294, (512,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf295, (512, 512), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf296, (512,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf297, (512, 512), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf298, (512,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf299, (512, 512), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf300, (512,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf301, (512,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf302, (512,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf303, (512, 512), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf304, (512,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf305, (512, 512), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf306, (512,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf307, (512, 512), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf308, (512,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf309, (512, 512), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf310, (512,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf311, (512,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf312, (512,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf313, (2048, 512), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf314, (2048,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf315, (512, 2048), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf316, (512,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf317, (512,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf318, (512,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf319, (512, 512), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf320, (512,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf321, (512, 512), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf322, (512,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf323, (512, 512), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf324, (512,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf325, (512, 512), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf326, (512,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf327, (512,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf328, (512,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf329, (512, 512), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf330, (512,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf331, (512, 512), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf332, (512,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf333, (512, 512), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf334, (512,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf335, (512, 512), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf336, (512,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf337, (512,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf338, (512,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf339, (2048, 512), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf340, (2048,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf341, (512, 2048), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf342, (512,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf343, (512,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf344, (512,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 201060, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1, 50265), is_leaf=True)  # arg345_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)