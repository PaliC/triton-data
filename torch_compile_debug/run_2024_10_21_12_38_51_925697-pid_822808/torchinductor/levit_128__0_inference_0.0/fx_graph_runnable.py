
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1):
        index = torch.ops.aten.index.Tensor(arg26_1, [None, arg27_1]);  arg26_1 = arg27_1 = None
        index_1 = torch.ops.aten.index.Tensor(arg48_1, [None, arg49_1]);  arg48_1 = arg49_1 = None
        index_2 = torch.ops.aten.index.Tensor(arg70_1, [None, arg71_1]);  arg70_1 = arg71_1 = None
        index_3 = torch.ops.aten.index.Tensor(arg92_1, [None, arg93_1]);  arg92_1 = arg93_1 = None
        index_4 = torch.ops.aten.index.Tensor(arg119_1, [None, arg120_1]);  arg119_1 = arg120_1 = None
        index_5 = torch.ops.aten.index.Tensor(arg141_1, [None, arg142_1]);  arg141_1 = arg142_1 = None
        index_6 = torch.ops.aten.index.Tensor(arg163_1, [None, arg164_1]);  arg163_1 = arg164_1 = None
        index_7 = torch.ops.aten.index.Tensor(arg185_1, [None, arg186_1]);  arg185_1 = arg186_1 = None
        index_8 = torch.ops.aten.index.Tensor(arg207_1, [None, arg208_1]);  arg207_1 = arg208_1 = None
        index_9 = torch.ops.aten.index.Tensor(arg234_1, [None, arg235_1]);  arg234_1 = arg235_1 = None
        index_10 = torch.ops.aten.index.Tensor(arg256_1, [None, arg257_1]);  arg256_1 = arg257_1 = None
        index_11 = torch.ops.aten.index.Tensor(arg278_1, [None, arg279_1]);  arg278_1 = arg279_1 = None
        index_12 = torch.ops.aten.index.Tensor(arg300_1, [None, arg301_1]);  arg300_1 = arg301_1 = None
        index_13 = torch.ops.aten.index.Tensor(arg322_1, [None, arg323_1]);  arg322_1 = arg323_1 = None
        convolution_4 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_200 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_237 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_35);  sub_78 = unsqueeze_35 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_37);  mul_238 = unsqueeze_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_201 = torch.ops.aten.add.Tensor(mul_239, unsqueeze_39);  mul_239 = unsqueeze_39 = None
        add_202 = torch.ops.aten.add.Tensor(add_201, 3)
        clamp_min_31 = torch.ops.aten.clamp_min.default(add_202, 0);  add_202 = None
        clamp_max_31 = torch.ops.aten.clamp_max.default(clamp_min_31, 6);  clamp_min_31 = None
        mul_240 = torch.ops.aten.mul.Tensor(add_201, clamp_max_31);  add_201 = clamp_max_31 = None
        div_46 = torch.ops.aten.div.Tensor(mul_240, 6);  mul_240 = None
        convolution_5 = torch.ops.aten.convolution.default(div_46, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_46 = arg6_1 = None
        add_203 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_203);  add_203 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_241 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(mul_241, -1);  mul_241 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_43);  sub_79 = unsqueeze_43 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_45);  mul_242 = unsqueeze_45 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_204 = torch.ops.aten.add.Tensor(mul_243, unsqueeze_47);  mul_243 = unsqueeze_47 = None
        add_205 = torch.ops.aten.add.Tensor(add_204, 3)
        clamp_min_32 = torch.ops.aten.clamp_min.default(add_205, 0);  add_205 = None
        clamp_max_32 = torch.ops.aten.clamp_max.default(clamp_min_32, 6);  clamp_min_32 = None
        mul_244 = torch.ops.aten.mul.Tensor(add_204, clamp_max_32);  add_204 = clamp_max_32 = None
        div_47 = torch.ops.aten.div.Tensor(mul_244, 6);  mul_244 = None
        convolution_6 = torch.ops.aten.convolution.default(div_47, arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_47 = arg11_1 = None
        add_206 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_245 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(mul_245, -1);  mul_245 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_51);  sub_80 = unsqueeze_51 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_53);  mul_246 = unsqueeze_53 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_207 = torch.ops.aten.add.Tensor(mul_247, unsqueeze_55);  mul_247 = unsqueeze_55 = None
        add_208 = torch.ops.aten.add.Tensor(add_207, 3)
        clamp_min_33 = torch.ops.aten.clamp_min.default(add_208, 0);  add_208 = None
        clamp_max_33 = torch.ops.aten.clamp_max.default(clamp_min_33, 6);  clamp_min_33 = None
        mul_248 = torch.ops.aten.mul.Tensor(add_207, clamp_max_33);  add_207 = clamp_max_33 = None
        div_48 = torch.ops.aten.div.Tensor(mul_248, 6);  mul_248 = None
        convolution_7 = torch.ops.aten.convolution.default(div_48, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_48 = arg16_1 = None
        add_209 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_209);  add_209 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_249 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        mul_250 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_59);  sub_81 = unsqueeze_59 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_61);  mul_250 = unsqueeze_61 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_210 = torch.ops.aten.add.Tensor(mul_251, unsqueeze_63);  mul_251 = unsqueeze_63 = None
        view_351 = torch.ops.aten.view.default(add_210, [8, 128, 196]);  add_210 = None
        permute_117 = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        permute_118 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        clone_83 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format)
        view_352 = torch.ops.aten.view.default(clone_83, [1568, 128]);  clone_83 = None
        mm_58 = torch.ops.aten.mm.default(view_352, permute_118);  view_352 = permute_118 = None
        view_353 = torch.ops.aten.view.default(mm_58, [8, 196, 256]);  mm_58 = None
        view_354 = torch.ops.aten.view.default(view_353, [1568, 256]);  view_353 = None
        add_211 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_252 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        sub_82 = torch.ops.aten.sub.Tensor(view_354, arg22_1);  view_354 = arg22_1 = None
        mul_253 = torch.ops.aten.mul.Tensor(sub_82, mul_252);  sub_82 = mul_252 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_253, arg24_1);  mul_253 = arg24_1 = None
        add_212 = torch.ops.aten.add.Tensor(mul_254, arg25_1);  mul_254 = arg25_1 = None
        view_355 = torch.ops.aten.view.default(add_212, [8, 196, 256]);  add_212 = None
        view_356 = torch.ops.aten.view.default(view_355, [8, 196, 4, -1]);  view_355 = None
        split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(view_356, [16, 16, 32], 3);  view_356 = None
        getitem_40 = split_with_sizes_14[0]
        getitem_41 = split_with_sizes_14[1]
        getitem_42 = split_with_sizes_14[2];  split_with_sizes_14 = None
        permute_119 = torch.ops.aten.permute.default(getitem_40, [0, 2, 1, 3]);  getitem_40 = None
        permute_120 = torch.ops.aten.permute.default(getitem_41, [0, 2, 3, 1]);  getitem_41 = None
        permute_121 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        expand_56 = torch.ops.aten.expand.default(permute_119, [8, 4, 196, 16]);  permute_119 = None
        clone_84 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_357 = torch.ops.aten.view.default(clone_84, [32, 196, 16]);  clone_84 = None
        expand_57 = torch.ops.aten.expand.default(permute_120, [8, 4, 16, 196]);  permute_120 = None
        clone_85 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_358 = torch.ops.aten.view.default(clone_85, [32, 16, 196]);  clone_85 = None
        bmm_28 = torch.ops.aten.bmm.default(view_357, view_358);  view_357 = view_358 = None
        view_359 = torch.ops.aten.view.default(bmm_28, [8, 4, 196, 196]);  bmm_28 = None
        mul_255 = torch.ops.aten.mul.Tensor(view_359, 0.25);  view_359 = None
        add_213 = torch.ops.aten.add.Tensor(mul_255, index);  mul_255 = None
        amax_14 = torch.ops.aten.amax.default(add_213, [-1], True)
        sub_83 = torch.ops.aten.sub.Tensor(add_213, amax_14);  add_213 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_83);  sub_83 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_49 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        expand_58 = torch.ops.aten.expand.default(div_49, [8, 4, 196, 196]);  div_49 = None
        view_360 = torch.ops.aten.view.default(expand_58, [32, 196, 196]);  expand_58 = None
        expand_59 = torch.ops.aten.expand.default(permute_121, [8, 4, 196, 32]);  permute_121 = None
        clone_86 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_361 = torch.ops.aten.view.default(clone_86, [32, 196, 32]);  clone_86 = None
        bmm_29 = torch.ops.aten.bmm.default(view_360, view_361);  view_360 = view_361 = None
        view_362 = torch.ops.aten.view.default(bmm_29, [8, 4, 196, 32]);  bmm_29 = None
        permute_122 = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_87 = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_363 = torch.ops.aten.view.default(clone_87, [8, 196, 128]);  clone_87 = None
        add_214 = torch.ops.aten.add.Tensor(view_363, 3)
        clamp_min_34 = torch.ops.aten.clamp_min.default(add_214, 0);  add_214 = None
        clamp_max_34 = torch.ops.aten.clamp_max.default(clamp_min_34, 6);  clamp_min_34 = None
        mul_256 = torch.ops.aten.mul.Tensor(view_363, clamp_max_34);  view_363 = clamp_max_34 = None
        div_50 = torch.ops.aten.div.Tensor(mul_256, 6);  mul_256 = None
        permute_123 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        view_364 = torch.ops.aten.view.default(div_50, [1568, 128]);  div_50 = None
        mm_59 = torch.ops.aten.mm.default(view_364, permute_123);  view_364 = permute_123 = None
        view_365 = torch.ops.aten.view.default(mm_59, [8, 196, 128]);  mm_59 = None
        view_366 = torch.ops.aten.view.default(view_365, [1568, 128]);  view_365 = None
        add_215 = torch.ops.aten.add.Tensor(arg30_1, 1e-05);  arg30_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_257 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        sub_84 = torch.ops.aten.sub.Tensor(view_366, arg29_1);  view_366 = arg29_1 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_84, mul_257);  sub_84 = mul_257 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, arg31_1);  mul_258 = arg31_1 = None
        add_216 = torch.ops.aten.add.Tensor(mul_259, arg32_1);  mul_259 = arg32_1 = None
        view_367 = torch.ops.aten.view.default(add_216, [8, 196, 128]);  add_216 = None
        add_217 = torch.ops.aten.add.Tensor(permute_117, view_367);  permute_117 = view_367 = None
        permute_124 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        clone_88 = torch.ops.aten.clone.default(add_217, memory_format = torch.contiguous_format)
        view_368 = torch.ops.aten.view.default(clone_88, [1568, 128]);  clone_88 = None
        mm_60 = torch.ops.aten.mm.default(view_368, permute_124);  view_368 = permute_124 = None
        view_369 = torch.ops.aten.view.default(mm_60, [8, 196, 256]);  mm_60 = None
        view_370 = torch.ops.aten.view.default(view_369, [1568, 256]);  view_369 = None
        add_218 = torch.ops.aten.add.Tensor(arg35_1, 1e-05);  arg35_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_260 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        sub_85 = torch.ops.aten.sub.Tensor(view_370, arg34_1);  view_370 = arg34_1 = None
        mul_261 = torch.ops.aten.mul.Tensor(sub_85, mul_260);  sub_85 = mul_260 = None
        mul_262 = torch.ops.aten.mul.Tensor(mul_261, arg36_1);  mul_261 = arg36_1 = None
        add_219 = torch.ops.aten.add.Tensor(mul_262, arg37_1);  mul_262 = arg37_1 = None
        view_371 = torch.ops.aten.view.default(add_219, [8, 196, 256]);  add_219 = None
        add_220 = torch.ops.aten.add.Tensor(view_371, 3)
        clamp_min_35 = torch.ops.aten.clamp_min.default(add_220, 0);  add_220 = None
        clamp_max_35 = torch.ops.aten.clamp_max.default(clamp_min_35, 6);  clamp_min_35 = None
        mul_263 = torch.ops.aten.mul.Tensor(view_371, clamp_max_35);  view_371 = clamp_max_35 = None
        div_51 = torch.ops.aten.div.Tensor(mul_263, 6);  mul_263 = None
        permute_125 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        view_372 = torch.ops.aten.view.default(div_51, [1568, 256]);  div_51 = None
        mm_61 = torch.ops.aten.mm.default(view_372, permute_125);  view_372 = permute_125 = None
        view_373 = torch.ops.aten.view.default(mm_61, [8, 196, 128]);  mm_61 = None
        view_374 = torch.ops.aten.view.default(view_373, [1568, 128]);  view_373 = None
        add_221 = torch.ops.aten.add.Tensor(arg40_1, 1e-05);  arg40_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_264 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        sub_86 = torch.ops.aten.sub.Tensor(view_374, arg39_1);  view_374 = arg39_1 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_86, mul_264);  sub_86 = mul_264 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, arg41_1);  mul_265 = arg41_1 = None
        add_222 = torch.ops.aten.add.Tensor(mul_266, arg42_1);  mul_266 = arg42_1 = None
        view_375 = torch.ops.aten.view.default(add_222, [8, 196, 128]);  add_222 = None
        add_223 = torch.ops.aten.add.Tensor(add_217, view_375);  add_217 = view_375 = None
        permute_126 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        clone_90 = torch.ops.aten.clone.default(add_223, memory_format = torch.contiguous_format)
        view_376 = torch.ops.aten.view.default(clone_90, [1568, 128]);  clone_90 = None
        mm_62 = torch.ops.aten.mm.default(view_376, permute_126);  view_376 = permute_126 = None
        view_377 = torch.ops.aten.view.default(mm_62, [8, 196, 256]);  mm_62 = None
        view_378 = torch.ops.aten.view.default(view_377, [1568, 256]);  view_377 = None
        add_224 = torch.ops.aten.add.Tensor(arg45_1, 1e-05);  arg45_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_267 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        sub_87 = torch.ops.aten.sub.Tensor(view_378, arg44_1);  view_378 = arg44_1 = None
        mul_268 = torch.ops.aten.mul.Tensor(sub_87, mul_267);  sub_87 = mul_267 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, arg46_1);  mul_268 = arg46_1 = None
        add_225 = torch.ops.aten.add.Tensor(mul_269, arg47_1);  mul_269 = arg47_1 = None
        view_379 = torch.ops.aten.view.default(add_225, [8, 196, 256]);  add_225 = None
        view_380 = torch.ops.aten.view.default(view_379, [8, 196, 4, -1]);  view_379 = None
        split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(view_380, [16, 16, 32], 3);  view_380 = None
        getitem_43 = split_with_sizes_15[0]
        getitem_44 = split_with_sizes_15[1]
        getitem_45 = split_with_sizes_15[2];  split_with_sizes_15 = None
        permute_127 = torch.ops.aten.permute.default(getitem_43, [0, 2, 1, 3]);  getitem_43 = None
        permute_128 = torch.ops.aten.permute.default(getitem_44, [0, 2, 3, 1]);  getitem_44 = None
        permute_129 = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None
        expand_60 = torch.ops.aten.expand.default(permute_127, [8, 4, 196, 16]);  permute_127 = None
        clone_91 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_381 = torch.ops.aten.view.default(clone_91, [32, 196, 16]);  clone_91 = None
        expand_61 = torch.ops.aten.expand.default(permute_128, [8, 4, 16, 196]);  permute_128 = None
        clone_92 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_382 = torch.ops.aten.view.default(clone_92, [32, 16, 196]);  clone_92 = None
        bmm_30 = torch.ops.aten.bmm.default(view_381, view_382);  view_381 = view_382 = None
        view_383 = torch.ops.aten.view.default(bmm_30, [8, 4, 196, 196]);  bmm_30 = None
        mul_270 = torch.ops.aten.mul.Tensor(view_383, 0.25);  view_383 = None
        add_226 = torch.ops.aten.add.Tensor(mul_270, index_1);  mul_270 = None
        amax_15 = torch.ops.aten.amax.default(add_226, [-1], True)
        sub_88 = torch.ops.aten.sub.Tensor(add_226, amax_15);  add_226 = amax_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_88);  sub_88 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_52 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        expand_62 = torch.ops.aten.expand.default(div_52, [8, 4, 196, 196]);  div_52 = None
        view_384 = torch.ops.aten.view.default(expand_62, [32, 196, 196]);  expand_62 = None
        expand_63 = torch.ops.aten.expand.default(permute_129, [8, 4, 196, 32]);  permute_129 = None
        clone_93 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_385 = torch.ops.aten.view.default(clone_93, [32, 196, 32]);  clone_93 = None
        bmm_31 = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
        view_386 = torch.ops.aten.view.default(bmm_31, [8, 4, 196, 32]);  bmm_31 = None
        permute_130 = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        clone_94 = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_387 = torch.ops.aten.view.default(clone_94, [8, 196, 128]);  clone_94 = None
        add_227 = torch.ops.aten.add.Tensor(view_387, 3)
        clamp_min_36 = torch.ops.aten.clamp_min.default(add_227, 0);  add_227 = None
        clamp_max_36 = torch.ops.aten.clamp_max.default(clamp_min_36, 6);  clamp_min_36 = None
        mul_271 = torch.ops.aten.mul.Tensor(view_387, clamp_max_36);  view_387 = clamp_max_36 = None
        div_53 = torch.ops.aten.div.Tensor(mul_271, 6);  mul_271 = None
        permute_131 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        view_388 = torch.ops.aten.view.default(div_53, [1568, 128]);  div_53 = None
        mm_63 = torch.ops.aten.mm.default(view_388, permute_131);  view_388 = permute_131 = None
        view_389 = torch.ops.aten.view.default(mm_63, [8, 196, 128]);  mm_63 = None
        view_390 = torch.ops.aten.view.default(view_389, [1568, 128]);  view_389 = None
        add_228 = torch.ops.aten.add.Tensor(arg52_1, 1e-05);  arg52_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_228);  add_228 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_272 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        sub_89 = torch.ops.aten.sub.Tensor(view_390, arg51_1);  view_390 = arg51_1 = None
        mul_273 = torch.ops.aten.mul.Tensor(sub_89, mul_272);  sub_89 = mul_272 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_273, arg53_1);  mul_273 = arg53_1 = None
        add_229 = torch.ops.aten.add.Tensor(mul_274, arg54_1);  mul_274 = arg54_1 = None
        view_391 = torch.ops.aten.view.default(add_229, [8, 196, 128]);  add_229 = None
        add_230 = torch.ops.aten.add.Tensor(add_223, view_391);  add_223 = view_391 = None
        permute_132 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        clone_95 = torch.ops.aten.clone.default(add_230, memory_format = torch.contiguous_format)
        view_392 = torch.ops.aten.view.default(clone_95, [1568, 128]);  clone_95 = None
        mm_64 = torch.ops.aten.mm.default(view_392, permute_132);  view_392 = permute_132 = None
        view_393 = torch.ops.aten.view.default(mm_64, [8, 196, 256]);  mm_64 = None
        view_394 = torch.ops.aten.view.default(view_393, [1568, 256]);  view_393 = None
        add_231 = torch.ops.aten.add.Tensor(arg57_1, 1e-05);  arg57_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_275 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        sub_90 = torch.ops.aten.sub.Tensor(view_394, arg56_1);  view_394 = arg56_1 = None
        mul_276 = torch.ops.aten.mul.Tensor(sub_90, mul_275);  sub_90 = mul_275 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, arg58_1);  mul_276 = arg58_1 = None
        add_232 = torch.ops.aten.add.Tensor(mul_277, arg59_1);  mul_277 = arg59_1 = None
        view_395 = torch.ops.aten.view.default(add_232, [8, 196, 256]);  add_232 = None
        add_233 = torch.ops.aten.add.Tensor(view_395, 3)
        clamp_min_37 = torch.ops.aten.clamp_min.default(add_233, 0);  add_233 = None
        clamp_max_37 = torch.ops.aten.clamp_max.default(clamp_min_37, 6);  clamp_min_37 = None
        mul_278 = torch.ops.aten.mul.Tensor(view_395, clamp_max_37);  view_395 = clamp_max_37 = None
        div_54 = torch.ops.aten.div.Tensor(mul_278, 6);  mul_278 = None
        permute_133 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        view_396 = torch.ops.aten.view.default(div_54, [1568, 256]);  div_54 = None
        mm_65 = torch.ops.aten.mm.default(view_396, permute_133);  view_396 = permute_133 = None
        view_397 = torch.ops.aten.view.default(mm_65, [8, 196, 128]);  mm_65 = None
        view_398 = torch.ops.aten.view.default(view_397, [1568, 128]);  view_397 = None
        add_234 = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_279 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        sub_91 = torch.ops.aten.sub.Tensor(view_398, arg61_1);  view_398 = arg61_1 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_91, mul_279);  sub_91 = mul_279 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, arg63_1);  mul_280 = arg63_1 = None
        add_235 = torch.ops.aten.add.Tensor(mul_281, arg64_1);  mul_281 = arg64_1 = None
        view_399 = torch.ops.aten.view.default(add_235, [8, 196, 128]);  add_235 = None
        add_236 = torch.ops.aten.add.Tensor(add_230, view_399);  add_230 = view_399 = None
        permute_134 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        clone_97 = torch.ops.aten.clone.default(add_236, memory_format = torch.contiguous_format)
        view_400 = torch.ops.aten.view.default(clone_97, [1568, 128]);  clone_97 = None
        mm_66 = torch.ops.aten.mm.default(view_400, permute_134);  view_400 = permute_134 = None
        view_401 = torch.ops.aten.view.default(mm_66, [8, 196, 256]);  mm_66 = None
        view_402 = torch.ops.aten.view.default(view_401, [1568, 256]);  view_401 = None
        add_237 = torch.ops.aten.add.Tensor(arg67_1, 1e-05);  arg67_1 = None
        sqrt_76 = torch.ops.aten.sqrt.default(add_237);  add_237 = None
        reciprocal_76 = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_282 = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        sub_92 = torch.ops.aten.sub.Tensor(view_402, arg66_1);  view_402 = arg66_1 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_92, mul_282);  sub_92 = mul_282 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, arg68_1);  mul_283 = arg68_1 = None
        add_238 = torch.ops.aten.add.Tensor(mul_284, arg69_1);  mul_284 = arg69_1 = None
        view_403 = torch.ops.aten.view.default(add_238, [8, 196, 256]);  add_238 = None
        view_404 = torch.ops.aten.view.default(view_403, [8, 196, 4, -1]);  view_403 = None
        split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(view_404, [16, 16, 32], 3);  view_404 = None
        getitem_46 = split_with_sizes_16[0]
        getitem_47 = split_with_sizes_16[1]
        getitem_48 = split_with_sizes_16[2];  split_with_sizes_16 = None
        permute_135 = torch.ops.aten.permute.default(getitem_46, [0, 2, 1, 3]);  getitem_46 = None
        permute_136 = torch.ops.aten.permute.default(getitem_47, [0, 2, 3, 1]);  getitem_47 = None
        permute_137 = torch.ops.aten.permute.default(getitem_48, [0, 2, 1, 3]);  getitem_48 = None
        expand_64 = torch.ops.aten.expand.default(permute_135, [8, 4, 196, 16]);  permute_135 = None
        clone_98 = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_405 = torch.ops.aten.view.default(clone_98, [32, 196, 16]);  clone_98 = None
        expand_65 = torch.ops.aten.expand.default(permute_136, [8, 4, 16, 196]);  permute_136 = None
        clone_99 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_406 = torch.ops.aten.view.default(clone_99, [32, 16, 196]);  clone_99 = None
        bmm_32 = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407 = torch.ops.aten.view.default(bmm_32, [8, 4, 196, 196]);  bmm_32 = None
        mul_285 = torch.ops.aten.mul.Tensor(view_407, 0.25);  view_407 = None
        add_239 = torch.ops.aten.add.Tensor(mul_285, index_2);  mul_285 = None
        amax_16 = torch.ops.aten.amax.default(add_239, [-1], True)
        sub_93 = torch.ops.aten.sub.Tensor(add_239, amax_16);  add_239 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_55 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        expand_66 = torch.ops.aten.expand.default(div_55, [8, 4, 196, 196]);  div_55 = None
        view_408 = torch.ops.aten.view.default(expand_66, [32, 196, 196]);  expand_66 = None
        expand_67 = torch.ops.aten.expand.default(permute_137, [8, 4, 196, 32]);  permute_137 = None
        clone_100 = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_409 = torch.ops.aten.view.default(clone_100, [32, 196, 32]);  clone_100 = None
        bmm_33 = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410 = torch.ops.aten.view.default(bmm_33, [8, 4, 196, 32]);  bmm_33 = None
        permute_138 = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_101 = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
        view_411 = torch.ops.aten.view.default(clone_101, [8, 196, 128]);  clone_101 = None
        add_240 = torch.ops.aten.add.Tensor(view_411, 3)
        clamp_min_38 = torch.ops.aten.clamp_min.default(add_240, 0);  add_240 = None
        clamp_max_38 = torch.ops.aten.clamp_max.default(clamp_min_38, 6);  clamp_min_38 = None
        mul_286 = torch.ops.aten.mul.Tensor(view_411, clamp_max_38);  view_411 = clamp_max_38 = None
        div_56 = torch.ops.aten.div.Tensor(mul_286, 6);  mul_286 = None
        permute_139 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        view_412 = torch.ops.aten.view.default(div_56, [1568, 128]);  div_56 = None
        mm_67 = torch.ops.aten.mm.default(view_412, permute_139);  view_412 = permute_139 = None
        view_413 = torch.ops.aten.view.default(mm_67, [8, 196, 128]);  mm_67 = None
        view_414 = torch.ops.aten.view.default(view_413, [1568, 128]);  view_413 = None
        add_241 = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_77 = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_77 = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_287 = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        sub_94 = torch.ops.aten.sub.Tensor(view_414, arg73_1);  view_414 = arg73_1 = None
        mul_288 = torch.ops.aten.mul.Tensor(sub_94, mul_287);  sub_94 = mul_287 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_288, arg75_1);  mul_288 = arg75_1 = None
        add_242 = torch.ops.aten.add.Tensor(mul_289, arg76_1);  mul_289 = arg76_1 = None
        view_415 = torch.ops.aten.view.default(add_242, [8, 196, 128]);  add_242 = None
        add_243 = torch.ops.aten.add.Tensor(add_236, view_415);  add_236 = view_415 = None
        permute_140 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        clone_102 = torch.ops.aten.clone.default(add_243, memory_format = torch.contiguous_format)
        view_416 = torch.ops.aten.view.default(clone_102, [1568, 128]);  clone_102 = None
        mm_68 = torch.ops.aten.mm.default(view_416, permute_140);  view_416 = permute_140 = None
        view_417 = torch.ops.aten.view.default(mm_68, [8, 196, 256]);  mm_68 = None
        view_418 = torch.ops.aten.view.default(view_417, [1568, 256]);  view_417 = None
        add_244 = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_78 = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_78 = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_290 = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        sub_95 = torch.ops.aten.sub.Tensor(view_418, arg78_1);  view_418 = arg78_1 = None
        mul_291 = torch.ops.aten.mul.Tensor(sub_95, mul_290);  sub_95 = mul_290 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_291, arg80_1);  mul_291 = arg80_1 = None
        add_245 = torch.ops.aten.add.Tensor(mul_292, arg81_1);  mul_292 = arg81_1 = None
        view_419 = torch.ops.aten.view.default(add_245, [8, 196, 256]);  add_245 = None
        add_246 = torch.ops.aten.add.Tensor(view_419, 3)
        clamp_min_39 = torch.ops.aten.clamp_min.default(add_246, 0);  add_246 = None
        clamp_max_39 = torch.ops.aten.clamp_max.default(clamp_min_39, 6);  clamp_min_39 = None
        mul_293 = torch.ops.aten.mul.Tensor(view_419, clamp_max_39);  view_419 = clamp_max_39 = None
        div_57 = torch.ops.aten.div.Tensor(mul_293, 6);  mul_293 = None
        permute_141 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        view_420 = torch.ops.aten.view.default(div_57, [1568, 256]);  div_57 = None
        mm_69 = torch.ops.aten.mm.default(view_420, permute_141);  view_420 = permute_141 = None
        view_421 = torch.ops.aten.view.default(mm_69, [8, 196, 128]);  mm_69 = None
        view_422 = torch.ops.aten.view.default(view_421, [1568, 128]);  view_421 = None
        add_247 = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_79 = torch.ops.aten.sqrt.default(add_247);  add_247 = None
        reciprocal_79 = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_294 = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        sub_96 = torch.ops.aten.sub.Tensor(view_422, arg83_1);  view_422 = arg83_1 = None
        mul_295 = torch.ops.aten.mul.Tensor(sub_96, mul_294);  sub_96 = mul_294 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, arg85_1);  mul_295 = arg85_1 = None
        add_248 = torch.ops.aten.add.Tensor(mul_296, arg86_1);  mul_296 = arg86_1 = None
        view_423 = torch.ops.aten.view.default(add_248, [8, 196, 128]);  add_248 = None
        add_249 = torch.ops.aten.add.Tensor(add_243, view_423);  add_243 = view_423 = None
        permute_142 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        clone_104 = torch.ops.aten.clone.default(add_249, memory_format = torch.contiguous_format)
        view_424 = torch.ops.aten.view.default(clone_104, [1568, 128]);  clone_104 = None
        mm_70 = torch.ops.aten.mm.default(view_424, permute_142);  view_424 = permute_142 = None
        view_425 = torch.ops.aten.view.default(mm_70, [8, 196, 256]);  mm_70 = None
        view_426 = torch.ops.aten.view.default(view_425, [1568, 256]);  view_425 = None
        add_250 = torch.ops.aten.add.Tensor(arg89_1, 1e-05);  arg89_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_297 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        sub_97 = torch.ops.aten.sub.Tensor(view_426, arg88_1);  view_426 = arg88_1 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_97, mul_297);  sub_97 = mul_297 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, arg90_1);  mul_298 = arg90_1 = None
        add_251 = torch.ops.aten.add.Tensor(mul_299, arg91_1);  mul_299 = arg91_1 = None
        view_427 = torch.ops.aten.view.default(add_251, [8, 196, 256]);  add_251 = None
        view_428 = torch.ops.aten.view.default(view_427, [8, 196, 4, -1]);  view_427 = None
        split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(view_428, [16, 16, 32], 3);  view_428 = None
        getitem_49 = split_with_sizes_17[0]
        getitem_50 = split_with_sizes_17[1]
        getitem_51 = split_with_sizes_17[2];  split_with_sizes_17 = None
        permute_143 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        permute_144 = torch.ops.aten.permute.default(getitem_50, [0, 2, 3, 1]);  getitem_50 = None
        permute_145 = torch.ops.aten.permute.default(getitem_51, [0, 2, 1, 3]);  getitem_51 = None
        expand_68 = torch.ops.aten.expand.default(permute_143, [8, 4, 196, 16]);  permute_143 = None
        clone_105 = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_429 = torch.ops.aten.view.default(clone_105, [32, 196, 16]);  clone_105 = None
        expand_69 = torch.ops.aten.expand.default(permute_144, [8, 4, 16, 196]);  permute_144 = None
        clone_106 = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_430 = torch.ops.aten.view.default(clone_106, [32, 16, 196]);  clone_106 = None
        bmm_34 = torch.ops.aten.bmm.default(view_429, view_430);  view_429 = view_430 = None
        view_431 = torch.ops.aten.view.default(bmm_34, [8, 4, 196, 196]);  bmm_34 = None
        mul_300 = torch.ops.aten.mul.Tensor(view_431, 0.25);  view_431 = None
        add_252 = torch.ops.aten.add.Tensor(mul_300, index_3);  mul_300 = None
        amax_17 = torch.ops.aten.amax.default(add_252, [-1], True)
        sub_98 = torch.ops.aten.sub.Tensor(add_252, amax_17);  add_252 = amax_17 = None
        exp_17 = torch.ops.aten.exp.default(sub_98);  sub_98 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_58 = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        expand_70 = torch.ops.aten.expand.default(div_58, [8, 4, 196, 196]);  div_58 = None
        view_432 = torch.ops.aten.view.default(expand_70, [32, 196, 196]);  expand_70 = None
        expand_71 = torch.ops.aten.expand.default(permute_145, [8, 4, 196, 32]);  permute_145 = None
        clone_107 = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_433 = torch.ops.aten.view.default(clone_107, [32, 196, 32]);  clone_107 = None
        bmm_35 = torch.ops.aten.bmm.default(view_432, view_433);  view_432 = view_433 = None
        view_434 = torch.ops.aten.view.default(bmm_35, [8, 4, 196, 32]);  bmm_35 = None
        permute_146 = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        clone_108 = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_435 = torch.ops.aten.view.default(clone_108, [8, 196, 128]);  clone_108 = None
        add_253 = torch.ops.aten.add.Tensor(view_435, 3)
        clamp_min_40 = torch.ops.aten.clamp_min.default(add_253, 0);  add_253 = None
        clamp_max_40 = torch.ops.aten.clamp_max.default(clamp_min_40, 6);  clamp_min_40 = None
        mul_301 = torch.ops.aten.mul.Tensor(view_435, clamp_max_40);  view_435 = clamp_max_40 = None
        div_59 = torch.ops.aten.div.Tensor(mul_301, 6);  mul_301 = None
        permute_147 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        view_436 = torch.ops.aten.view.default(div_59, [1568, 128]);  div_59 = None
        mm_71 = torch.ops.aten.mm.default(view_436, permute_147);  view_436 = permute_147 = None
        view_437 = torch.ops.aten.view.default(mm_71, [8, 196, 128]);  mm_71 = None
        view_438 = torch.ops.aten.view.default(view_437, [1568, 128]);  view_437 = None
        add_254 = torch.ops.aten.add.Tensor(arg96_1, 1e-05);  arg96_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_302 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        sub_99 = torch.ops.aten.sub.Tensor(view_438, arg95_1);  view_438 = arg95_1 = None
        mul_303 = torch.ops.aten.mul.Tensor(sub_99, mul_302);  sub_99 = mul_302 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_303, arg97_1);  mul_303 = arg97_1 = None
        add_255 = torch.ops.aten.add.Tensor(mul_304, arg98_1);  mul_304 = arg98_1 = None
        view_439 = torch.ops.aten.view.default(add_255, [8, 196, 128]);  add_255 = None
        add_256 = torch.ops.aten.add.Tensor(add_249, view_439);  add_249 = view_439 = None
        permute_148 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        clone_109 = torch.ops.aten.clone.default(add_256, memory_format = torch.contiguous_format)
        view_440 = torch.ops.aten.view.default(clone_109, [1568, 128]);  clone_109 = None
        mm_72 = torch.ops.aten.mm.default(view_440, permute_148);  view_440 = permute_148 = None
        view_441 = torch.ops.aten.view.default(mm_72, [8, 196, 256]);  mm_72 = None
        view_442 = torch.ops.aten.view.default(view_441, [1568, 256]);  view_441 = None
        add_257 = torch.ops.aten.add.Tensor(arg101_1, 1e-05);  arg101_1 = None
        sqrt_82 = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_82 = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_305 = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        sub_100 = torch.ops.aten.sub.Tensor(view_442, arg100_1);  view_442 = arg100_1 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_100, mul_305);  sub_100 = mul_305 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_306, arg102_1);  mul_306 = arg102_1 = None
        add_258 = torch.ops.aten.add.Tensor(mul_307, arg103_1);  mul_307 = arg103_1 = None
        view_443 = torch.ops.aten.view.default(add_258, [8, 196, 256]);  add_258 = None
        add_259 = torch.ops.aten.add.Tensor(view_443, 3)
        clamp_min_41 = torch.ops.aten.clamp_min.default(add_259, 0);  add_259 = None
        clamp_max_41 = torch.ops.aten.clamp_max.default(clamp_min_41, 6);  clamp_min_41 = None
        mul_308 = torch.ops.aten.mul.Tensor(view_443, clamp_max_41);  view_443 = clamp_max_41 = None
        div_60 = torch.ops.aten.div.Tensor(mul_308, 6);  mul_308 = None
        permute_149 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        view_444 = torch.ops.aten.view.default(div_60, [1568, 256]);  div_60 = None
        mm_73 = torch.ops.aten.mm.default(view_444, permute_149);  view_444 = permute_149 = None
        view_445 = torch.ops.aten.view.default(mm_73, [8, 196, 128]);  mm_73 = None
        view_446 = torch.ops.aten.view.default(view_445, [1568, 128]);  view_445 = None
        add_260 = torch.ops.aten.add.Tensor(arg106_1, 1e-05);  arg106_1 = None
        sqrt_83 = torch.ops.aten.sqrt.default(add_260);  add_260 = None
        reciprocal_83 = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_309 = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        sub_101 = torch.ops.aten.sub.Tensor(view_446, arg105_1);  view_446 = arg105_1 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_101, mul_309);  sub_101 = mul_309 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, arg107_1);  mul_310 = arg107_1 = None
        add_261 = torch.ops.aten.add.Tensor(mul_311, arg108_1);  mul_311 = arg108_1 = None
        view_447 = torch.ops.aten.view.default(add_261, [8, 196, 128]);  add_261 = None
        add_262 = torch.ops.aten.add.Tensor(add_256, view_447);  add_256 = view_447 = None
        permute_150 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        clone_111 = torch.ops.aten.clone.default(add_262, memory_format = torch.contiguous_format)
        view_448 = torch.ops.aten.view.default(clone_111, [1568, 128]);  clone_111 = None
        mm_74 = torch.ops.aten.mm.default(view_448, permute_150);  view_448 = permute_150 = None
        view_449 = torch.ops.aten.view.default(mm_74, [8, 196, 640]);  mm_74 = None
        view_450 = torch.ops.aten.view.default(view_449, [1568, 640]);  view_449 = None
        add_263 = torch.ops.aten.add.Tensor(arg111_1, 1e-05);  arg111_1 = None
        sqrt_84 = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_84 = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_312 = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        sub_102 = torch.ops.aten.sub.Tensor(view_450, arg110_1);  view_450 = arg110_1 = None
        mul_313 = torch.ops.aten.mul.Tensor(sub_102, mul_312);  sub_102 = mul_312 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, arg112_1);  mul_313 = arg112_1 = None
        add_264 = torch.ops.aten.add.Tensor(mul_314, arg113_1);  mul_314 = arg113_1 = None
        view_451 = torch.ops.aten.view.default(add_264, [8, 196, 640]);  add_264 = None
        view_452 = torch.ops.aten.view.default(view_451, [8, 196, 8, -1]);  view_451 = None
        split_with_sizes_18 = torch.ops.aten.split_with_sizes.default(view_452, [16, 64], 3);  view_452 = None
        getitem_52 = split_with_sizes_18[0]
        getitem_53 = split_with_sizes_18[1];  split_with_sizes_18 = None
        permute_151 = torch.ops.aten.permute.default(getitem_52, [0, 2, 3, 1]);  getitem_52 = None
        permute_152 = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        view_453 = torch.ops.aten.view.default(add_262, [8, 14, 14, 128]);  add_262 = None
        slice_22 = torch.ops.aten.slice.Tensor(view_453, 1, 0, 9223372036854775807, 2);  view_453 = None
        slice_23 = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 9223372036854775807, 2);  slice_22 = None
        clone_112 = torch.ops.aten.clone.default(slice_23, memory_format = torch.contiguous_format);  slice_23 = None
        view_454 = torch.ops.aten.view.default(clone_112, [8, 49, 128]);  clone_112 = None
        permute_153 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        view_455 = torch.ops.aten.view.default(view_454, [392, 128]);  view_454 = None
        mm_75 = torch.ops.aten.mm.default(view_455, permute_153);  view_455 = permute_153 = None
        view_456 = torch.ops.aten.view.default(mm_75, [8, 49, 128]);  mm_75 = None
        view_457 = torch.ops.aten.view.default(view_456, [392, 128]);  view_456 = None
        add_265 = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_315 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        sub_103 = torch.ops.aten.sub.Tensor(view_457, arg115_1);  view_457 = arg115_1 = None
        mul_316 = torch.ops.aten.mul.Tensor(sub_103, mul_315);  sub_103 = mul_315 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, arg117_1);  mul_316 = arg117_1 = None
        add_266 = torch.ops.aten.add.Tensor(mul_317, arg118_1);  mul_317 = arg118_1 = None
        view_458 = torch.ops.aten.view.default(add_266, [8, 49, 128]);  add_266 = None
        view_459 = torch.ops.aten.view.default(view_458, [8, -1, 8, 16]);  view_458 = None
        permute_154 = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        expand_72 = torch.ops.aten.expand.default(permute_154, [8, 8, 49, 16]);  permute_154 = None
        clone_113 = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_460 = torch.ops.aten.view.default(clone_113, [64, 49, 16]);  clone_113 = None
        expand_73 = torch.ops.aten.expand.default(permute_151, [8, 8, 16, 196]);  permute_151 = None
        clone_114 = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_461 = torch.ops.aten.view.default(clone_114, [64, 16, 196]);  clone_114 = None
        bmm_36 = torch.ops.aten.bmm.default(view_460, view_461);  view_460 = view_461 = None
        view_462 = torch.ops.aten.view.default(bmm_36, [8, 8, 49, 196]);  bmm_36 = None
        mul_318 = torch.ops.aten.mul.Tensor(view_462, 0.25);  view_462 = None
        add_267 = torch.ops.aten.add.Tensor(mul_318, index_4);  mul_318 = None
        amax_18 = torch.ops.aten.amax.default(add_267, [-1], True)
        sub_104 = torch.ops.aten.sub.Tensor(add_267, amax_18);  add_267 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_104);  sub_104 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_61 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        expand_74 = torch.ops.aten.expand.default(div_61, [8, 8, 49, 196]);  div_61 = None
        view_463 = torch.ops.aten.view.default(expand_74, [64, 49, 196]);  expand_74 = None
        expand_75 = torch.ops.aten.expand.default(permute_152, [8, 8, 196, 64]);  permute_152 = None
        clone_115 = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_464 = torch.ops.aten.view.default(clone_115, [64, 196, 64]);  clone_115 = None
        bmm_37 = torch.ops.aten.bmm.default(view_463, view_464);  view_463 = view_464 = None
        view_465 = torch.ops.aten.view.default(bmm_37, [8, 8, 49, 64]);  bmm_37 = None
        permute_155 = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        clone_116 = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_466 = torch.ops.aten.view.default(clone_116, [8, 49, 512]);  clone_116 = None
        add_268 = torch.ops.aten.add.Tensor(view_466, 3)
        clamp_min_42 = torch.ops.aten.clamp_min.default(add_268, 0);  add_268 = None
        clamp_max_42 = torch.ops.aten.clamp_max.default(clamp_min_42, 6);  clamp_min_42 = None
        mul_319 = torch.ops.aten.mul.Tensor(view_466, clamp_max_42);  view_466 = clamp_max_42 = None
        div_62 = torch.ops.aten.div.Tensor(mul_319, 6);  mul_319 = None
        permute_156 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        view_467 = torch.ops.aten.view.default(div_62, [392, 512]);  div_62 = None
        mm_76 = torch.ops.aten.mm.default(view_467, permute_156);  view_467 = permute_156 = None
        view_468 = torch.ops.aten.view.default(mm_76, [8, 49, 256]);  mm_76 = None
        view_469 = torch.ops.aten.view.default(view_468, [392, 256]);  view_468 = None
        add_269 = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_269);  add_269 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_320 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        sub_105 = torch.ops.aten.sub.Tensor(view_469, arg122_1);  view_469 = arg122_1 = None
        mul_321 = torch.ops.aten.mul.Tensor(sub_105, mul_320);  sub_105 = mul_320 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_321, arg124_1);  mul_321 = arg124_1 = None
        add_270 = torch.ops.aten.add.Tensor(mul_322, arg125_1);  mul_322 = arg125_1 = None
        view_470 = torch.ops.aten.view.default(add_270, [8, 49, 256]);  add_270 = None
        permute_157 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        view_471 = torch.ops.aten.view.default(view_470, [392, 256])
        mm_77 = torch.ops.aten.mm.default(view_471, permute_157);  view_471 = permute_157 = None
        view_472 = torch.ops.aten.view.default(mm_77, [8, 49, 512]);  mm_77 = None
        view_473 = torch.ops.aten.view.default(view_472, [392, 512]);  view_472 = None
        add_271 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_271);  add_271 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_323 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        sub_106 = torch.ops.aten.sub.Tensor(view_473, arg127_1);  view_473 = arg127_1 = None
        mul_324 = torch.ops.aten.mul.Tensor(sub_106, mul_323);  sub_106 = mul_323 = None
        mul_325 = torch.ops.aten.mul.Tensor(mul_324, arg129_1);  mul_324 = arg129_1 = None
        add_272 = torch.ops.aten.add.Tensor(mul_325, arg130_1);  mul_325 = arg130_1 = None
        view_474 = torch.ops.aten.view.default(add_272, [8, 49, 512]);  add_272 = None
        add_273 = torch.ops.aten.add.Tensor(view_474, 3)
        clamp_min_43 = torch.ops.aten.clamp_min.default(add_273, 0);  add_273 = None
        clamp_max_43 = torch.ops.aten.clamp_max.default(clamp_min_43, 6);  clamp_min_43 = None
        mul_326 = torch.ops.aten.mul.Tensor(view_474, clamp_max_43);  view_474 = clamp_max_43 = None
        div_63 = torch.ops.aten.div.Tensor(mul_326, 6);  mul_326 = None
        permute_158 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        view_475 = torch.ops.aten.view.default(div_63, [392, 512]);  div_63 = None
        mm_78 = torch.ops.aten.mm.default(view_475, permute_158);  view_475 = permute_158 = None
        view_476 = torch.ops.aten.view.default(mm_78, [8, 49, 256]);  mm_78 = None
        view_477 = torch.ops.aten.view.default(view_476, [392, 256]);  view_476 = None
        add_274 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_274);  add_274 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_327 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        sub_107 = torch.ops.aten.sub.Tensor(view_477, arg132_1);  view_477 = arg132_1 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_107, mul_327);  sub_107 = mul_327 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, arg134_1);  mul_328 = arg134_1 = None
        add_275 = torch.ops.aten.add.Tensor(mul_329, arg135_1);  mul_329 = arg135_1 = None
        view_478 = torch.ops.aten.view.default(add_275, [8, 49, 256]);  add_275 = None
        add_276 = torch.ops.aten.add.Tensor(view_470, view_478);  view_470 = view_478 = None
        permute_159 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        view_479 = torch.ops.aten.view.default(add_276, [392, 256])
        mm_79 = torch.ops.aten.mm.default(view_479, permute_159);  view_479 = permute_159 = None
        view_480 = torch.ops.aten.view.default(mm_79, [8, 49, 512]);  mm_79 = None
        view_481 = torch.ops.aten.view.default(view_480, [392, 512]);  view_480 = None
        add_277 = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_330 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        sub_108 = torch.ops.aten.sub.Tensor(view_481, arg137_1);  view_481 = arg137_1 = None
        mul_331 = torch.ops.aten.mul.Tensor(sub_108, mul_330);  sub_108 = mul_330 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, arg139_1);  mul_331 = arg139_1 = None
        add_278 = torch.ops.aten.add.Tensor(mul_332, arg140_1);  mul_332 = arg140_1 = None
        view_482 = torch.ops.aten.view.default(add_278, [8, 49, 512]);  add_278 = None
        view_483 = torch.ops.aten.view.default(view_482, [8, 49, 8, -1]);  view_482 = None
        split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(view_483, [16, 16, 32], 3);  view_483 = None
        getitem_54 = split_with_sizes_19[0]
        getitem_55 = split_with_sizes_19[1]
        getitem_56 = split_with_sizes_19[2];  split_with_sizes_19 = None
        permute_160 = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        permute_161 = torch.ops.aten.permute.default(getitem_55, [0, 2, 3, 1]);  getitem_55 = None
        permute_162 = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        expand_76 = torch.ops.aten.expand.default(permute_160, [8, 8, 49, 16]);  permute_160 = None
        clone_118 = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_484 = torch.ops.aten.view.default(clone_118, [64, 49, 16]);  clone_118 = None
        expand_77 = torch.ops.aten.expand.default(permute_161, [8, 8, 16, 49]);  permute_161 = None
        clone_119 = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_485 = torch.ops.aten.view.default(clone_119, [64, 16, 49]);  clone_119 = None
        bmm_38 = torch.ops.aten.bmm.default(view_484, view_485);  view_484 = view_485 = None
        view_486 = torch.ops.aten.view.default(bmm_38, [8, 8, 49, 49]);  bmm_38 = None
        mul_333 = torch.ops.aten.mul.Tensor(view_486, 0.25);  view_486 = None
        add_279 = torch.ops.aten.add.Tensor(mul_333, index_5);  mul_333 = None
        amax_19 = torch.ops.aten.amax.default(add_279, [-1], True)
        sub_109 = torch.ops.aten.sub.Tensor(add_279, amax_19);  add_279 = amax_19 = None
        exp_19 = torch.ops.aten.exp.default(sub_109);  sub_109 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_64 = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        expand_78 = torch.ops.aten.expand.default(div_64, [8, 8, 49, 49]);  div_64 = None
        view_487 = torch.ops.aten.view.default(expand_78, [64, 49, 49]);  expand_78 = None
        expand_79 = torch.ops.aten.expand.default(permute_162, [8, 8, 49, 32]);  permute_162 = None
        clone_120 = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_488 = torch.ops.aten.view.default(clone_120, [64, 49, 32]);  clone_120 = None
        bmm_39 = torch.ops.aten.bmm.default(view_487, view_488);  view_487 = view_488 = None
        view_489 = torch.ops.aten.view.default(bmm_39, [8, 8, 49, 32]);  bmm_39 = None
        permute_163 = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        clone_121 = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        view_490 = torch.ops.aten.view.default(clone_121, [8, 49, 256]);  clone_121 = None
        add_280 = torch.ops.aten.add.Tensor(view_490, 3)
        clamp_min_44 = torch.ops.aten.clamp_min.default(add_280, 0);  add_280 = None
        clamp_max_44 = torch.ops.aten.clamp_max.default(clamp_min_44, 6);  clamp_min_44 = None
        mul_334 = torch.ops.aten.mul.Tensor(view_490, clamp_max_44);  view_490 = clamp_max_44 = None
        div_65 = torch.ops.aten.div.Tensor(mul_334, 6);  mul_334 = None
        permute_164 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        view_491 = torch.ops.aten.view.default(div_65, [392, 256]);  div_65 = None
        mm_80 = torch.ops.aten.mm.default(view_491, permute_164);  view_491 = permute_164 = None
        view_492 = torch.ops.aten.view.default(mm_80, [8, 49, 256]);  mm_80 = None
        view_493 = torch.ops.aten.view.default(view_492, [392, 256]);  view_492 = None
        add_281 = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_281);  add_281 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_335 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        sub_110 = torch.ops.aten.sub.Tensor(view_493, arg144_1);  view_493 = arg144_1 = None
        mul_336 = torch.ops.aten.mul.Tensor(sub_110, mul_335);  sub_110 = mul_335 = None
        mul_337 = torch.ops.aten.mul.Tensor(mul_336, arg146_1);  mul_336 = arg146_1 = None
        add_282 = torch.ops.aten.add.Tensor(mul_337, arg147_1);  mul_337 = arg147_1 = None
        view_494 = torch.ops.aten.view.default(add_282, [8, 49, 256]);  add_282 = None
        add_283 = torch.ops.aten.add.Tensor(add_276, view_494);  add_276 = view_494 = None
        permute_165 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        view_495 = torch.ops.aten.view.default(add_283, [392, 256])
        mm_81 = torch.ops.aten.mm.default(view_495, permute_165);  view_495 = permute_165 = None
        view_496 = torch.ops.aten.view.default(mm_81, [8, 49, 512]);  mm_81 = None
        view_497 = torch.ops.aten.view.default(view_496, [392, 512]);  view_496 = None
        add_284 = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_284);  add_284 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_338 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        sub_111 = torch.ops.aten.sub.Tensor(view_497, arg149_1);  view_497 = arg149_1 = None
        mul_339 = torch.ops.aten.mul.Tensor(sub_111, mul_338);  sub_111 = mul_338 = None
        mul_340 = torch.ops.aten.mul.Tensor(mul_339, arg151_1);  mul_339 = arg151_1 = None
        add_285 = torch.ops.aten.add.Tensor(mul_340, arg152_1);  mul_340 = arg152_1 = None
        view_498 = torch.ops.aten.view.default(add_285, [8, 49, 512]);  add_285 = None
        add_286 = torch.ops.aten.add.Tensor(view_498, 3)
        clamp_min_45 = torch.ops.aten.clamp_min.default(add_286, 0);  add_286 = None
        clamp_max_45 = torch.ops.aten.clamp_max.default(clamp_min_45, 6);  clamp_min_45 = None
        mul_341 = torch.ops.aten.mul.Tensor(view_498, clamp_max_45);  view_498 = clamp_max_45 = None
        div_66 = torch.ops.aten.div.Tensor(mul_341, 6);  mul_341 = None
        permute_166 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        view_499 = torch.ops.aten.view.default(div_66, [392, 512]);  div_66 = None
        mm_82 = torch.ops.aten.mm.default(view_499, permute_166);  view_499 = permute_166 = None
        view_500 = torch.ops.aten.view.default(mm_82, [8, 49, 256]);  mm_82 = None
        view_501 = torch.ops.aten.view.default(view_500, [392, 256]);  view_500 = None
        add_287 = torch.ops.aten.add.Tensor(arg155_1, 1e-05);  arg155_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_287);  add_287 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_342 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        sub_112 = torch.ops.aten.sub.Tensor(view_501, arg154_1);  view_501 = arg154_1 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_112, mul_342);  sub_112 = mul_342 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, arg156_1);  mul_343 = arg156_1 = None
        add_288 = torch.ops.aten.add.Tensor(mul_344, arg157_1);  mul_344 = arg157_1 = None
        view_502 = torch.ops.aten.view.default(add_288, [8, 49, 256]);  add_288 = None
        add_289 = torch.ops.aten.add.Tensor(add_283, view_502);  add_283 = view_502 = None
        permute_167 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        view_503 = torch.ops.aten.view.default(add_289, [392, 256])
        mm_83 = torch.ops.aten.mm.default(view_503, permute_167);  view_503 = permute_167 = None
        view_504 = torch.ops.aten.view.default(mm_83, [8, 49, 512]);  mm_83 = None
        view_505 = torch.ops.aten.view.default(view_504, [392, 512]);  view_504 = None
        add_290 = torch.ops.aten.add.Tensor(arg160_1, 1e-05);  arg160_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_290);  add_290 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_345 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        sub_113 = torch.ops.aten.sub.Tensor(view_505, arg159_1);  view_505 = arg159_1 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_113, mul_345);  sub_113 = mul_345 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, arg161_1);  mul_346 = arg161_1 = None
        add_291 = torch.ops.aten.add.Tensor(mul_347, arg162_1);  mul_347 = arg162_1 = None
        view_506 = torch.ops.aten.view.default(add_291, [8, 49, 512]);  add_291 = None
        view_507 = torch.ops.aten.view.default(view_506, [8, 49, 8, -1]);  view_506 = None
        split_with_sizes_20 = torch.ops.aten.split_with_sizes.default(view_507, [16, 16, 32], 3);  view_507 = None
        getitem_57 = split_with_sizes_20[0]
        getitem_58 = split_with_sizes_20[1]
        getitem_59 = split_with_sizes_20[2];  split_with_sizes_20 = None
        permute_168 = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        permute_169 = torch.ops.aten.permute.default(getitem_58, [0, 2, 3, 1]);  getitem_58 = None
        permute_170 = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        expand_80 = torch.ops.aten.expand.default(permute_168, [8, 8, 49, 16]);  permute_168 = None
        clone_123 = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_508 = torch.ops.aten.view.default(clone_123, [64, 49, 16]);  clone_123 = None
        expand_81 = torch.ops.aten.expand.default(permute_169, [8, 8, 16, 49]);  permute_169 = None
        clone_124 = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_509 = torch.ops.aten.view.default(clone_124, [64, 16, 49]);  clone_124 = None
        bmm_40 = torch.ops.aten.bmm.default(view_508, view_509);  view_508 = view_509 = None
        view_510 = torch.ops.aten.view.default(bmm_40, [8, 8, 49, 49]);  bmm_40 = None
        mul_348 = torch.ops.aten.mul.Tensor(view_510, 0.25);  view_510 = None
        add_292 = torch.ops.aten.add.Tensor(mul_348, index_6);  mul_348 = None
        amax_20 = torch.ops.aten.amax.default(add_292, [-1], True)
        sub_114 = torch.ops.aten.sub.Tensor(add_292, amax_20);  add_292 = amax_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_67 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        expand_82 = torch.ops.aten.expand.default(div_67, [8, 8, 49, 49]);  div_67 = None
        view_511 = torch.ops.aten.view.default(expand_82, [64, 49, 49]);  expand_82 = None
        expand_83 = torch.ops.aten.expand.default(permute_170, [8, 8, 49, 32]);  permute_170 = None
        clone_125 = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_512 = torch.ops.aten.view.default(clone_125, [64, 49, 32]);  clone_125 = None
        bmm_41 = torch.ops.aten.bmm.default(view_511, view_512);  view_511 = view_512 = None
        view_513 = torch.ops.aten.view.default(bmm_41, [8, 8, 49, 32]);  bmm_41 = None
        permute_171 = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
        clone_126 = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_514 = torch.ops.aten.view.default(clone_126, [8, 49, 256]);  clone_126 = None
        add_293 = torch.ops.aten.add.Tensor(view_514, 3)
        clamp_min_46 = torch.ops.aten.clamp_min.default(add_293, 0);  add_293 = None
        clamp_max_46 = torch.ops.aten.clamp_max.default(clamp_min_46, 6);  clamp_min_46 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_514, clamp_max_46);  view_514 = clamp_max_46 = None
        div_68 = torch.ops.aten.div.Tensor(mul_349, 6);  mul_349 = None
        permute_172 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        view_515 = torch.ops.aten.view.default(div_68, [392, 256]);  div_68 = None
        mm_84 = torch.ops.aten.mm.default(view_515, permute_172);  view_515 = permute_172 = None
        view_516 = torch.ops.aten.view.default(mm_84, [8, 49, 256]);  mm_84 = None
        view_517 = torch.ops.aten.view.default(view_516, [392, 256]);  view_516 = None
        add_294 = torch.ops.aten.add.Tensor(arg167_1, 1e-05);  arg167_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_294);  add_294 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_350 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        sub_115 = torch.ops.aten.sub.Tensor(view_517, arg166_1);  view_517 = arg166_1 = None
        mul_351 = torch.ops.aten.mul.Tensor(sub_115, mul_350);  sub_115 = mul_350 = None
        mul_352 = torch.ops.aten.mul.Tensor(mul_351, arg168_1);  mul_351 = arg168_1 = None
        add_295 = torch.ops.aten.add.Tensor(mul_352, arg169_1);  mul_352 = arg169_1 = None
        view_518 = torch.ops.aten.view.default(add_295, [8, 49, 256]);  add_295 = None
        add_296 = torch.ops.aten.add.Tensor(add_289, view_518);  add_289 = view_518 = None
        permute_173 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        view_519 = torch.ops.aten.view.default(add_296, [392, 256])
        mm_85 = torch.ops.aten.mm.default(view_519, permute_173);  view_519 = permute_173 = None
        view_520 = torch.ops.aten.view.default(mm_85, [8, 49, 512]);  mm_85 = None
        view_521 = torch.ops.aten.view.default(view_520, [392, 512]);  view_520 = None
        add_297 = torch.ops.aten.add.Tensor(arg172_1, 1e-05);  arg172_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_297);  add_297 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_353 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        sub_116 = torch.ops.aten.sub.Tensor(view_521, arg171_1);  view_521 = arg171_1 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_116, mul_353);  sub_116 = mul_353 = None
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, arg173_1);  mul_354 = arg173_1 = None
        add_298 = torch.ops.aten.add.Tensor(mul_355, arg174_1);  mul_355 = arg174_1 = None
        view_522 = torch.ops.aten.view.default(add_298, [8, 49, 512]);  add_298 = None
        add_299 = torch.ops.aten.add.Tensor(view_522, 3)
        clamp_min_47 = torch.ops.aten.clamp_min.default(add_299, 0);  add_299 = None
        clamp_max_47 = torch.ops.aten.clamp_max.default(clamp_min_47, 6);  clamp_min_47 = None
        mul_356 = torch.ops.aten.mul.Tensor(view_522, clamp_max_47);  view_522 = clamp_max_47 = None
        div_69 = torch.ops.aten.div.Tensor(mul_356, 6);  mul_356 = None
        permute_174 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        view_523 = torch.ops.aten.view.default(div_69, [392, 512]);  div_69 = None
        mm_86 = torch.ops.aten.mm.default(view_523, permute_174);  view_523 = permute_174 = None
        view_524 = torch.ops.aten.view.default(mm_86, [8, 49, 256]);  mm_86 = None
        view_525 = torch.ops.aten.view.default(view_524, [392, 256]);  view_524 = None
        add_300 = torch.ops.aten.add.Tensor(arg177_1, 1e-05);  arg177_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_357 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        sub_117 = torch.ops.aten.sub.Tensor(view_525, arg176_1);  view_525 = arg176_1 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_117, mul_357);  sub_117 = mul_357 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, arg178_1);  mul_358 = arg178_1 = None
        add_301 = torch.ops.aten.add.Tensor(mul_359, arg179_1);  mul_359 = arg179_1 = None
        view_526 = torch.ops.aten.view.default(add_301, [8, 49, 256]);  add_301 = None
        add_302 = torch.ops.aten.add.Tensor(add_296, view_526);  add_296 = view_526 = None
        permute_175 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        view_527 = torch.ops.aten.view.default(add_302, [392, 256])
        mm_87 = torch.ops.aten.mm.default(view_527, permute_175);  view_527 = permute_175 = None
        view_528 = torch.ops.aten.view.default(mm_87, [8, 49, 512]);  mm_87 = None
        view_529 = torch.ops.aten.view.default(view_528, [392, 512]);  view_528 = None
        add_303 = torch.ops.aten.add.Tensor(arg182_1, 1e-05);  arg182_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_303);  add_303 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        sub_118 = torch.ops.aten.sub.Tensor(view_529, arg181_1);  view_529 = arg181_1 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_118, mul_360);  sub_118 = mul_360 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, arg183_1);  mul_361 = arg183_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_362, arg184_1);  mul_362 = arg184_1 = None
        view_530 = torch.ops.aten.view.default(add_304, [8, 49, 512]);  add_304 = None
        view_531 = torch.ops.aten.view.default(view_530, [8, 49, 8, -1]);  view_530 = None
        split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(view_531, [16, 16, 32], 3);  view_531 = None
        getitem_60 = split_with_sizes_21[0]
        getitem_61 = split_with_sizes_21[1]
        getitem_62 = split_with_sizes_21[2];  split_with_sizes_21 = None
        permute_176 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        permute_177 = torch.ops.aten.permute.default(getitem_61, [0, 2, 3, 1]);  getitem_61 = None
        permute_178 = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
        expand_84 = torch.ops.aten.expand.default(permute_176, [8, 8, 49, 16]);  permute_176 = None
        clone_128 = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_532 = torch.ops.aten.view.default(clone_128, [64, 49, 16]);  clone_128 = None
        expand_85 = torch.ops.aten.expand.default(permute_177, [8, 8, 16, 49]);  permute_177 = None
        clone_129 = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_533 = torch.ops.aten.view.default(clone_129, [64, 16, 49]);  clone_129 = None
        bmm_42 = torch.ops.aten.bmm.default(view_532, view_533);  view_532 = view_533 = None
        view_534 = torch.ops.aten.view.default(bmm_42, [8, 8, 49, 49]);  bmm_42 = None
        mul_363 = torch.ops.aten.mul.Tensor(view_534, 0.25);  view_534 = None
        add_305 = torch.ops.aten.add.Tensor(mul_363, index_7);  mul_363 = None
        amax_21 = torch.ops.aten.amax.default(add_305, [-1], True)
        sub_119 = torch.ops.aten.sub.Tensor(add_305, amax_21);  add_305 = amax_21 = None
        exp_21 = torch.ops.aten.exp.default(sub_119);  sub_119 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_70 = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        expand_86 = torch.ops.aten.expand.default(div_70, [8, 8, 49, 49]);  div_70 = None
        view_535 = torch.ops.aten.view.default(expand_86, [64, 49, 49]);  expand_86 = None
        expand_87 = torch.ops.aten.expand.default(permute_178, [8, 8, 49, 32]);  permute_178 = None
        clone_130 = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_536 = torch.ops.aten.view.default(clone_130, [64, 49, 32]);  clone_130 = None
        bmm_43 = torch.ops.aten.bmm.default(view_535, view_536);  view_535 = view_536 = None
        view_537 = torch.ops.aten.view.default(bmm_43, [8, 8, 49, 32]);  bmm_43 = None
        permute_179 = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
        clone_131 = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_538 = torch.ops.aten.view.default(clone_131, [8, 49, 256]);  clone_131 = None
        add_306 = torch.ops.aten.add.Tensor(view_538, 3)
        clamp_min_48 = torch.ops.aten.clamp_min.default(add_306, 0);  add_306 = None
        clamp_max_48 = torch.ops.aten.clamp_max.default(clamp_min_48, 6);  clamp_min_48 = None
        mul_364 = torch.ops.aten.mul.Tensor(view_538, clamp_max_48);  view_538 = clamp_max_48 = None
        div_71 = torch.ops.aten.div.Tensor(mul_364, 6);  mul_364 = None
        permute_180 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        view_539 = torch.ops.aten.view.default(div_71, [392, 256]);  div_71 = None
        mm_88 = torch.ops.aten.mm.default(view_539, permute_180);  view_539 = permute_180 = None
        view_540 = torch.ops.aten.view.default(mm_88, [8, 49, 256]);  mm_88 = None
        view_541 = torch.ops.aten.view.default(view_540, [392, 256]);  view_540 = None
        add_307 = torch.ops.aten.add.Tensor(arg189_1, 1e-05);  arg189_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_365 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        sub_120 = torch.ops.aten.sub.Tensor(view_541, arg188_1);  view_541 = arg188_1 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_120, mul_365);  sub_120 = mul_365 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_366, arg190_1);  mul_366 = arg190_1 = None
        add_308 = torch.ops.aten.add.Tensor(mul_367, arg191_1);  mul_367 = arg191_1 = None
        view_542 = torch.ops.aten.view.default(add_308, [8, 49, 256]);  add_308 = None
        add_309 = torch.ops.aten.add.Tensor(add_302, view_542);  add_302 = view_542 = None
        permute_181 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        view_543 = torch.ops.aten.view.default(add_309, [392, 256])
        mm_89 = torch.ops.aten.mm.default(view_543, permute_181);  view_543 = permute_181 = None
        view_544 = torch.ops.aten.view.default(mm_89, [8, 49, 512]);  mm_89 = None
        view_545 = torch.ops.aten.view.default(view_544, [392, 512]);  view_544 = None
        add_310 = torch.ops.aten.add.Tensor(arg194_1, 1e-05);  arg194_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_310);  add_310 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_368 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        sub_121 = torch.ops.aten.sub.Tensor(view_545, arg193_1);  view_545 = arg193_1 = None
        mul_369 = torch.ops.aten.mul.Tensor(sub_121, mul_368);  sub_121 = mul_368 = None
        mul_370 = torch.ops.aten.mul.Tensor(mul_369, arg195_1);  mul_369 = arg195_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_370, arg196_1);  mul_370 = arg196_1 = None
        view_546 = torch.ops.aten.view.default(add_311, [8, 49, 512]);  add_311 = None
        add_312 = torch.ops.aten.add.Tensor(view_546, 3)
        clamp_min_49 = torch.ops.aten.clamp_min.default(add_312, 0);  add_312 = None
        clamp_max_49 = torch.ops.aten.clamp_max.default(clamp_min_49, 6);  clamp_min_49 = None
        mul_371 = torch.ops.aten.mul.Tensor(view_546, clamp_max_49);  view_546 = clamp_max_49 = None
        div_72 = torch.ops.aten.div.Tensor(mul_371, 6);  mul_371 = None
        permute_182 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        view_547 = torch.ops.aten.view.default(div_72, [392, 512]);  div_72 = None
        mm_90 = torch.ops.aten.mm.default(view_547, permute_182);  view_547 = permute_182 = None
        view_548 = torch.ops.aten.view.default(mm_90, [8, 49, 256]);  mm_90 = None
        view_549 = torch.ops.aten.view.default(view_548, [392, 256]);  view_548 = None
        add_313 = torch.ops.aten.add.Tensor(arg199_1, 1e-05);  arg199_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_372 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        sub_122 = torch.ops.aten.sub.Tensor(view_549, arg198_1);  view_549 = arg198_1 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_122, mul_372);  sub_122 = mul_372 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, arg200_1);  mul_373 = arg200_1 = None
        add_314 = torch.ops.aten.add.Tensor(mul_374, arg201_1);  mul_374 = arg201_1 = None
        view_550 = torch.ops.aten.view.default(add_314, [8, 49, 256]);  add_314 = None
        add_315 = torch.ops.aten.add.Tensor(add_309, view_550);  add_309 = view_550 = None
        permute_183 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        view_551 = torch.ops.aten.view.default(add_315, [392, 256])
        mm_91 = torch.ops.aten.mm.default(view_551, permute_183);  view_551 = permute_183 = None
        view_552 = torch.ops.aten.view.default(mm_91, [8, 49, 512]);  mm_91 = None
        view_553 = torch.ops.aten.view.default(view_552, [392, 512]);  view_552 = None
        add_316 = torch.ops.aten.add.Tensor(arg204_1, 1e-05);  arg204_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_316);  add_316 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_375 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        sub_123 = torch.ops.aten.sub.Tensor(view_553, arg203_1);  view_553 = arg203_1 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_123, mul_375);  sub_123 = mul_375 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, arg205_1);  mul_376 = arg205_1 = None
        add_317 = torch.ops.aten.add.Tensor(mul_377, arg206_1);  mul_377 = arg206_1 = None
        view_554 = torch.ops.aten.view.default(add_317, [8, 49, 512]);  add_317 = None
        view_555 = torch.ops.aten.view.default(view_554, [8, 49, 8, -1]);  view_554 = None
        split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(view_555, [16, 16, 32], 3);  view_555 = None
        getitem_63 = split_with_sizes_22[0]
        getitem_64 = split_with_sizes_22[1]
        getitem_65 = split_with_sizes_22[2];  split_with_sizes_22 = None
        permute_184 = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
        permute_185 = torch.ops.aten.permute.default(getitem_64, [0, 2, 3, 1]);  getitem_64 = None
        permute_186 = torch.ops.aten.permute.default(getitem_65, [0, 2, 1, 3]);  getitem_65 = None
        expand_88 = torch.ops.aten.expand.default(permute_184, [8, 8, 49, 16]);  permute_184 = None
        clone_133 = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_556 = torch.ops.aten.view.default(clone_133, [64, 49, 16]);  clone_133 = None
        expand_89 = torch.ops.aten.expand.default(permute_185, [8, 8, 16, 49]);  permute_185 = None
        clone_134 = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_557 = torch.ops.aten.view.default(clone_134, [64, 16, 49]);  clone_134 = None
        bmm_44 = torch.ops.aten.bmm.default(view_556, view_557);  view_556 = view_557 = None
        view_558 = torch.ops.aten.view.default(bmm_44, [8, 8, 49, 49]);  bmm_44 = None
        mul_378 = torch.ops.aten.mul.Tensor(view_558, 0.25);  view_558 = None
        add_318 = torch.ops.aten.add.Tensor(mul_378, index_8);  mul_378 = None
        amax_22 = torch.ops.aten.amax.default(add_318, [-1], True)
        sub_124 = torch.ops.aten.sub.Tensor(add_318, amax_22);  add_318 = amax_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_124);  sub_124 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_73 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        expand_90 = torch.ops.aten.expand.default(div_73, [8, 8, 49, 49]);  div_73 = None
        view_559 = torch.ops.aten.view.default(expand_90, [64, 49, 49]);  expand_90 = None
        expand_91 = torch.ops.aten.expand.default(permute_186, [8, 8, 49, 32]);  permute_186 = None
        clone_135 = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_560 = torch.ops.aten.view.default(clone_135, [64, 49, 32]);  clone_135 = None
        bmm_45 = torch.ops.aten.bmm.default(view_559, view_560);  view_559 = view_560 = None
        view_561 = torch.ops.aten.view.default(bmm_45, [8, 8, 49, 32]);  bmm_45 = None
        permute_187 = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_136 = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_562 = torch.ops.aten.view.default(clone_136, [8, 49, 256]);  clone_136 = None
        add_319 = torch.ops.aten.add.Tensor(view_562, 3)
        clamp_min_50 = torch.ops.aten.clamp_min.default(add_319, 0);  add_319 = None
        clamp_max_50 = torch.ops.aten.clamp_max.default(clamp_min_50, 6);  clamp_min_50 = None
        mul_379 = torch.ops.aten.mul.Tensor(view_562, clamp_max_50);  view_562 = clamp_max_50 = None
        div_74 = torch.ops.aten.div.Tensor(mul_379, 6);  mul_379 = None
        permute_188 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        view_563 = torch.ops.aten.view.default(div_74, [392, 256]);  div_74 = None
        mm_92 = torch.ops.aten.mm.default(view_563, permute_188);  view_563 = permute_188 = None
        view_564 = torch.ops.aten.view.default(mm_92, [8, 49, 256]);  mm_92 = None
        view_565 = torch.ops.aten.view.default(view_564, [392, 256]);  view_564 = None
        add_320 = torch.ops.aten.add.Tensor(arg211_1, 1e-05);  arg211_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_320);  add_320 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_380 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        sub_125 = torch.ops.aten.sub.Tensor(view_565, arg210_1);  view_565 = arg210_1 = None
        mul_381 = torch.ops.aten.mul.Tensor(sub_125, mul_380);  sub_125 = mul_380 = None
        mul_382 = torch.ops.aten.mul.Tensor(mul_381, arg212_1);  mul_381 = arg212_1 = None
        add_321 = torch.ops.aten.add.Tensor(mul_382, arg213_1);  mul_382 = arg213_1 = None
        view_566 = torch.ops.aten.view.default(add_321, [8, 49, 256]);  add_321 = None
        add_322 = torch.ops.aten.add.Tensor(add_315, view_566);  add_315 = view_566 = None
        permute_189 = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        view_567 = torch.ops.aten.view.default(add_322, [392, 256])
        mm_93 = torch.ops.aten.mm.default(view_567, permute_189);  view_567 = permute_189 = None
        view_568 = torch.ops.aten.view.default(mm_93, [8, 49, 512]);  mm_93 = None
        view_569 = torch.ops.aten.view.default(view_568, [392, 512]);  view_568 = None
        add_323 = torch.ops.aten.add.Tensor(arg216_1, 1e-05);  arg216_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_383 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        sub_126 = torch.ops.aten.sub.Tensor(view_569, arg215_1);  view_569 = arg215_1 = None
        mul_384 = torch.ops.aten.mul.Tensor(sub_126, mul_383);  sub_126 = mul_383 = None
        mul_385 = torch.ops.aten.mul.Tensor(mul_384, arg217_1);  mul_384 = arg217_1 = None
        add_324 = torch.ops.aten.add.Tensor(mul_385, arg218_1);  mul_385 = arg218_1 = None
        view_570 = torch.ops.aten.view.default(add_324, [8, 49, 512]);  add_324 = None
        add_325 = torch.ops.aten.add.Tensor(view_570, 3)
        clamp_min_51 = torch.ops.aten.clamp_min.default(add_325, 0);  add_325 = None
        clamp_max_51 = torch.ops.aten.clamp_max.default(clamp_min_51, 6);  clamp_min_51 = None
        mul_386 = torch.ops.aten.mul.Tensor(view_570, clamp_max_51);  view_570 = clamp_max_51 = None
        div_75 = torch.ops.aten.div.Tensor(mul_386, 6);  mul_386 = None
        permute_190 = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        view_571 = torch.ops.aten.view.default(div_75, [392, 512]);  div_75 = None
        mm_94 = torch.ops.aten.mm.default(view_571, permute_190);  view_571 = permute_190 = None
        view_572 = torch.ops.aten.view.default(mm_94, [8, 49, 256]);  mm_94 = None
        view_573 = torch.ops.aten.view.default(view_572, [392, 256]);  view_572 = None
        add_326 = torch.ops.aten.add.Tensor(arg221_1, 1e-05);  arg221_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_326);  add_326 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_387 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        sub_127 = torch.ops.aten.sub.Tensor(view_573, arg220_1);  view_573 = arg220_1 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_127, mul_387);  sub_127 = mul_387 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, arg222_1);  mul_388 = arg222_1 = None
        add_327 = torch.ops.aten.add.Tensor(mul_389, arg223_1);  mul_389 = arg223_1 = None
        view_574 = torch.ops.aten.view.default(add_327, [8, 49, 256]);  add_327 = None
        add_328 = torch.ops.aten.add.Tensor(add_322, view_574);  add_322 = view_574 = None
        permute_191 = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        view_575 = torch.ops.aten.view.default(add_328, [392, 256])
        mm_95 = torch.ops.aten.mm.default(view_575, permute_191);  view_575 = permute_191 = None
        view_576 = torch.ops.aten.view.default(mm_95, [8, 49, 1280]);  mm_95 = None
        view_577 = torch.ops.aten.view.default(view_576, [392, 1280]);  view_576 = None
        add_329 = torch.ops.aten.add.Tensor(arg226_1, 1e-05);  arg226_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_329);  add_329 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_390 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        sub_128 = torch.ops.aten.sub.Tensor(view_577, arg225_1);  view_577 = arg225_1 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_128, mul_390);  sub_128 = mul_390 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, arg227_1);  mul_391 = arg227_1 = None
        add_330 = torch.ops.aten.add.Tensor(mul_392, arg228_1);  mul_392 = arg228_1 = None
        view_578 = torch.ops.aten.view.default(add_330, [8, 49, 1280]);  add_330 = None
        view_579 = torch.ops.aten.view.default(view_578, [8, 49, 16, -1]);  view_578 = None
        split_with_sizes_23 = torch.ops.aten.split_with_sizes.default(view_579, [16, 64], 3);  view_579 = None
        getitem_66 = split_with_sizes_23[0]
        getitem_67 = split_with_sizes_23[1];  split_with_sizes_23 = None
        permute_192 = torch.ops.aten.permute.default(getitem_66, [0, 2, 3, 1]);  getitem_66 = None
        permute_193 = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None
        view_580 = torch.ops.aten.view.default(add_328, [8, 7, 7, 256]);  add_328 = None
        slice_25 = torch.ops.aten.slice.Tensor(view_580, 1, 0, 9223372036854775807, 2);  view_580 = None
        slice_26 = torch.ops.aten.slice.Tensor(slice_25, 2, 0, 9223372036854775807, 2);  slice_25 = None
        clone_138 = torch.ops.aten.clone.default(slice_26, memory_format = torch.contiguous_format);  slice_26 = None
        view_581 = torch.ops.aten.view.default(clone_138, [8, 16, 256]);  clone_138 = None
        permute_194 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        view_582 = torch.ops.aten.view.default(view_581, [128, 256]);  view_581 = None
        mm_96 = torch.ops.aten.mm.default(view_582, permute_194);  view_582 = permute_194 = None
        view_583 = torch.ops.aten.view.default(mm_96, [8, 16, 256]);  mm_96 = None
        view_584 = torch.ops.aten.view.default(view_583, [128, 256]);  view_583 = None
        add_331 = torch.ops.aten.add.Tensor(arg231_1, 1e-05);  arg231_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_331);  add_331 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_393 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        sub_129 = torch.ops.aten.sub.Tensor(view_584, arg230_1);  view_584 = arg230_1 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_129, mul_393);  sub_129 = mul_393 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, arg232_1);  mul_394 = arg232_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_395, arg233_1);  mul_395 = arg233_1 = None
        view_585 = torch.ops.aten.view.default(add_332, [8, 16, 256]);  add_332 = None
        view_586 = torch.ops.aten.view.default(view_585, [8, -1, 16, 16]);  view_585 = None
        permute_195 = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
        expand_92 = torch.ops.aten.expand.default(permute_195, [8, 16, 16, 16]);  permute_195 = None
        clone_139 = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_587 = torch.ops.aten.view.default(clone_139, [128, 16, 16]);  clone_139 = None
        expand_93 = torch.ops.aten.expand.default(permute_192, [8, 16, 16, 49]);  permute_192 = None
        clone_140 = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_588 = torch.ops.aten.view.default(clone_140, [128, 16, 49]);  clone_140 = None
        bmm_46 = torch.ops.aten.bmm.default(view_587, view_588);  view_587 = view_588 = None
        view_589 = torch.ops.aten.view.default(bmm_46, [8, 16, 16, 49]);  bmm_46 = None
        mul_396 = torch.ops.aten.mul.Tensor(view_589, 0.25);  view_589 = None
        add_333 = torch.ops.aten.add.Tensor(mul_396, index_9);  mul_396 = None
        amax_23 = torch.ops.aten.amax.default(add_333, [-1], True)
        sub_130 = torch.ops.aten.sub.Tensor(add_333, amax_23);  add_333 = amax_23 = None
        exp_23 = torch.ops.aten.exp.default(sub_130);  sub_130 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_76 = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        expand_94 = torch.ops.aten.expand.default(div_76, [8, 16, 16, 49]);  div_76 = None
        view_590 = torch.ops.aten.view.default(expand_94, [128, 16, 49]);  expand_94 = None
        expand_95 = torch.ops.aten.expand.default(permute_193, [8, 16, 49, 64]);  permute_193 = None
        clone_141 = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_591 = torch.ops.aten.view.default(clone_141, [128, 49, 64]);  clone_141 = None
        bmm_47 = torch.ops.aten.bmm.default(view_590, view_591);  view_590 = view_591 = None
        view_592 = torch.ops.aten.view.default(bmm_47, [8, 16, 16, 64]);  bmm_47 = None
        permute_196 = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
        clone_142 = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        view_593 = torch.ops.aten.view.default(clone_142, [8, 16, 1024]);  clone_142 = None
        add_334 = torch.ops.aten.add.Tensor(view_593, 3)
        clamp_min_52 = torch.ops.aten.clamp_min.default(add_334, 0);  add_334 = None
        clamp_max_52 = torch.ops.aten.clamp_max.default(clamp_min_52, 6);  clamp_min_52 = None
        mul_397 = torch.ops.aten.mul.Tensor(view_593, clamp_max_52);  view_593 = clamp_max_52 = None
        div_77 = torch.ops.aten.div.Tensor(mul_397, 6);  mul_397 = None
        permute_197 = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        view_594 = torch.ops.aten.view.default(div_77, [128, 1024]);  div_77 = None
        mm_97 = torch.ops.aten.mm.default(view_594, permute_197);  view_594 = permute_197 = None
        view_595 = torch.ops.aten.view.default(mm_97, [8, 16, 384]);  mm_97 = None
        view_596 = torch.ops.aten.view.default(view_595, [128, 384]);  view_595 = None
        add_335 = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_335);  add_335 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_398 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        sub_131 = torch.ops.aten.sub.Tensor(view_596, arg237_1);  view_596 = arg237_1 = None
        mul_399 = torch.ops.aten.mul.Tensor(sub_131, mul_398);  sub_131 = mul_398 = None
        mul_400 = torch.ops.aten.mul.Tensor(mul_399, arg239_1);  mul_399 = arg239_1 = None
        add_336 = torch.ops.aten.add.Tensor(mul_400, arg240_1);  mul_400 = arg240_1 = None
        view_597 = torch.ops.aten.view.default(add_336, [8, 16, 384]);  add_336 = None
        permute_198 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        view_598 = torch.ops.aten.view.default(view_597, [128, 384])
        mm_98 = torch.ops.aten.mm.default(view_598, permute_198);  view_598 = permute_198 = None
        view_599 = torch.ops.aten.view.default(mm_98, [8, 16, 768]);  mm_98 = None
        view_600 = torch.ops.aten.view.default(view_599, [128, 768]);  view_599 = None
        add_337 = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_401 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        sub_132 = torch.ops.aten.sub.Tensor(view_600, arg242_1);  view_600 = arg242_1 = None
        mul_402 = torch.ops.aten.mul.Tensor(sub_132, mul_401);  sub_132 = mul_401 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_402, arg244_1);  mul_402 = arg244_1 = None
        add_338 = torch.ops.aten.add.Tensor(mul_403, arg245_1);  mul_403 = arg245_1 = None
        view_601 = torch.ops.aten.view.default(add_338, [8, 16, 768]);  add_338 = None
        add_339 = torch.ops.aten.add.Tensor(view_601, 3)
        clamp_min_53 = torch.ops.aten.clamp_min.default(add_339, 0);  add_339 = None
        clamp_max_53 = torch.ops.aten.clamp_max.default(clamp_min_53, 6);  clamp_min_53 = None
        mul_404 = torch.ops.aten.mul.Tensor(view_601, clamp_max_53);  view_601 = clamp_max_53 = None
        div_78 = torch.ops.aten.div.Tensor(mul_404, 6);  mul_404 = None
        permute_199 = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        view_602 = torch.ops.aten.view.default(div_78, [128, 768]);  div_78 = None
        mm_99 = torch.ops.aten.mm.default(view_602, permute_199);  view_602 = permute_199 = None
        view_603 = torch.ops.aten.view.default(mm_99, [8, 16, 384]);  mm_99 = None
        view_604 = torch.ops.aten.view.default(view_603, [128, 384]);  view_603 = None
        add_340 = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_340);  add_340 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_405 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        sub_133 = torch.ops.aten.sub.Tensor(view_604, arg247_1);  view_604 = arg247_1 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_133, mul_405);  sub_133 = mul_405 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, arg249_1);  mul_406 = arg249_1 = None
        add_341 = torch.ops.aten.add.Tensor(mul_407, arg250_1);  mul_407 = arg250_1 = None
        view_605 = torch.ops.aten.view.default(add_341, [8, 16, 384]);  add_341 = None
        add_342 = torch.ops.aten.add.Tensor(view_597, view_605);  view_597 = view_605 = None
        permute_200 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        view_606 = torch.ops.aten.view.default(add_342, [128, 384])
        mm_100 = torch.ops.aten.mm.default(view_606, permute_200);  view_606 = permute_200 = None
        view_607 = torch.ops.aten.view.default(mm_100, [8, 16, 768]);  mm_100 = None
        view_608 = torch.ops.aten.view.default(view_607, [128, 768]);  view_607 = None
        add_343 = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_343);  add_343 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_408 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        sub_134 = torch.ops.aten.sub.Tensor(view_608, arg252_1);  view_608 = arg252_1 = None
        mul_409 = torch.ops.aten.mul.Tensor(sub_134, mul_408);  sub_134 = mul_408 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, arg254_1);  mul_409 = arg254_1 = None
        add_344 = torch.ops.aten.add.Tensor(mul_410, arg255_1);  mul_410 = arg255_1 = None
        view_609 = torch.ops.aten.view.default(add_344, [8, 16, 768]);  add_344 = None
        view_610 = torch.ops.aten.view.default(view_609, [8, 16, 12, -1]);  view_609 = None
        split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(view_610, [16, 16, 32], 3);  view_610 = None
        getitem_68 = split_with_sizes_24[0]
        getitem_69 = split_with_sizes_24[1]
        getitem_70 = split_with_sizes_24[2];  split_with_sizes_24 = None
        permute_201 = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None
        permute_202 = torch.ops.aten.permute.default(getitem_69, [0, 2, 3, 1]);  getitem_69 = None
        permute_203 = torch.ops.aten.permute.default(getitem_70, [0, 2, 1, 3]);  getitem_70 = None
        expand_96 = torch.ops.aten.expand.default(permute_201, [8, 12, 16, 16]);  permute_201 = None
        clone_144 = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_611 = torch.ops.aten.view.default(clone_144, [96, 16, 16]);  clone_144 = None
        expand_97 = torch.ops.aten.expand.default(permute_202, [8, 12, 16, 16]);  permute_202 = None
        clone_145 = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_612 = torch.ops.aten.view.default(clone_145, [96, 16, 16]);  clone_145 = None
        bmm_48 = torch.ops.aten.bmm.default(view_611, view_612);  view_611 = view_612 = None
        view_613 = torch.ops.aten.view.default(bmm_48, [8, 12, 16, 16]);  bmm_48 = None
        mul_411 = torch.ops.aten.mul.Tensor(view_613, 0.25);  view_613 = None
        add_345 = torch.ops.aten.add.Tensor(mul_411, index_10);  mul_411 = None
        amax_24 = torch.ops.aten.amax.default(add_345, [-1], True)
        sub_135 = torch.ops.aten.sub.Tensor(add_345, amax_24);  add_345 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_79 = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        expand_98 = torch.ops.aten.expand.default(div_79, [8, 12, 16, 16]);  div_79 = None
        view_614 = torch.ops.aten.view.default(expand_98, [96, 16, 16]);  expand_98 = None
        expand_99 = torch.ops.aten.expand.default(permute_203, [8, 12, 16, 32]);  permute_203 = None
        clone_146 = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_615 = torch.ops.aten.view.default(clone_146, [96, 16, 32]);  clone_146 = None
        bmm_49 = torch.ops.aten.bmm.default(view_614, view_615);  view_614 = view_615 = None
        view_616 = torch.ops.aten.view.default(bmm_49, [8, 12, 16, 32]);  bmm_49 = None
        permute_204 = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        clone_147 = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_617 = torch.ops.aten.view.default(clone_147, [8, 16, 384]);  clone_147 = None
        add_346 = torch.ops.aten.add.Tensor(view_617, 3)
        clamp_min_54 = torch.ops.aten.clamp_min.default(add_346, 0);  add_346 = None
        clamp_max_54 = torch.ops.aten.clamp_max.default(clamp_min_54, 6);  clamp_min_54 = None
        mul_412 = torch.ops.aten.mul.Tensor(view_617, clamp_max_54);  view_617 = clamp_max_54 = None
        div_80 = torch.ops.aten.div.Tensor(mul_412, 6);  mul_412 = None
        permute_205 = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        view_618 = torch.ops.aten.view.default(div_80, [128, 384]);  div_80 = None
        mm_101 = torch.ops.aten.mm.default(view_618, permute_205);  view_618 = permute_205 = None
        view_619 = torch.ops.aten.view.default(mm_101, [8, 16, 384]);  mm_101 = None
        view_620 = torch.ops.aten.view.default(view_619, [128, 384]);  view_619 = None
        add_347 = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_347);  add_347 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_413 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        sub_136 = torch.ops.aten.sub.Tensor(view_620, arg259_1);  view_620 = arg259_1 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_136, mul_413);  sub_136 = mul_413 = None
        mul_415 = torch.ops.aten.mul.Tensor(mul_414, arg261_1);  mul_414 = arg261_1 = None
        add_348 = torch.ops.aten.add.Tensor(mul_415, arg262_1);  mul_415 = arg262_1 = None
        view_621 = torch.ops.aten.view.default(add_348, [8, 16, 384]);  add_348 = None
        add_349 = torch.ops.aten.add.Tensor(add_342, view_621);  add_342 = view_621 = None
        permute_206 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        view_622 = torch.ops.aten.view.default(add_349, [128, 384])
        mm_102 = torch.ops.aten.mm.default(view_622, permute_206);  view_622 = permute_206 = None
        view_623 = torch.ops.aten.view.default(mm_102, [8, 16, 768]);  mm_102 = None
        view_624 = torch.ops.aten.view.default(view_623, [128, 768]);  view_623 = None
        add_350 = torch.ops.aten.add.Tensor(arg265_1, 1e-05);  arg265_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_416 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        sub_137 = torch.ops.aten.sub.Tensor(view_624, arg264_1);  view_624 = arg264_1 = None
        mul_417 = torch.ops.aten.mul.Tensor(sub_137, mul_416);  sub_137 = mul_416 = None
        mul_418 = torch.ops.aten.mul.Tensor(mul_417, arg266_1);  mul_417 = arg266_1 = None
        add_351 = torch.ops.aten.add.Tensor(mul_418, arg267_1);  mul_418 = arg267_1 = None
        view_625 = torch.ops.aten.view.default(add_351, [8, 16, 768]);  add_351 = None
        add_352 = torch.ops.aten.add.Tensor(view_625, 3)
        clamp_min_55 = torch.ops.aten.clamp_min.default(add_352, 0);  add_352 = None
        clamp_max_55 = torch.ops.aten.clamp_max.default(clamp_min_55, 6);  clamp_min_55 = None
        mul_419 = torch.ops.aten.mul.Tensor(view_625, clamp_max_55);  view_625 = clamp_max_55 = None
        div_81 = torch.ops.aten.div.Tensor(mul_419, 6);  mul_419 = None
        permute_207 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        view_626 = torch.ops.aten.view.default(div_81, [128, 768]);  div_81 = None
        mm_103 = torch.ops.aten.mm.default(view_626, permute_207);  view_626 = permute_207 = None
        view_627 = torch.ops.aten.view.default(mm_103, [8, 16, 384]);  mm_103 = None
        view_628 = torch.ops.aten.view.default(view_627, [128, 384]);  view_627 = None
        add_353 = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_353);  add_353 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_420 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        sub_138 = torch.ops.aten.sub.Tensor(view_628, arg269_1);  view_628 = arg269_1 = None
        mul_421 = torch.ops.aten.mul.Tensor(sub_138, mul_420);  sub_138 = mul_420 = None
        mul_422 = torch.ops.aten.mul.Tensor(mul_421, arg271_1);  mul_421 = arg271_1 = None
        add_354 = torch.ops.aten.add.Tensor(mul_422, arg272_1);  mul_422 = arg272_1 = None
        view_629 = torch.ops.aten.view.default(add_354, [8, 16, 384]);  add_354 = None
        add_355 = torch.ops.aten.add.Tensor(add_349, view_629);  add_349 = view_629 = None
        permute_208 = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        view_630 = torch.ops.aten.view.default(add_355, [128, 384])
        mm_104 = torch.ops.aten.mm.default(view_630, permute_208);  view_630 = permute_208 = None
        view_631 = torch.ops.aten.view.default(mm_104, [8, 16, 768]);  mm_104 = None
        view_632 = torch.ops.aten.view.default(view_631, [128, 768]);  view_631 = None
        add_356 = torch.ops.aten.add.Tensor(arg275_1, 1e-05);  arg275_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_356);  add_356 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_423 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        sub_139 = torch.ops.aten.sub.Tensor(view_632, arg274_1);  view_632 = arg274_1 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_139, mul_423);  sub_139 = mul_423 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, arg276_1);  mul_424 = arg276_1 = None
        add_357 = torch.ops.aten.add.Tensor(mul_425, arg277_1);  mul_425 = arg277_1 = None
        view_633 = torch.ops.aten.view.default(add_357, [8, 16, 768]);  add_357 = None
        view_634 = torch.ops.aten.view.default(view_633, [8, 16, 12, -1]);  view_633 = None
        split_with_sizes_25 = torch.ops.aten.split_with_sizes.default(view_634, [16, 16, 32], 3);  view_634 = None
        getitem_71 = split_with_sizes_25[0]
        getitem_72 = split_with_sizes_25[1]
        getitem_73 = split_with_sizes_25[2];  split_with_sizes_25 = None
        permute_209 = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        permute_210 = torch.ops.aten.permute.default(getitem_72, [0, 2, 3, 1]);  getitem_72 = None
        permute_211 = torch.ops.aten.permute.default(getitem_73, [0, 2, 1, 3]);  getitem_73 = None
        expand_100 = torch.ops.aten.expand.default(permute_209, [8, 12, 16, 16]);  permute_209 = None
        clone_149 = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_635 = torch.ops.aten.view.default(clone_149, [96, 16, 16]);  clone_149 = None
        expand_101 = torch.ops.aten.expand.default(permute_210, [8, 12, 16, 16]);  permute_210 = None
        clone_150 = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_636 = torch.ops.aten.view.default(clone_150, [96, 16, 16]);  clone_150 = None
        bmm_50 = torch.ops.aten.bmm.default(view_635, view_636);  view_635 = view_636 = None
        view_637 = torch.ops.aten.view.default(bmm_50, [8, 12, 16, 16]);  bmm_50 = None
        mul_426 = torch.ops.aten.mul.Tensor(view_637, 0.25);  view_637 = None
        add_358 = torch.ops.aten.add.Tensor(mul_426, index_11);  mul_426 = None
        amax_25 = torch.ops.aten.amax.default(add_358, [-1], True)
        sub_140 = torch.ops.aten.sub.Tensor(add_358, amax_25);  add_358 = amax_25 = None
        exp_25 = torch.ops.aten.exp.default(sub_140);  sub_140 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_82 = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        expand_102 = torch.ops.aten.expand.default(div_82, [8, 12, 16, 16]);  div_82 = None
        view_638 = torch.ops.aten.view.default(expand_102, [96, 16, 16]);  expand_102 = None
        expand_103 = torch.ops.aten.expand.default(permute_211, [8, 12, 16, 32]);  permute_211 = None
        clone_151 = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_639 = torch.ops.aten.view.default(clone_151, [96, 16, 32]);  clone_151 = None
        bmm_51 = torch.ops.aten.bmm.default(view_638, view_639);  view_638 = view_639 = None
        view_640 = torch.ops.aten.view.default(bmm_51, [8, 12, 16, 32]);  bmm_51 = None
        permute_212 = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
        clone_152 = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_641 = torch.ops.aten.view.default(clone_152, [8, 16, 384]);  clone_152 = None
        add_359 = torch.ops.aten.add.Tensor(view_641, 3)
        clamp_min_56 = torch.ops.aten.clamp_min.default(add_359, 0);  add_359 = None
        clamp_max_56 = torch.ops.aten.clamp_max.default(clamp_min_56, 6);  clamp_min_56 = None
        mul_427 = torch.ops.aten.mul.Tensor(view_641, clamp_max_56);  view_641 = clamp_max_56 = None
        div_83 = torch.ops.aten.div.Tensor(mul_427, 6);  mul_427 = None
        permute_213 = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        view_642 = torch.ops.aten.view.default(div_83, [128, 384]);  div_83 = None
        mm_105 = torch.ops.aten.mm.default(view_642, permute_213);  view_642 = permute_213 = None
        view_643 = torch.ops.aten.view.default(mm_105, [8, 16, 384]);  mm_105 = None
        view_644 = torch.ops.aten.view.default(view_643, [128, 384]);  view_643 = None
        add_360 = torch.ops.aten.add.Tensor(arg282_1, 1e-05);  arg282_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_428 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        sub_141 = torch.ops.aten.sub.Tensor(view_644, arg281_1);  view_644 = arg281_1 = None
        mul_429 = torch.ops.aten.mul.Tensor(sub_141, mul_428);  sub_141 = mul_428 = None
        mul_430 = torch.ops.aten.mul.Tensor(mul_429, arg283_1);  mul_429 = arg283_1 = None
        add_361 = torch.ops.aten.add.Tensor(mul_430, arg284_1);  mul_430 = arg284_1 = None
        view_645 = torch.ops.aten.view.default(add_361, [8, 16, 384]);  add_361 = None
        add_362 = torch.ops.aten.add.Tensor(add_355, view_645);  add_355 = view_645 = None
        permute_214 = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        view_646 = torch.ops.aten.view.default(add_362, [128, 384])
        mm_106 = torch.ops.aten.mm.default(view_646, permute_214);  view_646 = permute_214 = None
        view_647 = torch.ops.aten.view.default(mm_106, [8, 16, 768]);  mm_106 = None
        view_648 = torch.ops.aten.view.default(view_647, [128, 768]);  view_647 = None
        add_363 = torch.ops.aten.add.Tensor(arg287_1, 1e-05);  arg287_1 = None
        sqrt_116 = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_116 = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_431 = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        sub_142 = torch.ops.aten.sub.Tensor(view_648, arg286_1);  view_648 = arg286_1 = None
        mul_432 = torch.ops.aten.mul.Tensor(sub_142, mul_431);  sub_142 = mul_431 = None
        mul_433 = torch.ops.aten.mul.Tensor(mul_432, arg288_1);  mul_432 = arg288_1 = None
        add_364 = torch.ops.aten.add.Tensor(mul_433, arg289_1);  mul_433 = arg289_1 = None
        view_649 = torch.ops.aten.view.default(add_364, [8, 16, 768]);  add_364 = None
        add_365 = torch.ops.aten.add.Tensor(view_649, 3)
        clamp_min_57 = torch.ops.aten.clamp_min.default(add_365, 0);  add_365 = None
        clamp_max_57 = torch.ops.aten.clamp_max.default(clamp_min_57, 6);  clamp_min_57 = None
        mul_434 = torch.ops.aten.mul.Tensor(view_649, clamp_max_57);  view_649 = clamp_max_57 = None
        div_84 = torch.ops.aten.div.Tensor(mul_434, 6);  mul_434 = None
        permute_215 = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        view_650 = torch.ops.aten.view.default(div_84, [128, 768]);  div_84 = None
        mm_107 = torch.ops.aten.mm.default(view_650, permute_215);  view_650 = permute_215 = None
        view_651 = torch.ops.aten.view.default(mm_107, [8, 16, 384]);  mm_107 = None
        view_652 = torch.ops.aten.view.default(view_651, [128, 384]);  view_651 = None
        add_366 = torch.ops.aten.add.Tensor(arg292_1, 1e-05);  arg292_1 = None
        sqrt_117 = torch.ops.aten.sqrt.default(add_366);  add_366 = None
        reciprocal_117 = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_435 = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        sub_143 = torch.ops.aten.sub.Tensor(view_652, arg291_1);  view_652 = arg291_1 = None
        mul_436 = torch.ops.aten.mul.Tensor(sub_143, mul_435);  sub_143 = mul_435 = None
        mul_437 = torch.ops.aten.mul.Tensor(mul_436, arg293_1);  mul_436 = arg293_1 = None
        add_367 = torch.ops.aten.add.Tensor(mul_437, arg294_1);  mul_437 = arg294_1 = None
        view_653 = torch.ops.aten.view.default(add_367, [8, 16, 384]);  add_367 = None
        add_368 = torch.ops.aten.add.Tensor(add_362, view_653);  add_362 = view_653 = None
        permute_216 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        view_654 = torch.ops.aten.view.default(add_368, [128, 384])
        mm_108 = torch.ops.aten.mm.default(view_654, permute_216);  view_654 = permute_216 = None
        view_655 = torch.ops.aten.view.default(mm_108, [8, 16, 768]);  mm_108 = None
        view_656 = torch.ops.aten.view.default(view_655, [128, 768]);  view_655 = None
        add_369 = torch.ops.aten.add.Tensor(arg297_1, 1e-05);  arg297_1 = None
        sqrt_118 = torch.ops.aten.sqrt.default(add_369);  add_369 = None
        reciprocal_118 = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_438 = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        sub_144 = torch.ops.aten.sub.Tensor(view_656, arg296_1);  view_656 = arg296_1 = None
        mul_439 = torch.ops.aten.mul.Tensor(sub_144, mul_438);  sub_144 = mul_438 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_439, arg298_1);  mul_439 = arg298_1 = None
        add_370 = torch.ops.aten.add.Tensor(mul_440, arg299_1);  mul_440 = arg299_1 = None
        view_657 = torch.ops.aten.view.default(add_370, [8, 16, 768]);  add_370 = None
        view_658 = torch.ops.aten.view.default(view_657, [8, 16, 12, -1]);  view_657 = None
        split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(view_658, [16, 16, 32], 3);  view_658 = None
        getitem_74 = split_with_sizes_26[0]
        getitem_75 = split_with_sizes_26[1]
        getitem_76 = split_with_sizes_26[2];  split_with_sizes_26 = None
        permute_217 = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        permute_218 = torch.ops.aten.permute.default(getitem_75, [0, 2, 3, 1]);  getitem_75 = None
        permute_219 = torch.ops.aten.permute.default(getitem_76, [0, 2, 1, 3]);  getitem_76 = None
        expand_104 = torch.ops.aten.expand.default(permute_217, [8, 12, 16, 16]);  permute_217 = None
        clone_154 = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_659 = torch.ops.aten.view.default(clone_154, [96, 16, 16]);  clone_154 = None
        expand_105 = torch.ops.aten.expand.default(permute_218, [8, 12, 16, 16]);  permute_218 = None
        clone_155 = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_660 = torch.ops.aten.view.default(clone_155, [96, 16, 16]);  clone_155 = None
        bmm_52 = torch.ops.aten.bmm.default(view_659, view_660);  view_659 = view_660 = None
        view_661 = torch.ops.aten.view.default(bmm_52, [8, 12, 16, 16]);  bmm_52 = None
        mul_441 = torch.ops.aten.mul.Tensor(view_661, 0.25);  view_661 = None
        add_371 = torch.ops.aten.add.Tensor(mul_441, index_12);  mul_441 = None
        amax_26 = torch.ops.aten.amax.default(add_371, [-1], True)
        sub_145 = torch.ops.aten.sub.Tensor(add_371, amax_26);  add_371 = amax_26 = None
        exp_26 = torch.ops.aten.exp.default(sub_145);  sub_145 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_85 = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        expand_106 = torch.ops.aten.expand.default(div_85, [8, 12, 16, 16]);  div_85 = None
        view_662 = torch.ops.aten.view.default(expand_106, [96, 16, 16]);  expand_106 = None
        expand_107 = torch.ops.aten.expand.default(permute_219, [8, 12, 16, 32]);  permute_219 = None
        clone_156 = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_663 = torch.ops.aten.view.default(clone_156, [96, 16, 32]);  clone_156 = None
        bmm_53 = torch.ops.aten.bmm.default(view_662, view_663);  view_662 = view_663 = None
        view_664 = torch.ops.aten.view.default(bmm_53, [8, 12, 16, 32]);  bmm_53 = None
        permute_220 = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None
        clone_157 = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_665 = torch.ops.aten.view.default(clone_157, [8, 16, 384]);  clone_157 = None
        add_372 = torch.ops.aten.add.Tensor(view_665, 3)
        clamp_min_58 = torch.ops.aten.clamp_min.default(add_372, 0);  add_372 = None
        clamp_max_58 = torch.ops.aten.clamp_max.default(clamp_min_58, 6);  clamp_min_58 = None
        mul_442 = torch.ops.aten.mul.Tensor(view_665, clamp_max_58);  view_665 = clamp_max_58 = None
        div_86 = torch.ops.aten.div.Tensor(mul_442, 6);  mul_442 = None
        permute_221 = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        view_666 = torch.ops.aten.view.default(div_86, [128, 384]);  div_86 = None
        mm_109 = torch.ops.aten.mm.default(view_666, permute_221);  view_666 = permute_221 = None
        view_667 = torch.ops.aten.view.default(mm_109, [8, 16, 384]);  mm_109 = None
        view_668 = torch.ops.aten.view.default(view_667, [128, 384]);  view_667 = None
        add_373 = torch.ops.aten.add.Tensor(arg304_1, 1e-05);  arg304_1 = None
        sqrt_119 = torch.ops.aten.sqrt.default(add_373);  add_373 = None
        reciprocal_119 = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_443 = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        sub_146 = torch.ops.aten.sub.Tensor(view_668, arg303_1);  view_668 = arg303_1 = None
        mul_444 = torch.ops.aten.mul.Tensor(sub_146, mul_443);  sub_146 = mul_443 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, arg305_1);  mul_444 = arg305_1 = None
        add_374 = torch.ops.aten.add.Tensor(mul_445, arg306_1);  mul_445 = arg306_1 = None
        view_669 = torch.ops.aten.view.default(add_374, [8, 16, 384]);  add_374 = None
        add_375 = torch.ops.aten.add.Tensor(add_368, view_669);  add_368 = view_669 = None
        permute_222 = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        view_670 = torch.ops.aten.view.default(add_375, [128, 384])
        mm_110 = torch.ops.aten.mm.default(view_670, permute_222);  view_670 = permute_222 = None
        view_671 = torch.ops.aten.view.default(mm_110, [8, 16, 768]);  mm_110 = None
        view_672 = torch.ops.aten.view.default(view_671, [128, 768]);  view_671 = None
        add_376 = torch.ops.aten.add.Tensor(arg309_1, 1e-05);  arg309_1 = None
        sqrt_120 = torch.ops.aten.sqrt.default(add_376);  add_376 = None
        reciprocal_120 = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_446 = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        sub_147 = torch.ops.aten.sub.Tensor(view_672, arg308_1);  view_672 = arg308_1 = None
        mul_447 = torch.ops.aten.mul.Tensor(sub_147, mul_446);  sub_147 = mul_446 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_447, arg310_1);  mul_447 = arg310_1 = None
        add_377 = torch.ops.aten.add.Tensor(mul_448, arg311_1);  mul_448 = arg311_1 = None
        view_673 = torch.ops.aten.view.default(add_377, [8, 16, 768]);  add_377 = None
        add_378 = torch.ops.aten.add.Tensor(view_673, 3)
        clamp_min_59 = torch.ops.aten.clamp_min.default(add_378, 0);  add_378 = None
        clamp_max_59 = torch.ops.aten.clamp_max.default(clamp_min_59, 6);  clamp_min_59 = None
        mul_449 = torch.ops.aten.mul.Tensor(view_673, clamp_max_59);  view_673 = clamp_max_59 = None
        div_87 = torch.ops.aten.div.Tensor(mul_449, 6);  mul_449 = None
        permute_223 = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        view_674 = torch.ops.aten.view.default(div_87, [128, 768]);  div_87 = None
        mm_111 = torch.ops.aten.mm.default(view_674, permute_223);  view_674 = permute_223 = None
        view_675 = torch.ops.aten.view.default(mm_111, [8, 16, 384]);  mm_111 = None
        view_676 = torch.ops.aten.view.default(view_675, [128, 384]);  view_675 = None
        add_379 = torch.ops.aten.add.Tensor(arg314_1, 1e-05);  arg314_1 = None
        sqrt_121 = torch.ops.aten.sqrt.default(add_379);  add_379 = None
        reciprocal_121 = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_450 = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        sub_148 = torch.ops.aten.sub.Tensor(view_676, arg313_1);  view_676 = arg313_1 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_148, mul_450);  sub_148 = mul_450 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, arg315_1);  mul_451 = arg315_1 = None
        add_380 = torch.ops.aten.add.Tensor(mul_452, arg316_1);  mul_452 = arg316_1 = None
        view_677 = torch.ops.aten.view.default(add_380, [8, 16, 384]);  add_380 = None
        add_381 = torch.ops.aten.add.Tensor(add_375, view_677);  add_375 = view_677 = None
        permute_224 = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        view_678 = torch.ops.aten.view.default(add_381, [128, 384])
        mm_112 = torch.ops.aten.mm.default(view_678, permute_224);  view_678 = permute_224 = None
        view_679 = torch.ops.aten.view.default(mm_112, [8, 16, 768]);  mm_112 = None
        view_680 = torch.ops.aten.view.default(view_679, [128, 768]);  view_679 = None
        add_382 = torch.ops.aten.add.Tensor(arg319_1, 1e-05);  arg319_1 = None
        sqrt_122 = torch.ops.aten.sqrt.default(add_382);  add_382 = None
        reciprocal_122 = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_453 = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        sub_149 = torch.ops.aten.sub.Tensor(view_680, arg318_1);  view_680 = arg318_1 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_149, mul_453);  sub_149 = mul_453 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, arg320_1);  mul_454 = arg320_1 = None
        add_383 = torch.ops.aten.add.Tensor(mul_455, arg321_1);  mul_455 = arg321_1 = None
        view_681 = torch.ops.aten.view.default(add_383, [8, 16, 768]);  add_383 = None
        view_682 = torch.ops.aten.view.default(view_681, [8, 16, 12, -1]);  view_681 = None
        split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(view_682, [16, 16, 32], 3);  view_682 = None
        getitem_77 = split_with_sizes_27[0]
        getitem_78 = split_with_sizes_27[1]
        getitem_79 = split_with_sizes_27[2];  split_with_sizes_27 = None
        permute_225 = torch.ops.aten.permute.default(getitem_77, [0, 2, 1, 3]);  getitem_77 = None
        permute_226 = torch.ops.aten.permute.default(getitem_78, [0, 2, 3, 1]);  getitem_78 = None
        permute_227 = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None
        expand_108 = torch.ops.aten.expand.default(permute_225, [8, 12, 16, 16]);  permute_225 = None
        clone_159 = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_683 = torch.ops.aten.view.default(clone_159, [96, 16, 16]);  clone_159 = None
        expand_109 = torch.ops.aten.expand.default(permute_226, [8, 12, 16, 16]);  permute_226 = None
        clone_160 = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_684 = torch.ops.aten.view.default(clone_160, [96, 16, 16]);  clone_160 = None
        bmm_54 = torch.ops.aten.bmm.default(view_683, view_684);  view_683 = view_684 = None
        view_685 = torch.ops.aten.view.default(bmm_54, [8, 12, 16, 16]);  bmm_54 = None
        mul_456 = torch.ops.aten.mul.Tensor(view_685, 0.25);  view_685 = None
        add_384 = torch.ops.aten.add.Tensor(mul_456, index_13);  mul_456 = None
        amax_27 = torch.ops.aten.amax.default(add_384, [-1], True)
        sub_150 = torch.ops.aten.sub.Tensor(add_384, amax_27);  add_384 = amax_27 = None
        exp_27 = torch.ops.aten.exp.default(sub_150);  sub_150 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_88 = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        expand_110 = torch.ops.aten.expand.default(div_88, [8, 12, 16, 16]);  div_88 = None
        view_686 = torch.ops.aten.view.default(expand_110, [96, 16, 16]);  expand_110 = None
        expand_111 = torch.ops.aten.expand.default(permute_227, [8, 12, 16, 32]);  permute_227 = None
        clone_161 = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_687 = torch.ops.aten.view.default(clone_161, [96, 16, 32]);  clone_161 = None
        bmm_55 = torch.ops.aten.bmm.default(view_686, view_687);  view_686 = view_687 = None
        view_688 = torch.ops.aten.view.default(bmm_55, [8, 12, 16, 32]);  bmm_55 = None
        permute_228 = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
        clone_162 = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_689 = torch.ops.aten.view.default(clone_162, [8, 16, 384]);  clone_162 = None
        add_385 = torch.ops.aten.add.Tensor(view_689, 3)
        clamp_min_60 = torch.ops.aten.clamp_min.default(add_385, 0);  add_385 = None
        clamp_max_60 = torch.ops.aten.clamp_max.default(clamp_min_60, 6);  clamp_min_60 = None
        mul_457 = torch.ops.aten.mul.Tensor(view_689, clamp_max_60);  view_689 = clamp_max_60 = None
        div_89 = torch.ops.aten.div.Tensor(mul_457, 6);  mul_457 = None
        permute_229 = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        view_690 = torch.ops.aten.view.default(div_89, [128, 384]);  div_89 = None
        mm_113 = torch.ops.aten.mm.default(view_690, permute_229);  view_690 = permute_229 = None
        view_691 = torch.ops.aten.view.default(mm_113, [8, 16, 384]);  mm_113 = None
        view_692 = torch.ops.aten.view.default(view_691, [128, 384]);  view_691 = None
        add_386 = torch.ops.aten.add.Tensor(arg326_1, 1e-05);  arg326_1 = None
        sqrt_123 = torch.ops.aten.sqrt.default(add_386);  add_386 = None
        reciprocal_123 = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_458 = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        sub_151 = torch.ops.aten.sub.Tensor(view_692, arg325_1);  view_692 = arg325_1 = None
        mul_459 = torch.ops.aten.mul.Tensor(sub_151, mul_458);  sub_151 = mul_458 = None
        mul_460 = torch.ops.aten.mul.Tensor(mul_459, arg327_1);  mul_459 = arg327_1 = None
        add_387 = torch.ops.aten.add.Tensor(mul_460, arg328_1);  mul_460 = arg328_1 = None
        view_693 = torch.ops.aten.view.default(add_387, [8, 16, 384]);  add_387 = None
        add_388 = torch.ops.aten.add.Tensor(add_381, view_693);  add_381 = view_693 = None
        permute_230 = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        view_694 = torch.ops.aten.view.default(add_388, [128, 384])
        mm_114 = torch.ops.aten.mm.default(view_694, permute_230);  view_694 = permute_230 = None
        view_695 = torch.ops.aten.view.default(mm_114, [8, 16, 768]);  mm_114 = None
        view_696 = torch.ops.aten.view.default(view_695, [128, 768]);  view_695 = None
        add_389 = torch.ops.aten.add.Tensor(arg331_1, 1e-05);  arg331_1 = None
        sqrt_124 = torch.ops.aten.sqrt.default(add_389);  add_389 = None
        reciprocal_124 = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_461 = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        sub_152 = torch.ops.aten.sub.Tensor(view_696, arg330_1);  view_696 = arg330_1 = None
        mul_462 = torch.ops.aten.mul.Tensor(sub_152, mul_461);  sub_152 = mul_461 = None
        mul_463 = torch.ops.aten.mul.Tensor(mul_462, arg332_1);  mul_462 = arg332_1 = None
        add_390 = torch.ops.aten.add.Tensor(mul_463, arg333_1);  mul_463 = arg333_1 = None
        view_697 = torch.ops.aten.view.default(add_390, [8, 16, 768]);  add_390 = None
        add_391 = torch.ops.aten.add.Tensor(view_697, 3)
        clamp_min_61 = torch.ops.aten.clamp_min.default(add_391, 0);  add_391 = None
        clamp_max_61 = torch.ops.aten.clamp_max.default(clamp_min_61, 6);  clamp_min_61 = None
        mul_464 = torch.ops.aten.mul.Tensor(view_697, clamp_max_61);  view_697 = clamp_max_61 = None
        div_90 = torch.ops.aten.div.Tensor(mul_464, 6);  mul_464 = None
        permute_231 = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        view_698 = torch.ops.aten.view.default(div_90, [128, 768]);  div_90 = None
        mm_115 = torch.ops.aten.mm.default(view_698, permute_231);  view_698 = permute_231 = None
        view_699 = torch.ops.aten.view.default(mm_115, [8, 16, 384]);  mm_115 = None
        view_700 = torch.ops.aten.view.default(view_699, [128, 384]);  view_699 = None
        add_392 = torch.ops.aten.add.Tensor(arg336_1, 1e-05);  arg336_1 = None
        sqrt_125 = torch.ops.aten.sqrt.default(add_392);  add_392 = None
        reciprocal_125 = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_465 = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        sub_153 = torch.ops.aten.sub.Tensor(view_700, arg335_1);  view_700 = arg335_1 = None
        mul_466 = torch.ops.aten.mul.Tensor(sub_153, mul_465);  sub_153 = mul_465 = None
        mul_467 = torch.ops.aten.mul.Tensor(mul_466, arg337_1);  mul_466 = arg337_1 = None
        add_393 = torch.ops.aten.add.Tensor(mul_467, arg338_1);  mul_467 = arg338_1 = None
        view_701 = torch.ops.aten.view.default(add_393, [8, 16, 384]);  add_393 = None
        add_394 = torch.ops.aten.add.Tensor(add_388, view_701);  add_388 = view_701 = None
        mean_1 = torch.ops.aten.mean.dim(add_394, [1]);  add_394 = None
        add_395 = torch.ops.aten.add.Tensor(arg340_1, 1e-05);  arg340_1 = None
        sqrt_126 = torch.ops.aten.sqrt.default(add_395);  add_395 = None
        reciprocal_126 = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_468 = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        sub_154 = torch.ops.aten.sub.Tensor(mean_1, arg339_1);  arg339_1 = None
        mul_469 = torch.ops.aten.mul.Tensor(sub_154, mul_468);  sub_154 = mul_468 = None
        mul_470 = torch.ops.aten.mul.Tensor(mul_469, arg341_1);  mul_469 = arg341_1 = None
        add_396 = torch.ops.aten.add.Tensor(mul_470, arg342_1);  mul_470 = arg342_1 = None
        permute_232 = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg344_1, add_396, permute_232);  arg344_1 = add_396 = permute_232 = None
        add_397 = torch.ops.aten.add.Tensor(arg346_1, 1e-05);  arg346_1 = None
        sqrt_127 = torch.ops.aten.sqrt.default(add_397);  add_397 = None
        reciprocal_127 = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_471 = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        sub_155 = torch.ops.aten.sub.Tensor(mean_1, arg345_1);  mean_1 = arg345_1 = None
        mul_472 = torch.ops.aten.mul.Tensor(sub_155, mul_471);  sub_155 = mul_471 = None
        mul_473 = torch.ops.aten.mul.Tensor(mul_472, arg347_1);  mul_472 = arg347_1 = None
        add_398 = torch.ops.aten.add.Tensor(mul_473, arg348_1);  mul_473 = arg348_1 = None
        permute_233 = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg350_1, add_398, permute_233);  arg350_1 = add_398 = permute_233 = None
        add_399 = torch.ops.aten.add.Tensor(addmm_2, addmm_3);  addmm_2 = addmm_3 = None
        div_91 = torch.ops.aten.div.Tensor(add_399, 2);  add_399 = None
        return (div_91, index, index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, index_10, index_11, index_12, index_13)
        
def load_args(reader):
    buf0 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf0, (16, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 224, 224), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf2, (16,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf6, (32, 16, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf7, (32,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf8, (32,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf9, (32,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf10, (32,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 32, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128, 64, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 128), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3136, device=device(type='cuda', index=0))
    reader.tensor(buf26, (4, 196), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 307328, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf27, (196, 196), dtype=torch.int64, is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128, 128), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf31, (128,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256, 128), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128, 256), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf39, (128,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf40, (128,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf41, (128,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf42, (128,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256, 128), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf46, (256,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf47, (256,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3136, device=device(type='cuda', index=0))
    reader.tensor(buf48, (4, 196), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 307328, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf49, (196, 196), dtype=torch.int64, is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf50, (128, 128), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf51, (128,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf52, (128,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf53, (128,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf54, (128,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256, 128), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (128, 256), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (256, 128), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf67, (256,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf68, (256,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf69, (256,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3136, device=device(type='cuda', index=0))
    reader.tensor(buf70, (4, 196), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 307328, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf71, (196, 196), dtype=torch.int64, is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf72, (128, 128), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf73, (128,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (256, 128), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf78, (256,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf79, (256,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf80, (256,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf81, (256,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (128, 256), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (256, 128), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf88, (256,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf89, (256,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf90, (256,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3136, device=device(type='cuda', index=0))
    reader.tensor(buf92, (4, 196), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 307328, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf93, (196, 196), dtype=torch.int64, is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf94, (128, 128), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf95, (128,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf96, (128,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (256, 128), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf100, (256,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf101, (256,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf102, (256,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf103, (256,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (128, 256), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf105, (128,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (128,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 327680, device=device(type='cuda', index=0))
    reader.tensor(buf109, (640, 128), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf110, (640,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf111, (640,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf112, (640,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf113, (640,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (128, 128), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf116, (128,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf117, (128,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf118, (128,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 6272, device=device(type='cuda', index=0))
    reader.tensor(buf119, (8, 196), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 76832, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf120, (49, 196), dtype=torch.int64, is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256, 512), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf126, (512, 256), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf127, (512,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf130, (512,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf131, (256, 512), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf132, (256,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf133, (256,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf135, (256,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf136, (512, 256), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1568, device=device(type='cuda', index=0))
    reader.tensor(buf141, (8, 49), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf142, (49, 49), dtype=torch.int64, is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256, 256), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf144, (256,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf145, (256,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf146, (256,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf147, (256,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf148, (512, 256), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf149, (512,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf150, (512,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf151, (512,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf152, (512,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf153, (256, 512), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf154, (256,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf155, (256,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf156, (256,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf157, (256,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf158, (512, 256), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf159, (512,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf160, (512,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf161, (512,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1568, device=device(type='cuda', index=0))
    reader.tensor(buf163, (8, 49), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf164, (49, 49), dtype=torch.int64, is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf165, (256, 256), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf166, (256,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf167, (256,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf168, (256,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf169, (256,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf170, (512, 256), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf171, (512,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf172, (512,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf173, (512,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf175, (256, 512), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf176, (256,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf177, (256,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf178, (256,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf179, (256,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512, 256), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf183, (512,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf184, (512,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1568, device=device(type='cuda', index=0))
    reader.tensor(buf185, (8, 49), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf186, (49, 49), dtype=torch.int64, is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf187, (256, 256), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf188, (256,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf189, (256,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf190, (256,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf191, (256,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512, 256), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf194, (512,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf195, (512,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf196, (512,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf197, (256, 512), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf198, (256,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf199, (256,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf200, (256,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf201, (256,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf202, (512, 256), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf203, (512,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf205, (512,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf206, (512,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 1568, device=device(type='cuda', index=0))
    reader.tensor(buf207, (8, 49), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf208, (49, 49), dtype=torch.int64, is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf209, (256, 256), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf210, (256,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf211, (256,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf212, (256,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf213, (256,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf214, (512, 256), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf215, (512,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf216, (512,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf217, (512,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf218, (512,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf219, (256, 512), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf220, (256,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf221, (256,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf222, (256,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf223, (256,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 1310720, device=device(type='cuda', index=0))
    reader.tensor(buf224, (1280, 256), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf225, (1280,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf226, (1280,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1280,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf228, (1280,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf229, (256, 256), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf230, (256,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf231, (256,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf232, (256,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf233, (256,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 3136, device=device(type='cuda', index=0))
    reader.tensor(buf234, (16, 49), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 6272, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf235, (16, 49), dtype=torch.int64, is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf236, (384, 1024), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf237, (384,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf238, (384,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf239, (384,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf240, (384,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf241, (768, 384), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf242, (768,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf243, (768,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf244, (768,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf245, (768,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf246, (384, 768), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf247, (384,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf248, (384,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf249, (384,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf250, (384,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf251, (768, 384), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf252, (768,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf253, (768,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf254, (768,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf255, (768,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf256, (12, 16), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf257, (16, 16), dtype=torch.int64, is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf258, (384, 384), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf259, (384,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf260, (384,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf261, (384,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf262, (384,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf263, (768, 384), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf264, (768,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf265, (768,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf266, (768,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf267, (768,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf268, (384, 768), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf269, (384,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf270, (384,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf271, (384,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf272, (384,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf273, (768, 384), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf274, (768,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf275, (768,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf276, (768,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf277, (768,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf278, (12, 16), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf279, (16, 16), dtype=torch.int64, is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf280, (384, 384), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf281, (384,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf282, (384,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf283, (384,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf284, (384,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf285, (768, 384), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf286, (768,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf287, (768,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf288, (768,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf289, (768,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf290, (384, 768), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf291, (384,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf292, (384,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf293, (384,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf294, (384,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf295, (768, 384), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf296, (768,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf297, (768,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf298, (768,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf299, (768,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf300, (12, 16), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf301, (16, 16), dtype=torch.int64, is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf302, (384, 384), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf303, (384,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf304, (384,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf305, (384,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf306, (384,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf307, (768, 384), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf308, (768,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf309, (768,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf310, (768,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf311, (768,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf312, (384, 768), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf313, (384,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf314, (384,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf315, (384,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf316, (384,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf317, (768, 384), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf318, (768,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf319, (768,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf320, (768,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf321, (768,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf322, (12, 16), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf323, (16, 16), dtype=torch.int64, is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf324, (384, 384), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf325, (384,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf326, (384,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf327, (384,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf328, (384,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf329, (768, 384), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf330, (768,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf331, (768,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf332, (768,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf333, (768,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf334, (384, 768), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf335, (384,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf336, (384,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf337, (384,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf338, (384,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf339, (384,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf340, (384,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf341, (384,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf342, (384,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf343, (1000, 384), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1000,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf345, (384,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf346, (384,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf347, (384,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf348, (384,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1000, 384), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1000,), is_leaf=True)  # arg350_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)