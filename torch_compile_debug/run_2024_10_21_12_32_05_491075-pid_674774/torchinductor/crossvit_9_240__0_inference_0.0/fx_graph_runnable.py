
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1):
        convolution_2 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [12, 12], [0, 0], [1, 1], False, [0, 0], 1);  arg1_1 = arg2_1 = None
        view_255 = torch.ops.aten.view.default(convolution_2, [8, 128, 400]);  convolution_2 = None
        permute_142 = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        expand_26 = torch.ops.aten.expand.default(arg3_1, [8, -1, -1]);  arg3_1 = None
        cat_15 = torch.ops.aten.cat.default([expand_26, permute_142], 1);  expand_26 = permute_142 = None
        add_181 = torch.ops.aten.add.Tensor(cat_15, arg4_1);  cat_15 = arg4_1 = None
        iota_2 = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(iota_2, torch.float32);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(iota_3, torch.float32);  iota_3 = None
        add_182 = torch.ops.aten.add.Tensor(convert_element_type_5, 0.5);  convert_element_type_5 = None
        mul_212 = torch.ops.aten.mul.Tensor(add_182, 1.0714285714285714);  add_182 = None
        sub_72 = torch.ops.aten.sub.Tensor(mul_212, 0.5);  mul_212 = None
        add_183 = torch.ops.aten.add.Tensor(convert_element_type_4, 0.5);  convert_element_type_4 = None
        mul_213 = torch.ops.aten.mul.Tensor(add_183, 1.0714285714285714);  add_183 = None
        sub_73 = torch.ops.aten.sub.Tensor(mul_213, 0.5);  mul_213 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(sub_73, -1);  sub_73 = None
        floor_2 = torch.ops.aten.floor.default(sub_72)
        floor_3 = torch.ops.aten.floor.default(unsqueeze_1)
        sub_74 = torch.ops.aten.sub.Tensor(unsqueeze_1, floor_3);  unsqueeze_1 = None
        clamp_min_34 = torch.ops.aten.clamp_min.default(sub_74, 0.0);  sub_74 = None
        clamp_max_34 = torch.ops.aten.clamp_max.default(clamp_min_34, 1.0);  clamp_min_34 = None
        sub_75 = torch.ops.aten.sub.Tensor(sub_72, floor_2);  sub_72 = None
        clamp_min_35 = torch.ops.aten.clamp_min.default(sub_75, 0.0);  sub_75 = None
        clamp_max_35 = torch.ops.aten.clamp_max.default(clamp_min_35, 1.0);  clamp_min_35 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(floor_2, torch.int64);  floor_2 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(floor_3, torch.int64);  floor_3 = None
        sub_76 = torch.ops.aten.sub.Tensor(convert_element_type_7, 1)
        add_184 = torch.ops.aten.add.Tensor(convert_element_type_7, 1)
        add_185 = torch.ops.aten.add.Tensor(convert_element_type_7, 2)
        sub_77 = torch.ops.aten.sub.Tensor(convert_element_type_6, 1)
        add_186 = torch.ops.aten.add.Tensor(convert_element_type_6, 1)
        add_187 = torch.ops.aten.add.Tensor(convert_element_type_6, 2)
        add_188 = torch.ops.aten.add.Tensor(clamp_max_35, 1.0)
        mul_214 = torch.ops.aten.mul.Tensor(add_188, -0.75)
        sub_78 = torch.ops.aten.sub.Tensor(mul_214, -3.75);  mul_214 = None
        mul_215 = torch.ops.aten.mul.Tensor(sub_78, add_188);  sub_78 = None
        add_189 = torch.ops.aten.add.Tensor(mul_215, -6.0);  mul_215 = None
        mul_216 = torch.ops.aten.mul.Tensor(add_189, add_188);  add_189 = add_188 = None
        sub_79 = torch.ops.aten.sub.Tensor(mul_216, -3.0);  mul_216 = None
        mul_217 = torch.ops.aten.mul.Tensor(clamp_max_35, 1.25)
        sub_80 = torch.ops.aten.sub.Tensor(mul_217, 2.25);  mul_217 = None
        mul_218 = torch.ops.aten.mul.Tensor(sub_80, clamp_max_35);  sub_80 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, clamp_max_35);  mul_218 = None
        add_190 = torch.ops.aten.add.Tensor(mul_219, 1);  mul_219 = None
        sub_81 = torch.ops.aten.sub.Tensor(1.0, clamp_max_35)
        mul_220 = torch.ops.aten.mul.Tensor(sub_81, 1.25)
        sub_82 = torch.ops.aten.sub.Tensor(mul_220, 2.25);  mul_220 = None
        mul_221 = torch.ops.aten.mul.Tensor(sub_82, sub_81);  sub_82 = None
        mul_222 = torch.ops.aten.mul.Tensor(mul_221, sub_81);  mul_221 = sub_81 = None
        add_191 = torch.ops.aten.add.Tensor(mul_222, 1);  mul_222 = None
        sub_83 = torch.ops.aten.sub.Tensor(2.0, clamp_max_35);  clamp_max_35 = None
        mul_223 = torch.ops.aten.mul.Tensor(sub_83, -0.75)
        sub_84 = torch.ops.aten.sub.Tensor(mul_223, -3.75);  mul_223 = None
        mul_224 = torch.ops.aten.mul.Tensor(sub_84, sub_83);  sub_84 = None
        add_192 = torch.ops.aten.add.Tensor(mul_224, -6.0);  mul_224 = None
        mul_225 = torch.ops.aten.mul.Tensor(add_192, sub_83);  add_192 = sub_83 = None
        sub_85 = torch.ops.aten.sub.Tensor(mul_225, -3.0);  mul_225 = None
        add_193 = torch.ops.aten.add.Tensor(clamp_max_34, 1.0)
        mul_226 = torch.ops.aten.mul.Tensor(add_193, -0.75)
        sub_86 = torch.ops.aten.sub.Tensor(mul_226, -3.75);  mul_226 = None
        mul_227 = torch.ops.aten.mul.Tensor(sub_86, add_193);  sub_86 = None
        add_194 = torch.ops.aten.add.Tensor(mul_227, -6.0);  mul_227 = None
        mul_228 = torch.ops.aten.mul.Tensor(add_194, add_193);  add_194 = add_193 = None
        sub_87 = torch.ops.aten.sub.Tensor(mul_228, -3.0);  mul_228 = None
        mul_229 = torch.ops.aten.mul.Tensor(clamp_max_34, 1.25)
        sub_88 = torch.ops.aten.sub.Tensor(mul_229, 2.25);  mul_229 = None
        mul_230 = torch.ops.aten.mul.Tensor(sub_88, clamp_max_34);  sub_88 = None
        mul_231 = torch.ops.aten.mul.Tensor(mul_230, clamp_max_34);  mul_230 = None
        add_195 = torch.ops.aten.add.Tensor(mul_231, 1);  mul_231 = None
        sub_89 = torch.ops.aten.sub.Tensor(1.0, clamp_max_34)
        mul_232 = torch.ops.aten.mul.Tensor(sub_89, 1.25)
        sub_90 = torch.ops.aten.sub.Tensor(mul_232, 2.25);  mul_232 = None
        mul_233 = torch.ops.aten.mul.Tensor(sub_90, sub_89);  sub_90 = None
        mul_234 = torch.ops.aten.mul.Tensor(mul_233, sub_89);  mul_233 = sub_89 = None
        add_196 = torch.ops.aten.add.Tensor(mul_234, 1);  mul_234 = None
        sub_91 = torch.ops.aten.sub.Tensor(2.0, clamp_max_34);  clamp_max_34 = None
        mul_235 = torch.ops.aten.mul.Tensor(sub_91, -0.75)
        sub_92 = torch.ops.aten.sub.Tensor(mul_235, -3.75);  mul_235 = None
        mul_236 = torch.ops.aten.mul.Tensor(sub_92, sub_91);  sub_92 = None
        add_197 = torch.ops.aten.add.Tensor(mul_236, -6.0);  mul_236 = None
        mul_237 = torch.ops.aten.mul.Tensor(add_197, sub_91);  add_197 = sub_91 = None
        sub_93 = torch.ops.aten.sub.Tensor(mul_237, -3.0);  mul_237 = None
        clamp_min_36 = torch.ops.aten.clamp_min.default(sub_76, 0)
        clamp_max_36 = torch.ops.aten.clamp_max.default(clamp_min_36, 239);  clamp_min_36 = None
        clamp_min_37 = torch.ops.aten.clamp_min.default(sub_77, 0)
        clamp_max_37 = torch.ops.aten.clamp_max.default(clamp_min_37, 239);  clamp_min_37 = None
        _unsafe_index_16 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_36, clamp_max_37]);  clamp_max_36 = clamp_max_37 = None
        clamp_min_38 = torch.ops.aten.clamp_min.default(sub_76, 0)
        clamp_max_38 = torch.ops.aten.clamp_max.default(clamp_min_38, 239);  clamp_min_38 = None
        clamp_min_39 = torch.ops.aten.clamp_min.default(convert_element_type_6, 0)
        clamp_max_39 = torch.ops.aten.clamp_max.default(clamp_min_39, 239);  clamp_min_39 = None
        _unsafe_index_17 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_38, clamp_max_39]);  clamp_max_38 = clamp_max_39 = None
        clamp_min_40 = torch.ops.aten.clamp_min.default(sub_76, 0)
        clamp_max_40 = torch.ops.aten.clamp_max.default(clamp_min_40, 239);  clamp_min_40 = None
        clamp_min_41 = torch.ops.aten.clamp_min.default(add_186, 0)
        clamp_max_41 = torch.ops.aten.clamp_max.default(clamp_min_41, 239);  clamp_min_41 = None
        _unsafe_index_18 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_40, clamp_max_41]);  clamp_max_40 = clamp_max_41 = None
        clamp_min_42 = torch.ops.aten.clamp_min.default(sub_76, 0);  sub_76 = None
        clamp_max_42 = torch.ops.aten.clamp_max.default(clamp_min_42, 239);  clamp_min_42 = None
        clamp_min_43 = torch.ops.aten.clamp_min.default(add_187, 0)
        clamp_max_43 = torch.ops.aten.clamp_max.default(clamp_min_43, 239);  clamp_min_43 = None
        _unsafe_index_19 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_42, clamp_max_43]);  clamp_max_42 = clamp_max_43 = None
        mul_238 = torch.ops.aten.mul.Tensor(_unsafe_index_16, sub_79);  _unsafe_index_16 = None
        mul_239 = torch.ops.aten.mul.Tensor(_unsafe_index_17, add_190);  _unsafe_index_17 = None
        add_198 = torch.ops.aten.add.Tensor(mul_238, mul_239);  mul_238 = mul_239 = None
        mul_240 = torch.ops.aten.mul.Tensor(_unsafe_index_18, add_191);  _unsafe_index_18 = None
        add_199 = torch.ops.aten.add.Tensor(add_198, mul_240);  add_198 = mul_240 = None
        mul_241 = torch.ops.aten.mul.Tensor(_unsafe_index_19, sub_85);  _unsafe_index_19 = None
        add_200 = torch.ops.aten.add.Tensor(add_199, mul_241);  add_199 = mul_241 = None
        clamp_min_44 = torch.ops.aten.clamp_min.default(convert_element_type_7, 0)
        clamp_max_44 = torch.ops.aten.clamp_max.default(clamp_min_44, 239);  clamp_min_44 = None
        clamp_min_45 = torch.ops.aten.clamp_min.default(sub_77, 0)
        clamp_max_45 = torch.ops.aten.clamp_max.default(clamp_min_45, 239);  clamp_min_45 = None
        _unsafe_index_20 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_44, clamp_max_45]);  clamp_max_44 = clamp_max_45 = None
        clamp_min_46 = torch.ops.aten.clamp_min.default(convert_element_type_7, 0)
        clamp_max_46 = torch.ops.aten.clamp_max.default(clamp_min_46, 239);  clamp_min_46 = None
        clamp_min_47 = torch.ops.aten.clamp_min.default(convert_element_type_6, 0)
        clamp_max_47 = torch.ops.aten.clamp_max.default(clamp_min_47, 239);  clamp_min_47 = None
        _unsafe_index_21 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_46, clamp_max_47]);  clamp_max_46 = clamp_max_47 = None
        clamp_min_48 = torch.ops.aten.clamp_min.default(convert_element_type_7, 0)
        clamp_max_48 = torch.ops.aten.clamp_max.default(clamp_min_48, 239);  clamp_min_48 = None
        clamp_min_49 = torch.ops.aten.clamp_min.default(add_186, 0)
        clamp_max_49 = torch.ops.aten.clamp_max.default(clamp_min_49, 239);  clamp_min_49 = None
        _unsafe_index_22 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_48, clamp_max_49]);  clamp_max_48 = clamp_max_49 = None
        clamp_min_50 = torch.ops.aten.clamp_min.default(convert_element_type_7, 0);  convert_element_type_7 = None
        clamp_max_50 = torch.ops.aten.clamp_max.default(clamp_min_50, 239);  clamp_min_50 = None
        clamp_min_51 = torch.ops.aten.clamp_min.default(add_187, 0)
        clamp_max_51 = torch.ops.aten.clamp_max.default(clamp_min_51, 239);  clamp_min_51 = None
        _unsafe_index_23 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_50, clamp_max_51]);  clamp_max_50 = clamp_max_51 = None
        mul_242 = torch.ops.aten.mul.Tensor(_unsafe_index_20, sub_79);  _unsafe_index_20 = None
        mul_243 = torch.ops.aten.mul.Tensor(_unsafe_index_21, add_190);  _unsafe_index_21 = None
        add_201 = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        mul_244 = torch.ops.aten.mul.Tensor(_unsafe_index_22, add_191);  _unsafe_index_22 = None
        add_202 = torch.ops.aten.add.Tensor(add_201, mul_244);  add_201 = mul_244 = None
        mul_245 = torch.ops.aten.mul.Tensor(_unsafe_index_23, sub_85);  _unsafe_index_23 = None
        add_203 = torch.ops.aten.add.Tensor(add_202, mul_245);  add_202 = mul_245 = None
        clamp_min_52 = torch.ops.aten.clamp_min.default(add_184, 0)
        clamp_max_52 = torch.ops.aten.clamp_max.default(clamp_min_52, 239);  clamp_min_52 = None
        clamp_min_53 = torch.ops.aten.clamp_min.default(sub_77, 0)
        clamp_max_53 = torch.ops.aten.clamp_max.default(clamp_min_53, 239);  clamp_min_53 = None
        _unsafe_index_24 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_52, clamp_max_53]);  clamp_max_52 = clamp_max_53 = None
        clamp_min_54 = torch.ops.aten.clamp_min.default(add_184, 0)
        clamp_max_54 = torch.ops.aten.clamp_max.default(clamp_min_54, 239);  clamp_min_54 = None
        clamp_min_55 = torch.ops.aten.clamp_min.default(convert_element_type_6, 0)
        clamp_max_55 = torch.ops.aten.clamp_max.default(clamp_min_55, 239);  clamp_min_55 = None
        _unsafe_index_25 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_54, clamp_max_55]);  clamp_max_54 = clamp_max_55 = None
        clamp_min_56 = torch.ops.aten.clamp_min.default(add_184, 0)
        clamp_max_56 = torch.ops.aten.clamp_max.default(clamp_min_56, 239);  clamp_min_56 = None
        clamp_min_57 = torch.ops.aten.clamp_min.default(add_186, 0)
        clamp_max_57 = torch.ops.aten.clamp_max.default(clamp_min_57, 239);  clamp_min_57 = None
        _unsafe_index_26 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_56, clamp_max_57]);  clamp_max_56 = clamp_max_57 = None
        clamp_min_58 = torch.ops.aten.clamp_min.default(add_184, 0);  add_184 = None
        clamp_max_58 = torch.ops.aten.clamp_max.default(clamp_min_58, 239);  clamp_min_58 = None
        clamp_min_59 = torch.ops.aten.clamp_min.default(add_187, 0)
        clamp_max_59 = torch.ops.aten.clamp_max.default(clamp_min_59, 239);  clamp_min_59 = None
        _unsafe_index_27 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_58, clamp_max_59]);  clamp_max_58 = clamp_max_59 = None
        mul_246 = torch.ops.aten.mul.Tensor(_unsafe_index_24, sub_79);  _unsafe_index_24 = None
        mul_247 = torch.ops.aten.mul.Tensor(_unsafe_index_25, add_190);  _unsafe_index_25 = None
        add_204 = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        mul_248 = torch.ops.aten.mul.Tensor(_unsafe_index_26, add_191);  _unsafe_index_26 = None
        add_205 = torch.ops.aten.add.Tensor(add_204, mul_248);  add_204 = mul_248 = None
        mul_249 = torch.ops.aten.mul.Tensor(_unsafe_index_27, sub_85);  _unsafe_index_27 = None
        add_206 = torch.ops.aten.add.Tensor(add_205, mul_249);  add_205 = mul_249 = None
        clamp_min_60 = torch.ops.aten.clamp_min.default(add_185, 0)
        clamp_max_60 = torch.ops.aten.clamp_max.default(clamp_min_60, 239);  clamp_min_60 = None
        clamp_min_61 = torch.ops.aten.clamp_min.default(sub_77, 0);  sub_77 = None
        clamp_max_61 = torch.ops.aten.clamp_max.default(clamp_min_61, 239);  clamp_min_61 = None
        _unsafe_index_28 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_60, clamp_max_61]);  clamp_max_60 = clamp_max_61 = None
        clamp_min_62 = torch.ops.aten.clamp_min.default(add_185, 0)
        clamp_max_62 = torch.ops.aten.clamp_max.default(clamp_min_62, 239);  clamp_min_62 = None
        clamp_min_63 = torch.ops.aten.clamp_min.default(convert_element_type_6, 0);  convert_element_type_6 = None
        clamp_max_63 = torch.ops.aten.clamp_max.default(clamp_min_63, 239);  clamp_min_63 = None
        _unsafe_index_29 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_62, clamp_max_63]);  clamp_max_62 = clamp_max_63 = None
        clamp_min_64 = torch.ops.aten.clamp_min.default(add_185, 0)
        clamp_max_64 = torch.ops.aten.clamp_max.default(clamp_min_64, 239);  clamp_min_64 = None
        clamp_min_65 = torch.ops.aten.clamp_min.default(add_186, 0);  add_186 = None
        clamp_max_65 = torch.ops.aten.clamp_max.default(clamp_min_65, 239);  clamp_min_65 = None
        _unsafe_index_30 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_64, clamp_max_65]);  clamp_max_64 = clamp_max_65 = None
        clamp_min_66 = torch.ops.aten.clamp_min.default(add_185, 0);  add_185 = None
        clamp_max_66 = torch.ops.aten.clamp_max.default(clamp_min_66, 239);  clamp_min_66 = None
        clamp_min_67 = torch.ops.aten.clamp_min.default(add_187, 0);  add_187 = None
        clamp_max_67 = torch.ops.aten.clamp_max.default(clamp_min_67, 239);  clamp_min_67 = None
        _unsafe_index_31 = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, clamp_max_66, clamp_max_67]);  arg0_1 = clamp_max_66 = clamp_max_67 = None
        mul_250 = torch.ops.aten.mul.Tensor(_unsafe_index_28, sub_79);  _unsafe_index_28 = sub_79 = None
        mul_251 = torch.ops.aten.mul.Tensor(_unsafe_index_29, add_190);  _unsafe_index_29 = add_190 = None
        add_207 = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        mul_252 = torch.ops.aten.mul.Tensor(_unsafe_index_30, add_191);  _unsafe_index_30 = add_191 = None
        add_208 = torch.ops.aten.add.Tensor(add_207, mul_252);  add_207 = mul_252 = None
        mul_253 = torch.ops.aten.mul.Tensor(_unsafe_index_31, sub_85);  _unsafe_index_31 = sub_85 = None
        add_209 = torch.ops.aten.add.Tensor(add_208, mul_253);  add_208 = mul_253 = None
        mul_254 = torch.ops.aten.mul.Tensor(add_200, sub_87);  add_200 = sub_87 = None
        mul_255 = torch.ops.aten.mul.Tensor(add_203, add_195);  add_203 = add_195 = None
        add_210 = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        mul_256 = torch.ops.aten.mul.Tensor(add_206, add_196);  add_206 = add_196 = None
        add_211 = torch.ops.aten.add.Tensor(add_210, mul_256);  add_210 = mul_256 = None
        mul_257 = torch.ops.aten.mul.Tensor(add_209, sub_93);  add_209 = sub_93 = None
        add_212 = torch.ops.aten.add.Tensor(add_211, mul_257);  add_211 = mul_257 = None
        convolution_3 = torch.ops.aten.convolution.default(add_212, arg5_1, arg6_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  add_212 = arg5_1 = arg6_1 = None
        view_256 = torch.ops.aten.view.default(convolution_3, [8, 256, 196]);  convolution_3 = None
        permute_143 = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        expand_27 = torch.ops.aten.expand.default(arg7_1, [8, -1, -1]);  arg7_1 = None
        cat_16 = torch.ops.aten.cat.default([expand_27, permute_143], 1);  expand_27 = permute_143 = None
        add_213 = torch.ops.aten.add.Tensor(cat_16, arg8_1);  cat_16 = arg8_1 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
        getitem_172 = var_mean_44[0]
        getitem_173 = var_mean_44[1];  var_mean_44 = None
        add_214 = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_94 = torch.ops.aten.sub.Tensor(add_181, getitem_173);  getitem_173 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_44);  sub_94 = rsqrt_44 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, arg9_1);  mul_258 = arg9_1 = None
        add_215 = torch.ops.aten.add.Tensor(mul_259, arg10_1);  mul_259 = arg10_1 = None
        view_257 = torch.ops.aten.view.default(add_215, [3208, 128]);  add_215 = None
        permute_144 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg12_1, view_257, permute_144);  arg12_1 = view_257 = permute_144 = None
        view_258 = torch.ops.aten.view.default(addmm_80, [8, 401, 384]);  addmm_80 = None
        view_259 = torch.ops.aten.view.default(view_258, [8, 401, 3, 4, 32]);  view_258 = None
        permute_145 = torch.ops.aten.permute.default(view_259, [2, 0, 3, 1, 4]);  view_259 = None
        unbind_12 = torch.ops.aten.unbind.int(permute_145);  permute_145 = None
        getitem_174 = unbind_12[0]
        getitem_175 = unbind_12[1]
        getitem_176 = unbind_12[2];  unbind_12 = None
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_174, getitem_175, getitem_176, None, False);  getitem_174 = getitem_175 = getitem_176 = None
        getitem_177 = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
        permute_146 = torch.ops.aten.permute.default(getitem_177, [0, 2, 1, 3]);  getitem_177 = None
        view_260 = torch.ops.aten.view.default(permute_146, [8, 401, 128]);  permute_146 = None
        view_261 = torch.ops.aten.view.default(view_260, [3208, 128]);  view_260 = None
        permute_147 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg14_1, view_261, permute_147);  arg14_1 = view_261 = permute_147 = None
        view_262 = torch.ops.aten.view.default(addmm_81, [8, 401, 128]);  addmm_81 = None
        add_216 = torch.ops.aten.add.Tensor(add_181, view_262);  add_181 = view_262 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_216, [2], correction = 0, keepdim = True)
        getitem_181 = var_mean_45[0]
        getitem_182 = var_mean_45[1];  var_mean_45 = None
        add_217 = torch.ops.aten.add.Tensor(getitem_181, 1e-06);  getitem_181 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        sub_95 = torch.ops.aten.sub.Tensor(add_216, getitem_182);  getitem_182 = None
        mul_260 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_45);  sub_95 = rsqrt_45 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, arg15_1);  mul_260 = arg15_1 = None
        add_218 = torch.ops.aten.add.Tensor(mul_261, arg16_1);  mul_261 = arg16_1 = None
        view_263 = torch.ops.aten.view.default(add_218, [3208, 128]);  add_218 = None
        permute_148 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg18_1, view_263, permute_148);  arg18_1 = view_263 = permute_148 = None
        view_264 = torch.ops.aten.view.default(addmm_82, [8, 401, 384]);  addmm_82 = None
        mul_262 = torch.ops.aten.mul.Tensor(view_264, 0.5)
        mul_263 = torch.ops.aten.mul.Tensor(view_264, 0.7071067811865476);  view_264 = None
        erf_24 = torch.ops.aten.erf.default(mul_263);  mul_263 = None
        add_219 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_262, add_219);  mul_262 = add_219 = None
        view_265 = torch.ops.aten.view.default(mul_264, [3208, 384]);  mul_264 = None
        permute_149 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg20_1, view_265, permute_149);  arg20_1 = view_265 = permute_149 = None
        view_266 = torch.ops.aten.view.default(addmm_83, [8, 401, 128]);  addmm_83 = None
        add_220 = torch.ops.aten.add.Tensor(add_216, view_266);  add_216 = view_266 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_213, [2], correction = 0, keepdim = True)
        getitem_183 = var_mean_46[0]
        getitem_184 = var_mean_46[1];  var_mean_46 = None
        add_221 = torch.ops.aten.add.Tensor(getitem_183, 1e-06);  getitem_183 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        sub_96 = torch.ops.aten.sub.Tensor(add_213, getitem_184);  getitem_184 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_46);  sub_96 = rsqrt_46 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, arg21_1);  mul_265 = arg21_1 = None
        add_222 = torch.ops.aten.add.Tensor(mul_266, arg22_1);  mul_266 = arg22_1 = None
        view_267 = torch.ops.aten.view.default(add_222, [1576, 256]);  add_222 = None
        permute_150 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg24_1, view_267, permute_150);  arg24_1 = view_267 = permute_150 = None
        view_268 = torch.ops.aten.view.default(addmm_84, [8, 197, 768]);  addmm_84 = None
        view_269 = torch.ops.aten.view.default(view_268, [8, 197, 3, 4, 64]);  view_268 = None
        permute_151 = torch.ops.aten.permute.default(view_269, [2, 0, 3, 1, 4]);  view_269 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_151);  permute_151 = None
        getitem_185 = unbind_13[0]
        getitem_186 = unbind_13[1]
        getitem_187 = unbind_13[2];  unbind_13 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_185, getitem_186, getitem_187, None, False);  getitem_185 = getitem_186 = getitem_187 = None
        getitem_188 = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        permute_152 = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None
        view_270 = torch.ops.aten.view.default(permute_152, [8, 197, 256]);  permute_152 = None
        view_271 = torch.ops.aten.view.default(view_270, [1576, 256]);  view_270 = None
        permute_153 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg26_1, view_271, permute_153);  arg26_1 = view_271 = permute_153 = None
        view_272 = torch.ops.aten.view.default(addmm_85, [8, 197, 256]);  addmm_85 = None
        add_223 = torch.ops.aten.add.Tensor(add_213, view_272);  add_213 = view_272 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_223, [2], correction = 0, keepdim = True)
        getitem_192 = var_mean_47[0]
        getitem_193 = var_mean_47[1];  var_mean_47 = None
        add_224 = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        sub_97 = torch.ops.aten.sub.Tensor(add_223, getitem_193);  getitem_193 = None
        mul_267 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_47);  sub_97 = rsqrt_47 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, arg27_1);  mul_267 = arg27_1 = None
        add_225 = torch.ops.aten.add.Tensor(mul_268, arg28_1);  mul_268 = arg28_1 = None
        view_273 = torch.ops.aten.view.default(add_225, [1576, 256]);  add_225 = None
        permute_154 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg30_1, view_273, permute_154);  arg30_1 = view_273 = permute_154 = None
        view_274 = torch.ops.aten.view.default(addmm_86, [8, 197, 768]);  addmm_86 = None
        mul_269 = torch.ops.aten.mul.Tensor(view_274, 0.5)
        mul_270 = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
        erf_25 = torch.ops.aten.erf.default(mul_270);  mul_270 = None
        add_226 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_271 = torch.ops.aten.mul.Tensor(mul_269, add_226);  mul_269 = add_226 = None
        view_275 = torch.ops.aten.view.default(mul_271, [1576, 768]);  mul_271 = None
        permute_155 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg32_1, view_275, permute_155);  arg32_1 = view_275 = permute_155 = None
        view_276 = torch.ops.aten.view.default(addmm_87, [8, 197, 256]);  addmm_87 = None
        add_227 = torch.ops.aten.add.Tensor(add_223, view_276);  add_223 = view_276 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_227, [2], correction = 0, keepdim = True)
        getitem_194 = var_mean_48[0]
        getitem_195 = var_mean_48[1];  var_mean_48 = None
        add_228 = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        sub_98 = torch.ops.aten.sub.Tensor(add_227, getitem_195);  getitem_195 = None
        mul_272 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_48);  sub_98 = rsqrt_48 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, arg33_1);  mul_272 = arg33_1 = None
        add_229 = torch.ops.aten.add.Tensor(mul_273, arg34_1);  mul_273 = arg34_1 = None
        view_277 = torch.ops.aten.view.default(add_229, [1576, 256]);  add_229 = None
        permute_156 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg36_1, view_277, permute_156);  arg36_1 = view_277 = permute_156 = None
        view_278 = torch.ops.aten.view.default(addmm_88, [8, 197, 768]);  addmm_88 = None
        view_279 = torch.ops.aten.view.default(view_278, [8, 197, 3, 4, 64]);  view_278 = None
        permute_157 = torch.ops.aten.permute.default(view_279, [2, 0, 3, 1, 4]);  view_279 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
        getitem_196 = unbind_14[0]
        getitem_197 = unbind_14[1]
        getitem_198 = unbind_14[2];  unbind_14 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_196, getitem_197, getitem_198, None, False);  getitem_196 = getitem_197 = getitem_198 = None
        getitem_199 = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        permute_158 = torch.ops.aten.permute.default(getitem_199, [0, 2, 1, 3]);  getitem_199 = None
        view_280 = torch.ops.aten.view.default(permute_158, [8, 197, 256]);  permute_158 = None
        view_281 = torch.ops.aten.view.default(view_280, [1576, 256]);  view_280 = None
        permute_159 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg38_1, view_281, permute_159);  arg38_1 = view_281 = permute_159 = None
        view_282 = torch.ops.aten.view.default(addmm_89, [8, 197, 256]);  addmm_89 = None
        add_230 = torch.ops.aten.add.Tensor(add_227, view_282);  add_227 = view_282 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(add_230, [2], correction = 0, keepdim = True)
        getitem_203 = var_mean_49[0]
        getitem_204 = var_mean_49[1];  var_mean_49 = None
        add_231 = torch.ops.aten.add.Tensor(getitem_203, 1e-06);  getitem_203 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        sub_99 = torch.ops.aten.sub.Tensor(add_230, getitem_204);  getitem_204 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_99, rsqrt_49);  sub_99 = rsqrt_49 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, arg39_1);  mul_274 = arg39_1 = None
        add_232 = torch.ops.aten.add.Tensor(mul_275, arg40_1);  mul_275 = arg40_1 = None
        view_283 = torch.ops.aten.view.default(add_232, [1576, 256]);  add_232 = None
        permute_160 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg42_1, view_283, permute_160);  arg42_1 = view_283 = permute_160 = None
        view_284 = torch.ops.aten.view.default(addmm_90, [8, 197, 768]);  addmm_90 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_284, 0.5)
        mul_277 = torch.ops.aten.mul.Tensor(view_284, 0.7071067811865476);  view_284 = None
        erf_26 = torch.ops.aten.erf.default(mul_277);  mul_277 = None
        add_233 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_276, add_233);  mul_276 = add_233 = None
        view_285 = torch.ops.aten.view.default(mul_278, [1576, 768]);  mul_278 = None
        permute_161 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg44_1, view_285, permute_161);  arg44_1 = view_285 = permute_161 = None
        view_286 = torch.ops.aten.view.default(addmm_91, [8, 197, 256]);  addmm_91 = None
        add_234 = torch.ops.aten.add.Tensor(add_230, view_286);  add_230 = view_286 = None
        var_mean_50 = torch.ops.aten.var_mean.correction(add_234, [2], correction = 0, keepdim = True)
        getitem_205 = var_mean_50[0]
        getitem_206 = var_mean_50[1];  var_mean_50 = None
        add_235 = torch.ops.aten.add.Tensor(getitem_205, 1e-06);  getitem_205 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        sub_100 = torch.ops.aten.sub.Tensor(add_234, getitem_206);  getitem_206 = None
        mul_279 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_50);  sub_100 = rsqrt_50 = None
        mul_280 = torch.ops.aten.mul.Tensor(mul_279, arg45_1);  mul_279 = arg45_1 = None
        add_236 = torch.ops.aten.add.Tensor(mul_280, arg46_1);  mul_280 = arg46_1 = None
        view_287 = torch.ops.aten.view.default(add_236, [1576, 256]);  add_236 = None
        permute_162 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg48_1, view_287, permute_162);  arg48_1 = view_287 = permute_162 = None
        view_288 = torch.ops.aten.view.default(addmm_92, [8, 197, 768]);  addmm_92 = None
        view_289 = torch.ops.aten.view.default(view_288, [8, 197, 3, 4, 64]);  view_288 = None
        permute_163 = torch.ops.aten.permute.default(view_289, [2, 0, 3, 1, 4]);  view_289 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_163);  permute_163 = None
        getitem_207 = unbind_15[0]
        getitem_208 = unbind_15[1]
        getitem_209 = unbind_15[2];  unbind_15 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_207, getitem_208, getitem_209, None, False);  getitem_207 = getitem_208 = getitem_209 = None
        getitem_210 = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        permute_164 = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None
        view_290 = torch.ops.aten.view.default(permute_164, [8, 197, 256]);  permute_164 = None
        view_291 = torch.ops.aten.view.default(view_290, [1576, 256]);  view_290 = None
        permute_165 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg50_1, view_291, permute_165);  arg50_1 = view_291 = permute_165 = None
        view_292 = torch.ops.aten.view.default(addmm_93, [8, 197, 256]);  addmm_93 = None
        add_237 = torch.ops.aten.add.Tensor(add_234, view_292);  add_234 = view_292 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(add_237, [2], correction = 0, keepdim = True)
        getitem_214 = var_mean_51[0]
        getitem_215 = var_mean_51[1];  var_mean_51 = None
        add_238 = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        sub_101 = torch.ops.aten.sub.Tensor(add_237, getitem_215);  getitem_215 = None
        mul_281 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_51);  sub_101 = rsqrt_51 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_281, arg51_1);  mul_281 = arg51_1 = None
        add_239 = torch.ops.aten.add.Tensor(mul_282, arg52_1);  mul_282 = arg52_1 = None
        view_293 = torch.ops.aten.view.default(add_239, [1576, 256]);  add_239 = None
        permute_166 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg54_1, view_293, permute_166);  arg54_1 = view_293 = permute_166 = None
        view_294 = torch.ops.aten.view.default(addmm_94, [8, 197, 768]);  addmm_94 = None
        mul_283 = torch.ops.aten.mul.Tensor(view_294, 0.5)
        mul_284 = torch.ops.aten.mul.Tensor(view_294, 0.7071067811865476);  view_294 = None
        erf_27 = torch.ops.aten.erf.default(mul_284);  mul_284 = None
        add_240 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_285 = torch.ops.aten.mul.Tensor(mul_283, add_240);  mul_283 = add_240 = None
        view_295 = torch.ops.aten.view.default(mul_285, [1576, 768]);  mul_285 = None
        permute_167 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg56_1, view_295, permute_167);  arg56_1 = view_295 = permute_167 = None
        view_296 = torch.ops.aten.view.default(addmm_95, [8, 197, 256]);  addmm_95 = None
        add_241 = torch.ops.aten.add.Tensor(add_237, view_296);  add_237 = view_296 = None
        slice_70 = torch.ops.aten.slice.Tensor(add_220, 1, 0, 1)
        clone_84 = torch.ops.aten.clone.default(slice_70, memory_format = torch.contiguous_format);  slice_70 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
        getitem_216 = var_mean_52[0]
        getitem_217 = var_mean_52[1];  var_mean_52 = None
        add_242 = torch.ops.aten.add.Tensor(getitem_216, 1e-06);  getitem_216 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
        sub_102 = torch.ops.aten.sub.Tensor(clone_84, getitem_217);  clone_84 = getitem_217 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_102, rsqrt_52);  sub_102 = rsqrt_52 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, arg57_1);  mul_286 = arg57_1 = None
        add_243 = torch.ops.aten.add.Tensor(mul_287, arg58_1);  mul_287 = arg58_1 = None
        mul_288 = torch.ops.aten.mul.Tensor(add_243, 0.5)
        mul_289 = torch.ops.aten.mul.Tensor(add_243, 0.7071067811865476);  add_243 = None
        erf_28 = torch.ops.aten.erf.default(mul_289);  mul_289 = None
        add_244 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_288, add_244);  mul_288 = add_244 = None
        view_297 = torch.ops.aten.view.default(mul_290, [8, 128]);  mul_290 = None
        permute_168 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg60_1, view_297, permute_168);  arg60_1 = view_297 = permute_168 = None
        view_298 = torch.ops.aten.view.default(addmm_96, [8, 1, 256]);  addmm_96 = None
        slice_72 = torch.ops.aten.slice.Tensor(add_241, 1, 0, 1)
        clone_85 = torch.ops.aten.clone.default(slice_72, memory_format = torch.contiguous_format);  slice_72 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
        getitem_218 = var_mean_53[0]
        getitem_219 = var_mean_53[1];  var_mean_53 = None
        add_245 = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        sub_103 = torch.ops.aten.sub.Tensor(clone_85, getitem_219);  clone_85 = getitem_219 = None
        mul_291 = torch.ops.aten.mul.Tensor(sub_103, rsqrt_53);  sub_103 = rsqrt_53 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_291, arg61_1);  mul_291 = arg61_1 = None
        add_246 = torch.ops.aten.add.Tensor(mul_292, arg62_1);  mul_292 = arg62_1 = None
        mul_293 = torch.ops.aten.mul.Tensor(add_246, 0.5)
        mul_294 = torch.ops.aten.mul.Tensor(add_246, 0.7071067811865476);  add_246 = None
        erf_29 = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_247 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_293, add_247);  mul_293 = add_247 = None
        view_299 = torch.ops.aten.view.default(mul_295, [8, 256]);  mul_295 = None
        permute_169 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg64_1, view_299, permute_169);  arg64_1 = view_299 = permute_169 = None
        view_300 = torch.ops.aten.view.default(addmm_97, [8, 1, 128]);  addmm_97 = None
        slice_74 = torch.ops.aten.slice.Tensor(add_241, 1, 1, 9223372036854775807)
        cat_17 = torch.ops.aten.cat.default([view_298, slice_74], 1);  view_298 = slice_74 = None
        slice_76 = torch.ops.aten.slice.Tensor(cat_17, 1, 0, 1)
        var_mean_54 = torch.ops.aten.var_mean.correction(cat_17, [2], correction = 0, keepdim = True)
        getitem_220 = var_mean_54[0]
        getitem_221 = var_mean_54[1];  var_mean_54 = None
        add_248 = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
        sub_104 = torch.ops.aten.sub.Tensor(cat_17, getitem_221);  cat_17 = getitem_221 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_104, rsqrt_54);  sub_104 = rsqrt_54 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, arg65_1);  mul_296 = arg65_1 = None
        add_249 = torch.ops.aten.add.Tensor(mul_297, arg66_1);  mul_297 = arg66_1 = None
        slice_78 = torch.ops.aten.slice.Tensor(add_249, 1, 0, 1)
        permute_170 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        view_301 = torch.ops.aten.view.default(slice_78, [8, 256]);  slice_78 = None
        mm_6 = torch.ops.aten.mm.default(view_301, permute_170);  view_301 = permute_170 = None
        view_302 = torch.ops.aten.view.default(mm_6, [8, 1, 256]);  mm_6 = None
        add_250 = torch.ops.aten.add.Tensor(view_302, arg68_1);  view_302 = arg68_1 = None
        view_303 = torch.ops.aten.view.default(add_250, [8, 1, 4, 64]);  add_250 = None
        permute_171 = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        view_304 = torch.ops.aten.view.default(add_249, [1576, 256])
        permute_172 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg70_1, view_304, permute_172);  arg70_1 = view_304 = permute_172 = None
        view_305 = torch.ops.aten.view.default(addmm_98, [8, 197, 256]);  addmm_98 = None
        view_306 = torch.ops.aten.view.default(view_305, [8, 197, 4, 64]);  view_305 = None
        permute_173 = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
        view_307 = torch.ops.aten.view.default(add_249, [1576, 256]);  add_249 = None
        permute_174 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg72_1, view_307, permute_174);  arg72_1 = view_307 = permute_174 = None
        view_308 = torch.ops.aten.view.default(addmm_99, [8, 197, 256]);  addmm_99 = None
        view_309 = torch.ops.aten.view.default(view_308, [8, 197, 4, 64]);  view_308 = None
        permute_175 = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
        permute_176 = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2]);  permute_173 = None
        expand_28 = torch.ops.aten.expand.default(permute_171, [8, 4, 1, 64]);  permute_171 = None
        view_310 = torch.ops.aten.view.default(expand_28, [32, 1, 64]);  expand_28 = None
        expand_29 = torch.ops.aten.expand.default(permute_176, [8, 4, 64, 197]);  permute_176 = None
        clone_86 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_311 = torch.ops.aten.view.default(clone_86, [32, 64, 197]);  clone_86 = None
        bmm_12 = torch.ops.aten.bmm.default(view_310, view_311);  view_310 = view_311 = None
        view_312 = torch.ops.aten.view.default(bmm_12, [8, 4, 1, 197]);  bmm_12 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_312, 1);  view_312 = None
        amax_default_5 = torch.ops.aten.amax.default(mul_tensor_10, [-1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, 0.125);  sub_tensor_5 = None
        exp_6 = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        expand_30 = torch.ops.aten.expand.default(div_6, [8, 4, 1, 197]);  div_6 = None
        view_313 = torch.ops.aten.view.default(expand_30, [32, 1, 197]);  expand_30 = None
        expand_31 = torch.ops.aten.expand.default(permute_175, [8, 4, 197, 64]);  permute_175 = None
        clone_88 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_314 = torch.ops.aten.view.default(clone_88, [32, 197, 64]);  clone_88 = None
        bmm_13 = torch.ops.aten.bmm.default(view_313, view_314);  view_313 = view_314 = None
        view_315 = torch.ops.aten.view.default(bmm_13, [8, 4, 1, 64]);  bmm_13 = None
        permute_177 = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        view_316 = torch.ops.aten.view.default(permute_177, [8, 1, 256]);  permute_177 = None
        view_317 = torch.ops.aten.view.default(view_316, [8, 256]);  view_316 = None
        permute_178 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg74_1, view_317, permute_178);  arg74_1 = view_317 = permute_178 = None
        view_318 = torch.ops.aten.view.default(addmm_100, [8, 1, 256]);  addmm_100 = None
        add_251 = torch.ops.aten.add.Tensor(slice_76, view_318);  slice_76 = view_318 = None
        var_mean_55 = torch.ops.aten.var_mean.correction(add_251, [2], correction = 0, keepdim = True)
        getitem_222 = var_mean_55[0]
        getitem_223 = var_mean_55[1];  var_mean_55 = None
        add_252 = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        sub_106 = torch.ops.aten.sub.Tensor(add_251, getitem_223);  add_251 = getitem_223 = None
        mul_299 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_55);  sub_106 = rsqrt_55 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_299, arg75_1);  mul_299 = arg75_1 = None
        add_253 = torch.ops.aten.add.Tensor(mul_300, arg76_1);  mul_300 = arg76_1 = None
        mul_301 = torch.ops.aten.mul.Tensor(add_253, 0.5)
        mul_302 = torch.ops.aten.mul.Tensor(add_253, 0.7071067811865476);  add_253 = None
        erf_30 = torch.ops.aten.erf.default(mul_302);  mul_302 = None
        add_254 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_301, add_254);  mul_301 = add_254 = None
        view_319 = torch.ops.aten.view.default(mul_303, [8, 256]);  mul_303 = None
        permute_179 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg78_1, view_319, permute_179);  arg78_1 = view_319 = permute_179 = None
        view_320 = torch.ops.aten.view.default(addmm_101, [8, 1, 128]);  addmm_101 = None
        slice_81 = torch.ops.aten.slice.Tensor(add_220, 1, 1, 9223372036854775807)
        cat_18 = torch.ops.aten.cat.default([view_320, slice_81], 1);  view_320 = slice_81 = None
        slice_83 = torch.ops.aten.slice.Tensor(add_220, 1, 1, 9223372036854775807);  add_220 = None
        cat_19 = torch.ops.aten.cat.default([view_300, slice_83], 1);  view_300 = slice_83 = None
        slice_85 = torch.ops.aten.slice.Tensor(cat_19, 1, 0, 1)
        var_mean_56 = torch.ops.aten.var_mean.correction(cat_19, [2], correction = 0, keepdim = True)
        getitem_224 = var_mean_56[0]
        getitem_225 = var_mean_56[1];  var_mean_56 = None
        add_255 = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_107 = torch.ops.aten.sub.Tensor(cat_19, getitem_225);  cat_19 = getitem_225 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_56);  sub_107 = rsqrt_56 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, arg79_1);  mul_304 = arg79_1 = None
        add_256 = torch.ops.aten.add.Tensor(mul_305, arg80_1);  mul_305 = arg80_1 = None
        slice_87 = torch.ops.aten.slice.Tensor(add_256, 1, 0, 1)
        permute_180 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        view_321 = torch.ops.aten.view.default(slice_87, [8, 128]);  slice_87 = None
        mm_7 = torch.ops.aten.mm.default(view_321, permute_180);  view_321 = permute_180 = None
        view_322 = torch.ops.aten.view.default(mm_7, [8, 1, 128]);  mm_7 = None
        add_257 = torch.ops.aten.add.Tensor(view_322, arg82_1);  view_322 = arg82_1 = None
        view_323 = torch.ops.aten.view.default(add_257, [8, 1, 4, 32]);  add_257 = None
        permute_181 = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        view_324 = torch.ops.aten.view.default(add_256, [3208, 128])
        permute_182 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg84_1, view_324, permute_182);  arg84_1 = view_324 = permute_182 = None
        view_325 = torch.ops.aten.view.default(addmm_102, [8, 401, 128]);  addmm_102 = None
        view_326 = torch.ops.aten.view.default(view_325, [8, 401, 4, 32]);  view_325 = None
        permute_183 = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        view_327 = torch.ops.aten.view.default(add_256, [3208, 128]);  add_256 = None
        permute_184 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg86_1, view_327, permute_184);  arg86_1 = view_327 = permute_184 = None
        view_328 = torch.ops.aten.view.default(addmm_103, [8, 401, 128]);  addmm_103 = None
        view_329 = torch.ops.aten.view.default(view_328, [8, 401, 4, 32]);  view_328 = None
        permute_185 = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        permute_186 = torch.ops.aten.permute.default(permute_183, [0, 1, 3, 2]);  permute_183 = None
        expand_32 = torch.ops.aten.expand.default(permute_181, [8, 4, 1, 32]);  permute_181 = None
        view_330 = torch.ops.aten.view.default(expand_32, [32, 1, 32]);  expand_32 = None
        expand_33 = torch.ops.aten.expand.default(permute_186, [8, 4, 32, 401]);  permute_186 = None
        clone_90 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_331 = torch.ops.aten.view.default(clone_90, [32, 32, 401]);  clone_90 = None
        bmm_14 = torch.ops.aten.bmm.default(view_330, view_331);  view_330 = view_331 = None
        view_332 = torch.ops.aten.view.default(bmm_14, [8, 4, 1, 401]);  bmm_14 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_332, 1);  view_332 = None
        amax_default_4 = torch.ops.aten.amax.default(mul_tensor_8, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.1767766952966369);  sub_tensor_4 = None
        exp_7 = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        expand_34 = torch.ops.aten.expand.default(div_7, [8, 4, 1, 401]);  div_7 = None
        view_333 = torch.ops.aten.view.default(expand_34, [32, 1, 401]);  expand_34 = None
        expand_35 = torch.ops.aten.expand.default(permute_185, [8, 4, 401, 32]);  permute_185 = None
        clone_92 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_334 = torch.ops.aten.view.default(clone_92, [32, 401, 32]);  clone_92 = None
        bmm_15 = torch.ops.aten.bmm.default(view_333, view_334);  view_333 = view_334 = None
        view_335 = torch.ops.aten.view.default(bmm_15, [8, 4, 1, 32]);  bmm_15 = None
        permute_187 = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
        view_336 = torch.ops.aten.view.default(permute_187, [8, 1, 128]);  permute_187 = None
        view_337 = torch.ops.aten.view.default(view_336, [8, 128]);  view_336 = None
        permute_188 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg88_1, view_337, permute_188);  arg88_1 = view_337 = permute_188 = None
        view_338 = torch.ops.aten.view.default(addmm_104, [8, 1, 128]);  addmm_104 = None
        add_258 = torch.ops.aten.add.Tensor(slice_85, view_338);  slice_85 = view_338 = None
        var_mean_57 = torch.ops.aten.var_mean.correction(add_258, [2], correction = 0, keepdim = True)
        getitem_226 = var_mean_57[0]
        getitem_227 = var_mean_57[1];  var_mean_57 = None
        add_259 = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
        sub_109 = torch.ops.aten.sub.Tensor(add_258, getitem_227);  add_258 = getitem_227 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_57);  sub_109 = rsqrt_57 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, arg89_1);  mul_307 = arg89_1 = None
        add_260 = torch.ops.aten.add.Tensor(mul_308, arg90_1);  mul_308 = arg90_1 = None
        mul_309 = torch.ops.aten.mul.Tensor(add_260, 0.5)
        mul_310 = torch.ops.aten.mul.Tensor(add_260, 0.7071067811865476);  add_260 = None
        erf_31 = torch.ops.aten.erf.default(mul_310);  mul_310 = None
        add_261 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_309, add_261);  mul_309 = add_261 = None
        view_339 = torch.ops.aten.view.default(mul_311, [8, 128]);  mul_311 = None
        permute_189 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg92_1, view_339, permute_189);  arg92_1 = view_339 = permute_189 = None
        view_340 = torch.ops.aten.view.default(addmm_105, [8, 1, 256]);  addmm_105 = None
        slice_90 = torch.ops.aten.slice.Tensor(add_241, 1, 1, 9223372036854775807);  add_241 = None
        cat_20 = torch.ops.aten.cat.default([view_340, slice_90], 1);  view_340 = slice_90 = None
        var_mean_58 = torch.ops.aten.var_mean.correction(cat_18, [2], correction = 0, keepdim = True)
        getitem_228 = var_mean_58[0]
        getitem_229 = var_mean_58[1];  var_mean_58 = None
        add_262 = torch.ops.aten.add.Tensor(getitem_228, 1e-06);  getitem_228 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        sub_110 = torch.ops.aten.sub.Tensor(cat_18, getitem_229);  getitem_229 = None
        mul_312 = torch.ops.aten.mul.Tensor(sub_110, rsqrt_58);  sub_110 = rsqrt_58 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_312, arg93_1);  mul_312 = arg93_1 = None
        add_263 = torch.ops.aten.add.Tensor(mul_313, arg94_1);  mul_313 = arg94_1 = None
        view_341 = torch.ops.aten.view.default(add_263, [3208, 128]);  add_263 = None
        permute_190 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg96_1, view_341, permute_190);  arg96_1 = view_341 = permute_190 = None
        view_342 = torch.ops.aten.view.default(addmm_106, [8, 401, 384]);  addmm_106 = None
        view_343 = torch.ops.aten.view.default(view_342, [8, 401, 3, 4, 32]);  view_342 = None
        permute_191 = torch.ops.aten.permute.default(view_343, [2, 0, 3, 1, 4]);  view_343 = None
        unbind_16 = torch.ops.aten.unbind.int(permute_191);  permute_191 = None
        getitem_230 = unbind_16[0]
        getitem_231 = unbind_16[1]
        getitem_232 = unbind_16[2];  unbind_16 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_230, getitem_231, getitem_232, None, False);  getitem_230 = getitem_231 = getitem_232 = None
        getitem_233 = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        permute_192 = torch.ops.aten.permute.default(getitem_233, [0, 2, 1, 3]);  getitem_233 = None
        view_344 = torch.ops.aten.view.default(permute_192, [8, 401, 128]);  permute_192 = None
        view_345 = torch.ops.aten.view.default(view_344, [3208, 128]);  view_344 = None
        permute_193 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg98_1, view_345, permute_193);  arg98_1 = view_345 = permute_193 = None
        view_346 = torch.ops.aten.view.default(addmm_107, [8, 401, 128]);  addmm_107 = None
        add_264 = torch.ops.aten.add.Tensor(cat_18, view_346);  cat_18 = view_346 = None
        var_mean_59 = torch.ops.aten.var_mean.correction(add_264, [2], correction = 0, keepdim = True)
        getitem_237 = var_mean_59[0]
        getitem_238 = var_mean_59[1];  var_mean_59 = None
        add_265 = torch.ops.aten.add.Tensor(getitem_237, 1e-06);  getitem_237 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
        sub_111 = torch.ops.aten.sub.Tensor(add_264, getitem_238);  getitem_238 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_111, rsqrt_59);  sub_111 = rsqrt_59 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, arg99_1);  mul_314 = arg99_1 = None
        add_266 = torch.ops.aten.add.Tensor(mul_315, arg100_1);  mul_315 = arg100_1 = None
        view_347 = torch.ops.aten.view.default(add_266, [3208, 128]);  add_266 = None
        permute_194 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg102_1, view_347, permute_194);  arg102_1 = view_347 = permute_194 = None
        view_348 = torch.ops.aten.view.default(addmm_108, [8, 401, 384]);  addmm_108 = None
        mul_316 = torch.ops.aten.mul.Tensor(view_348, 0.5)
        mul_317 = torch.ops.aten.mul.Tensor(view_348, 0.7071067811865476);  view_348 = None
        erf_32 = torch.ops.aten.erf.default(mul_317);  mul_317 = None
        add_267 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_316, add_267);  mul_316 = add_267 = None
        view_349 = torch.ops.aten.view.default(mul_318, [3208, 384]);  mul_318 = None
        permute_195 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg104_1, view_349, permute_195);  arg104_1 = view_349 = permute_195 = None
        view_350 = torch.ops.aten.view.default(addmm_109, [8, 401, 128]);  addmm_109 = None
        add_268 = torch.ops.aten.add.Tensor(add_264, view_350);  add_264 = view_350 = None
        var_mean_60 = torch.ops.aten.var_mean.correction(cat_20, [2], correction = 0, keepdim = True)
        getitem_239 = var_mean_60[0]
        getitem_240 = var_mean_60[1];  var_mean_60 = None
        add_269 = torch.ops.aten.add.Tensor(getitem_239, 1e-06);  getitem_239 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        sub_112 = torch.ops.aten.sub.Tensor(cat_20, getitem_240);  getitem_240 = None
        mul_319 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_60);  sub_112 = rsqrt_60 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, arg105_1);  mul_319 = arg105_1 = None
        add_270 = torch.ops.aten.add.Tensor(mul_320, arg106_1);  mul_320 = arg106_1 = None
        view_351 = torch.ops.aten.view.default(add_270, [1576, 256]);  add_270 = None
        permute_196 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg108_1, view_351, permute_196);  arg108_1 = view_351 = permute_196 = None
        view_352 = torch.ops.aten.view.default(addmm_110, [8, 197, 768]);  addmm_110 = None
        view_353 = torch.ops.aten.view.default(view_352, [8, 197, 3, 4, 64]);  view_352 = None
        permute_197 = torch.ops.aten.permute.default(view_353, [2, 0, 3, 1, 4]);  view_353 = None
        unbind_17 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_241 = unbind_17[0]
        getitem_242 = unbind_17[1]
        getitem_243 = unbind_17[2];  unbind_17 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_241, getitem_242, getitem_243, None, False);  getitem_241 = getitem_242 = getitem_243 = None
        getitem_244 = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        permute_198 = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3]);  getitem_244 = None
        view_354 = torch.ops.aten.view.default(permute_198, [8, 197, 256]);  permute_198 = None
        view_355 = torch.ops.aten.view.default(view_354, [1576, 256]);  view_354 = None
        permute_199 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg110_1, view_355, permute_199);  arg110_1 = view_355 = permute_199 = None
        view_356 = torch.ops.aten.view.default(addmm_111, [8, 197, 256]);  addmm_111 = None
        add_271 = torch.ops.aten.add.Tensor(cat_20, view_356);  cat_20 = view_356 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(add_271, [2], correction = 0, keepdim = True)
        getitem_248 = var_mean_61[0]
        getitem_249 = var_mean_61[1];  var_mean_61 = None
        add_272 = torch.ops.aten.add.Tensor(getitem_248, 1e-06);  getitem_248 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        sub_113 = torch.ops.aten.sub.Tensor(add_271, getitem_249);  getitem_249 = None
        mul_321 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_61);  sub_113 = rsqrt_61 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_321, arg111_1);  mul_321 = arg111_1 = None
        add_273 = torch.ops.aten.add.Tensor(mul_322, arg112_1);  mul_322 = arg112_1 = None
        view_357 = torch.ops.aten.view.default(add_273, [1576, 256]);  add_273 = None
        permute_200 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg114_1, view_357, permute_200);  arg114_1 = view_357 = permute_200 = None
        view_358 = torch.ops.aten.view.default(addmm_112, [8, 197, 768]);  addmm_112 = None
        mul_323 = torch.ops.aten.mul.Tensor(view_358, 0.5)
        mul_324 = torch.ops.aten.mul.Tensor(view_358, 0.7071067811865476);  view_358 = None
        erf_33 = torch.ops.aten.erf.default(mul_324);  mul_324 = None
        add_274 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_325 = torch.ops.aten.mul.Tensor(mul_323, add_274);  mul_323 = add_274 = None
        view_359 = torch.ops.aten.view.default(mul_325, [1576, 768]);  mul_325 = None
        permute_201 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg116_1, view_359, permute_201);  arg116_1 = view_359 = permute_201 = None
        view_360 = torch.ops.aten.view.default(addmm_113, [8, 197, 256]);  addmm_113 = None
        add_275 = torch.ops.aten.add.Tensor(add_271, view_360);  add_271 = view_360 = None
        var_mean_62 = torch.ops.aten.var_mean.correction(add_275, [2], correction = 0, keepdim = True)
        getitem_250 = var_mean_62[0]
        getitem_251 = var_mean_62[1];  var_mean_62 = None
        add_276 = torch.ops.aten.add.Tensor(getitem_250, 1e-06);  getitem_250 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_114 = torch.ops.aten.sub.Tensor(add_275, getitem_251);  getitem_251 = None
        mul_326 = torch.ops.aten.mul.Tensor(sub_114, rsqrt_62);  sub_114 = rsqrt_62 = None
        mul_327 = torch.ops.aten.mul.Tensor(mul_326, arg117_1);  mul_326 = arg117_1 = None
        add_277 = torch.ops.aten.add.Tensor(mul_327, arg118_1);  mul_327 = arg118_1 = None
        view_361 = torch.ops.aten.view.default(add_277, [1576, 256]);  add_277 = None
        permute_202 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg120_1, view_361, permute_202);  arg120_1 = view_361 = permute_202 = None
        view_362 = torch.ops.aten.view.default(addmm_114, [8, 197, 768]);  addmm_114 = None
        view_363 = torch.ops.aten.view.default(view_362, [8, 197, 3, 4, 64]);  view_362 = None
        permute_203 = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None
        unbind_18 = torch.ops.aten.unbind.int(permute_203);  permute_203 = None
        getitem_252 = unbind_18[0]
        getitem_253 = unbind_18[1]
        getitem_254 = unbind_18[2];  unbind_18 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_252, getitem_253, getitem_254, None, False);  getitem_252 = getitem_253 = getitem_254 = None
        getitem_255 = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        permute_204 = torch.ops.aten.permute.default(getitem_255, [0, 2, 1, 3]);  getitem_255 = None
        view_364 = torch.ops.aten.view.default(permute_204, [8, 197, 256]);  permute_204 = None
        view_365 = torch.ops.aten.view.default(view_364, [1576, 256]);  view_364 = None
        permute_205 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg122_1, view_365, permute_205);  arg122_1 = view_365 = permute_205 = None
        view_366 = torch.ops.aten.view.default(addmm_115, [8, 197, 256]);  addmm_115 = None
        add_278 = torch.ops.aten.add.Tensor(add_275, view_366);  add_275 = view_366 = None
        var_mean_63 = torch.ops.aten.var_mean.correction(add_278, [2], correction = 0, keepdim = True)
        getitem_259 = var_mean_63[0]
        getitem_260 = var_mean_63[1];  var_mean_63 = None
        add_279 = torch.ops.aten.add.Tensor(getitem_259, 1e-06);  getitem_259 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
        sub_115 = torch.ops.aten.sub.Tensor(add_278, getitem_260);  getitem_260 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_115, rsqrt_63);  sub_115 = rsqrt_63 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, arg123_1);  mul_328 = arg123_1 = None
        add_280 = torch.ops.aten.add.Tensor(mul_329, arg124_1);  mul_329 = arg124_1 = None
        view_367 = torch.ops.aten.view.default(add_280, [1576, 256]);  add_280 = None
        permute_206 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg126_1, view_367, permute_206);  arg126_1 = view_367 = permute_206 = None
        view_368 = torch.ops.aten.view.default(addmm_116, [8, 197, 768]);  addmm_116 = None
        mul_330 = torch.ops.aten.mul.Tensor(view_368, 0.5)
        mul_331 = torch.ops.aten.mul.Tensor(view_368, 0.7071067811865476);  view_368 = None
        erf_34 = torch.ops.aten.erf.default(mul_331);  mul_331 = None
        add_281 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_330, add_281);  mul_330 = add_281 = None
        view_369 = torch.ops.aten.view.default(mul_332, [1576, 768]);  mul_332 = None
        permute_207 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg128_1, view_369, permute_207);  arg128_1 = view_369 = permute_207 = None
        view_370 = torch.ops.aten.view.default(addmm_117, [8, 197, 256]);  addmm_117 = None
        add_282 = torch.ops.aten.add.Tensor(add_278, view_370);  add_278 = view_370 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(add_282, [2], correction = 0, keepdim = True)
        getitem_261 = var_mean_64[0]
        getitem_262 = var_mean_64[1];  var_mean_64 = None
        add_283 = torch.ops.aten.add.Tensor(getitem_261, 1e-06);  getitem_261 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        sub_116 = torch.ops.aten.sub.Tensor(add_282, getitem_262);  getitem_262 = None
        mul_333 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_64);  sub_116 = rsqrt_64 = None
        mul_334 = torch.ops.aten.mul.Tensor(mul_333, arg129_1);  mul_333 = arg129_1 = None
        add_284 = torch.ops.aten.add.Tensor(mul_334, arg130_1);  mul_334 = arg130_1 = None
        view_371 = torch.ops.aten.view.default(add_284, [1576, 256]);  add_284 = None
        permute_208 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg132_1, view_371, permute_208);  arg132_1 = view_371 = permute_208 = None
        view_372 = torch.ops.aten.view.default(addmm_118, [8, 197, 768]);  addmm_118 = None
        view_373 = torch.ops.aten.view.default(view_372, [8, 197, 3, 4, 64]);  view_372 = None
        permute_209 = torch.ops.aten.permute.default(view_373, [2, 0, 3, 1, 4]);  view_373 = None
        unbind_19 = torch.ops.aten.unbind.int(permute_209);  permute_209 = None
        getitem_263 = unbind_19[0]
        getitem_264 = unbind_19[1]
        getitem_265 = unbind_19[2];  unbind_19 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_263, getitem_264, getitem_265, None, False);  getitem_263 = getitem_264 = getitem_265 = None
        getitem_266 = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        permute_210 = torch.ops.aten.permute.default(getitem_266, [0, 2, 1, 3]);  getitem_266 = None
        view_374 = torch.ops.aten.view.default(permute_210, [8, 197, 256]);  permute_210 = None
        view_375 = torch.ops.aten.view.default(view_374, [1576, 256]);  view_374 = None
        permute_211 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg134_1, view_375, permute_211);  arg134_1 = view_375 = permute_211 = None
        view_376 = torch.ops.aten.view.default(addmm_119, [8, 197, 256]);  addmm_119 = None
        add_285 = torch.ops.aten.add.Tensor(add_282, view_376);  add_282 = view_376 = None
        var_mean_65 = torch.ops.aten.var_mean.correction(add_285, [2], correction = 0, keepdim = True)
        getitem_270 = var_mean_65[0]
        getitem_271 = var_mean_65[1];  var_mean_65 = None
        add_286 = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_117 = torch.ops.aten.sub.Tensor(add_285, getitem_271);  getitem_271 = None
        mul_335 = torch.ops.aten.mul.Tensor(sub_117, rsqrt_65);  sub_117 = rsqrt_65 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_335, arg135_1);  mul_335 = arg135_1 = None
        add_287 = torch.ops.aten.add.Tensor(mul_336, arg136_1);  mul_336 = arg136_1 = None
        view_377 = torch.ops.aten.view.default(add_287, [1576, 256]);  add_287 = None
        permute_212 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg138_1, view_377, permute_212);  arg138_1 = view_377 = permute_212 = None
        view_378 = torch.ops.aten.view.default(addmm_120, [8, 197, 768]);  addmm_120 = None
        mul_337 = torch.ops.aten.mul.Tensor(view_378, 0.5)
        mul_338 = torch.ops.aten.mul.Tensor(view_378, 0.7071067811865476);  view_378 = None
        erf_35 = torch.ops.aten.erf.default(mul_338);  mul_338 = None
        add_288 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_337, add_288);  mul_337 = add_288 = None
        view_379 = torch.ops.aten.view.default(mul_339, [1576, 768]);  mul_339 = None
        permute_213 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg140_1, view_379, permute_213);  arg140_1 = view_379 = permute_213 = None
        view_380 = torch.ops.aten.view.default(addmm_121, [8, 197, 256]);  addmm_121 = None
        add_289 = torch.ops.aten.add.Tensor(add_285, view_380);  add_285 = view_380 = None
        slice_92 = torch.ops.aten.slice.Tensor(add_268, 1, 0, 1)
        clone_106 = torch.ops.aten.clone.default(slice_92, memory_format = torch.contiguous_format);  slice_92 = None
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_106, [2], correction = 0, keepdim = True)
        getitem_272 = var_mean_66[0]
        getitem_273 = var_mean_66[1];  var_mean_66 = None
        add_290 = torch.ops.aten.add.Tensor(getitem_272, 1e-06);  getitem_272 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        sub_118 = torch.ops.aten.sub.Tensor(clone_106, getitem_273);  clone_106 = getitem_273 = None
        mul_340 = torch.ops.aten.mul.Tensor(sub_118, rsqrt_66);  sub_118 = rsqrt_66 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, arg141_1);  mul_340 = arg141_1 = None
        add_291 = torch.ops.aten.add.Tensor(mul_341, arg142_1);  mul_341 = arg142_1 = None
        mul_342 = torch.ops.aten.mul.Tensor(add_291, 0.5)
        mul_343 = torch.ops.aten.mul.Tensor(add_291, 0.7071067811865476);  add_291 = None
        erf_36 = torch.ops.aten.erf.default(mul_343);  mul_343 = None
        add_292 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_342, add_292);  mul_342 = add_292 = None
        view_381 = torch.ops.aten.view.default(mul_344, [8, 128]);  mul_344 = None
        permute_214 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg144_1, view_381, permute_214);  arg144_1 = view_381 = permute_214 = None
        view_382 = torch.ops.aten.view.default(addmm_122, [8, 1, 256]);  addmm_122 = None
        slice_94 = torch.ops.aten.slice.Tensor(add_289, 1, 0, 1)
        clone_107 = torch.ops.aten.clone.default(slice_94, memory_format = torch.contiguous_format);  slice_94 = None
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_107, [2], correction = 0, keepdim = True)
        getitem_274 = var_mean_67[0]
        getitem_275 = var_mean_67[1];  var_mean_67 = None
        add_293 = torch.ops.aten.add.Tensor(getitem_274, 1e-06);  getitem_274 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        sub_119 = torch.ops.aten.sub.Tensor(clone_107, getitem_275);  clone_107 = getitem_275 = None
        mul_345 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_67);  sub_119 = rsqrt_67 = None
        mul_346 = torch.ops.aten.mul.Tensor(mul_345, arg145_1);  mul_345 = arg145_1 = None
        add_294 = torch.ops.aten.add.Tensor(mul_346, arg146_1);  mul_346 = arg146_1 = None
        mul_347 = torch.ops.aten.mul.Tensor(add_294, 0.5)
        mul_348 = torch.ops.aten.mul.Tensor(add_294, 0.7071067811865476);  add_294 = None
        erf_37 = torch.ops.aten.erf.default(mul_348);  mul_348 = None
        add_295 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_347, add_295);  mul_347 = add_295 = None
        view_383 = torch.ops.aten.view.default(mul_349, [8, 256]);  mul_349 = None
        permute_215 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg148_1, view_383, permute_215);  arg148_1 = view_383 = permute_215 = None
        view_384 = torch.ops.aten.view.default(addmm_123, [8, 1, 128]);  addmm_123 = None
        slice_96 = torch.ops.aten.slice.Tensor(add_289, 1, 1, 9223372036854775807)
        cat_21 = torch.ops.aten.cat.default([view_382, slice_96], 1);  view_382 = slice_96 = None
        slice_98 = torch.ops.aten.slice.Tensor(cat_21, 1, 0, 1)
        var_mean_68 = torch.ops.aten.var_mean.correction(cat_21, [2], correction = 0, keepdim = True)
        getitem_276 = var_mean_68[0]
        getitem_277 = var_mean_68[1];  var_mean_68 = None
        add_296 = torch.ops.aten.add.Tensor(getitem_276, 1e-06);  getitem_276 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_120 = torch.ops.aten.sub.Tensor(cat_21, getitem_277);  cat_21 = getitem_277 = None
        mul_350 = torch.ops.aten.mul.Tensor(sub_120, rsqrt_68);  sub_120 = rsqrt_68 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_350, arg149_1);  mul_350 = arg149_1 = None
        add_297 = torch.ops.aten.add.Tensor(mul_351, arg150_1);  mul_351 = arg150_1 = None
        slice_100 = torch.ops.aten.slice.Tensor(add_297, 1, 0, 1)
        permute_216 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        view_385 = torch.ops.aten.view.default(slice_100, [8, 256]);  slice_100 = None
        mm_8 = torch.ops.aten.mm.default(view_385, permute_216);  view_385 = permute_216 = None
        view_386 = torch.ops.aten.view.default(mm_8, [8, 1, 256]);  mm_8 = None
        add_298 = torch.ops.aten.add.Tensor(view_386, arg152_1);  view_386 = arg152_1 = None
        view_387 = torch.ops.aten.view.default(add_298, [8, 1, 4, 64]);  add_298 = None
        permute_217 = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        view_388 = torch.ops.aten.view.default(add_297, [1576, 256])
        permute_218 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg154_1, view_388, permute_218);  arg154_1 = view_388 = permute_218 = None
        view_389 = torch.ops.aten.view.default(addmm_124, [8, 197, 256]);  addmm_124 = None
        view_390 = torch.ops.aten.view.default(view_389, [8, 197, 4, 64]);  view_389 = None
        permute_219 = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
        view_391 = torch.ops.aten.view.default(add_297, [1576, 256]);  add_297 = None
        permute_220 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg156_1, view_391, permute_220);  arg156_1 = view_391 = permute_220 = None
        view_392 = torch.ops.aten.view.default(addmm_125, [8, 197, 256]);  addmm_125 = None
        view_393 = torch.ops.aten.view.default(view_392, [8, 197, 4, 64]);  view_392 = None
        permute_221 = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        permute_222 = torch.ops.aten.permute.default(permute_219, [0, 1, 3, 2]);  permute_219 = None
        expand_36 = torch.ops.aten.expand.default(permute_217, [8, 4, 1, 64]);  permute_217 = None
        view_394 = torch.ops.aten.view.default(expand_36, [32, 1, 64]);  expand_36 = None
        expand_37 = torch.ops.aten.expand.default(permute_222, [8, 4, 64, 197]);  permute_222 = None
        clone_108 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_395 = torch.ops.aten.view.default(clone_108, [32, 64, 197]);  clone_108 = None
        bmm_16 = torch.ops.aten.bmm.default(view_394, view_395);  view_394 = view_395 = None
        view_396 = torch.ops.aten.view.default(bmm_16, [8, 4, 1, 197]);  bmm_16 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_396, 1);  view_396 = None
        amax_default_3 = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, 0.125);  sub_tensor_3 = None
        exp_8 = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        expand_38 = torch.ops.aten.expand.default(div_8, [8, 4, 1, 197]);  div_8 = None
        view_397 = torch.ops.aten.view.default(expand_38, [32, 1, 197]);  expand_38 = None
        expand_39 = torch.ops.aten.expand.default(permute_221, [8, 4, 197, 64]);  permute_221 = None
        clone_110 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_398 = torch.ops.aten.view.default(clone_110, [32, 197, 64]);  clone_110 = None
        bmm_17 = torch.ops.aten.bmm.default(view_397, view_398);  view_397 = view_398 = None
        view_399 = torch.ops.aten.view.default(bmm_17, [8, 4, 1, 64]);  bmm_17 = None
        permute_223 = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        view_400 = torch.ops.aten.view.default(permute_223, [8, 1, 256]);  permute_223 = None
        view_401 = torch.ops.aten.view.default(view_400, [8, 256]);  view_400 = None
        permute_224 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg158_1, view_401, permute_224);  arg158_1 = view_401 = permute_224 = None
        view_402 = torch.ops.aten.view.default(addmm_126, [8, 1, 256]);  addmm_126 = None
        add_299 = torch.ops.aten.add.Tensor(slice_98, view_402);  slice_98 = view_402 = None
        var_mean_69 = torch.ops.aten.var_mean.correction(add_299, [2], correction = 0, keepdim = True)
        getitem_278 = var_mean_69[0]
        getitem_279 = var_mean_69[1];  var_mean_69 = None
        add_300 = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
        sub_122 = torch.ops.aten.sub.Tensor(add_299, getitem_279);  add_299 = getitem_279 = None
        mul_353 = torch.ops.aten.mul.Tensor(sub_122, rsqrt_69);  sub_122 = rsqrt_69 = None
        mul_354 = torch.ops.aten.mul.Tensor(mul_353, arg159_1);  mul_353 = arg159_1 = None
        add_301 = torch.ops.aten.add.Tensor(mul_354, arg160_1);  mul_354 = arg160_1 = None
        mul_355 = torch.ops.aten.mul.Tensor(add_301, 0.5)
        mul_356 = torch.ops.aten.mul.Tensor(add_301, 0.7071067811865476);  add_301 = None
        erf_38 = torch.ops.aten.erf.default(mul_356);  mul_356 = None
        add_302 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_355, add_302);  mul_355 = add_302 = None
        view_403 = torch.ops.aten.view.default(mul_357, [8, 256]);  mul_357 = None
        permute_225 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg162_1, view_403, permute_225);  arg162_1 = view_403 = permute_225 = None
        view_404 = torch.ops.aten.view.default(addmm_127, [8, 1, 128]);  addmm_127 = None
        slice_103 = torch.ops.aten.slice.Tensor(add_268, 1, 1, 9223372036854775807)
        cat_22 = torch.ops.aten.cat.default([view_404, slice_103], 1);  view_404 = slice_103 = None
        slice_105 = torch.ops.aten.slice.Tensor(add_268, 1, 1, 9223372036854775807);  add_268 = None
        cat_23 = torch.ops.aten.cat.default([view_384, slice_105], 1);  view_384 = slice_105 = None
        slice_107 = torch.ops.aten.slice.Tensor(cat_23, 1, 0, 1)
        var_mean_70 = torch.ops.aten.var_mean.correction(cat_23, [2], correction = 0, keepdim = True)
        getitem_280 = var_mean_70[0]
        getitem_281 = var_mean_70[1];  var_mean_70 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_280, 1e-06);  getitem_280 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_123 = torch.ops.aten.sub.Tensor(cat_23, getitem_281);  cat_23 = getitem_281 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_123, rsqrt_70);  sub_123 = rsqrt_70 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, arg163_1);  mul_358 = arg163_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_359, arg164_1);  mul_359 = arg164_1 = None
        slice_109 = torch.ops.aten.slice.Tensor(add_304, 1, 0, 1)
        permute_226 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        view_405 = torch.ops.aten.view.default(slice_109, [8, 128]);  slice_109 = None
        mm_9 = torch.ops.aten.mm.default(view_405, permute_226);  view_405 = permute_226 = None
        view_406 = torch.ops.aten.view.default(mm_9, [8, 1, 128]);  mm_9 = None
        add_305 = torch.ops.aten.add.Tensor(view_406, arg166_1);  view_406 = arg166_1 = None
        view_407 = torch.ops.aten.view.default(add_305, [8, 1, 4, 32]);  add_305 = None
        permute_227 = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
        view_408 = torch.ops.aten.view.default(add_304, [3208, 128])
        permute_228 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg168_1, view_408, permute_228);  arg168_1 = view_408 = permute_228 = None
        view_409 = torch.ops.aten.view.default(addmm_128, [8, 401, 128]);  addmm_128 = None
        view_410 = torch.ops.aten.view.default(view_409, [8, 401, 4, 32]);  view_409 = None
        permute_229 = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        view_411 = torch.ops.aten.view.default(add_304, [3208, 128]);  add_304 = None
        permute_230 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg170_1, view_411, permute_230);  arg170_1 = view_411 = permute_230 = None
        view_412 = torch.ops.aten.view.default(addmm_129, [8, 401, 128]);  addmm_129 = None
        view_413 = torch.ops.aten.view.default(view_412, [8, 401, 4, 32]);  view_412 = None
        permute_231 = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
        permute_232 = torch.ops.aten.permute.default(permute_229, [0, 1, 3, 2]);  permute_229 = None
        expand_40 = torch.ops.aten.expand.default(permute_227, [8, 4, 1, 32]);  permute_227 = None
        view_414 = torch.ops.aten.view.default(expand_40, [32, 1, 32]);  expand_40 = None
        expand_41 = torch.ops.aten.expand.default(permute_232, [8, 4, 32, 401]);  permute_232 = None
        clone_112 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_415 = torch.ops.aten.view.default(clone_112, [32, 32, 401]);  clone_112 = None
        bmm_18 = torch.ops.aten.bmm.default(view_414, view_415);  view_414 = view_415 = None
        view_416 = torch.ops.aten.view.default(bmm_18, [8, 4, 1, 401]);  bmm_18 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_416, 1);  view_416 = None
        amax_default_2 = torch.ops.aten.amax.default(mul_tensor_4, [-1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.1767766952966369);  sub_tensor_2 = None
        exp_9 = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        expand_42 = torch.ops.aten.expand.default(div_9, [8, 4, 1, 401]);  div_9 = None
        view_417 = torch.ops.aten.view.default(expand_42, [32, 1, 401]);  expand_42 = None
        expand_43 = torch.ops.aten.expand.default(permute_231, [8, 4, 401, 32]);  permute_231 = None
        clone_114 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_418 = torch.ops.aten.view.default(clone_114, [32, 401, 32]);  clone_114 = None
        bmm_19 = torch.ops.aten.bmm.default(view_417, view_418);  view_417 = view_418 = None
        view_419 = torch.ops.aten.view.default(bmm_19, [8, 4, 1, 32]);  bmm_19 = None
        permute_233 = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
        view_420 = torch.ops.aten.view.default(permute_233, [8, 1, 128]);  permute_233 = None
        view_421 = torch.ops.aten.view.default(view_420, [8, 128]);  view_420 = None
        permute_234 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg172_1, view_421, permute_234);  arg172_1 = view_421 = permute_234 = None
        view_422 = torch.ops.aten.view.default(addmm_130, [8, 1, 128]);  addmm_130 = None
        add_306 = torch.ops.aten.add.Tensor(slice_107, view_422);  slice_107 = view_422 = None
        var_mean_71 = torch.ops.aten.var_mean.correction(add_306, [2], correction = 0, keepdim = True)
        getitem_282 = var_mean_71[0]
        getitem_283 = var_mean_71[1];  var_mean_71 = None
        add_307 = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
        sub_125 = torch.ops.aten.sub.Tensor(add_306, getitem_283);  add_306 = getitem_283 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_125, rsqrt_71);  sub_125 = rsqrt_71 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, arg173_1);  mul_361 = arg173_1 = None
        add_308 = torch.ops.aten.add.Tensor(mul_362, arg174_1);  mul_362 = arg174_1 = None
        mul_363 = torch.ops.aten.mul.Tensor(add_308, 0.5)
        mul_364 = torch.ops.aten.mul.Tensor(add_308, 0.7071067811865476);  add_308 = None
        erf_39 = torch.ops.aten.erf.default(mul_364);  mul_364 = None
        add_309 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_363, add_309);  mul_363 = add_309 = None
        view_423 = torch.ops.aten.view.default(mul_365, [8, 128]);  mul_365 = None
        permute_235 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg176_1, view_423, permute_235);  arg176_1 = view_423 = permute_235 = None
        view_424 = torch.ops.aten.view.default(addmm_131, [8, 1, 256]);  addmm_131 = None
        slice_112 = torch.ops.aten.slice.Tensor(add_289, 1, 1, 9223372036854775807);  add_289 = None
        cat_24 = torch.ops.aten.cat.default([view_424, slice_112], 1);  view_424 = slice_112 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(cat_22, [2], correction = 0, keepdim = True)
        getitem_284 = var_mean_72[0]
        getitem_285 = var_mean_72[1];  var_mean_72 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_126 = torch.ops.aten.sub.Tensor(cat_22, getitem_285);  getitem_285 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_126, rsqrt_72);  sub_126 = rsqrt_72 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_366, arg177_1);  mul_366 = arg177_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_367, arg178_1);  mul_367 = arg178_1 = None
        view_425 = torch.ops.aten.view.default(add_311, [3208, 128]);  add_311 = None
        permute_236 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg180_1, view_425, permute_236);  arg180_1 = view_425 = permute_236 = None
        view_426 = torch.ops.aten.view.default(addmm_132, [8, 401, 384]);  addmm_132 = None
        view_427 = torch.ops.aten.view.default(view_426, [8, 401, 3, 4, 32]);  view_426 = None
        permute_237 = torch.ops.aten.permute.default(view_427, [2, 0, 3, 1, 4]);  view_427 = None
        unbind_20 = torch.ops.aten.unbind.int(permute_237);  permute_237 = None
        getitem_286 = unbind_20[0]
        getitem_287 = unbind_20[1]
        getitem_288 = unbind_20[2];  unbind_20 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_286, getitem_287, getitem_288, None, False);  getitem_286 = getitem_287 = getitem_288 = None
        getitem_289 = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        permute_238 = torch.ops.aten.permute.default(getitem_289, [0, 2, 1, 3]);  getitem_289 = None
        view_428 = torch.ops.aten.view.default(permute_238, [8, 401, 128]);  permute_238 = None
        view_429 = torch.ops.aten.view.default(view_428, [3208, 128]);  view_428 = None
        permute_239 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg182_1, view_429, permute_239);  arg182_1 = view_429 = permute_239 = None
        view_430 = torch.ops.aten.view.default(addmm_133, [8, 401, 128]);  addmm_133 = None
        add_312 = torch.ops.aten.add.Tensor(cat_22, view_430);  cat_22 = view_430 = None
        var_mean_73 = torch.ops.aten.var_mean.correction(add_312, [2], correction = 0, keepdim = True)
        getitem_293 = var_mean_73[0]
        getitem_294 = var_mean_73[1];  var_mean_73 = None
        add_313 = torch.ops.aten.add.Tensor(getitem_293, 1e-06);  getitem_293 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_127 = torch.ops.aten.sub.Tensor(add_312, getitem_294);  getitem_294 = None
        mul_368 = torch.ops.aten.mul.Tensor(sub_127, rsqrt_73);  sub_127 = rsqrt_73 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_368, arg183_1);  mul_368 = arg183_1 = None
        add_314 = torch.ops.aten.add.Tensor(mul_369, arg184_1);  mul_369 = arg184_1 = None
        view_431 = torch.ops.aten.view.default(add_314, [3208, 128]);  add_314 = None
        permute_240 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg186_1, view_431, permute_240);  arg186_1 = view_431 = permute_240 = None
        view_432 = torch.ops.aten.view.default(addmm_134, [8, 401, 384]);  addmm_134 = None
        mul_370 = torch.ops.aten.mul.Tensor(view_432, 0.5)
        mul_371 = torch.ops.aten.mul.Tensor(view_432, 0.7071067811865476);  view_432 = None
        erf_40 = torch.ops.aten.erf.default(mul_371);  mul_371 = None
        add_315 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_372 = torch.ops.aten.mul.Tensor(mul_370, add_315);  mul_370 = add_315 = None
        view_433 = torch.ops.aten.view.default(mul_372, [3208, 384]);  mul_372 = None
        permute_241 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg188_1, view_433, permute_241);  arg188_1 = view_433 = permute_241 = None
        view_434 = torch.ops.aten.view.default(addmm_135, [8, 401, 128]);  addmm_135 = None
        add_316 = torch.ops.aten.add.Tensor(add_312, view_434);  add_312 = view_434 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(cat_24, [2], correction = 0, keepdim = True)
        getitem_295 = var_mean_74[0]
        getitem_296 = var_mean_74[1];  var_mean_74 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_295, 1e-06);  getitem_295 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_128 = torch.ops.aten.sub.Tensor(cat_24, getitem_296);  getitem_296 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_128, rsqrt_74);  sub_128 = rsqrt_74 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, arg189_1);  mul_373 = arg189_1 = None
        add_318 = torch.ops.aten.add.Tensor(mul_374, arg190_1);  mul_374 = arg190_1 = None
        view_435 = torch.ops.aten.view.default(add_318, [1576, 256]);  add_318 = None
        permute_242 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg192_1, view_435, permute_242);  arg192_1 = view_435 = permute_242 = None
        view_436 = torch.ops.aten.view.default(addmm_136, [8, 197, 768]);  addmm_136 = None
        view_437 = torch.ops.aten.view.default(view_436, [8, 197, 3, 4, 64]);  view_436 = None
        permute_243 = torch.ops.aten.permute.default(view_437, [2, 0, 3, 1, 4]);  view_437 = None
        unbind_21 = torch.ops.aten.unbind.int(permute_243);  permute_243 = None
        getitem_297 = unbind_21[0]
        getitem_298 = unbind_21[1]
        getitem_299 = unbind_21[2];  unbind_21 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_297, getitem_298, getitem_299, None, False);  getitem_297 = getitem_298 = getitem_299 = None
        getitem_300 = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        permute_244 = torch.ops.aten.permute.default(getitem_300, [0, 2, 1, 3]);  getitem_300 = None
        view_438 = torch.ops.aten.view.default(permute_244, [8, 197, 256]);  permute_244 = None
        view_439 = torch.ops.aten.view.default(view_438, [1576, 256]);  view_438 = None
        permute_245 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg194_1, view_439, permute_245);  arg194_1 = view_439 = permute_245 = None
        view_440 = torch.ops.aten.view.default(addmm_137, [8, 197, 256]);  addmm_137 = None
        add_319 = torch.ops.aten.add.Tensor(cat_24, view_440);  cat_24 = view_440 = None
        var_mean_75 = torch.ops.aten.var_mean.correction(add_319, [2], correction = 0, keepdim = True)
        getitem_304 = var_mean_75[0]
        getitem_305 = var_mean_75[1];  var_mean_75 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_304, 1e-06);  getitem_304 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_129 = torch.ops.aten.sub.Tensor(add_319, getitem_305);  getitem_305 = None
        mul_375 = torch.ops.aten.mul.Tensor(sub_129, rsqrt_75);  sub_129 = rsqrt_75 = None
        mul_376 = torch.ops.aten.mul.Tensor(mul_375, arg195_1);  mul_375 = arg195_1 = None
        add_321 = torch.ops.aten.add.Tensor(mul_376, arg196_1);  mul_376 = arg196_1 = None
        view_441 = torch.ops.aten.view.default(add_321, [1576, 256]);  add_321 = None
        permute_246 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg198_1, view_441, permute_246);  arg198_1 = view_441 = permute_246 = None
        view_442 = torch.ops.aten.view.default(addmm_138, [8, 197, 768]);  addmm_138 = None
        mul_377 = torch.ops.aten.mul.Tensor(view_442, 0.5)
        mul_378 = torch.ops.aten.mul.Tensor(view_442, 0.7071067811865476);  view_442 = None
        erf_41 = torch.ops.aten.erf.default(mul_378);  mul_378 = None
        add_322 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_379 = torch.ops.aten.mul.Tensor(mul_377, add_322);  mul_377 = add_322 = None
        view_443 = torch.ops.aten.view.default(mul_379, [1576, 768]);  mul_379 = None
        permute_247 = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg200_1, view_443, permute_247);  arg200_1 = view_443 = permute_247 = None
        view_444 = torch.ops.aten.view.default(addmm_139, [8, 197, 256]);  addmm_139 = None
        add_323 = torch.ops.aten.add.Tensor(add_319, view_444);  add_319 = view_444 = None
        var_mean_76 = torch.ops.aten.var_mean.correction(add_323, [2], correction = 0, keepdim = True)
        getitem_306 = var_mean_76[0]
        getitem_307 = var_mean_76[1];  var_mean_76 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_306, 1e-06);  getitem_306 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_130 = torch.ops.aten.sub.Tensor(add_323, getitem_307);  getitem_307 = None
        mul_380 = torch.ops.aten.mul.Tensor(sub_130, rsqrt_76);  sub_130 = rsqrt_76 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_380, arg201_1);  mul_380 = arg201_1 = None
        add_325 = torch.ops.aten.add.Tensor(mul_381, arg202_1);  mul_381 = arg202_1 = None
        view_445 = torch.ops.aten.view.default(add_325, [1576, 256]);  add_325 = None
        permute_248 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg204_1, view_445, permute_248);  arg204_1 = view_445 = permute_248 = None
        view_446 = torch.ops.aten.view.default(addmm_140, [8, 197, 768]);  addmm_140 = None
        view_447 = torch.ops.aten.view.default(view_446, [8, 197, 3, 4, 64]);  view_446 = None
        permute_249 = torch.ops.aten.permute.default(view_447, [2, 0, 3, 1, 4]);  view_447 = None
        unbind_22 = torch.ops.aten.unbind.int(permute_249);  permute_249 = None
        getitem_308 = unbind_22[0]
        getitem_309 = unbind_22[1]
        getitem_310 = unbind_22[2];  unbind_22 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_308, getitem_309, getitem_310, None, False);  getitem_308 = getitem_309 = getitem_310 = None
        getitem_311 = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        permute_250 = torch.ops.aten.permute.default(getitem_311, [0, 2, 1, 3]);  getitem_311 = None
        view_448 = torch.ops.aten.view.default(permute_250, [8, 197, 256]);  permute_250 = None
        view_449 = torch.ops.aten.view.default(view_448, [1576, 256]);  view_448 = None
        permute_251 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg206_1, view_449, permute_251);  arg206_1 = view_449 = permute_251 = None
        view_450 = torch.ops.aten.view.default(addmm_141, [8, 197, 256]);  addmm_141 = None
        add_326 = torch.ops.aten.add.Tensor(add_323, view_450);  add_323 = view_450 = None
        var_mean_77 = torch.ops.aten.var_mean.correction(add_326, [2], correction = 0, keepdim = True)
        getitem_315 = var_mean_77[0]
        getitem_316 = var_mean_77[1];  var_mean_77 = None
        add_327 = torch.ops.aten.add.Tensor(getitem_315, 1e-06);  getitem_315 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        sub_131 = torch.ops.aten.sub.Tensor(add_326, getitem_316);  getitem_316 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_131, rsqrt_77);  sub_131 = rsqrt_77 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, arg207_1);  mul_382 = arg207_1 = None
        add_328 = torch.ops.aten.add.Tensor(mul_383, arg208_1);  mul_383 = arg208_1 = None
        view_451 = torch.ops.aten.view.default(add_328, [1576, 256]);  add_328 = None
        permute_252 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg210_1, view_451, permute_252);  arg210_1 = view_451 = permute_252 = None
        view_452 = torch.ops.aten.view.default(addmm_142, [8, 197, 768]);  addmm_142 = None
        mul_384 = torch.ops.aten.mul.Tensor(view_452, 0.5)
        mul_385 = torch.ops.aten.mul.Tensor(view_452, 0.7071067811865476);  view_452 = None
        erf_42 = torch.ops.aten.erf.default(mul_385);  mul_385 = None
        add_329 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_386 = torch.ops.aten.mul.Tensor(mul_384, add_329);  mul_384 = add_329 = None
        view_453 = torch.ops.aten.view.default(mul_386, [1576, 768]);  mul_386 = None
        permute_253 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg212_1, view_453, permute_253);  arg212_1 = view_453 = permute_253 = None
        view_454 = torch.ops.aten.view.default(addmm_143, [8, 197, 256]);  addmm_143 = None
        add_330 = torch.ops.aten.add.Tensor(add_326, view_454);  add_326 = view_454 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(add_330, [2], correction = 0, keepdim = True)
        getitem_317 = var_mean_78[0]
        getitem_318 = var_mean_78[1];  var_mean_78 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_317, 1e-06);  getitem_317 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_132 = torch.ops.aten.sub.Tensor(add_330, getitem_318);  getitem_318 = None
        mul_387 = torch.ops.aten.mul.Tensor(sub_132, rsqrt_78);  sub_132 = rsqrt_78 = None
        mul_388 = torch.ops.aten.mul.Tensor(mul_387, arg213_1);  mul_387 = arg213_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_388, arg214_1);  mul_388 = arg214_1 = None
        view_455 = torch.ops.aten.view.default(add_332, [1576, 256]);  add_332 = None
        permute_254 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg216_1, view_455, permute_254);  arg216_1 = view_455 = permute_254 = None
        view_456 = torch.ops.aten.view.default(addmm_144, [8, 197, 768]);  addmm_144 = None
        view_457 = torch.ops.aten.view.default(view_456, [8, 197, 3, 4, 64]);  view_456 = None
        permute_255 = torch.ops.aten.permute.default(view_457, [2, 0, 3, 1, 4]);  view_457 = None
        unbind_23 = torch.ops.aten.unbind.int(permute_255);  permute_255 = None
        getitem_319 = unbind_23[0]
        getitem_320 = unbind_23[1]
        getitem_321 = unbind_23[2];  unbind_23 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_319, getitem_320, getitem_321, None, False);  getitem_319 = getitem_320 = getitem_321 = None
        getitem_322 = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        permute_256 = torch.ops.aten.permute.default(getitem_322, [0, 2, 1, 3]);  getitem_322 = None
        view_458 = torch.ops.aten.view.default(permute_256, [8, 197, 256]);  permute_256 = None
        view_459 = torch.ops.aten.view.default(view_458, [1576, 256]);  view_458 = None
        permute_257 = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg218_1, view_459, permute_257);  arg218_1 = view_459 = permute_257 = None
        view_460 = torch.ops.aten.view.default(addmm_145, [8, 197, 256]);  addmm_145 = None
        add_333 = torch.ops.aten.add.Tensor(add_330, view_460);  add_330 = view_460 = None
        var_mean_79 = torch.ops.aten.var_mean.correction(add_333, [2], correction = 0, keepdim = True)
        getitem_326 = var_mean_79[0]
        getitem_327 = var_mean_79[1];  var_mean_79 = None
        add_334 = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_133 = torch.ops.aten.sub.Tensor(add_333, getitem_327);  getitem_327 = None
        mul_389 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_79);  sub_133 = rsqrt_79 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_389, arg219_1);  mul_389 = arg219_1 = None
        add_335 = torch.ops.aten.add.Tensor(mul_390, arg220_1);  mul_390 = arg220_1 = None
        view_461 = torch.ops.aten.view.default(add_335, [1576, 256]);  add_335 = None
        permute_258 = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg222_1, view_461, permute_258);  arg222_1 = view_461 = permute_258 = None
        view_462 = torch.ops.aten.view.default(addmm_146, [8, 197, 768]);  addmm_146 = None
        mul_391 = torch.ops.aten.mul.Tensor(view_462, 0.5)
        mul_392 = torch.ops.aten.mul.Tensor(view_462, 0.7071067811865476);  view_462 = None
        erf_43 = torch.ops.aten.erf.default(mul_392);  mul_392 = None
        add_336 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_393 = torch.ops.aten.mul.Tensor(mul_391, add_336);  mul_391 = add_336 = None
        view_463 = torch.ops.aten.view.default(mul_393, [1576, 768]);  mul_393 = None
        permute_259 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg224_1, view_463, permute_259);  arg224_1 = view_463 = permute_259 = None
        view_464 = torch.ops.aten.view.default(addmm_147, [8, 197, 256]);  addmm_147 = None
        add_337 = torch.ops.aten.add.Tensor(add_333, view_464);  add_333 = view_464 = None
        slice_114 = torch.ops.aten.slice.Tensor(add_316, 1, 0, 1)
        clone_128 = torch.ops.aten.clone.default(slice_114, memory_format = torch.contiguous_format);  slice_114 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_128, [2], correction = 0, keepdim = True)
        getitem_328 = var_mean_80[0]
        getitem_329 = var_mean_80[1];  var_mean_80 = None
        add_338 = torch.ops.aten.add.Tensor(getitem_328, 1e-06);  getitem_328 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        sub_134 = torch.ops.aten.sub.Tensor(clone_128, getitem_329);  clone_128 = getitem_329 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_134, rsqrt_80);  sub_134 = rsqrt_80 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, arg225_1);  mul_394 = arg225_1 = None
        add_339 = torch.ops.aten.add.Tensor(mul_395, arg226_1);  mul_395 = arg226_1 = None
        mul_396 = torch.ops.aten.mul.Tensor(add_339, 0.5)
        mul_397 = torch.ops.aten.mul.Tensor(add_339, 0.7071067811865476);  add_339 = None
        erf_44 = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_340 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_396, add_340);  mul_396 = add_340 = None
        view_465 = torch.ops.aten.view.default(mul_398, [8, 128]);  mul_398 = None
        permute_260 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg228_1, view_465, permute_260);  arg228_1 = view_465 = permute_260 = None
        view_466 = torch.ops.aten.view.default(addmm_148, [8, 1, 256]);  addmm_148 = None
        slice_116 = torch.ops.aten.slice.Tensor(add_337, 1, 0, 1)
        clone_129 = torch.ops.aten.clone.default(slice_116, memory_format = torch.contiguous_format);  slice_116 = None
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_129, [2], correction = 0, keepdim = True)
        getitem_330 = var_mean_81[0]
        getitem_331 = var_mean_81[1];  var_mean_81 = None
        add_341 = torch.ops.aten.add.Tensor(getitem_330, 1e-06);  getitem_330 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_135 = torch.ops.aten.sub.Tensor(clone_129, getitem_331);  clone_129 = getitem_331 = None
        mul_399 = torch.ops.aten.mul.Tensor(sub_135, rsqrt_81);  sub_135 = rsqrt_81 = None
        mul_400 = torch.ops.aten.mul.Tensor(mul_399, arg229_1);  mul_399 = arg229_1 = None
        add_342 = torch.ops.aten.add.Tensor(mul_400, arg230_1);  mul_400 = arg230_1 = None
        mul_401 = torch.ops.aten.mul.Tensor(add_342, 0.5)
        mul_402 = torch.ops.aten.mul.Tensor(add_342, 0.7071067811865476);  add_342 = None
        erf_45 = torch.ops.aten.erf.default(mul_402);  mul_402 = None
        add_343 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_401, add_343);  mul_401 = add_343 = None
        view_467 = torch.ops.aten.view.default(mul_403, [8, 256]);  mul_403 = None
        permute_261 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg232_1, view_467, permute_261);  arg232_1 = view_467 = permute_261 = None
        view_468 = torch.ops.aten.view.default(addmm_149, [8, 1, 128]);  addmm_149 = None
        slice_118 = torch.ops.aten.slice.Tensor(add_337, 1, 1, 9223372036854775807)
        cat_25 = torch.ops.aten.cat.default([view_466, slice_118], 1);  view_466 = slice_118 = None
        slice_120 = torch.ops.aten.slice.Tensor(cat_25, 1, 0, 1)
        var_mean_82 = torch.ops.aten.var_mean.correction(cat_25, [2], correction = 0, keepdim = True)
        getitem_332 = var_mean_82[0]
        getitem_333 = var_mean_82[1];  var_mean_82 = None
        add_344 = torch.ops.aten.add.Tensor(getitem_332, 1e-06);  getitem_332 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_344);  add_344 = None
        sub_136 = torch.ops.aten.sub.Tensor(cat_25, getitem_333);  cat_25 = getitem_333 = None
        mul_404 = torch.ops.aten.mul.Tensor(sub_136, rsqrt_82);  sub_136 = rsqrt_82 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_404, arg233_1);  mul_404 = arg233_1 = None
        add_345 = torch.ops.aten.add.Tensor(mul_405, arg234_1);  mul_405 = arg234_1 = None
        slice_122 = torch.ops.aten.slice.Tensor(add_345, 1, 0, 1)
        permute_262 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        view_469 = torch.ops.aten.view.default(slice_122, [8, 256]);  slice_122 = None
        mm_10 = torch.ops.aten.mm.default(view_469, permute_262);  view_469 = permute_262 = None
        view_470 = torch.ops.aten.view.default(mm_10, [8, 1, 256]);  mm_10 = None
        add_346 = torch.ops.aten.add.Tensor(view_470, arg236_1);  view_470 = arg236_1 = None
        view_471 = torch.ops.aten.view.default(add_346, [8, 1, 4, 64]);  add_346 = None
        permute_263 = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
        view_472 = torch.ops.aten.view.default(add_345, [1576, 256])
        permute_264 = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg238_1, view_472, permute_264);  arg238_1 = view_472 = permute_264 = None
        view_473 = torch.ops.aten.view.default(addmm_150, [8, 197, 256]);  addmm_150 = None
        view_474 = torch.ops.aten.view.default(view_473, [8, 197, 4, 64]);  view_473 = None
        permute_265 = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None
        view_475 = torch.ops.aten.view.default(add_345, [1576, 256]);  add_345 = None
        permute_266 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg240_1, view_475, permute_266);  arg240_1 = view_475 = permute_266 = None
        view_476 = torch.ops.aten.view.default(addmm_151, [8, 197, 256]);  addmm_151 = None
        view_477 = torch.ops.aten.view.default(view_476, [8, 197, 4, 64]);  view_476 = None
        permute_267 = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        permute_268 = torch.ops.aten.permute.default(permute_265, [0, 1, 3, 2]);  permute_265 = None
        expand_44 = torch.ops.aten.expand.default(permute_263, [8, 4, 1, 64]);  permute_263 = None
        view_478 = torch.ops.aten.view.default(expand_44, [32, 1, 64]);  expand_44 = None
        expand_45 = torch.ops.aten.expand.default(permute_268, [8, 4, 64, 197]);  permute_268 = None
        clone_130 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_479 = torch.ops.aten.view.default(clone_130, [32, 64, 197]);  clone_130 = None
        bmm_20 = torch.ops.aten.bmm.default(view_478, view_479);  view_478 = view_479 = None
        view_480 = torch.ops.aten.view.default(bmm_20, [8, 4, 1, 197]);  bmm_20 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_480, 1);  view_480 = None
        amax_default_1 = torch.ops.aten.amax.default(mul_tensor_2, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, 0.125);  sub_tensor_1 = None
        exp_10 = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        expand_46 = torch.ops.aten.expand.default(div_10, [8, 4, 1, 197]);  div_10 = None
        view_481 = torch.ops.aten.view.default(expand_46, [32, 1, 197]);  expand_46 = None
        expand_47 = torch.ops.aten.expand.default(permute_267, [8, 4, 197, 64]);  permute_267 = None
        clone_132 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_482 = torch.ops.aten.view.default(clone_132, [32, 197, 64]);  clone_132 = None
        bmm_21 = torch.ops.aten.bmm.default(view_481, view_482);  view_481 = view_482 = None
        view_483 = torch.ops.aten.view.default(bmm_21, [8, 4, 1, 64]);  bmm_21 = None
        permute_269 = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
        view_484 = torch.ops.aten.view.default(permute_269, [8, 1, 256]);  permute_269 = None
        view_485 = torch.ops.aten.view.default(view_484, [8, 256]);  view_484 = None
        permute_270 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg242_1, view_485, permute_270);  arg242_1 = view_485 = permute_270 = None
        view_486 = torch.ops.aten.view.default(addmm_152, [8, 1, 256]);  addmm_152 = None
        add_347 = torch.ops.aten.add.Tensor(slice_120, view_486);  slice_120 = view_486 = None
        var_mean_83 = torch.ops.aten.var_mean.correction(add_347, [2], correction = 0, keepdim = True)
        getitem_334 = var_mean_83[0]
        getitem_335 = var_mean_83[1];  var_mean_83 = None
        add_348 = torch.ops.aten.add.Tensor(getitem_334, 1e-06);  getitem_334 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        sub_138 = torch.ops.aten.sub.Tensor(add_347, getitem_335);  add_347 = getitem_335 = None
        mul_407 = torch.ops.aten.mul.Tensor(sub_138, rsqrt_83);  sub_138 = rsqrt_83 = None
        mul_408 = torch.ops.aten.mul.Tensor(mul_407, arg243_1);  mul_407 = arg243_1 = None
        add_349 = torch.ops.aten.add.Tensor(mul_408, arg244_1);  mul_408 = arg244_1 = None
        mul_409 = torch.ops.aten.mul.Tensor(add_349, 0.5)
        mul_410 = torch.ops.aten.mul.Tensor(add_349, 0.7071067811865476);  add_349 = None
        erf_46 = torch.ops.aten.erf.default(mul_410);  mul_410 = None
        add_350 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_409, add_350);  mul_409 = add_350 = None
        view_487 = torch.ops.aten.view.default(mul_411, [8, 256]);  mul_411 = None
        permute_271 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg246_1, view_487, permute_271);  arg246_1 = view_487 = permute_271 = None
        view_488 = torch.ops.aten.view.default(addmm_153, [8, 1, 128]);  addmm_153 = None
        slice_125 = torch.ops.aten.slice.Tensor(add_316, 1, 1, 9223372036854775807)
        cat_26 = torch.ops.aten.cat.default([view_488, slice_125], 1);  view_488 = slice_125 = None
        slice_127 = torch.ops.aten.slice.Tensor(add_316, 1, 1, 9223372036854775807);  add_316 = None
        cat_27 = torch.ops.aten.cat.default([view_468, slice_127], 1);  view_468 = slice_127 = None
        slice_129 = torch.ops.aten.slice.Tensor(cat_27, 1, 0, 1)
        var_mean_84 = torch.ops.aten.var_mean.correction(cat_27, [2], correction = 0, keepdim = True)
        getitem_336 = var_mean_84[0]
        getitem_337 = var_mean_84[1];  var_mean_84 = None
        add_351 = torch.ops.aten.add.Tensor(getitem_336, 1e-06);  getitem_336 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_139 = torch.ops.aten.sub.Tensor(cat_27, getitem_337);  cat_27 = getitem_337 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_139, rsqrt_84);  sub_139 = rsqrt_84 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, arg247_1);  mul_412 = arg247_1 = None
        add_352 = torch.ops.aten.add.Tensor(mul_413, arg248_1);  mul_413 = arg248_1 = None
        slice_131 = torch.ops.aten.slice.Tensor(add_352, 1, 0, 1)
        permute_272 = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        view_489 = torch.ops.aten.view.default(slice_131, [8, 128]);  slice_131 = None
        mm_11 = torch.ops.aten.mm.default(view_489, permute_272);  view_489 = permute_272 = None
        view_490 = torch.ops.aten.view.default(mm_11, [8, 1, 128]);  mm_11 = None
        add_353 = torch.ops.aten.add.Tensor(view_490, arg250_1);  view_490 = arg250_1 = None
        view_491 = torch.ops.aten.view.default(add_353, [8, 1, 4, 32]);  add_353 = None
        permute_273 = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
        view_492 = torch.ops.aten.view.default(add_352, [3208, 128])
        permute_274 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg252_1, view_492, permute_274);  arg252_1 = view_492 = permute_274 = None
        view_493 = torch.ops.aten.view.default(addmm_154, [8, 401, 128]);  addmm_154 = None
        view_494 = torch.ops.aten.view.default(view_493, [8, 401, 4, 32]);  view_493 = None
        permute_275 = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        view_495 = torch.ops.aten.view.default(add_352, [3208, 128]);  add_352 = None
        permute_276 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg254_1, view_495, permute_276);  arg254_1 = view_495 = permute_276 = None
        view_496 = torch.ops.aten.view.default(addmm_155, [8, 401, 128]);  addmm_155 = None
        view_497 = torch.ops.aten.view.default(view_496, [8, 401, 4, 32]);  view_496 = None
        permute_277 = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
        permute_278 = torch.ops.aten.permute.default(permute_275, [0, 1, 3, 2]);  permute_275 = None
        expand_48 = torch.ops.aten.expand.default(permute_273, [8, 4, 1, 32]);  permute_273 = None
        view_498 = torch.ops.aten.view.default(expand_48, [32, 1, 32]);  expand_48 = None
        expand_49 = torch.ops.aten.expand.default(permute_278, [8, 4, 32, 401]);  permute_278 = None
        clone_134 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_499 = torch.ops.aten.view.default(clone_134, [32, 32, 401]);  clone_134 = None
        bmm_22 = torch.ops.aten.bmm.default(view_498, view_499);  view_498 = view_499 = None
        view_500 = torch.ops.aten.view.default(bmm_22, [8, 4, 1, 401]);  bmm_22 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_500, 1);  view_500 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, 0.1767766952966369);  sub_tensor = None
        exp_11 = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        expand_50 = torch.ops.aten.expand.default(div_11, [8, 4, 1, 401]);  div_11 = None
        view_501 = torch.ops.aten.view.default(expand_50, [32, 1, 401]);  expand_50 = None
        expand_51 = torch.ops.aten.expand.default(permute_277, [8, 4, 401, 32]);  permute_277 = None
        clone_136 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_502 = torch.ops.aten.view.default(clone_136, [32, 401, 32]);  clone_136 = None
        bmm_23 = torch.ops.aten.bmm.default(view_501, view_502);  view_501 = view_502 = None
        view_503 = torch.ops.aten.view.default(bmm_23, [8, 4, 1, 32]);  bmm_23 = None
        permute_279 = torch.ops.aten.permute.default(view_503, [0, 2, 1, 3]);  view_503 = None
        view_504 = torch.ops.aten.view.default(permute_279, [8, 1, 128]);  permute_279 = None
        view_505 = torch.ops.aten.view.default(view_504, [8, 128]);  view_504 = None
        permute_280 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg256_1, view_505, permute_280);  arg256_1 = view_505 = permute_280 = None
        view_506 = torch.ops.aten.view.default(addmm_156, [8, 1, 128]);  addmm_156 = None
        add_354 = torch.ops.aten.add.Tensor(slice_129, view_506);  slice_129 = view_506 = None
        var_mean_85 = torch.ops.aten.var_mean.correction(add_354, [2], correction = 0, keepdim = True)
        getitem_338 = var_mean_85[0]
        getitem_339 = var_mean_85[1];  var_mean_85 = None
        add_355 = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        sub_141 = torch.ops.aten.sub.Tensor(add_354, getitem_339);  add_354 = getitem_339 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_141, rsqrt_85);  sub_141 = rsqrt_85 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, arg257_1);  mul_415 = arg257_1 = None
        add_356 = torch.ops.aten.add.Tensor(mul_416, arg258_1);  mul_416 = arg258_1 = None
        mul_417 = torch.ops.aten.mul.Tensor(add_356, 0.5)
        mul_418 = torch.ops.aten.mul.Tensor(add_356, 0.7071067811865476);  add_356 = None
        erf_47 = torch.ops.aten.erf.default(mul_418);  mul_418 = None
        add_357 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_417, add_357);  mul_417 = add_357 = None
        view_507 = torch.ops.aten.view.default(mul_419, [8, 128]);  mul_419 = None
        permute_281 = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg260_1, view_507, permute_281);  arg260_1 = view_507 = permute_281 = None
        view_508 = torch.ops.aten.view.default(addmm_157, [8, 1, 256]);  addmm_157 = None
        slice_134 = torch.ops.aten.slice.Tensor(add_337, 1, 1, 9223372036854775807);  add_337 = None
        cat_28 = torch.ops.aten.cat.default([view_508, slice_134], 1);  view_508 = slice_134 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(cat_26, [2], correction = 0, keepdim = True)
        getitem_340 = var_mean_86[0]
        getitem_341 = var_mean_86[1];  var_mean_86 = None
        add_358 = torch.ops.aten.add.Tensor(getitem_340, 1e-06);  getitem_340 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_358);  add_358 = None
        sub_142 = torch.ops.aten.sub.Tensor(cat_26, getitem_341);  cat_26 = getitem_341 = None
        mul_420 = torch.ops.aten.mul.Tensor(sub_142, rsqrt_86);  sub_142 = rsqrt_86 = None
        mul_421 = torch.ops.aten.mul.Tensor(mul_420, arg261_1);  mul_420 = arg261_1 = None
        add_359 = torch.ops.aten.add.Tensor(mul_421, arg262_1);  mul_421 = arg262_1 = None
        var_mean_87 = torch.ops.aten.var_mean.correction(cat_28, [2], correction = 0, keepdim = True)
        getitem_342 = var_mean_87[0]
        getitem_343 = var_mean_87[1];  var_mean_87 = None
        add_360 = torch.ops.aten.add.Tensor(getitem_342, 1e-06);  getitem_342 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_360);  add_360 = None
        sub_143 = torch.ops.aten.sub.Tensor(cat_28, getitem_343);  cat_28 = getitem_343 = None
        mul_422 = torch.ops.aten.mul.Tensor(sub_143, rsqrt_87);  sub_143 = rsqrt_87 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_422, arg263_1);  mul_422 = arg263_1 = None
        add_361 = torch.ops.aten.add.Tensor(mul_423, arg264_1);  mul_423 = arg264_1 = None
        select_2 = torch.ops.aten.select.int(add_359, 1, 0);  add_359 = None
        select_3 = torch.ops.aten.select.int(add_361, 1, 0);  add_361 = None
        clone_138 = torch.ops.aten.clone.default(select_2);  select_2 = None
        clone_139 = torch.ops.aten.clone.default(select_3);  select_3 = None
        permute_282 = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg266_1, clone_138, permute_282);  arg266_1 = clone_138 = permute_282 = None
        permute_283 = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg268_1, clone_139, permute_283);  arg268_1 = clone_139 = permute_283 = None
        cat_29 = torch.ops.aten.cat.default([addmm_158, addmm_159]);  addmm_158 = addmm_159 = None
        view_509 = torch.ops.aten.view.default(cat_29, [2, 8, 1000]);  cat_29 = None
        mean_1 = torch.ops.aten.mean.dim(view_509, [0]);  view_509 = None
        return (mean_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 5529600, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 240, 240), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 3, 12, 12), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 1, 128), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 205312, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1, 401, 128), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf5, (256, 3, 16, 16), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf6, (256,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1, 1, 256), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 201728, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1, 197, 256), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf11, (384, 128), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf12, (384,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128, 128), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf17, (384, 128), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 384), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768, 256), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256, 256), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768, 256), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256, 768), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768, 256), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256, 256), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 256), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256, 768), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf46, (256,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768, 256), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256, 256), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf50, (256,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768, 256), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256, 768), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf57, (128,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf58, (128,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256, 128), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf61, (256,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf62, (256,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128, 256), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf65, (256,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf67, (256, 256), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf68, (256,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf69, (256, 256), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf70, (256,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf71, (256, 256), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf72, (256,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256, 256), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf74, (256,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf75, (256,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf76, (256,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128, 256), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf79, (128,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (128,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (128, 128), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf82, (128,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128, 128), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128, 128), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (128, 128), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf88, (128,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf89, (128,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256, 128), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf92, (256,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf94, (128,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf95, (384, 128), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf96, (384,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128, 128), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf99, (128,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf100, (128,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf101, (384, 128), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf102, (384,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf103, (128, 384), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf104, (128,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf105, (256,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf106, (256,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768, 256), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf109, (256, 256), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf110, (256,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768, 256), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf115, (256, 768), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf116, (256,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf117, (256,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf118, (256,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768, 256), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256, 256), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768, 256), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256, 768), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf128, (256,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768, 256), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (256, 256), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf135, (256,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf136, (256,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768, 256), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf139, (256, 768), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf140, (256,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf141, (128,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf142, (128,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256, 128), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf144, (256,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf145, (256,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf146, (256,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (128, 256), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf148, (128,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf149, (256,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf150, (256,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf151, (256, 256), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf152, (256,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (256, 256), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf154, (256,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf155, (256, 256), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf156, (256,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf157, (256, 256), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf158, (256,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf159, (256,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf160, (256,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (128, 256), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf162, (128,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf163, (128,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf164, (128,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf165, (128, 128), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf166, (128,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf167, (128, 128), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf168, (128,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf169, (128, 128), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf170, (128,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf171, (128, 128), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf172, (128,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf173, (128,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf174, (128,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf175, (256, 128), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf176, (256,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf177, (128,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf178, (128,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf179, (384, 128), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf180, (384,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf181, (128, 128), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf182, (128,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf183, (128,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf184, (128,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf185, (384, 128), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf186, (384,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf187, (128, 384), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf188, (128,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf189, (256,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf190, (256,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768, 256), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf192, (768,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf193, (256, 256), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf194, (256,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf195, (256,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf196, (256,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768, 256), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf199, (256, 768), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf200, (256,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf201, (256,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf202, (256,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf203, (768, 256), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf204, (768,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf205, (256, 256), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf206, (256,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf207, (256,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf208, (256,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf209, (768, 256), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf210, (768,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf211, (256, 768), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf212, (256,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf213, (256,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf214, (256,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf215, (768, 256), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf216, (768,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf217, (256, 256), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf218, (256,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf219, (256,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf220, (256,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf221, (768, 256), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf222, (768,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf223, (256, 768), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf224, (256,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf225, (128,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf226, (128,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf227, (256, 128), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf228, (256,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf229, (256,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf230, (256,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf231, (128, 256), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf232, (128,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf233, (256,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf234, (256,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf235, (256, 256), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf236, (256,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf237, (256, 256), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf238, (256,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf239, (256, 256), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf240, (256,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf241, (256, 256), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf242, (256,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf243, (256,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf244, (256,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf245, (128, 256), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf246, (128,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf247, (128,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf248, (128,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf249, (128, 128), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf250, (128,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf251, (128, 128), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf252, (128,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf253, (128, 128), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf254, (128,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf255, (128, 128), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf256, (128,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf257, (128,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf258, (128,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf259, (256, 128), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf260, (256,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf261, (128,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf262, (128,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf263, (256,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf264, (256,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 512000, device=device(type='cuda', index=0))
    reader.tensor(buf265, (1000, 128), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf266, (1000,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1000, 256), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1000,), is_leaf=True)  # arg268_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)