
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1):
        convolution_41 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_50 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_23 = torch.ops.aten.sqrt.default(add_50);  add_50 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_73 = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        sub_23 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_185);  convolution_41 = unsqueeze_185 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_189);  mul_74 = unsqueeze_189 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_51 = torch.ops.aten.add.Tensor(mul_75, unsqueeze_191);  mul_75 = unsqueeze_191 = None
        relu_23 = torch.ops.aten.relu.default(add_51);  add_51 = None
        convolution_42 = torch.ops.aten.convolution.default(relu_23, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  relu_23 = arg6_1 = None
        convolution_43 = torch.ops.aten.convolution.default(convolution_42, arg7_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_42 = arg7_1 = None
        add_52 = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_24 = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_24 = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_76 = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        sub_24 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_193);  convolution_43 = unsqueeze_193 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_197);  mul_77 = unsqueeze_197 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_53 = torch.ops.aten.add.Tensor(mul_78, unsqueeze_199);  mul_78 = unsqueeze_199 = None
        relu_24 = torch.ops.aten.relu.default(add_53);  add_53 = None
        convolution_44 = torch.ops.aten.convolution.default(relu_24, arg12_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  relu_24 = arg12_1 = None
        convolution_45 = torch.ops.aten.convolution.default(convolution_44, arg13_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_44 = arg13_1 = None
        add_54 = torch.ops.aten.add.Tensor(arg15_1, 1e-05);  arg15_1 = None
        sqrt_25 = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_25 = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_79 = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        sub_25 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_201);  convolution_45 = unsqueeze_201 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_205);  mul_80 = unsqueeze_205 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_55 = torch.ops.aten.add.Tensor(mul_81, unsqueeze_207);  mul_81 = unsqueeze_207 = None
        relu_25 = torch.ops.aten.relu.default(add_55);  add_55 = None
        convolution_46 = torch.ops.aten.convolution.default(relu_25, arg18_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg18_1 = None
        add_56 = torch.ops.aten.add.Tensor(arg20_1, 1e-05);  arg20_1 = None
        sqrt_26 = torch.ops.aten.sqrt.default(add_56);  add_56 = None
        reciprocal_26 = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_82 = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        sub_26 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_209);  convolution_46 = unsqueeze_209 = None
        mul_83 = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_213);  mul_83 = unsqueeze_213 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_57 = torch.ops.aten.add.Tensor(mul_84, unsqueeze_215);  mul_84 = unsqueeze_215 = None
        relu_26 = torch.ops.aten.relu.default(add_57);  add_57 = None
        convolution_47 = torch.ops.aten.convolution.default(relu_26, arg23_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_26 = arg23_1 = None
        convolution_48 = torch.ops.aten.convolution.default(convolution_47, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_47 = arg24_1 = None
        add_58 = torch.ops.aten.add.Tensor(arg26_1, 1e-05);  arg26_1 = None
        sqrt_27 = torch.ops.aten.sqrt.default(add_58);  add_58 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_85 = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(mul_85, -1);  mul_85 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        sub_27 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_217);  convolution_48 = unsqueeze_217 = None
        mul_86 = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_221);  mul_86 = unsqueeze_221 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_59 = torch.ops.aten.add.Tensor(mul_87, unsqueeze_223);  mul_87 = unsqueeze_223 = None
        relu_27 = torch.ops.aten.relu.default(add_59);  add_59 = None
        convolution_49 = torch.ops.aten.convolution.default(relu_27, arg29_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg29_1 = None
        convolution_50 = torch.ops.aten.convolution.default(convolution_49, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_49 = arg30_1 = None
        add_60 = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_28 = torch.ops.aten.sqrt.default(add_60);  add_60 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_88 = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        sub_28 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_225);  convolution_50 = unsqueeze_225 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_229);  mul_89 = unsqueeze_229 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_61 = torch.ops.aten.add.Tensor(mul_90, unsqueeze_231);  mul_90 = unsqueeze_231 = None
        relu_28 = torch.ops.aten.relu.default(add_61);  add_61 = None
        convolution_51 = torch.ops.aten.convolution.default(relu_28, arg35_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg35_1 = None
        convolution_52 = torch.ops.aten.convolution.default(convolution_51, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_51 = arg36_1 = None
        add_62 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_62);  add_62 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_91 = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        sub_29 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_233);  convolution_52 = unsqueeze_233 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_237);  mul_92 = unsqueeze_237 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_63 = torch.ops.aten.add.Tensor(mul_93, unsqueeze_239);  mul_93 = unsqueeze_239 = None
        relu_29 = torch.ops.aten.relu.default(add_63);  add_63 = None
        cat_4 = torch.ops.aten.cat.default([relu_25, relu_27, relu_28, relu_29], 1);  relu_25 = relu_27 = relu_28 = relu_29 = None
        convolution_53 = torch.ops.aten.convolution.default(cat_4, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg41_1 = None
        add_64 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_94 = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(mul_94, -1);  mul_94 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        sub_30 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_241);  convolution_53 = unsqueeze_241 = None
        mul_95 = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_245);  mul_95 = unsqueeze_245 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_65 = torch.ops.aten.add.Tensor(mul_96, unsqueeze_247);  mul_96 = unsqueeze_247 = None
        relu_30 = torch.ops.aten.relu.default(add_65);  add_65 = None
        mean_5 = torch.ops.aten.mean.dim(relu_30, [2, 3], True)
        convolution_54 = torch.ops.aten.convolution.default(mean_5, arg46_1, arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg46_1 = arg47_1 = None
        add_66 = torch.ops.aten.add.Tensor(convolution_54, 3);  convolution_54 = None
        clamp_min_4 = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
        clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
        div_4 = torch.ops.aten.div.Tensor(clamp_max_4, 6);  clamp_max_4 = None
        mul_97 = torch.ops.aten.mul.Tensor(relu_30, div_4);  relu_30 = div_4 = None
        _low_memory_max_pool2d_with_offsets_3 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_97, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_97 = None
        getitem_6 = _low_memory_max_pool2d_with_offsets_3[0];  _low_memory_max_pool2d_with_offsets_3 = None
        convolution_55 = torch.ops.aten.convolution.default(getitem_6, arg48_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg48_1 = None
        add_67 = torch.ops.aten.add.Tensor(arg50_1, 1e-05);  arg50_1 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_67);  add_67 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_98 = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(mul_98, -1);  mul_98 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        sub_31 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_249);  convolution_55 = unsqueeze_249 = None
        mul_99 = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_253);  mul_99 = unsqueeze_253 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_68 = torch.ops.aten.add.Tensor(mul_100, unsqueeze_255);  mul_100 = unsqueeze_255 = None
        relu_31 = torch.ops.aten.relu.default(add_68);  add_68 = None
        convolution_56 = torch.ops.aten.convolution.default(relu_31, arg53_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  relu_31 = arg53_1 = None
        convolution_57 = torch.ops.aten.convolution.default(convolution_56, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_56 = arg54_1 = None
        add_69 = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_69);  add_69 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_101 = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_101, -1);  mul_101 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_32 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_257);  convolution_57 = unsqueeze_257 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_261);  mul_102 = unsqueeze_261 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_70 = torch.ops.aten.add.Tensor(mul_103, unsqueeze_263);  mul_103 = unsqueeze_263 = None
        relu_32 = torch.ops.aten.relu.default(add_70);  add_70 = None
        convolution_58 = torch.ops.aten.convolution.default(relu_32, arg59_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  arg59_1 = None
        convolution_59 = torch.ops.aten.convolution.default(convolution_58, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_58 = arg60_1 = None
        add_71 = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_104 = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_104, -1);  mul_104 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_33 = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_265);  convolution_59 = unsqueeze_265 = None
        mul_105 = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_269);  mul_105 = unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_72 = torch.ops.aten.add.Tensor(mul_106, unsqueeze_271);  mul_106 = unsqueeze_271 = None
        relu_33 = torch.ops.aten.relu.default(add_72);  add_72 = None
        convolution_60 = torch.ops.aten.convolution.default(relu_33, arg65_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  arg65_1 = None
        convolution_61 = torch.ops.aten.convolution.default(convolution_60, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_60 = arg66_1 = None
        add_73 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_73);  add_73 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_107 = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(mul_107, -1);  mul_107 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_34 = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_273);  convolution_61 = unsqueeze_273 = None
        mul_108 = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_277);  mul_108 = unsqueeze_277 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_74 = torch.ops.aten.add.Tensor(mul_109, unsqueeze_279);  mul_109 = unsqueeze_279 = None
        relu_34 = torch.ops.aten.relu.default(add_74);  add_74 = None
        cat_5 = torch.ops.aten.cat.default([getitem_6, relu_32, relu_33, relu_34], 1);  getitem_6 = relu_32 = relu_33 = relu_34 = None
        convolution_62 = torch.ops.aten.convolution.default(cat_5, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg71_1 = None
        add_75 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_75);  add_75 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_110 = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_110, -1);  mul_110 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_35 = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_281);  convolution_62 = unsqueeze_281 = None
        mul_111 = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_285);  mul_111 = unsqueeze_285 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_76 = torch.ops.aten.add.Tensor(mul_112, unsqueeze_287);  mul_112 = unsqueeze_287 = None
        relu_35 = torch.ops.aten.relu.default(add_76);  add_76 = None
        mean_6 = torch.ops.aten.mean.dim(relu_35, [2, 3], True)
        convolution_63 = torch.ops.aten.convolution.default(mean_6, arg76_1, arg77_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg76_1 = arg77_1 = None
        add_77 = torch.ops.aten.add.Tensor(convolution_63, 3);  convolution_63 = None
        clamp_min_5 = torch.ops.aten.clamp_min.default(add_77, 0);  add_77 = None
        clamp_max_5 = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
        div_5 = torch.ops.aten.div.Tensor(clamp_max_5, 6);  clamp_max_5 = None
        mul_113 = torch.ops.aten.mul.Tensor(relu_35, div_5);  relu_35 = div_5 = None
        _low_memory_max_pool2d_with_offsets_4 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_113, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_113 = None
        getitem_8 = _low_memory_max_pool2d_with_offsets_4[0];  _low_memory_max_pool2d_with_offsets_4 = None
        convolution_64 = torch.ops.aten.convolution.default(getitem_8, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg78_1 = None
        add_78 = torch.ops.aten.add.Tensor(arg80_1, 1e-05);  arg80_1 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_114 = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_36 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_289);  convolution_64 = unsqueeze_289 = None
        mul_115 = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_293);  mul_115 = unsqueeze_293 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_79 = torch.ops.aten.add.Tensor(mul_116, unsqueeze_295);  mul_116 = unsqueeze_295 = None
        relu_36 = torch.ops.aten.relu.default(add_79);  add_79 = None
        convolution_65 = torch.ops.aten.convolution.default(relu_36, arg83_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  relu_36 = arg83_1 = None
        convolution_66 = torch.ops.aten.convolution.default(convolution_65, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_65 = arg84_1 = None
        add_80 = torch.ops.aten.add.Tensor(arg86_1, 1e-05);  arg86_1 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_117 = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_37 = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_297);  convolution_66 = unsqueeze_297 = None
        mul_118 = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_301);  mul_118 = unsqueeze_301 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_81 = torch.ops.aten.add.Tensor(mul_119, unsqueeze_303);  mul_119 = unsqueeze_303 = None
        relu_37 = torch.ops.aten.relu.default(add_81);  add_81 = None
        convolution_67 = torch.ops.aten.convolution.default(relu_37, arg89_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  arg89_1 = None
        convolution_68 = torch.ops.aten.convolution.default(convolution_67, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_67 = arg90_1 = None
        add_82 = torch.ops.aten.add.Tensor(arg92_1, 1e-05);  arg92_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_120 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_38 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_305);  convolution_68 = unsqueeze_305 = None
        mul_121 = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_309);  mul_121 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_83 = torch.ops.aten.add.Tensor(mul_122, unsqueeze_311);  mul_122 = unsqueeze_311 = None
        relu_38 = torch.ops.aten.relu.default(add_83);  add_83 = None
        convolution_69 = torch.ops.aten.convolution.default(relu_38, arg95_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  arg95_1 = None
        convolution_70 = torch.ops.aten.convolution.default(convolution_69, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_69 = arg96_1 = None
        add_84 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_123 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_39 = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_313);  convolution_70 = unsqueeze_313 = None
        mul_124 = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_317);  mul_124 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_85 = torch.ops.aten.add.Tensor(mul_125, unsqueeze_319);  mul_125 = unsqueeze_319 = None
        relu_39 = torch.ops.aten.relu.default(add_85);  add_85 = None
        cat_6 = torch.ops.aten.cat.default([getitem_8, relu_37, relu_38, relu_39], 1);  getitem_8 = relu_37 = relu_38 = relu_39 = None
        convolution_71 = torch.ops.aten.convolution.default(cat_6, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg101_1 = None
        add_86 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_126 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_40 = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_321);  convolution_71 = unsqueeze_321 = None
        mul_127 = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_325);  mul_127 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_87 = torch.ops.aten.add.Tensor(mul_128, unsqueeze_327);  mul_128 = unsqueeze_327 = None
        relu_40 = torch.ops.aten.relu.default(add_87);  add_87 = None
        mean_7 = torch.ops.aten.mean.dim(relu_40, [2, 3], True)
        convolution_72 = torch.ops.aten.convolution.default(mean_7, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg106_1 = arg107_1 = None
        add_88 = torch.ops.aten.add.Tensor(convolution_72, 3);  convolution_72 = None
        clamp_min_6 = torch.ops.aten.clamp_min.default(add_88, 0);  add_88 = None
        clamp_max_6 = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
        div_6 = torch.ops.aten.div.Tensor(clamp_max_6, 6);  clamp_max_6 = None
        mul_129 = torch.ops.aten.mul.Tensor(relu_40, div_6);  relu_40 = div_6 = None
        _low_memory_max_pool2d_with_offsets_5 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_129, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_129 = None
        getitem_10 = _low_memory_max_pool2d_with_offsets_5[0];  _low_memory_max_pool2d_with_offsets_5 = None
        convolution_73 = torch.ops.aten.convolution.default(getitem_10, arg108_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg108_1 = None
        add_89 = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_89);  add_89 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_130 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_329);  convolution_73 = unsqueeze_329 = None
        mul_131 = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_333);  mul_131 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_90 = torch.ops.aten.add.Tensor(mul_132, unsqueeze_335);  mul_132 = unsqueeze_335 = None
        relu_41 = torch.ops.aten.relu.default(add_90);  add_90 = None
        convolution_74 = torch.ops.aten.convolution.default(relu_41, arg113_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  relu_41 = arg113_1 = None
        convolution_75 = torch.ops.aten.convolution.default(convolution_74, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_74 = arg114_1 = None
        add_91 = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_91);  add_91 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_133 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_133, -1);  mul_133 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_337);  convolution_75 = unsqueeze_337 = None
        mul_134 = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, unsqueeze_341);  mul_134 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_92 = torch.ops.aten.add.Tensor(mul_135, unsqueeze_343);  mul_135 = unsqueeze_343 = None
        relu_42 = torch.ops.aten.relu.default(add_92);  add_92 = None
        convolution_76 = torch.ops.aten.convolution.default(relu_42, arg119_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  arg119_1 = None
        convolution_77 = torch.ops.aten.convolution.default(convolution_76, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_76 = arg120_1 = None
        add_93 = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_136 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_136, -1);  mul_136 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_345);  convolution_77 = unsqueeze_345 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_349);  mul_137 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_94 = torch.ops.aten.add.Tensor(mul_138, unsqueeze_351);  mul_138 = unsqueeze_351 = None
        relu_43 = torch.ops.aten.relu.default(add_94);  add_94 = None
        convolution_78 = torch.ops.aten.convolution.default(relu_43, arg125_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  arg125_1 = None
        convolution_79 = torch.ops.aten.convolution.default(convolution_78, arg126_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_78 = arg126_1 = None
        add_95 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_139 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_353);  convolution_79 = unsqueeze_353 = None
        mul_140 = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_357);  mul_140 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_96 = torch.ops.aten.add.Tensor(mul_141, unsqueeze_359);  mul_141 = unsqueeze_359 = None
        relu_44 = torch.ops.aten.relu.default(add_96);  add_96 = None
        cat_7 = torch.ops.aten.cat.default([getitem_10, relu_42, relu_43, relu_44], 1);  getitem_10 = relu_42 = relu_43 = relu_44 = None
        convolution_80 = torch.ops.aten.convolution.default(cat_7, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg131_1 = None
        add_97 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_97);  add_97 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_142 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_361);  convolution_80 = unsqueeze_361 = None
        mul_143 = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_365);  mul_143 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_98 = torch.ops.aten.add.Tensor(mul_144, unsqueeze_367);  mul_144 = unsqueeze_367 = None
        relu_45 = torch.ops.aten.relu.default(add_98);  add_98 = None
        mean_8 = torch.ops.aten.mean.dim(relu_45, [2, 3], True)
        convolution_81 = torch.ops.aten.convolution.default(mean_8, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg136_1 = arg137_1 = None
        add_99 = torch.ops.aten.add.Tensor(convolution_81, 3);  convolution_81 = None
        clamp_min_7 = torch.ops.aten.clamp_min.default(add_99, 0);  add_99 = None
        clamp_max_7 = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        div_7 = torch.ops.aten.div.Tensor(clamp_max_7, 6);  clamp_max_7 = None
        mul_145 = torch.ops.aten.mul.Tensor(relu_45, div_7);  relu_45 = div_7 = None
        mean_9 = torch.ops.aten.mean.dim(mul_145, [-1, -2], True);  mul_145 = None
        view_1 = torch.ops.aten.view.default(mean_9, [8, 1024]);  mean_9 = None
        permute_1 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg139_1, view_1, permute_1);  arg139_1 = view_1 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 7962624, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 288, 288), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 1, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64, 64, 1, 1), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64, 1, 3, 3), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64, 64, 1, 1), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128, 64, 1, 1), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128, 1, 3, 3), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf24, (128, 128, 1, 1), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128, 1, 3, 3), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128, 128, 1, 1), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf31, (128,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf34, (128,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf35, (128, 1, 3, 3), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf36, (128, 128, 1, 1), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf37, (128,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf39, (128,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf40, (128,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 458752, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256, 448, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf46, (256, 256, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf47, (256,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 163840, device=device(type='cuda', index=0))
    reader.tensor(buf48, (160, 256, 1, 1), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf49, (160,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf50, (160,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf51, (160,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf52, (160,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 5760, device=device(type='cuda', index=0))
    reader.tensor(buf53, (160, 1, 3, 3), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 102400, device=device(type='cuda', index=0))
    reader.tensor(buf54, (160, 160, 1, 1), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf55, (160,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf56, (160,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf57, (160,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf58, (160,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 5760, device=device(type='cuda', index=0))
    reader.tensor(buf59, (160, 1, 3, 3), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 102400, device=device(type='cuda', index=0))
    reader.tensor(buf60, (160, 160, 1, 1), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf61, (160,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf62, (160,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf63, (160,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf64, (160,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 5760, device=device(type='cuda', index=0))
    reader.tensor(buf65, (160, 1, 3, 3), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 102400, device=device(type='cuda', index=0))
    reader.tensor(buf66, (160, 160, 1, 1), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf67, (160,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf68, (160,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf69, (160,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf70, (160,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1507328, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512, 736, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512, 512, 1, 1), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf77, (512,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf78, (192, 512, 1, 1), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf79, (192,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf80, (192,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf81, (192,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf82, (192,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf83, (192, 1, 3, 3), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf84, (192, 192, 1, 1), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf85, (192,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf86, (192,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf87, (192,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf88, (192,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf89, (192, 1, 3, 3), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf90, (192, 192, 1, 1), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf91, (192,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf92, (192,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf93, (192,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf94, (192,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf95, (192, 1, 3, 3), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf96, (192, 192, 1, 1), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf97, (192,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf98, (192,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf99, (192,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf100, (192,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 3342336, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768, 1088, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 768, 1, 1), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 688128, device=device(type='cuda', index=0))
    reader.tensor(buf108, (224, 768, 1, 1), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf109, (224,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf110, (224,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf111, (224,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf112, (224,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 8064, device=device(type='cuda', index=0))
    reader.tensor(buf113, (224, 1, 3, 3), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf114, (224, 224, 1, 1), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf115, (224,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf116, (224,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf117, (224,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf118, (224,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 8064, device=device(type='cuda', index=0))
    reader.tensor(buf119, (224, 1, 3, 3), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf120, (224, 224, 1, 1), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf121, (224,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf122, (224,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf123, (224,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf124, (224,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 8064, device=device(type='cuda', index=0))
    reader.tensor(buf125, (224, 1, 3, 3), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf126, (224, 224, 1, 1), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf127, (224,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf128, (224,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf129, (224,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf130, (224,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 5898240, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024, 1440, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1024,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1024,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1024,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024, 1024, 1, 1), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1000, 1024), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1000,), is_leaf=True)  # arg139_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)