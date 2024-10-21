
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1):
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_193 = torch.ops.aten.view.default(convolution_1, [8, 384, 196]);  convolution_1 = None
        permute_146 = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        clone_170 = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format)
        var_mean_49 = torch.ops.aten.var_mean.correction(clone_170, [2], correction = 0, keepdim = True)
        getitem_194 = var_mean_49[0]
        getitem_195 = var_mean_49[1];  var_mean_49 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_49 = torch.ops.aten.sub.Tensor(clone_170, getitem_195);  clone_170 = getitem_195 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, arg3_1);  mul_194 = arg3_1 = None
        add_171 = torch.ops.aten.add.Tensor(mul_195, arg4_1);  mul_195 = arg4_1 = None
        permute_147 = torch.ops.aten.permute.default(add_171, [0, 2, 1]);  add_171 = None
        permute_148 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        clone_171 = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_194 = torch.ops.aten.view.default(clone_171, [3072, 196]);  clone_171 = None
        mm_24 = torch.ops.aten.mm.default(view_194, permute_148);  view_194 = permute_148 = None
        view_195 = torch.ops.aten.view.default(mm_24, [8, 384, 384]);  mm_24 = None
        add_172 = torch.ops.aten.add.Tensor(view_195, arg6_1);  view_195 = arg6_1 = None
        split_48 = torch.ops.aten.split.Tensor(add_172, 192, -1);  add_172 = None
        getitem_196 = split_48[0]
        getitem_197 = split_48[1];  split_48 = None
        sigmoid_48 = torch.ops.aten.sigmoid.default(getitem_197)
        mul_196 = torch.ops.aten.mul.Tensor(getitem_197, sigmoid_48);  getitem_197 = sigmoid_48 = None
        mul_197 = torch.ops.aten.mul.Tensor(getitem_196, mul_196);  getitem_196 = mul_196 = None
        view_196 = torch.ops.aten.view.default(mul_197, [3072, 192]);  mul_197 = None
        permute_149 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg8_1, view_196, permute_149);  arg8_1 = view_196 = permute_149 = None
        view_197 = torch.ops.aten.view.default(addmm_73, [8, 384, 196]);  addmm_73 = None
        permute_150 = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
        add_173 = torch.ops.aten.add.Tensor(permute_146, permute_150);  permute_146 = permute_150 = None
        clone_174 = torch.ops.aten.clone.default(add_173, memory_format = torch.contiguous_format)
        var_mean_50 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
        getitem_198 = var_mean_50[0]
        getitem_199 = var_mean_50[1];  var_mean_50 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_50 = torch.ops.aten.sub.Tensor(clone_174, getitem_199);  clone_174 = getitem_199 = None
        mul_198 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_199 = torch.ops.aten.mul.Tensor(mul_198, arg9_1);  mul_198 = arg9_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_199, arg10_1);  mul_199 = arg10_1 = None
        view_198 = torch.ops.aten.view.default(add_175, [1568, 384]);  add_175 = None
        permute_151 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg12_1, view_198, permute_151);  arg12_1 = view_198 = permute_151 = None
        view_199 = torch.ops.aten.view.default(addmm_74, [8, 196, 1536]);  addmm_74 = None
        split_49 = torch.ops.aten.split.Tensor(view_199, 768, -1);  view_199 = None
        getitem_200 = split_49[0]
        getitem_201 = split_49[1];  split_49 = None
        sigmoid_49 = torch.ops.aten.sigmoid.default(getitem_201)
        mul_200 = torch.ops.aten.mul.Tensor(getitem_201, sigmoid_49);  getitem_201 = sigmoid_49 = None
        mul_201 = torch.ops.aten.mul.Tensor(getitem_200, mul_200);  getitem_200 = mul_200 = None
        view_200 = torch.ops.aten.view.default(mul_201, [1568, 768]);  mul_201 = None
        permute_152 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg14_1, view_200, permute_152);  arg14_1 = view_200 = permute_152 = None
        view_201 = torch.ops.aten.view.default(addmm_75, [8, 196, 384]);  addmm_75 = None
        add_176 = torch.ops.aten.add.Tensor(add_173, view_201);  add_173 = view_201 = None
        clone_177 = torch.ops.aten.clone.default(add_176, memory_format = torch.contiguous_format)
        var_mean_51 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
        getitem_202 = var_mean_51[0]
        getitem_203 = var_mean_51[1];  var_mean_51 = None
        add_177 = torch.ops.aten.add.Tensor(getitem_202, 1e-06);  getitem_202 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_51 = torch.ops.aten.sub.Tensor(clone_177, getitem_203);  clone_177 = getitem_203 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, arg15_1);  mul_202 = arg15_1 = None
        add_178 = torch.ops.aten.add.Tensor(mul_203, arg16_1);  mul_203 = arg16_1 = None
        permute_153 = torch.ops.aten.permute.default(add_178, [0, 2, 1]);  add_178 = None
        permute_154 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        clone_178 = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_202 = torch.ops.aten.view.default(clone_178, [3072, 196]);  clone_178 = None
        mm_25 = torch.ops.aten.mm.default(view_202, permute_154);  view_202 = permute_154 = None
        view_203 = torch.ops.aten.view.default(mm_25, [8, 384, 384]);  mm_25 = None
        add_179 = torch.ops.aten.add.Tensor(view_203, arg18_1);  view_203 = arg18_1 = None
        split_50 = torch.ops.aten.split.Tensor(add_179, 192, -1);  add_179 = None
        getitem_204 = split_50[0]
        getitem_205 = split_50[1];  split_50 = None
        sigmoid_50 = torch.ops.aten.sigmoid.default(getitem_205)
        mul_204 = torch.ops.aten.mul.Tensor(getitem_205, sigmoid_50);  getitem_205 = sigmoid_50 = None
        mul_205 = torch.ops.aten.mul.Tensor(getitem_204, mul_204);  getitem_204 = mul_204 = None
        view_204 = torch.ops.aten.view.default(mul_205, [3072, 192]);  mul_205 = None
        permute_155 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg20_1, view_204, permute_155);  arg20_1 = view_204 = permute_155 = None
        view_205 = torch.ops.aten.view.default(addmm_76, [8, 384, 196]);  addmm_76 = None
        permute_156 = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
        add_180 = torch.ops.aten.add.Tensor(add_176, permute_156);  add_176 = permute_156 = None
        clone_181 = torch.ops.aten.clone.default(add_180, memory_format = torch.contiguous_format)
        var_mean_52 = torch.ops.aten.var_mean.correction(clone_181, [2], correction = 0, keepdim = True)
        getitem_206 = var_mean_52[0]
        getitem_207 = var_mean_52[1];  var_mean_52 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_206, 1e-06);  getitem_206 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_52 = torch.ops.aten.sub.Tensor(clone_181, getitem_207);  clone_181 = getitem_207 = None
        mul_206 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_206, arg21_1);  mul_206 = arg21_1 = None
        add_182 = torch.ops.aten.add.Tensor(mul_207, arg22_1);  mul_207 = arg22_1 = None
        view_206 = torch.ops.aten.view.default(add_182, [1568, 384]);  add_182 = None
        permute_157 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg24_1, view_206, permute_157);  arg24_1 = view_206 = permute_157 = None
        view_207 = torch.ops.aten.view.default(addmm_77, [8, 196, 1536]);  addmm_77 = None
        split_51 = torch.ops.aten.split.Tensor(view_207, 768, -1);  view_207 = None
        getitem_208 = split_51[0]
        getitem_209 = split_51[1];  split_51 = None
        sigmoid_51 = torch.ops.aten.sigmoid.default(getitem_209)
        mul_208 = torch.ops.aten.mul.Tensor(getitem_209, sigmoid_51);  getitem_209 = sigmoid_51 = None
        mul_209 = torch.ops.aten.mul.Tensor(getitem_208, mul_208);  getitem_208 = mul_208 = None
        view_208 = torch.ops.aten.view.default(mul_209, [1568, 768]);  mul_209 = None
        permute_158 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg26_1, view_208, permute_158);  arg26_1 = view_208 = permute_158 = None
        view_209 = torch.ops.aten.view.default(addmm_78, [8, 196, 384]);  addmm_78 = None
        add_183 = torch.ops.aten.add.Tensor(add_180, view_209);  add_180 = view_209 = None
        clone_184 = torch.ops.aten.clone.default(add_183, memory_format = torch.contiguous_format)
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_184, [2], correction = 0, keepdim = True)
        getitem_210 = var_mean_53[0]
        getitem_211 = var_mean_53[1];  var_mean_53 = None
        add_184 = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        sub_53 = torch.ops.aten.sub.Tensor(clone_184, getitem_211);  clone_184 = getitem_211 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_210, arg27_1);  mul_210 = arg27_1 = None
        add_185 = torch.ops.aten.add.Tensor(mul_211, arg28_1);  mul_211 = arg28_1 = None
        permute_159 = torch.ops.aten.permute.default(add_185, [0, 2, 1]);  add_185 = None
        permute_160 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        clone_185 = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_210 = torch.ops.aten.view.default(clone_185, [3072, 196]);  clone_185 = None
        mm_26 = torch.ops.aten.mm.default(view_210, permute_160);  view_210 = permute_160 = None
        view_211 = torch.ops.aten.view.default(mm_26, [8, 384, 384]);  mm_26 = None
        add_186 = torch.ops.aten.add.Tensor(view_211, arg30_1);  view_211 = arg30_1 = None
        split_52 = torch.ops.aten.split.Tensor(add_186, 192, -1);  add_186 = None
        getitem_212 = split_52[0]
        getitem_213 = split_52[1];  split_52 = None
        sigmoid_52 = torch.ops.aten.sigmoid.default(getitem_213)
        mul_212 = torch.ops.aten.mul.Tensor(getitem_213, sigmoid_52);  getitem_213 = sigmoid_52 = None
        mul_213 = torch.ops.aten.mul.Tensor(getitem_212, mul_212);  getitem_212 = mul_212 = None
        view_212 = torch.ops.aten.view.default(mul_213, [3072, 192]);  mul_213 = None
        permute_161 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg32_1, view_212, permute_161);  arg32_1 = view_212 = permute_161 = None
        view_213 = torch.ops.aten.view.default(addmm_79, [8, 384, 196]);  addmm_79 = None
        permute_162 = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        add_187 = torch.ops.aten.add.Tensor(add_183, permute_162);  add_183 = permute_162 = None
        clone_188 = torch.ops.aten.clone.default(add_187, memory_format = torch.contiguous_format)
        var_mean_54 = torch.ops.aten.var_mean.correction(clone_188, [2], correction = 0, keepdim = True)
        getitem_214 = var_mean_54[0]
        getitem_215 = var_mean_54[1];  var_mean_54 = None
        add_188 = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_54 = torch.ops.aten.sub.Tensor(clone_188, getitem_215);  clone_188 = getitem_215 = None
        mul_214 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, arg33_1);  mul_214 = arg33_1 = None
        add_189 = torch.ops.aten.add.Tensor(mul_215, arg34_1);  mul_215 = arg34_1 = None
        view_214 = torch.ops.aten.view.default(add_189, [1568, 384]);  add_189 = None
        permute_163 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg36_1, view_214, permute_163);  arg36_1 = view_214 = permute_163 = None
        view_215 = torch.ops.aten.view.default(addmm_80, [8, 196, 1536]);  addmm_80 = None
        split_53 = torch.ops.aten.split.Tensor(view_215, 768, -1);  view_215 = None
        getitem_216 = split_53[0]
        getitem_217 = split_53[1];  split_53 = None
        sigmoid_53 = torch.ops.aten.sigmoid.default(getitem_217)
        mul_216 = torch.ops.aten.mul.Tensor(getitem_217, sigmoid_53);  getitem_217 = sigmoid_53 = None
        mul_217 = torch.ops.aten.mul.Tensor(getitem_216, mul_216);  getitem_216 = mul_216 = None
        view_216 = torch.ops.aten.view.default(mul_217, [1568, 768]);  mul_217 = None
        permute_164 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg38_1, view_216, permute_164);  arg38_1 = view_216 = permute_164 = None
        view_217 = torch.ops.aten.view.default(addmm_81, [8, 196, 384]);  addmm_81 = None
        add_190 = torch.ops.aten.add.Tensor(add_187, view_217);  add_187 = view_217 = None
        clone_191 = torch.ops.aten.clone.default(add_190, memory_format = torch.contiguous_format)
        var_mean_55 = torch.ops.aten.var_mean.correction(clone_191, [2], correction = 0, keepdim = True)
        getitem_218 = var_mean_55[0]
        getitem_219 = var_mean_55[1];  var_mean_55 = None
        add_191 = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_55 = torch.ops.aten.sub.Tensor(clone_191, getitem_219);  clone_191 = getitem_219 = None
        mul_218 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, arg39_1);  mul_218 = arg39_1 = None
        add_192 = torch.ops.aten.add.Tensor(mul_219, arg40_1);  mul_219 = arg40_1 = None
        permute_165 = torch.ops.aten.permute.default(add_192, [0, 2, 1]);  add_192 = None
        permute_166 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        clone_192 = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_218 = torch.ops.aten.view.default(clone_192, [3072, 196]);  clone_192 = None
        mm_27 = torch.ops.aten.mm.default(view_218, permute_166);  view_218 = permute_166 = None
        view_219 = torch.ops.aten.view.default(mm_27, [8, 384, 384]);  mm_27 = None
        add_193 = torch.ops.aten.add.Tensor(view_219, arg42_1);  view_219 = arg42_1 = None
        split_54 = torch.ops.aten.split.Tensor(add_193, 192, -1);  add_193 = None
        getitem_220 = split_54[0]
        getitem_221 = split_54[1];  split_54 = None
        sigmoid_54 = torch.ops.aten.sigmoid.default(getitem_221)
        mul_220 = torch.ops.aten.mul.Tensor(getitem_221, sigmoid_54);  getitem_221 = sigmoid_54 = None
        mul_221 = torch.ops.aten.mul.Tensor(getitem_220, mul_220);  getitem_220 = mul_220 = None
        view_220 = torch.ops.aten.view.default(mul_221, [3072, 192]);  mul_221 = None
        permute_167 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg44_1, view_220, permute_167);  arg44_1 = view_220 = permute_167 = None
        view_221 = torch.ops.aten.view.default(addmm_82, [8, 384, 196]);  addmm_82 = None
        permute_168 = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
        add_194 = torch.ops.aten.add.Tensor(add_190, permute_168);  add_190 = permute_168 = None
        clone_195 = torch.ops.aten.clone.default(add_194, memory_format = torch.contiguous_format)
        var_mean_56 = torch.ops.aten.var_mean.correction(clone_195, [2], correction = 0, keepdim = True)
        getitem_222 = var_mean_56[0]
        getitem_223 = var_mean_56[1];  var_mean_56 = None
        add_195 = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_56 = torch.ops.aten.sub.Tensor(clone_195, getitem_223);  clone_195 = getitem_223 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, arg45_1);  mul_222 = arg45_1 = None
        add_196 = torch.ops.aten.add.Tensor(mul_223, arg46_1);  mul_223 = arg46_1 = None
        view_222 = torch.ops.aten.view.default(add_196, [1568, 384]);  add_196 = None
        permute_169 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg48_1, view_222, permute_169);  arg48_1 = view_222 = permute_169 = None
        view_223 = torch.ops.aten.view.default(addmm_83, [8, 196, 1536]);  addmm_83 = None
        split_55 = torch.ops.aten.split.Tensor(view_223, 768, -1);  view_223 = None
        getitem_224 = split_55[0]
        getitem_225 = split_55[1];  split_55 = None
        sigmoid_55 = torch.ops.aten.sigmoid.default(getitem_225)
        mul_224 = torch.ops.aten.mul.Tensor(getitem_225, sigmoid_55);  getitem_225 = sigmoid_55 = None
        mul_225 = torch.ops.aten.mul.Tensor(getitem_224, mul_224);  getitem_224 = mul_224 = None
        view_224 = torch.ops.aten.view.default(mul_225, [1568, 768]);  mul_225 = None
        permute_170 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg50_1, view_224, permute_170);  arg50_1 = view_224 = permute_170 = None
        view_225 = torch.ops.aten.view.default(addmm_84, [8, 196, 384]);  addmm_84 = None
        add_197 = torch.ops.aten.add.Tensor(add_194, view_225);  add_194 = view_225 = None
        clone_198 = torch.ops.aten.clone.default(add_197, memory_format = torch.contiguous_format)
        var_mean_57 = torch.ops.aten.var_mean.correction(clone_198, [2], correction = 0, keepdim = True)
        getitem_226 = var_mean_57[0]
        getitem_227 = var_mean_57[1];  var_mean_57 = None
        add_198 = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        sub_57 = torch.ops.aten.sub.Tensor(clone_198, getitem_227);  clone_198 = getitem_227 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, arg51_1);  mul_226 = arg51_1 = None
        add_199 = torch.ops.aten.add.Tensor(mul_227, arg52_1);  mul_227 = arg52_1 = None
        permute_171 = torch.ops.aten.permute.default(add_199, [0, 2, 1]);  add_199 = None
        permute_172 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        clone_199 = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_226 = torch.ops.aten.view.default(clone_199, [3072, 196]);  clone_199 = None
        mm_28 = torch.ops.aten.mm.default(view_226, permute_172);  view_226 = permute_172 = None
        view_227 = torch.ops.aten.view.default(mm_28, [8, 384, 384]);  mm_28 = None
        add_200 = torch.ops.aten.add.Tensor(view_227, arg54_1);  view_227 = arg54_1 = None
        split_56 = torch.ops.aten.split.Tensor(add_200, 192, -1);  add_200 = None
        getitem_228 = split_56[0]
        getitem_229 = split_56[1];  split_56 = None
        sigmoid_56 = torch.ops.aten.sigmoid.default(getitem_229)
        mul_228 = torch.ops.aten.mul.Tensor(getitem_229, sigmoid_56);  getitem_229 = sigmoid_56 = None
        mul_229 = torch.ops.aten.mul.Tensor(getitem_228, mul_228);  getitem_228 = mul_228 = None
        view_228 = torch.ops.aten.view.default(mul_229, [3072, 192]);  mul_229 = None
        permute_173 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg56_1, view_228, permute_173);  arg56_1 = view_228 = permute_173 = None
        view_229 = torch.ops.aten.view.default(addmm_85, [8, 384, 196]);  addmm_85 = None
        permute_174 = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        add_201 = torch.ops.aten.add.Tensor(add_197, permute_174);  add_197 = permute_174 = None
        clone_202 = torch.ops.aten.clone.default(add_201, memory_format = torch.contiguous_format)
        var_mean_58 = torch.ops.aten.var_mean.correction(clone_202, [2], correction = 0, keepdim = True)
        getitem_230 = var_mean_58[0]
        getitem_231 = var_mean_58[1];  var_mean_58 = None
        add_202 = torch.ops.aten.add.Tensor(getitem_230, 1e-06);  getitem_230 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_58 = torch.ops.aten.sub.Tensor(clone_202, getitem_231);  clone_202 = getitem_231 = None
        mul_230 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        mul_231 = torch.ops.aten.mul.Tensor(mul_230, arg57_1);  mul_230 = arg57_1 = None
        add_203 = torch.ops.aten.add.Tensor(mul_231, arg58_1);  mul_231 = arg58_1 = None
        view_230 = torch.ops.aten.view.default(add_203, [1568, 384]);  add_203 = None
        permute_175 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg60_1, view_230, permute_175);  arg60_1 = view_230 = permute_175 = None
        view_231 = torch.ops.aten.view.default(addmm_86, [8, 196, 1536]);  addmm_86 = None
        split_57 = torch.ops.aten.split.Tensor(view_231, 768, -1);  view_231 = None
        getitem_232 = split_57[0]
        getitem_233 = split_57[1];  split_57 = None
        sigmoid_57 = torch.ops.aten.sigmoid.default(getitem_233)
        mul_232 = torch.ops.aten.mul.Tensor(getitem_233, sigmoid_57);  getitem_233 = sigmoid_57 = None
        mul_233 = torch.ops.aten.mul.Tensor(getitem_232, mul_232);  getitem_232 = mul_232 = None
        view_232 = torch.ops.aten.view.default(mul_233, [1568, 768]);  mul_233 = None
        permute_176 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg62_1, view_232, permute_176);  arg62_1 = view_232 = permute_176 = None
        view_233 = torch.ops.aten.view.default(addmm_87, [8, 196, 384]);  addmm_87 = None
        add_204 = torch.ops.aten.add.Tensor(add_201, view_233);  add_201 = view_233 = None
        clone_205 = torch.ops.aten.clone.default(add_204, memory_format = torch.contiguous_format)
        var_mean_59 = torch.ops.aten.var_mean.correction(clone_205, [2], correction = 0, keepdim = True)
        getitem_234 = var_mean_59[0]
        getitem_235 = var_mean_59[1];  var_mean_59 = None
        add_205 = torch.ops.aten.add.Tensor(getitem_234, 1e-06);  getitem_234 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        sub_59 = torch.ops.aten.sub.Tensor(clone_205, getitem_235);  clone_205 = getitem_235 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, arg63_1);  mul_234 = arg63_1 = None
        add_206 = torch.ops.aten.add.Tensor(mul_235, arg64_1);  mul_235 = arg64_1 = None
        permute_177 = torch.ops.aten.permute.default(add_206, [0, 2, 1]);  add_206 = None
        permute_178 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        clone_206 = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_234 = torch.ops.aten.view.default(clone_206, [3072, 196]);  clone_206 = None
        mm_29 = torch.ops.aten.mm.default(view_234, permute_178);  view_234 = permute_178 = None
        view_235 = torch.ops.aten.view.default(mm_29, [8, 384, 384]);  mm_29 = None
        add_207 = torch.ops.aten.add.Tensor(view_235, arg66_1);  view_235 = arg66_1 = None
        split_58 = torch.ops.aten.split.Tensor(add_207, 192, -1);  add_207 = None
        getitem_236 = split_58[0]
        getitem_237 = split_58[1];  split_58 = None
        sigmoid_58 = torch.ops.aten.sigmoid.default(getitem_237)
        mul_236 = torch.ops.aten.mul.Tensor(getitem_237, sigmoid_58);  getitem_237 = sigmoid_58 = None
        mul_237 = torch.ops.aten.mul.Tensor(getitem_236, mul_236);  getitem_236 = mul_236 = None
        view_236 = torch.ops.aten.view.default(mul_237, [3072, 192]);  mul_237 = None
        permute_179 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg68_1, view_236, permute_179);  arg68_1 = view_236 = permute_179 = None
        view_237 = torch.ops.aten.view.default(addmm_88, [8, 384, 196]);  addmm_88 = None
        permute_180 = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
        add_208 = torch.ops.aten.add.Tensor(add_204, permute_180);  add_204 = permute_180 = None
        clone_209 = torch.ops.aten.clone.default(add_208, memory_format = torch.contiguous_format)
        var_mean_60 = torch.ops.aten.var_mean.correction(clone_209, [2], correction = 0, keepdim = True)
        getitem_238 = var_mean_60[0]
        getitem_239 = var_mean_60[1];  var_mean_60 = None
        add_209 = torch.ops.aten.add.Tensor(getitem_238, 1e-06);  getitem_238 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_60 = torch.ops.aten.sub.Tensor(clone_209, getitem_239);  clone_209 = getitem_239 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, arg69_1);  mul_238 = arg69_1 = None
        add_210 = torch.ops.aten.add.Tensor(mul_239, arg70_1);  mul_239 = arg70_1 = None
        view_238 = torch.ops.aten.view.default(add_210, [1568, 384]);  add_210 = None
        permute_181 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg72_1, view_238, permute_181);  arg72_1 = view_238 = permute_181 = None
        view_239 = torch.ops.aten.view.default(addmm_89, [8, 196, 1536]);  addmm_89 = None
        split_59 = torch.ops.aten.split.Tensor(view_239, 768, -1);  view_239 = None
        getitem_240 = split_59[0]
        getitem_241 = split_59[1];  split_59 = None
        sigmoid_59 = torch.ops.aten.sigmoid.default(getitem_241)
        mul_240 = torch.ops.aten.mul.Tensor(getitem_241, sigmoid_59);  getitem_241 = sigmoid_59 = None
        mul_241 = torch.ops.aten.mul.Tensor(getitem_240, mul_240);  getitem_240 = mul_240 = None
        view_240 = torch.ops.aten.view.default(mul_241, [1568, 768]);  mul_241 = None
        permute_182 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg74_1, view_240, permute_182);  arg74_1 = view_240 = permute_182 = None
        view_241 = torch.ops.aten.view.default(addmm_90, [8, 196, 384]);  addmm_90 = None
        add_211 = torch.ops.aten.add.Tensor(add_208, view_241);  add_208 = view_241 = None
        clone_212 = torch.ops.aten.clone.default(add_211, memory_format = torch.contiguous_format)
        var_mean_61 = torch.ops.aten.var_mean.correction(clone_212, [2], correction = 0, keepdim = True)
        getitem_242 = var_mean_61[0]
        getitem_243 = var_mean_61[1];  var_mean_61 = None
        add_212 = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_61 = torch.ops.aten.sub.Tensor(clone_212, getitem_243);  clone_212 = getitem_243 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, arg75_1);  mul_242 = arg75_1 = None
        add_213 = torch.ops.aten.add.Tensor(mul_243, arg76_1);  mul_243 = arg76_1 = None
        permute_183 = torch.ops.aten.permute.default(add_213, [0, 2, 1]);  add_213 = None
        permute_184 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        clone_213 = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_242 = torch.ops.aten.view.default(clone_213, [3072, 196]);  clone_213 = None
        mm_30 = torch.ops.aten.mm.default(view_242, permute_184);  view_242 = permute_184 = None
        view_243 = torch.ops.aten.view.default(mm_30, [8, 384, 384]);  mm_30 = None
        add_214 = torch.ops.aten.add.Tensor(view_243, arg78_1);  view_243 = arg78_1 = None
        split_60 = torch.ops.aten.split.Tensor(add_214, 192, -1);  add_214 = None
        getitem_244 = split_60[0]
        getitem_245 = split_60[1];  split_60 = None
        sigmoid_60 = torch.ops.aten.sigmoid.default(getitem_245)
        mul_244 = torch.ops.aten.mul.Tensor(getitem_245, sigmoid_60);  getitem_245 = sigmoid_60 = None
        mul_245 = torch.ops.aten.mul.Tensor(getitem_244, mul_244);  getitem_244 = mul_244 = None
        view_244 = torch.ops.aten.view.default(mul_245, [3072, 192]);  mul_245 = None
        permute_185 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg80_1, view_244, permute_185);  arg80_1 = view_244 = permute_185 = None
        view_245 = torch.ops.aten.view.default(addmm_91, [8, 384, 196]);  addmm_91 = None
        permute_186 = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
        add_215 = torch.ops.aten.add.Tensor(add_211, permute_186);  add_211 = permute_186 = None
        clone_216 = torch.ops.aten.clone.default(add_215, memory_format = torch.contiguous_format)
        var_mean_62 = torch.ops.aten.var_mean.correction(clone_216, [2], correction = 0, keepdim = True)
        getitem_246 = var_mean_62[0]
        getitem_247 = var_mean_62[1];  var_mean_62 = None
        add_216 = torch.ops.aten.add.Tensor(getitem_246, 1e-06);  getitem_246 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_62 = torch.ops.aten.sub.Tensor(clone_216, getitem_247);  clone_216 = getitem_247 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_246, arg81_1);  mul_246 = arg81_1 = None
        add_217 = torch.ops.aten.add.Tensor(mul_247, arg82_1);  mul_247 = arg82_1 = None
        view_246 = torch.ops.aten.view.default(add_217, [1568, 384]);  add_217 = None
        permute_187 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg84_1, view_246, permute_187);  arg84_1 = view_246 = permute_187 = None
        view_247 = torch.ops.aten.view.default(addmm_92, [8, 196, 1536]);  addmm_92 = None
        split_61 = torch.ops.aten.split.Tensor(view_247, 768, -1);  view_247 = None
        getitem_248 = split_61[0]
        getitem_249 = split_61[1];  split_61 = None
        sigmoid_61 = torch.ops.aten.sigmoid.default(getitem_249)
        mul_248 = torch.ops.aten.mul.Tensor(getitem_249, sigmoid_61);  getitem_249 = sigmoid_61 = None
        mul_249 = torch.ops.aten.mul.Tensor(getitem_248, mul_248);  getitem_248 = mul_248 = None
        view_248 = torch.ops.aten.view.default(mul_249, [1568, 768]);  mul_249 = None
        permute_188 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg86_1, view_248, permute_188);  arg86_1 = view_248 = permute_188 = None
        view_249 = torch.ops.aten.view.default(addmm_93, [8, 196, 384]);  addmm_93 = None
        add_218 = torch.ops.aten.add.Tensor(add_215, view_249);  add_215 = view_249 = None
        clone_219 = torch.ops.aten.clone.default(add_218, memory_format = torch.contiguous_format)
        var_mean_63 = torch.ops.aten.var_mean.correction(clone_219, [2], correction = 0, keepdim = True)
        getitem_250 = var_mean_63[0]
        getitem_251 = var_mean_63[1];  var_mean_63 = None
        add_219 = torch.ops.aten.add.Tensor(getitem_250, 1e-06);  getitem_250 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        sub_63 = torch.ops.aten.sub.Tensor(clone_219, getitem_251);  clone_219 = getitem_251 = None
        mul_250 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, arg87_1);  mul_250 = arg87_1 = None
        add_220 = torch.ops.aten.add.Tensor(mul_251, arg88_1);  mul_251 = arg88_1 = None
        permute_189 = torch.ops.aten.permute.default(add_220, [0, 2, 1]);  add_220 = None
        permute_190 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        clone_220 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_250 = torch.ops.aten.view.default(clone_220, [3072, 196]);  clone_220 = None
        mm_31 = torch.ops.aten.mm.default(view_250, permute_190);  view_250 = permute_190 = None
        view_251 = torch.ops.aten.view.default(mm_31, [8, 384, 384]);  mm_31 = None
        add_221 = torch.ops.aten.add.Tensor(view_251, arg90_1);  view_251 = arg90_1 = None
        split_62 = torch.ops.aten.split.Tensor(add_221, 192, -1);  add_221 = None
        getitem_252 = split_62[0]
        getitem_253 = split_62[1];  split_62 = None
        sigmoid_62 = torch.ops.aten.sigmoid.default(getitem_253)
        mul_252 = torch.ops.aten.mul.Tensor(getitem_253, sigmoid_62);  getitem_253 = sigmoid_62 = None
        mul_253 = torch.ops.aten.mul.Tensor(getitem_252, mul_252);  getitem_252 = mul_252 = None
        view_252 = torch.ops.aten.view.default(mul_253, [3072, 192]);  mul_253 = None
        permute_191 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg92_1, view_252, permute_191);  arg92_1 = view_252 = permute_191 = None
        view_253 = torch.ops.aten.view.default(addmm_94, [8, 384, 196]);  addmm_94 = None
        permute_192 = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        add_222 = torch.ops.aten.add.Tensor(add_218, permute_192);  add_218 = permute_192 = None
        clone_223 = torch.ops.aten.clone.default(add_222, memory_format = torch.contiguous_format)
        var_mean_64 = torch.ops.aten.var_mean.correction(clone_223, [2], correction = 0, keepdim = True)
        getitem_254 = var_mean_64[0]
        getitem_255 = var_mean_64[1];  var_mean_64 = None
        add_223 = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_64 = torch.ops.aten.sub.Tensor(clone_223, getitem_255);  clone_223 = getitem_255 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, arg93_1);  mul_254 = arg93_1 = None
        add_224 = torch.ops.aten.add.Tensor(mul_255, arg94_1);  mul_255 = arg94_1 = None
        view_254 = torch.ops.aten.view.default(add_224, [1568, 384]);  add_224 = None
        permute_193 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg96_1, view_254, permute_193);  arg96_1 = view_254 = permute_193 = None
        view_255 = torch.ops.aten.view.default(addmm_95, [8, 196, 1536]);  addmm_95 = None
        split_63 = torch.ops.aten.split.Tensor(view_255, 768, -1);  view_255 = None
        getitem_256 = split_63[0]
        getitem_257 = split_63[1];  split_63 = None
        sigmoid_63 = torch.ops.aten.sigmoid.default(getitem_257)
        mul_256 = torch.ops.aten.mul.Tensor(getitem_257, sigmoid_63);  getitem_257 = sigmoid_63 = None
        mul_257 = torch.ops.aten.mul.Tensor(getitem_256, mul_256);  getitem_256 = mul_256 = None
        view_256 = torch.ops.aten.view.default(mul_257, [1568, 768]);  mul_257 = None
        permute_194 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg98_1, view_256, permute_194);  arg98_1 = view_256 = permute_194 = None
        view_257 = torch.ops.aten.view.default(addmm_96, [8, 196, 384]);  addmm_96 = None
        add_225 = torch.ops.aten.add.Tensor(add_222, view_257);  add_222 = view_257 = None
        clone_226 = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_226, [2], correction = 0, keepdim = True)
        getitem_258 = var_mean_65[0]
        getitem_259 = var_mean_65[1];  var_mean_65 = None
        add_226 = torch.ops.aten.add.Tensor(getitem_258, 1e-06);  getitem_258 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_65 = torch.ops.aten.sub.Tensor(clone_226, getitem_259);  clone_226 = getitem_259 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, arg99_1);  mul_258 = arg99_1 = None
        add_227 = torch.ops.aten.add.Tensor(mul_259, arg100_1);  mul_259 = arg100_1 = None
        permute_195 = torch.ops.aten.permute.default(add_227, [0, 2, 1]);  add_227 = None
        permute_196 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        clone_227 = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_258 = torch.ops.aten.view.default(clone_227, [3072, 196]);  clone_227 = None
        mm_32 = torch.ops.aten.mm.default(view_258, permute_196);  view_258 = permute_196 = None
        view_259 = torch.ops.aten.view.default(mm_32, [8, 384, 384]);  mm_32 = None
        add_228 = torch.ops.aten.add.Tensor(view_259, arg102_1);  view_259 = arg102_1 = None
        split_64 = torch.ops.aten.split.Tensor(add_228, 192, -1);  add_228 = None
        getitem_260 = split_64[0]
        getitem_261 = split_64[1];  split_64 = None
        sigmoid_64 = torch.ops.aten.sigmoid.default(getitem_261)
        mul_260 = torch.ops.aten.mul.Tensor(getitem_261, sigmoid_64);  getitem_261 = sigmoid_64 = None
        mul_261 = torch.ops.aten.mul.Tensor(getitem_260, mul_260);  getitem_260 = mul_260 = None
        view_260 = torch.ops.aten.view.default(mul_261, [3072, 192]);  mul_261 = None
        permute_197 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg104_1, view_260, permute_197);  arg104_1 = view_260 = permute_197 = None
        view_261 = torch.ops.aten.view.default(addmm_97, [8, 384, 196]);  addmm_97 = None
        permute_198 = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None
        add_229 = torch.ops.aten.add.Tensor(add_225, permute_198);  add_225 = permute_198 = None
        clone_230 = torch.ops.aten.clone.default(add_229, memory_format = torch.contiguous_format)
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_230, [2], correction = 0, keepdim = True)
        getitem_262 = var_mean_66[0]
        getitem_263 = var_mean_66[1];  var_mean_66 = None
        add_230 = torch.ops.aten.add.Tensor(getitem_262, 1e-06);  getitem_262 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
        sub_66 = torch.ops.aten.sub.Tensor(clone_230, getitem_263);  clone_230 = getitem_263 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, arg105_1);  mul_262 = arg105_1 = None
        add_231 = torch.ops.aten.add.Tensor(mul_263, arg106_1);  mul_263 = arg106_1 = None
        view_262 = torch.ops.aten.view.default(add_231, [1568, 384]);  add_231 = None
        permute_199 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg108_1, view_262, permute_199);  arg108_1 = view_262 = permute_199 = None
        view_263 = torch.ops.aten.view.default(addmm_98, [8, 196, 1536]);  addmm_98 = None
        split_65 = torch.ops.aten.split.Tensor(view_263, 768, -1);  view_263 = None
        getitem_264 = split_65[0]
        getitem_265 = split_65[1];  split_65 = None
        sigmoid_65 = torch.ops.aten.sigmoid.default(getitem_265)
        mul_264 = torch.ops.aten.mul.Tensor(getitem_265, sigmoid_65);  getitem_265 = sigmoid_65 = None
        mul_265 = torch.ops.aten.mul.Tensor(getitem_264, mul_264);  getitem_264 = mul_264 = None
        view_264 = torch.ops.aten.view.default(mul_265, [1568, 768]);  mul_265 = None
        permute_200 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg110_1, view_264, permute_200);  arg110_1 = view_264 = permute_200 = None
        view_265 = torch.ops.aten.view.default(addmm_99, [8, 196, 384]);  addmm_99 = None
        add_232 = torch.ops.aten.add.Tensor(add_229, view_265);  add_229 = view_265 = None
        clone_233 = torch.ops.aten.clone.default(add_232, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_233, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_67[0]
        getitem_267 = var_mean_67[1];  var_mean_67 = None
        add_233 = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        sub_67 = torch.ops.aten.sub.Tensor(clone_233, getitem_267);  clone_233 = getitem_267 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, arg111_1);  mul_266 = arg111_1 = None
        add_234 = torch.ops.aten.add.Tensor(mul_267, arg112_1);  mul_267 = arg112_1 = None
        permute_201 = torch.ops.aten.permute.default(add_234, [0, 2, 1]);  add_234 = None
        permute_202 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        clone_234 = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_266 = torch.ops.aten.view.default(clone_234, [3072, 196]);  clone_234 = None
        mm_33 = torch.ops.aten.mm.default(view_266, permute_202);  view_266 = permute_202 = None
        view_267 = torch.ops.aten.view.default(mm_33, [8, 384, 384]);  mm_33 = None
        add_235 = torch.ops.aten.add.Tensor(view_267, arg114_1);  view_267 = arg114_1 = None
        split_66 = torch.ops.aten.split.Tensor(add_235, 192, -1);  add_235 = None
        getitem_268 = split_66[0]
        getitem_269 = split_66[1];  split_66 = None
        sigmoid_66 = torch.ops.aten.sigmoid.default(getitem_269)
        mul_268 = torch.ops.aten.mul.Tensor(getitem_269, sigmoid_66);  getitem_269 = sigmoid_66 = None
        mul_269 = torch.ops.aten.mul.Tensor(getitem_268, mul_268);  getitem_268 = mul_268 = None
        view_268 = torch.ops.aten.view.default(mul_269, [3072, 192]);  mul_269 = None
        permute_203 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg116_1, view_268, permute_203);  arg116_1 = view_268 = permute_203 = None
        view_269 = torch.ops.aten.view.default(addmm_100, [8, 384, 196]);  addmm_100 = None
        permute_204 = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
        add_236 = torch.ops.aten.add.Tensor(add_232, permute_204);  add_232 = permute_204 = None
        clone_237 = torch.ops.aten.clone.default(add_236, memory_format = torch.contiguous_format)
        var_mean_68 = torch.ops.aten.var_mean.correction(clone_237, [2], correction = 0, keepdim = True)
        getitem_270 = var_mean_68[0]
        getitem_271 = var_mean_68[1];  var_mean_68 = None
        add_237 = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        sub_68 = torch.ops.aten.sub.Tensor(clone_237, getitem_271);  clone_237 = getitem_271 = None
        mul_270 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        mul_271 = torch.ops.aten.mul.Tensor(mul_270, arg117_1);  mul_270 = arg117_1 = None
        add_238 = torch.ops.aten.add.Tensor(mul_271, arg118_1);  mul_271 = arg118_1 = None
        view_270 = torch.ops.aten.view.default(add_238, [1568, 384]);  add_238 = None
        permute_205 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg120_1, view_270, permute_205);  arg120_1 = view_270 = permute_205 = None
        view_271 = torch.ops.aten.view.default(addmm_101, [8, 196, 1536]);  addmm_101 = None
        split_67 = torch.ops.aten.split.Tensor(view_271, 768, -1);  view_271 = None
        getitem_272 = split_67[0]
        getitem_273 = split_67[1];  split_67 = None
        sigmoid_67 = torch.ops.aten.sigmoid.default(getitem_273)
        mul_272 = torch.ops.aten.mul.Tensor(getitem_273, sigmoid_67);  getitem_273 = sigmoid_67 = None
        mul_273 = torch.ops.aten.mul.Tensor(getitem_272, mul_272);  getitem_272 = mul_272 = None
        view_272 = torch.ops.aten.view.default(mul_273, [1568, 768]);  mul_273 = None
        permute_206 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg122_1, view_272, permute_206);  arg122_1 = view_272 = permute_206 = None
        view_273 = torch.ops.aten.view.default(addmm_102, [8, 196, 384]);  addmm_102 = None
        add_239 = torch.ops.aten.add.Tensor(add_236, view_273);  add_236 = view_273 = None
        clone_240 = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
        var_mean_69 = torch.ops.aten.var_mean.correction(clone_240, [2], correction = 0, keepdim = True)
        getitem_274 = var_mean_69[0]
        getitem_275 = var_mean_69[1];  var_mean_69 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_274, 1e-06);  getitem_274 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_69 = torch.ops.aten.sub.Tensor(clone_240, getitem_275);  clone_240 = getitem_275 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, arg123_1);  mul_274 = arg123_1 = None
        add_241 = torch.ops.aten.add.Tensor(mul_275, arg124_1);  mul_275 = arg124_1 = None
        permute_207 = torch.ops.aten.permute.default(add_241, [0, 2, 1]);  add_241 = None
        permute_208 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        clone_241 = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        view_274 = torch.ops.aten.view.default(clone_241, [3072, 196]);  clone_241 = None
        mm_34 = torch.ops.aten.mm.default(view_274, permute_208);  view_274 = permute_208 = None
        view_275 = torch.ops.aten.view.default(mm_34, [8, 384, 384]);  mm_34 = None
        add_242 = torch.ops.aten.add.Tensor(view_275, arg126_1);  view_275 = arg126_1 = None
        split_68 = torch.ops.aten.split.Tensor(add_242, 192, -1);  add_242 = None
        getitem_276 = split_68[0]
        getitem_277 = split_68[1];  split_68 = None
        sigmoid_68 = torch.ops.aten.sigmoid.default(getitem_277)
        mul_276 = torch.ops.aten.mul.Tensor(getitem_277, sigmoid_68);  getitem_277 = sigmoid_68 = None
        mul_277 = torch.ops.aten.mul.Tensor(getitem_276, mul_276);  getitem_276 = mul_276 = None
        view_276 = torch.ops.aten.view.default(mul_277, [3072, 192]);  mul_277 = None
        permute_209 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg128_1, view_276, permute_209);  arg128_1 = view_276 = permute_209 = None
        view_277 = torch.ops.aten.view.default(addmm_103, [8, 384, 196]);  addmm_103 = None
        permute_210 = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
        add_243 = torch.ops.aten.add.Tensor(add_239, permute_210);  add_239 = permute_210 = None
        clone_244 = torch.ops.aten.clone.default(add_243, memory_format = torch.contiguous_format)
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_244, [2], correction = 0, keepdim = True)
        getitem_278 = var_mean_70[0]
        getitem_279 = var_mean_70[1];  var_mean_70 = None
        add_244 = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_70 = torch.ops.aten.sub.Tensor(clone_244, getitem_279);  clone_244 = getitem_279 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, arg129_1);  mul_278 = arg129_1 = None
        add_245 = torch.ops.aten.add.Tensor(mul_279, arg130_1);  mul_279 = arg130_1 = None
        view_278 = torch.ops.aten.view.default(add_245, [1568, 384]);  add_245 = None
        permute_211 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg132_1, view_278, permute_211);  arg132_1 = view_278 = permute_211 = None
        view_279 = torch.ops.aten.view.default(addmm_104, [8, 196, 1536]);  addmm_104 = None
        split_69 = torch.ops.aten.split.Tensor(view_279, 768, -1);  view_279 = None
        getitem_280 = split_69[0]
        getitem_281 = split_69[1];  split_69 = None
        sigmoid_69 = torch.ops.aten.sigmoid.default(getitem_281)
        mul_280 = torch.ops.aten.mul.Tensor(getitem_281, sigmoid_69);  getitem_281 = sigmoid_69 = None
        mul_281 = torch.ops.aten.mul.Tensor(getitem_280, mul_280);  getitem_280 = mul_280 = None
        view_280 = torch.ops.aten.view.default(mul_281, [1568, 768]);  mul_281 = None
        permute_212 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg134_1, view_280, permute_212);  arg134_1 = view_280 = permute_212 = None
        view_281 = torch.ops.aten.view.default(addmm_105, [8, 196, 384]);  addmm_105 = None
        add_246 = torch.ops.aten.add.Tensor(add_243, view_281);  add_243 = view_281 = None
        clone_247 = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_247, [2], correction = 0, keepdim = True)
        getitem_282 = var_mean_71[0]
        getitem_283 = var_mean_71[1];  var_mean_71 = None
        add_247 = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_71 = torch.ops.aten.sub.Tensor(clone_247, getitem_283);  clone_247 = getitem_283 = None
        mul_282 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        mul_283 = torch.ops.aten.mul.Tensor(mul_282, arg135_1);  mul_282 = arg135_1 = None
        add_248 = torch.ops.aten.add.Tensor(mul_283, arg136_1);  mul_283 = arg136_1 = None
        permute_213 = torch.ops.aten.permute.default(add_248, [0, 2, 1]);  add_248 = None
        permute_214 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        clone_248 = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_282 = torch.ops.aten.view.default(clone_248, [3072, 196]);  clone_248 = None
        mm_35 = torch.ops.aten.mm.default(view_282, permute_214);  view_282 = permute_214 = None
        view_283 = torch.ops.aten.view.default(mm_35, [8, 384, 384]);  mm_35 = None
        add_249 = torch.ops.aten.add.Tensor(view_283, arg138_1);  view_283 = arg138_1 = None
        split_70 = torch.ops.aten.split.Tensor(add_249, 192, -1);  add_249 = None
        getitem_284 = split_70[0]
        getitem_285 = split_70[1];  split_70 = None
        sigmoid_70 = torch.ops.aten.sigmoid.default(getitem_285)
        mul_284 = torch.ops.aten.mul.Tensor(getitem_285, sigmoid_70);  getitem_285 = sigmoid_70 = None
        mul_285 = torch.ops.aten.mul.Tensor(getitem_284, mul_284);  getitem_284 = mul_284 = None
        view_284 = torch.ops.aten.view.default(mul_285, [3072, 192]);  mul_285 = None
        permute_215 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg140_1, view_284, permute_215);  arg140_1 = view_284 = permute_215 = None
        view_285 = torch.ops.aten.view.default(addmm_106, [8, 384, 196]);  addmm_106 = None
        permute_216 = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
        add_250 = torch.ops.aten.add.Tensor(add_246, permute_216);  add_246 = permute_216 = None
        clone_251 = torch.ops.aten.clone.default(add_250, memory_format = torch.contiguous_format)
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_251, [2], correction = 0, keepdim = True)
        getitem_286 = var_mean_72[0]
        getitem_287 = var_mean_72[1];  var_mean_72 = None
        add_251 = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_72 = torch.ops.aten.sub.Tensor(clone_251, getitem_287);  clone_251 = getitem_287 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, arg141_1);  mul_286 = arg141_1 = None
        add_252 = torch.ops.aten.add.Tensor(mul_287, arg142_1);  mul_287 = arg142_1 = None
        view_286 = torch.ops.aten.view.default(add_252, [1568, 384]);  add_252 = None
        permute_217 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg144_1, view_286, permute_217);  arg144_1 = view_286 = permute_217 = None
        view_287 = torch.ops.aten.view.default(addmm_107, [8, 196, 1536]);  addmm_107 = None
        split_71 = torch.ops.aten.split.Tensor(view_287, 768, -1);  view_287 = None
        getitem_288 = split_71[0]
        getitem_289 = split_71[1];  split_71 = None
        sigmoid_71 = torch.ops.aten.sigmoid.default(getitem_289)
        mul_288 = torch.ops.aten.mul.Tensor(getitem_289, sigmoid_71);  getitem_289 = sigmoid_71 = None
        mul_289 = torch.ops.aten.mul.Tensor(getitem_288, mul_288);  getitem_288 = mul_288 = None
        view_288 = torch.ops.aten.view.default(mul_289, [1568, 768]);  mul_289 = None
        permute_218 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg146_1, view_288, permute_218);  arg146_1 = view_288 = permute_218 = None
        view_289 = torch.ops.aten.view.default(addmm_108, [8, 196, 384]);  addmm_108 = None
        add_253 = torch.ops.aten.add.Tensor(add_250, view_289);  add_250 = view_289 = None
        clone_254 = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format)
        var_mean_73 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
        getitem_290 = var_mean_73[0]
        getitem_291 = var_mean_73[1];  var_mean_73 = None
        add_254 = torch.ops.aten.add.Tensor(getitem_290, 1e-06);  getitem_290 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_73 = torch.ops.aten.sub.Tensor(clone_254, getitem_291);  clone_254 = getitem_291 = None
        mul_290 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, arg147_1);  mul_290 = arg147_1 = None
        add_255 = torch.ops.aten.add.Tensor(mul_291, arg148_1);  mul_291 = arg148_1 = None
        permute_219 = torch.ops.aten.permute.default(add_255, [0, 2, 1]);  add_255 = None
        permute_220 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        clone_255 = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_290 = torch.ops.aten.view.default(clone_255, [3072, 196]);  clone_255 = None
        mm_36 = torch.ops.aten.mm.default(view_290, permute_220);  view_290 = permute_220 = None
        view_291 = torch.ops.aten.view.default(mm_36, [8, 384, 384]);  mm_36 = None
        add_256 = torch.ops.aten.add.Tensor(view_291, arg150_1);  view_291 = arg150_1 = None
        split_72 = torch.ops.aten.split.Tensor(add_256, 192, -1);  add_256 = None
        getitem_292 = split_72[0]
        getitem_293 = split_72[1];  split_72 = None
        sigmoid_72 = torch.ops.aten.sigmoid.default(getitem_293)
        mul_292 = torch.ops.aten.mul.Tensor(getitem_293, sigmoid_72);  getitem_293 = sigmoid_72 = None
        mul_293 = torch.ops.aten.mul.Tensor(getitem_292, mul_292);  getitem_292 = mul_292 = None
        view_292 = torch.ops.aten.view.default(mul_293, [3072, 192]);  mul_293 = None
        permute_221 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg152_1, view_292, permute_221);  arg152_1 = view_292 = permute_221 = None
        view_293 = torch.ops.aten.view.default(addmm_109, [8, 384, 196]);  addmm_109 = None
        permute_222 = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
        add_257 = torch.ops.aten.add.Tensor(add_253, permute_222);  add_253 = permute_222 = None
        clone_258 = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format)
        var_mean_74 = torch.ops.aten.var_mean.correction(clone_258, [2], correction = 0, keepdim = True)
        getitem_294 = var_mean_74[0]
        getitem_295 = var_mean_74[1];  var_mean_74 = None
        add_258 = torch.ops.aten.add.Tensor(getitem_294, 1e-06);  getitem_294 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        sub_74 = torch.ops.aten.sub.Tensor(clone_258, getitem_295);  clone_258 = getitem_295 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_294, arg153_1);  mul_294 = arg153_1 = None
        add_259 = torch.ops.aten.add.Tensor(mul_295, arg154_1);  mul_295 = arg154_1 = None
        view_294 = torch.ops.aten.view.default(add_259, [1568, 384]);  add_259 = None
        permute_223 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg156_1, view_294, permute_223);  arg156_1 = view_294 = permute_223 = None
        view_295 = torch.ops.aten.view.default(addmm_110, [8, 196, 1536]);  addmm_110 = None
        split_73 = torch.ops.aten.split.Tensor(view_295, 768, -1);  view_295 = None
        getitem_296 = split_73[0]
        getitem_297 = split_73[1];  split_73 = None
        sigmoid_73 = torch.ops.aten.sigmoid.default(getitem_297)
        mul_296 = torch.ops.aten.mul.Tensor(getitem_297, sigmoid_73);  getitem_297 = sigmoid_73 = None
        mul_297 = torch.ops.aten.mul.Tensor(getitem_296, mul_296);  getitem_296 = mul_296 = None
        view_296 = torch.ops.aten.view.default(mul_297, [1568, 768]);  mul_297 = None
        permute_224 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg158_1, view_296, permute_224);  arg158_1 = view_296 = permute_224 = None
        view_297 = torch.ops.aten.view.default(addmm_111, [8, 196, 384]);  addmm_111 = None
        add_260 = torch.ops.aten.add.Tensor(add_257, view_297);  add_257 = view_297 = None
        clone_261 = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_261, [2], correction = 0, keepdim = True)
        getitem_298 = var_mean_75[0]
        getitem_299 = var_mean_75[1];  var_mean_75 = None
        add_261 = torch.ops.aten.add.Tensor(getitem_298, 1e-06);  getitem_298 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_75 = torch.ops.aten.sub.Tensor(clone_261, getitem_299);  clone_261 = getitem_299 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, arg159_1);  mul_298 = arg159_1 = None
        add_262 = torch.ops.aten.add.Tensor(mul_299, arg160_1);  mul_299 = arg160_1 = None
        permute_225 = torch.ops.aten.permute.default(add_262, [0, 2, 1]);  add_262 = None
        permute_226 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        clone_262 = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_298 = torch.ops.aten.view.default(clone_262, [3072, 196]);  clone_262 = None
        mm_37 = torch.ops.aten.mm.default(view_298, permute_226);  view_298 = permute_226 = None
        view_299 = torch.ops.aten.view.default(mm_37, [8, 384, 384]);  mm_37 = None
        add_263 = torch.ops.aten.add.Tensor(view_299, arg162_1);  view_299 = arg162_1 = None
        split_74 = torch.ops.aten.split.Tensor(add_263, 192, -1);  add_263 = None
        getitem_300 = split_74[0]
        getitem_301 = split_74[1];  split_74 = None
        sigmoid_74 = torch.ops.aten.sigmoid.default(getitem_301)
        mul_300 = torch.ops.aten.mul.Tensor(getitem_301, sigmoid_74);  getitem_301 = sigmoid_74 = None
        mul_301 = torch.ops.aten.mul.Tensor(getitem_300, mul_300);  getitem_300 = mul_300 = None
        view_300 = torch.ops.aten.view.default(mul_301, [3072, 192]);  mul_301 = None
        permute_227 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg164_1, view_300, permute_227);  arg164_1 = view_300 = permute_227 = None
        view_301 = torch.ops.aten.view.default(addmm_112, [8, 384, 196]);  addmm_112 = None
        permute_228 = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
        add_264 = torch.ops.aten.add.Tensor(add_260, permute_228);  add_260 = permute_228 = None
        clone_265 = torch.ops.aten.clone.default(add_264, memory_format = torch.contiguous_format)
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_265, [2], correction = 0, keepdim = True)
        getitem_302 = var_mean_76[0]
        getitem_303 = var_mean_76[1];  var_mean_76 = None
        add_265 = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
        sub_76 = torch.ops.aten.sub.Tensor(clone_265, getitem_303);  clone_265 = getitem_303 = None
        mul_302 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_302, arg165_1);  mul_302 = arg165_1 = None
        add_266 = torch.ops.aten.add.Tensor(mul_303, arg166_1);  mul_303 = arg166_1 = None
        view_302 = torch.ops.aten.view.default(add_266, [1568, 384]);  add_266 = None
        permute_229 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg168_1, view_302, permute_229);  arg168_1 = view_302 = permute_229 = None
        view_303 = torch.ops.aten.view.default(addmm_113, [8, 196, 1536]);  addmm_113 = None
        split_75 = torch.ops.aten.split.Tensor(view_303, 768, -1);  view_303 = None
        getitem_304 = split_75[0]
        getitem_305 = split_75[1];  split_75 = None
        sigmoid_75 = torch.ops.aten.sigmoid.default(getitem_305)
        mul_304 = torch.ops.aten.mul.Tensor(getitem_305, sigmoid_75);  getitem_305 = sigmoid_75 = None
        mul_305 = torch.ops.aten.mul.Tensor(getitem_304, mul_304);  getitem_304 = mul_304 = None
        view_304 = torch.ops.aten.view.default(mul_305, [1568, 768]);  mul_305 = None
        permute_230 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg170_1, view_304, permute_230);  arg170_1 = view_304 = permute_230 = None
        view_305 = torch.ops.aten.view.default(addmm_114, [8, 196, 384]);  addmm_114 = None
        add_267 = torch.ops.aten.add.Tensor(add_264, view_305);  add_264 = view_305 = None
        clone_268 = torch.ops.aten.clone.default(add_267, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_268, [2], correction = 0, keepdim = True)
        getitem_306 = var_mean_77[0]
        getitem_307 = var_mean_77[1];  var_mean_77 = None
        add_268 = torch.ops.aten.add.Tensor(getitem_306, 1e-06);  getitem_306 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_77 = torch.ops.aten.sub.Tensor(clone_268, getitem_307);  clone_268 = getitem_307 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_306, arg171_1);  mul_306 = arg171_1 = None
        add_269 = torch.ops.aten.add.Tensor(mul_307, arg172_1);  mul_307 = arg172_1 = None
        permute_231 = torch.ops.aten.permute.default(add_269, [0, 2, 1]);  add_269 = None
        permute_232 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        clone_269 = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_306 = torch.ops.aten.view.default(clone_269, [3072, 196]);  clone_269 = None
        mm_38 = torch.ops.aten.mm.default(view_306, permute_232);  view_306 = permute_232 = None
        view_307 = torch.ops.aten.view.default(mm_38, [8, 384, 384]);  mm_38 = None
        add_270 = torch.ops.aten.add.Tensor(view_307, arg174_1);  view_307 = arg174_1 = None
        split_76 = torch.ops.aten.split.Tensor(add_270, 192, -1);  add_270 = None
        getitem_308 = split_76[0]
        getitem_309 = split_76[1];  split_76 = None
        sigmoid_76 = torch.ops.aten.sigmoid.default(getitem_309)
        mul_308 = torch.ops.aten.mul.Tensor(getitem_309, sigmoid_76);  getitem_309 = sigmoid_76 = None
        mul_309 = torch.ops.aten.mul.Tensor(getitem_308, mul_308);  getitem_308 = mul_308 = None
        view_308 = torch.ops.aten.view.default(mul_309, [3072, 192]);  mul_309 = None
        permute_233 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg176_1, view_308, permute_233);  arg176_1 = view_308 = permute_233 = None
        view_309 = torch.ops.aten.view.default(addmm_115, [8, 384, 196]);  addmm_115 = None
        permute_234 = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        add_271 = torch.ops.aten.add.Tensor(add_267, permute_234);  add_267 = permute_234 = None
        clone_272 = torch.ops.aten.clone.default(add_271, memory_format = torch.contiguous_format)
        var_mean_78 = torch.ops.aten.var_mean.correction(clone_272, [2], correction = 0, keepdim = True)
        getitem_310 = var_mean_78[0]
        getitem_311 = var_mean_78[1];  var_mean_78 = None
        add_272 = torch.ops.aten.add.Tensor(getitem_310, 1e-06);  getitem_310 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        sub_78 = torch.ops.aten.sub.Tensor(clone_272, getitem_311);  clone_272 = getitem_311 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, arg177_1);  mul_310 = arg177_1 = None
        add_273 = torch.ops.aten.add.Tensor(mul_311, arg178_1);  mul_311 = arg178_1 = None
        view_310 = torch.ops.aten.view.default(add_273, [1568, 384]);  add_273 = None
        permute_235 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg180_1, view_310, permute_235);  arg180_1 = view_310 = permute_235 = None
        view_311 = torch.ops.aten.view.default(addmm_116, [8, 196, 1536]);  addmm_116 = None
        split_77 = torch.ops.aten.split.Tensor(view_311, 768, -1);  view_311 = None
        getitem_312 = split_77[0]
        getitem_313 = split_77[1];  split_77 = None
        sigmoid_77 = torch.ops.aten.sigmoid.default(getitem_313)
        mul_312 = torch.ops.aten.mul.Tensor(getitem_313, sigmoid_77);  getitem_313 = sigmoid_77 = None
        mul_313 = torch.ops.aten.mul.Tensor(getitem_312, mul_312);  getitem_312 = mul_312 = None
        view_312 = torch.ops.aten.view.default(mul_313, [1568, 768]);  mul_313 = None
        permute_236 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg182_1, view_312, permute_236);  arg182_1 = view_312 = permute_236 = None
        view_313 = torch.ops.aten.view.default(addmm_117, [8, 196, 384]);  addmm_117 = None
        add_274 = torch.ops.aten.add.Tensor(add_271, view_313);  add_271 = view_313 = None
        clone_275 = torch.ops.aten.clone.default(add_274, memory_format = torch.contiguous_format)
        var_mean_79 = torch.ops.aten.var_mean.correction(clone_275, [2], correction = 0, keepdim = True)
        getitem_314 = var_mean_79[0]
        getitem_315 = var_mean_79[1];  var_mean_79 = None
        add_275 = torch.ops.aten.add.Tensor(getitem_314, 1e-06);  getitem_314 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        sub_79 = torch.ops.aten.sub.Tensor(clone_275, getitem_315);  clone_275 = getitem_315 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, arg183_1);  mul_314 = arg183_1 = None
        add_276 = torch.ops.aten.add.Tensor(mul_315, arg184_1);  mul_315 = arg184_1 = None
        permute_237 = torch.ops.aten.permute.default(add_276, [0, 2, 1]);  add_276 = None
        permute_238 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        clone_276 = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_314 = torch.ops.aten.view.default(clone_276, [3072, 196]);  clone_276 = None
        mm_39 = torch.ops.aten.mm.default(view_314, permute_238);  view_314 = permute_238 = None
        view_315 = torch.ops.aten.view.default(mm_39, [8, 384, 384]);  mm_39 = None
        add_277 = torch.ops.aten.add.Tensor(view_315, arg186_1);  view_315 = arg186_1 = None
        split_78 = torch.ops.aten.split.Tensor(add_277, 192, -1);  add_277 = None
        getitem_316 = split_78[0]
        getitem_317 = split_78[1];  split_78 = None
        sigmoid_78 = torch.ops.aten.sigmoid.default(getitem_317)
        mul_316 = torch.ops.aten.mul.Tensor(getitem_317, sigmoid_78);  getitem_317 = sigmoid_78 = None
        mul_317 = torch.ops.aten.mul.Tensor(getitem_316, mul_316);  getitem_316 = mul_316 = None
        view_316 = torch.ops.aten.view.default(mul_317, [3072, 192]);  mul_317 = None
        permute_239 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg188_1, view_316, permute_239);  arg188_1 = view_316 = permute_239 = None
        view_317 = torch.ops.aten.view.default(addmm_118, [8, 384, 196]);  addmm_118 = None
        permute_240 = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        add_278 = torch.ops.aten.add.Tensor(add_274, permute_240);  add_274 = permute_240 = None
        clone_279 = torch.ops.aten.clone.default(add_278, memory_format = torch.contiguous_format)
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_279, [2], correction = 0, keepdim = True)
        getitem_318 = var_mean_80[0]
        getitem_319 = var_mean_80[1];  var_mean_80 = None
        add_279 = torch.ops.aten.add.Tensor(getitem_318, 1e-06);  getitem_318 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
        sub_80 = torch.ops.aten.sub.Tensor(clone_279, getitem_319);  clone_279 = getitem_319 = None
        mul_318 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, arg189_1);  mul_318 = arg189_1 = None
        add_280 = torch.ops.aten.add.Tensor(mul_319, arg190_1);  mul_319 = arg190_1 = None
        view_318 = torch.ops.aten.view.default(add_280, [1568, 384]);  add_280 = None
        permute_241 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg192_1, view_318, permute_241);  arg192_1 = view_318 = permute_241 = None
        view_319 = torch.ops.aten.view.default(addmm_119, [8, 196, 1536]);  addmm_119 = None
        split_79 = torch.ops.aten.split.Tensor(view_319, 768, -1);  view_319 = None
        getitem_320 = split_79[0]
        getitem_321 = split_79[1];  split_79 = None
        sigmoid_79 = torch.ops.aten.sigmoid.default(getitem_321)
        mul_320 = torch.ops.aten.mul.Tensor(getitem_321, sigmoid_79);  getitem_321 = sigmoid_79 = None
        mul_321 = torch.ops.aten.mul.Tensor(getitem_320, mul_320);  getitem_320 = mul_320 = None
        view_320 = torch.ops.aten.view.default(mul_321, [1568, 768]);  mul_321 = None
        permute_242 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg194_1, view_320, permute_242);  arg194_1 = view_320 = permute_242 = None
        view_321 = torch.ops.aten.view.default(addmm_120, [8, 196, 384]);  addmm_120 = None
        add_281 = torch.ops.aten.add.Tensor(add_278, view_321);  add_278 = view_321 = None
        clone_282 = torch.ops.aten.clone.default(add_281, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_282, [2], correction = 0, keepdim = True)
        getitem_322 = var_mean_81[0]
        getitem_323 = var_mean_81[1];  var_mean_81 = None
        add_282 = torch.ops.aten.add.Tensor(getitem_322, 1e-06);  getitem_322 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_81 = torch.ops.aten.sub.Tensor(clone_282, getitem_323);  clone_282 = getitem_323 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, arg195_1);  mul_322 = arg195_1 = None
        add_283 = torch.ops.aten.add.Tensor(mul_323, arg196_1);  mul_323 = arg196_1 = None
        permute_243 = torch.ops.aten.permute.default(add_283, [0, 2, 1]);  add_283 = None
        permute_244 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        clone_283 = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_322 = torch.ops.aten.view.default(clone_283, [3072, 196]);  clone_283 = None
        mm_40 = torch.ops.aten.mm.default(view_322, permute_244);  view_322 = permute_244 = None
        view_323 = torch.ops.aten.view.default(mm_40, [8, 384, 384]);  mm_40 = None
        add_284 = torch.ops.aten.add.Tensor(view_323, arg198_1);  view_323 = arg198_1 = None
        split_80 = torch.ops.aten.split.Tensor(add_284, 192, -1);  add_284 = None
        getitem_324 = split_80[0]
        getitem_325 = split_80[1];  split_80 = None
        sigmoid_80 = torch.ops.aten.sigmoid.default(getitem_325)
        mul_324 = torch.ops.aten.mul.Tensor(getitem_325, sigmoid_80);  getitem_325 = sigmoid_80 = None
        mul_325 = torch.ops.aten.mul.Tensor(getitem_324, mul_324);  getitem_324 = mul_324 = None
        view_324 = torch.ops.aten.view.default(mul_325, [3072, 192]);  mul_325 = None
        permute_245 = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg200_1, view_324, permute_245);  arg200_1 = view_324 = permute_245 = None
        view_325 = torch.ops.aten.view.default(addmm_121, [8, 384, 196]);  addmm_121 = None
        permute_246 = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
        add_285 = torch.ops.aten.add.Tensor(add_281, permute_246);  add_281 = permute_246 = None
        clone_286 = torch.ops.aten.clone.default(add_285, memory_format = torch.contiguous_format)
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_286, [2], correction = 0, keepdim = True)
        getitem_326 = var_mean_82[0]
        getitem_327 = var_mean_82[1];  var_mean_82 = None
        add_286 = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_82 = torch.ops.aten.sub.Tensor(clone_286, getitem_327);  clone_286 = getitem_327 = None
        mul_326 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
        mul_327 = torch.ops.aten.mul.Tensor(mul_326, arg201_1);  mul_326 = arg201_1 = None
        add_287 = torch.ops.aten.add.Tensor(mul_327, arg202_1);  mul_327 = arg202_1 = None
        view_326 = torch.ops.aten.view.default(add_287, [1568, 384]);  add_287 = None
        permute_247 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg204_1, view_326, permute_247);  arg204_1 = view_326 = permute_247 = None
        view_327 = torch.ops.aten.view.default(addmm_122, [8, 196, 1536]);  addmm_122 = None
        split_81 = torch.ops.aten.split.Tensor(view_327, 768, -1);  view_327 = None
        getitem_328 = split_81[0]
        getitem_329 = split_81[1];  split_81 = None
        sigmoid_81 = torch.ops.aten.sigmoid.default(getitem_329)
        mul_328 = torch.ops.aten.mul.Tensor(getitem_329, sigmoid_81);  getitem_329 = sigmoid_81 = None
        mul_329 = torch.ops.aten.mul.Tensor(getitem_328, mul_328);  getitem_328 = mul_328 = None
        view_328 = torch.ops.aten.view.default(mul_329, [1568, 768]);  mul_329 = None
        permute_248 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg206_1, view_328, permute_248);  arg206_1 = view_328 = permute_248 = None
        view_329 = torch.ops.aten.view.default(addmm_123, [8, 196, 384]);  addmm_123 = None
        add_288 = torch.ops.aten.add.Tensor(add_285, view_329);  add_285 = view_329 = None
        clone_289 = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
        var_mean_83 = torch.ops.aten.var_mean.correction(clone_289, [2], correction = 0, keepdim = True)
        getitem_330 = var_mean_83[0]
        getitem_331 = var_mean_83[1];  var_mean_83 = None
        add_289 = torch.ops.aten.add.Tensor(getitem_330, 1e-06);  getitem_330 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        sub_83 = torch.ops.aten.sub.Tensor(clone_289, getitem_331);  clone_289 = getitem_331 = None
        mul_330 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
        mul_331 = torch.ops.aten.mul.Tensor(mul_330, arg207_1);  mul_330 = arg207_1 = None
        add_290 = torch.ops.aten.add.Tensor(mul_331, arg208_1);  mul_331 = arg208_1 = None
        permute_249 = torch.ops.aten.permute.default(add_290, [0, 2, 1]);  add_290 = None
        permute_250 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        clone_290 = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_330 = torch.ops.aten.view.default(clone_290, [3072, 196]);  clone_290 = None
        mm_41 = torch.ops.aten.mm.default(view_330, permute_250);  view_330 = permute_250 = None
        view_331 = torch.ops.aten.view.default(mm_41, [8, 384, 384]);  mm_41 = None
        add_291 = torch.ops.aten.add.Tensor(view_331, arg210_1);  view_331 = arg210_1 = None
        split_82 = torch.ops.aten.split.Tensor(add_291, 192, -1);  add_291 = None
        getitem_332 = split_82[0]
        getitem_333 = split_82[1];  split_82 = None
        sigmoid_82 = torch.ops.aten.sigmoid.default(getitem_333)
        mul_332 = torch.ops.aten.mul.Tensor(getitem_333, sigmoid_82);  getitem_333 = sigmoid_82 = None
        mul_333 = torch.ops.aten.mul.Tensor(getitem_332, mul_332);  getitem_332 = mul_332 = None
        view_332 = torch.ops.aten.view.default(mul_333, [3072, 192]);  mul_333 = None
        permute_251 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg212_1, view_332, permute_251);  arg212_1 = view_332 = permute_251 = None
        view_333 = torch.ops.aten.view.default(addmm_124, [8, 384, 196]);  addmm_124 = None
        permute_252 = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        add_292 = torch.ops.aten.add.Tensor(add_288, permute_252);  add_288 = permute_252 = None
        clone_293 = torch.ops.aten.clone.default(add_292, memory_format = torch.contiguous_format)
        var_mean_84 = torch.ops.aten.var_mean.correction(clone_293, [2], correction = 0, keepdim = True)
        getitem_334 = var_mean_84[0]
        getitem_335 = var_mean_84[1];  var_mean_84 = None
        add_293 = torch.ops.aten.add.Tensor(getitem_334, 1e-06);  getitem_334 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        sub_84 = torch.ops.aten.sub.Tensor(clone_293, getitem_335);  clone_293 = getitem_335 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, arg213_1);  mul_334 = arg213_1 = None
        add_294 = torch.ops.aten.add.Tensor(mul_335, arg214_1);  mul_335 = arg214_1 = None
        view_334 = torch.ops.aten.view.default(add_294, [1568, 384]);  add_294 = None
        permute_253 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg216_1, view_334, permute_253);  arg216_1 = view_334 = permute_253 = None
        view_335 = torch.ops.aten.view.default(addmm_125, [8, 196, 1536]);  addmm_125 = None
        split_83 = torch.ops.aten.split.Tensor(view_335, 768, -1);  view_335 = None
        getitem_336 = split_83[0]
        getitem_337 = split_83[1];  split_83 = None
        sigmoid_83 = torch.ops.aten.sigmoid.default(getitem_337)
        mul_336 = torch.ops.aten.mul.Tensor(getitem_337, sigmoid_83);  getitem_337 = sigmoid_83 = None
        mul_337 = torch.ops.aten.mul.Tensor(getitem_336, mul_336);  getitem_336 = mul_336 = None
        view_336 = torch.ops.aten.view.default(mul_337, [1568, 768]);  mul_337 = None
        permute_254 = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg218_1, view_336, permute_254);  arg218_1 = view_336 = permute_254 = None
        view_337 = torch.ops.aten.view.default(addmm_126, [8, 196, 384]);  addmm_126 = None
        add_295 = torch.ops.aten.add.Tensor(add_292, view_337);  add_292 = view_337 = None
        clone_296 = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_296, [2], correction = 0, keepdim = True)
        getitem_338 = var_mean_85[0]
        getitem_339 = var_mean_85[1];  var_mean_85 = None
        add_296 = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_85 = torch.ops.aten.sub.Tensor(clone_296, getitem_339);  clone_296 = getitem_339 = None
        mul_338 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, arg219_1);  mul_338 = arg219_1 = None
        add_297 = torch.ops.aten.add.Tensor(mul_339, arg220_1);  mul_339 = arg220_1 = None
        permute_255 = torch.ops.aten.permute.default(add_297, [0, 2, 1]);  add_297 = None
        permute_256 = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        clone_297 = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
        view_338 = torch.ops.aten.view.default(clone_297, [3072, 196]);  clone_297 = None
        mm_42 = torch.ops.aten.mm.default(view_338, permute_256);  view_338 = permute_256 = None
        view_339 = torch.ops.aten.view.default(mm_42, [8, 384, 384]);  mm_42 = None
        add_298 = torch.ops.aten.add.Tensor(view_339, arg222_1);  view_339 = arg222_1 = None
        split_84 = torch.ops.aten.split.Tensor(add_298, 192, -1);  add_298 = None
        getitem_340 = split_84[0]
        getitem_341 = split_84[1];  split_84 = None
        sigmoid_84 = torch.ops.aten.sigmoid.default(getitem_341)
        mul_340 = torch.ops.aten.mul.Tensor(getitem_341, sigmoid_84);  getitem_341 = sigmoid_84 = None
        mul_341 = torch.ops.aten.mul.Tensor(getitem_340, mul_340);  getitem_340 = mul_340 = None
        view_340 = torch.ops.aten.view.default(mul_341, [3072, 192]);  mul_341 = None
        permute_257 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg224_1, view_340, permute_257);  arg224_1 = view_340 = permute_257 = None
        view_341 = torch.ops.aten.view.default(addmm_127, [8, 384, 196]);  addmm_127 = None
        permute_258 = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
        add_299 = torch.ops.aten.add.Tensor(add_295, permute_258);  add_295 = permute_258 = None
        clone_300 = torch.ops.aten.clone.default(add_299, memory_format = torch.contiguous_format)
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_300, [2], correction = 0, keepdim = True)
        getitem_342 = var_mean_86[0]
        getitem_343 = var_mean_86[1];  var_mean_86 = None
        add_300 = torch.ops.aten.add.Tensor(getitem_342, 1e-06);  getitem_342 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
        sub_86 = torch.ops.aten.sub.Tensor(clone_300, getitem_343);  clone_300 = getitem_343 = None
        mul_342 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_342, arg225_1);  mul_342 = arg225_1 = None
        add_301 = torch.ops.aten.add.Tensor(mul_343, arg226_1);  mul_343 = arg226_1 = None
        view_342 = torch.ops.aten.view.default(add_301, [1568, 384]);  add_301 = None
        permute_259 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg228_1, view_342, permute_259);  arg228_1 = view_342 = permute_259 = None
        view_343 = torch.ops.aten.view.default(addmm_128, [8, 196, 1536]);  addmm_128 = None
        split_85 = torch.ops.aten.split.Tensor(view_343, 768, -1);  view_343 = None
        getitem_344 = split_85[0]
        getitem_345 = split_85[1];  split_85 = None
        sigmoid_85 = torch.ops.aten.sigmoid.default(getitem_345)
        mul_344 = torch.ops.aten.mul.Tensor(getitem_345, sigmoid_85);  getitem_345 = sigmoid_85 = None
        mul_345 = torch.ops.aten.mul.Tensor(getitem_344, mul_344);  getitem_344 = mul_344 = None
        view_344 = torch.ops.aten.view.default(mul_345, [1568, 768]);  mul_345 = None
        permute_260 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg230_1, view_344, permute_260);  arg230_1 = view_344 = permute_260 = None
        view_345 = torch.ops.aten.view.default(addmm_129, [8, 196, 384]);  addmm_129 = None
        add_302 = torch.ops.aten.add.Tensor(add_299, view_345);  add_299 = view_345 = None
        clone_303 = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_303, [2], correction = 0, keepdim = True)
        getitem_346 = var_mean_87[0]
        getitem_347 = var_mean_87[1];  var_mean_87 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_346, 1e-06);  getitem_346 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_87 = torch.ops.aten.sub.Tensor(clone_303, getitem_347);  clone_303 = getitem_347 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, arg231_1);  mul_346 = arg231_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_347, arg232_1);  mul_347 = arg232_1 = None
        permute_261 = torch.ops.aten.permute.default(add_304, [0, 2, 1]);  add_304 = None
        permute_262 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        clone_304 = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_346 = torch.ops.aten.view.default(clone_304, [3072, 196]);  clone_304 = None
        mm_43 = torch.ops.aten.mm.default(view_346, permute_262);  view_346 = permute_262 = None
        view_347 = torch.ops.aten.view.default(mm_43, [8, 384, 384]);  mm_43 = None
        add_305 = torch.ops.aten.add.Tensor(view_347, arg234_1);  view_347 = arg234_1 = None
        split_86 = torch.ops.aten.split.Tensor(add_305, 192, -1);  add_305 = None
        getitem_348 = split_86[0]
        getitem_349 = split_86[1];  split_86 = None
        sigmoid_86 = torch.ops.aten.sigmoid.default(getitem_349)
        mul_348 = torch.ops.aten.mul.Tensor(getitem_349, sigmoid_86);  getitem_349 = sigmoid_86 = None
        mul_349 = torch.ops.aten.mul.Tensor(getitem_348, mul_348);  getitem_348 = mul_348 = None
        view_348 = torch.ops.aten.view.default(mul_349, [3072, 192]);  mul_349 = None
        permute_263 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg236_1, view_348, permute_263);  arg236_1 = view_348 = permute_263 = None
        view_349 = torch.ops.aten.view.default(addmm_130, [8, 384, 196]);  addmm_130 = None
        permute_264 = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
        add_306 = torch.ops.aten.add.Tensor(add_302, permute_264);  add_302 = permute_264 = None
        clone_307 = torch.ops.aten.clone.default(add_306, memory_format = torch.contiguous_format)
        var_mean_88 = torch.ops.aten.var_mean.correction(clone_307, [2], correction = 0, keepdim = True)
        getitem_350 = var_mean_88[0]
        getitem_351 = var_mean_88[1];  var_mean_88 = None
        add_307 = torch.ops.aten.add.Tensor(getitem_350, 1e-06);  getitem_350 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
        sub_88 = torch.ops.aten.sub.Tensor(clone_307, getitem_351);  clone_307 = getitem_351 = None
        mul_350 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_350, arg237_1);  mul_350 = arg237_1 = None
        add_308 = torch.ops.aten.add.Tensor(mul_351, arg238_1);  mul_351 = arg238_1 = None
        view_350 = torch.ops.aten.view.default(add_308, [1568, 384]);  add_308 = None
        permute_265 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg240_1, view_350, permute_265);  arg240_1 = view_350 = permute_265 = None
        view_351 = torch.ops.aten.view.default(addmm_131, [8, 196, 1536]);  addmm_131 = None
        split_87 = torch.ops.aten.split.Tensor(view_351, 768, -1);  view_351 = None
        getitem_352 = split_87[0]
        getitem_353 = split_87[1];  split_87 = None
        sigmoid_87 = torch.ops.aten.sigmoid.default(getitem_353)
        mul_352 = torch.ops.aten.mul.Tensor(getitem_353, sigmoid_87);  getitem_353 = sigmoid_87 = None
        mul_353 = torch.ops.aten.mul.Tensor(getitem_352, mul_352);  getitem_352 = mul_352 = None
        view_352 = torch.ops.aten.view.default(mul_353, [1568, 768]);  mul_353 = None
        permute_266 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg242_1, view_352, permute_266);  arg242_1 = view_352 = permute_266 = None
        view_353 = torch.ops.aten.view.default(addmm_132, [8, 196, 384]);  addmm_132 = None
        add_309 = torch.ops.aten.add.Tensor(add_306, view_353);  add_306 = view_353 = None
        clone_310 = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_89 = torch.ops.aten.var_mean.correction(clone_310, [2], correction = 0, keepdim = True)
        getitem_354 = var_mean_89[0]
        getitem_355 = var_mean_89[1];  var_mean_89 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_354, 1e-06);  getitem_354 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_89 = torch.ops.aten.sub.Tensor(clone_310, getitem_355);  clone_310 = getitem_355 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, arg243_1);  mul_354 = arg243_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_355, arg244_1);  mul_355 = arg244_1 = None
        permute_267 = torch.ops.aten.permute.default(add_311, [0, 2, 1]);  add_311 = None
        permute_268 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        clone_311 = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_354 = torch.ops.aten.view.default(clone_311, [3072, 196]);  clone_311 = None
        mm_44 = torch.ops.aten.mm.default(view_354, permute_268);  view_354 = permute_268 = None
        view_355 = torch.ops.aten.view.default(mm_44, [8, 384, 384]);  mm_44 = None
        add_312 = torch.ops.aten.add.Tensor(view_355, arg246_1);  view_355 = arg246_1 = None
        split_88 = torch.ops.aten.split.Tensor(add_312, 192, -1);  add_312 = None
        getitem_356 = split_88[0]
        getitem_357 = split_88[1];  split_88 = None
        sigmoid_88 = torch.ops.aten.sigmoid.default(getitem_357)
        mul_356 = torch.ops.aten.mul.Tensor(getitem_357, sigmoid_88);  getitem_357 = sigmoid_88 = None
        mul_357 = torch.ops.aten.mul.Tensor(getitem_356, mul_356);  getitem_356 = mul_356 = None
        view_356 = torch.ops.aten.view.default(mul_357, [3072, 192]);  mul_357 = None
        permute_269 = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg248_1, view_356, permute_269);  arg248_1 = view_356 = permute_269 = None
        view_357 = torch.ops.aten.view.default(addmm_133, [8, 384, 196]);  addmm_133 = None
        permute_270 = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
        add_313 = torch.ops.aten.add.Tensor(add_309, permute_270);  add_309 = permute_270 = None
        clone_314 = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_314, [2], correction = 0, keepdim = True)
        getitem_358 = var_mean_90[0]
        getitem_359 = var_mean_90[1];  var_mean_90 = None
        add_314 = torch.ops.aten.add.Tensor(getitem_358, 1e-06);  getitem_358 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        sub_90 = torch.ops.aten.sub.Tensor(clone_314, getitem_359);  clone_314 = getitem_359 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, arg249_1);  mul_358 = arg249_1 = None
        add_315 = torch.ops.aten.add.Tensor(mul_359, arg250_1);  mul_359 = arg250_1 = None
        view_358 = torch.ops.aten.view.default(add_315, [1568, 384]);  add_315 = None
        permute_271 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg252_1, view_358, permute_271);  arg252_1 = view_358 = permute_271 = None
        view_359 = torch.ops.aten.view.default(addmm_134, [8, 196, 1536]);  addmm_134 = None
        split_89 = torch.ops.aten.split.Tensor(view_359, 768, -1);  view_359 = None
        getitem_360 = split_89[0]
        getitem_361 = split_89[1];  split_89 = None
        sigmoid_89 = torch.ops.aten.sigmoid.default(getitem_361)
        mul_360 = torch.ops.aten.mul.Tensor(getitem_361, sigmoid_89);  getitem_361 = sigmoid_89 = None
        mul_361 = torch.ops.aten.mul.Tensor(getitem_360, mul_360);  getitem_360 = mul_360 = None
        view_360 = torch.ops.aten.view.default(mul_361, [1568, 768]);  mul_361 = None
        permute_272 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg254_1, view_360, permute_272);  arg254_1 = view_360 = permute_272 = None
        view_361 = torch.ops.aten.view.default(addmm_135, [8, 196, 384]);  addmm_135 = None
        add_316 = torch.ops.aten.add.Tensor(add_313, view_361);  add_313 = view_361 = None
        clone_317 = torch.ops.aten.clone.default(add_316, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_317, [2], correction = 0, keepdim = True)
        getitem_362 = var_mean_91[0]
        getitem_363 = var_mean_91[1];  var_mean_91 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_362, 1e-06);  getitem_362 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_91 = torch.ops.aten.sub.Tensor(clone_317, getitem_363);  clone_317 = getitem_363 = None
        mul_362 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_362, arg255_1);  mul_362 = arg255_1 = None
        add_318 = torch.ops.aten.add.Tensor(mul_363, arg256_1);  mul_363 = arg256_1 = None
        permute_273 = torch.ops.aten.permute.default(add_318, [0, 2, 1]);  add_318 = None
        permute_274 = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        clone_318 = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
        view_362 = torch.ops.aten.view.default(clone_318, [3072, 196]);  clone_318 = None
        mm_45 = torch.ops.aten.mm.default(view_362, permute_274);  view_362 = permute_274 = None
        view_363 = torch.ops.aten.view.default(mm_45, [8, 384, 384]);  mm_45 = None
        add_319 = torch.ops.aten.add.Tensor(view_363, arg258_1);  view_363 = arg258_1 = None
        split_90 = torch.ops.aten.split.Tensor(add_319, 192, -1);  add_319 = None
        getitem_364 = split_90[0]
        getitem_365 = split_90[1];  split_90 = None
        sigmoid_90 = torch.ops.aten.sigmoid.default(getitem_365)
        mul_364 = torch.ops.aten.mul.Tensor(getitem_365, sigmoid_90);  getitem_365 = sigmoid_90 = None
        mul_365 = torch.ops.aten.mul.Tensor(getitem_364, mul_364);  getitem_364 = mul_364 = None
        view_364 = torch.ops.aten.view.default(mul_365, [3072, 192]);  mul_365 = None
        permute_275 = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg260_1, view_364, permute_275);  arg260_1 = view_364 = permute_275 = None
        view_365 = torch.ops.aten.view.default(addmm_136, [8, 384, 196]);  addmm_136 = None
        permute_276 = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
        add_320 = torch.ops.aten.add.Tensor(add_316, permute_276);  add_316 = permute_276 = None
        clone_321 = torch.ops.aten.clone.default(add_320, memory_format = torch.contiguous_format)
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_321, [2], correction = 0, keepdim = True)
        getitem_366 = var_mean_92[0]
        getitem_367 = var_mean_92[1];  var_mean_92 = None
        add_321 = torch.ops.aten.add.Tensor(getitem_366, 1e-06);  getitem_366 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
        sub_92 = torch.ops.aten.sub.Tensor(clone_321, getitem_367);  clone_321 = getitem_367 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_366, arg261_1);  mul_366 = arg261_1 = None
        add_322 = torch.ops.aten.add.Tensor(mul_367, arg262_1);  mul_367 = arg262_1 = None
        view_366 = torch.ops.aten.view.default(add_322, [1568, 384]);  add_322 = None
        permute_277 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg264_1, view_366, permute_277);  arg264_1 = view_366 = permute_277 = None
        view_367 = torch.ops.aten.view.default(addmm_137, [8, 196, 1536]);  addmm_137 = None
        split_91 = torch.ops.aten.split.Tensor(view_367, 768, -1);  view_367 = None
        getitem_368 = split_91[0]
        getitem_369 = split_91[1];  split_91 = None
        sigmoid_91 = torch.ops.aten.sigmoid.default(getitem_369)
        mul_368 = torch.ops.aten.mul.Tensor(getitem_369, sigmoid_91);  getitem_369 = sigmoid_91 = None
        mul_369 = torch.ops.aten.mul.Tensor(getitem_368, mul_368);  getitem_368 = mul_368 = None
        view_368 = torch.ops.aten.view.default(mul_369, [1568, 768]);  mul_369 = None
        permute_278 = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg266_1, view_368, permute_278);  arg266_1 = view_368 = permute_278 = None
        view_369 = torch.ops.aten.view.default(addmm_138, [8, 196, 384]);  addmm_138 = None
        add_323 = torch.ops.aten.add.Tensor(add_320, view_369);  add_320 = view_369 = None
        clone_324 = torch.ops.aten.clone.default(add_323, memory_format = torch.contiguous_format)
        var_mean_93 = torch.ops.aten.var_mean.correction(clone_324, [2], correction = 0, keepdim = True)
        getitem_370 = var_mean_93[0]
        getitem_371 = var_mean_93[1];  var_mean_93 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_370, 1e-06);  getitem_370 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_93 = torch.ops.aten.sub.Tensor(clone_324, getitem_371);  clone_324 = getitem_371 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, arg267_1);  mul_370 = arg267_1 = None
        add_325 = torch.ops.aten.add.Tensor(mul_371, arg268_1);  mul_371 = arg268_1 = None
        permute_279 = torch.ops.aten.permute.default(add_325, [0, 2, 1]);  add_325 = None
        permute_280 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        clone_325 = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_370 = torch.ops.aten.view.default(clone_325, [3072, 196]);  clone_325 = None
        mm_46 = torch.ops.aten.mm.default(view_370, permute_280);  view_370 = permute_280 = None
        view_371 = torch.ops.aten.view.default(mm_46, [8, 384, 384]);  mm_46 = None
        add_326 = torch.ops.aten.add.Tensor(view_371, arg270_1);  view_371 = arg270_1 = None
        split_92 = torch.ops.aten.split.Tensor(add_326, 192, -1);  add_326 = None
        getitem_372 = split_92[0]
        getitem_373 = split_92[1];  split_92 = None
        sigmoid_92 = torch.ops.aten.sigmoid.default(getitem_373)
        mul_372 = torch.ops.aten.mul.Tensor(getitem_373, sigmoid_92);  getitem_373 = sigmoid_92 = None
        mul_373 = torch.ops.aten.mul.Tensor(getitem_372, mul_372);  getitem_372 = mul_372 = None
        view_372 = torch.ops.aten.view.default(mul_373, [3072, 192]);  mul_373 = None
        permute_281 = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg272_1, view_372, permute_281);  arg272_1 = view_372 = permute_281 = None
        view_373 = torch.ops.aten.view.default(addmm_139, [8, 384, 196]);  addmm_139 = None
        permute_282 = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
        add_327 = torch.ops.aten.add.Tensor(add_323, permute_282);  add_323 = permute_282 = None
        clone_328 = torch.ops.aten.clone.default(add_327, memory_format = torch.contiguous_format)
        var_mean_94 = torch.ops.aten.var_mean.correction(clone_328, [2], correction = 0, keepdim = True)
        getitem_374 = var_mean_94[0]
        getitem_375 = var_mean_94[1];  var_mean_94 = None
        add_328 = torch.ops.aten.add.Tensor(getitem_374, 1e-06);  getitem_374 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_328);  add_328 = None
        sub_94 = torch.ops.aten.sub.Tensor(clone_328, getitem_375);  clone_328 = getitem_375 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        mul_375 = torch.ops.aten.mul.Tensor(mul_374, arg273_1);  mul_374 = arg273_1 = None
        add_329 = torch.ops.aten.add.Tensor(mul_375, arg274_1);  mul_375 = arg274_1 = None
        view_374 = torch.ops.aten.view.default(add_329, [1568, 384]);  add_329 = None
        permute_283 = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg276_1, view_374, permute_283);  arg276_1 = view_374 = permute_283 = None
        view_375 = torch.ops.aten.view.default(addmm_140, [8, 196, 1536]);  addmm_140 = None
        split_93 = torch.ops.aten.split.Tensor(view_375, 768, -1);  view_375 = None
        getitem_376 = split_93[0]
        getitem_377 = split_93[1];  split_93 = None
        sigmoid_93 = torch.ops.aten.sigmoid.default(getitem_377)
        mul_376 = torch.ops.aten.mul.Tensor(getitem_377, sigmoid_93);  getitem_377 = sigmoid_93 = None
        mul_377 = torch.ops.aten.mul.Tensor(getitem_376, mul_376);  getitem_376 = mul_376 = None
        view_376 = torch.ops.aten.view.default(mul_377, [1568, 768]);  mul_377 = None
        permute_284 = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg278_1, view_376, permute_284);  arg278_1 = view_376 = permute_284 = None
        view_377 = torch.ops.aten.view.default(addmm_141, [8, 196, 384]);  addmm_141 = None
        add_330 = torch.ops.aten.add.Tensor(add_327, view_377);  add_327 = view_377 = None
        clone_331 = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_331, [2], correction = 0, keepdim = True)
        getitem_378 = var_mean_95[0]
        getitem_379 = var_mean_95[1];  var_mean_95 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_378, 1e-06);  getitem_378 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_95 = torch.ops.aten.sub.Tensor(clone_331, getitem_379);  clone_331 = getitem_379 = None
        mul_378 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        mul_379 = torch.ops.aten.mul.Tensor(mul_378, arg279_1);  mul_378 = arg279_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_379, arg280_1);  mul_379 = arg280_1 = None
        permute_285 = torch.ops.aten.permute.default(add_332, [0, 2, 1]);  add_332 = None
        permute_286 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        clone_332 = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
        view_378 = torch.ops.aten.view.default(clone_332, [3072, 196]);  clone_332 = None
        mm_47 = torch.ops.aten.mm.default(view_378, permute_286);  view_378 = permute_286 = None
        view_379 = torch.ops.aten.view.default(mm_47, [8, 384, 384]);  mm_47 = None
        add_333 = torch.ops.aten.add.Tensor(view_379, arg282_1);  view_379 = arg282_1 = None
        split_94 = torch.ops.aten.split.Tensor(add_333, 192, -1);  add_333 = None
        getitem_380 = split_94[0]
        getitem_381 = split_94[1];  split_94 = None
        sigmoid_94 = torch.ops.aten.sigmoid.default(getitem_381)
        mul_380 = torch.ops.aten.mul.Tensor(getitem_381, sigmoid_94);  getitem_381 = sigmoid_94 = None
        mul_381 = torch.ops.aten.mul.Tensor(getitem_380, mul_380);  getitem_380 = mul_380 = None
        view_380 = torch.ops.aten.view.default(mul_381, [3072, 192]);  mul_381 = None
        permute_287 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg284_1, view_380, permute_287);  arg284_1 = view_380 = permute_287 = None
        view_381 = torch.ops.aten.view.default(addmm_142, [8, 384, 196]);  addmm_142 = None
        permute_288 = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
        add_334 = torch.ops.aten.add.Tensor(add_330, permute_288);  add_330 = permute_288 = None
        clone_335 = torch.ops.aten.clone.default(add_334, memory_format = torch.contiguous_format)
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_335, [2], correction = 0, keepdim = True)
        getitem_382 = var_mean_96[0]
        getitem_383 = var_mean_96[1];  var_mean_96 = None
        add_335 = torch.ops.aten.add.Tensor(getitem_382, 1e-06);  getitem_382 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
        sub_96 = torch.ops.aten.sub.Tensor(clone_335, getitem_383);  clone_335 = getitem_383 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, arg285_1);  mul_382 = arg285_1 = None
        add_336 = torch.ops.aten.add.Tensor(mul_383, arg286_1);  mul_383 = arg286_1 = None
        view_382 = torch.ops.aten.view.default(add_336, [1568, 384]);  add_336 = None
        permute_289 = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg288_1, view_382, permute_289);  arg288_1 = view_382 = permute_289 = None
        view_383 = torch.ops.aten.view.default(addmm_143, [8, 196, 1536]);  addmm_143 = None
        split_95 = torch.ops.aten.split.Tensor(view_383, 768, -1);  view_383 = None
        getitem_384 = split_95[0]
        getitem_385 = split_95[1];  split_95 = None
        sigmoid_95 = torch.ops.aten.sigmoid.default(getitem_385)
        mul_384 = torch.ops.aten.mul.Tensor(getitem_385, sigmoid_95);  getitem_385 = sigmoid_95 = None
        mul_385 = torch.ops.aten.mul.Tensor(getitem_384, mul_384);  getitem_384 = mul_384 = None
        view_384 = torch.ops.aten.view.default(mul_385, [1568, 768]);  mul_385 = None
        permute_290 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg290_1, view_384, permute_290);  arg290_1 = view_384 = permute_290 = None
        view_385 = torch.ops.aten.view.default(addmm_144, [8, 196, 384]);  addmm_144 = None
        add_337 = torch.ops.aten.add.Tensor(add_334, view_385);  add_334 = view_385 = None
        clone_338 = torch.ops.aten.clone.default(add_337, memory_format = torch.contiguous_format);  add_337 = None
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_338, [2], correction = 0, keepdim = True)
        getitem_386 = var_mean_97[0]
        getitem_387 = var_mean_97[1];  var_mean_97 = None
        add_338 = torch.ops.aten.add.Tensor(getitem_386, 1e-06);  getitem_386 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        sub_97 = torch.ops.aten.sub.Tensor(clone_338, getitem_387);  clone_338 = getitem_387 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, arg291_1);  mul_386 = arg291_1 = None
        add_339 = torch.ops.aten.add.Tensor(mul_387, arg292_1);  mul_387 = arg292_1 = None
        mean_1 = torch.ops.aten.mean.dim(add_339, [1]);  add_339 = None
        permute_291 = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg294_1, mean_1, permute_291);  arg294_1 = mean_1 = permute_291 = None
        return (addmm_145,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf1, (384, 3, 16, 16), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf2, (384,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (384,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf4, (384,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf5, (384, 196), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf7, (196, 192), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf8, (196,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf9, (384,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf10, (384,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1536, 384), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1536,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf13, (384, 768), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf14, (384,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf15, (384,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf16, (384,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf17, (384, 196), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf19, (196, 192), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf20, (196,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf21, (384,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf22, (384,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf23, (1536, 384), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1536,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf25, (384, 768), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf26, (384,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf27, (384,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf28, (384,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf29, (384, 196), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf30, (384,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf31, (196, 192), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf32, (196,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf33, (384,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf34, (384,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1536, 384), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1536,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf37, (384, 768), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf38, (384,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf39, (384,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf40, (384,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf41, (384, 196), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf42, (384,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf43, (196, 192), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf44, (196,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf45, (384,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf46, (384,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1536, 384), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1536,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf49, (384, 768), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf50, (384,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf51, (384,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf52, (384,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf53, (384, 196), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf54, (384,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf55, (196, 192), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf56, (196,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf57, (384,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf58, (384,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1536, 384), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1536,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf61, (384, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf62, (384,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf63, (384,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf64, (384,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf65, (384, 196), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf66, (384,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf67, (196, 192), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf68, (196,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf69, (384,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf70, (384,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1536, 384), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1536,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf73, (384, 768), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf74, (384,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf75, (384,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf76, (384,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf77, (384, 196), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf78, (384,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf79, (196, 192), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf80, (196,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (384,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf82, (384,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1536, 384), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1536,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf85, (384, 768), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf86, (384,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf88, (384,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf89, (384, 196), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf90, (384,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf91, (196, 192), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf92, (196,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf93, (384,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf94, (384,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1536, 384), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1536,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf97, (384, 768), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf98, (384,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf99, (384,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf100, (384,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf101, (384, 196), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf102, (384,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf103, (196, 192), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf104, (196,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (384,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf106, (384,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1536, 384), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1536,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf109, (384, 768), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf112, (384,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf113, (384, 196), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (384,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf115, (196, 192), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf116, (196,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf117, (384,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf118, (384,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1536, 384), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1536,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf121, (384, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf124, (384,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf125, (384, 196), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf127, (196, 192), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf128, (196,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf129, (384,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf130, (384,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1536, 384), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1536,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf133, (384, 768), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf134, (384,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf135, (384,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf136, (384,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf137, (384, 196), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf138, (384,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf139, (196, 192), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf140, (196,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf141, (384,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf142, (384,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf143, (1536, 384), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1536,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384, 768), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf146, (384,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf147, (384,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf148, (384,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf149, (384, 196), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf150, (384,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf151, (196, 192), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf152, (196,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf153, (384,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf154, (384,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1536, 384), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1536,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf157, (384, 768), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf158, (384,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf159, (384,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf160, (384,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf161, (384, 196), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf162, (384,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf163, (196, 192), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf164, (196,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf165, (384,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf166, (384,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1536, 384), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1536,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf169, (384, 768), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf170, (384,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf171, (384,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf172, (384,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf173, (384, 196), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf174, (384,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf175, (196, 192), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf176, (196,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf177, (384,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf178, (384,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1536, 384), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1536,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf181, (384, 768), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf182, (384,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf183, (384,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf184, (384,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf185, (384, 196), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf186, (384,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf187, (196, 192), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf188, (196,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf189, (384,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf190, (384,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1536, 384), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf192, (1536,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf193, (384, 768), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf194, (384,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf195, (384,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf196, (384,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf197, (384, 196), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf198, (384,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf199, (196, 192), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf200, (196,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf201, (384,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf202, (384,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1536, 384), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1536,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf205, (384, 768), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf206, (384,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf207, (384,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf208, (384,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf209, (384, 196), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf210, (384,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf211, (196, 192), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf212, (196,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf213, (384,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf214, (384,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1536, 384), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1536,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf217, (384, 768), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf218, (384,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf219, (384,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf220, (384,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf221, (384, 196), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf222, (384,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf223, (196, 192), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf224, (196,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf225, (384,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf226, (384,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1536, 384), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf228, (1536,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf229, (384, 768), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf230, (384,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf231, (384,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf232, (384,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf233, (384, 196), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf234, (384,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf235, (196, 192), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf236, (196,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf237, (384,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf238, (384,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf239, (1536, 384), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf240, (1536,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf241, (384, 768), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf242, (384,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf243, (384,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf244, (384,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf245, (384, 196), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf246, (384,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf247, (196, 192), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf248, (196,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf249, (384,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf250, (384,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf251, (1536, 384), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf252, (1536,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf253, (384, 768), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf254, (384,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf255, (384,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf256, (384,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf257, (384, 196), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf258, (384,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf259, (196, 192), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf260, (196,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf261, (384,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf262, (384,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1536, 384), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf264, (1536,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf265, (384, 768), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf266, (384,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf267, (384,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf268, (384,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf269, (384, 196), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf270, (384,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf271, (196, 192), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf272, (196,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf273, (384,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf274, (384,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1536, 384), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1536,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf277, (384, 768), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf278, (384,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf279, (384,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf280, (384,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf281, (384, 196), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf282, (384,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf283, (196, 192), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf284, (196,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf285, (384,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf286, (384,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf287, (1536, 384), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf288, (1536,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf289, (384, 768), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf290, (384,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf291, (384,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf292, (384,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1000, 384), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1000,), is_leaf=True)  # arg294_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)