
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1):
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_181 = torch.ops.aten.view.default(convolution_1, [8, 256, 196]);  convolution_1 = None
        permute_152 = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        clone_152 = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format)
        var_mean_61 = torch.ops.aten.var_mean.correction(clone_152, [2], correction = 0, keepdim = True)
        getitem_182 = var_mean_61[0]
        getitem_183 = var_mean_61[1];  var_mean_61 = None
        add_212 = torch.ops.aten.add.Tensor(getitem_182, 1e-06);  getitem_182 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_61 = torch.ops.aten.sub.Tensor(clone_152, getitem_183);  clone_152 = getitem_183 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, arg3_1);  mul_242 = arg3_1 = None
        add_213 = torch.ops.aten.add.Tensor(mul_243, arg4_1);  mul_243 = arg4_1 = None
        view_182 = torch.ops.aten.view.default(add_213, [1568, 256]);  add_213 = None
        permute_153 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg6_1, view_182, permute_153);  arg6_1 = view_182 = permute_153 = None
        view_183 = torch.ops.aten.view.default(addmm_61, [8, 196, 1536]);  addmm_61 = None
        mul_244 = torch.ops.aten.mul.Tensor(view_183, 0.5)
        mul_245 = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476);  view_183 = None
        erf_30 = torch.ops.aten.erf.default(mul_245);  mul_245 = None
        add_214 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_246 = torch.ops.aten.mul.Tensor(mul_244, add_214);  mul_244 = add_214 = None
        split_30 = torch.ops.aten.split.Tensor(mul_246, 768, -1);  mul_246 = None
        getitem_184 = split_30[0]
        getitem_185 = split_30[1];  split_30 = None
        clone_154 = torch.ops.aten.clone.default(getitem_185, memory_format = torch.contiguous_format);  getitem_185 = None
        var_mean_62 = torch.ops.aten.var_mean.correction(clone_154, [2], correction = 0, keepdim = True)
        getitem_186 = var_mean_62[0]
        getitem_187 = var_mean_62[1];  var_mean_62 = None
        add_215 = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_62 = torch.ops.aten.sub.Tensor(clone_154, getitem_187);  clone_154 = getitem_187 = None
        mul_247 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_247, arg7_1);  mul_247 = arg7_1 = None
        add_216 = torch.ops.aten.add.Tensor(mul_248, arg8_1);  mul_248 = arg8_1 = None
        permute_154 = torch.ops.aten.permute.default(add_216, [0, 2, 1]);  add_216 = None
        permute_155 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        clone_155 = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_184 = torch.ops.aten.view.default(clone_155, [6144, 196]);  clone_155 = None
        mm_30 = torch.ops.aten.mm.default(view_184, permute_155);  view_184 = permute_155 = None
        view_185 = torch.ops.aten.view.default(mm_30, [8, 768, 196]);  mm_30 = None
        add_217 = torch.ops.aten.add.Tensor(view_185, arg10_1);  view_185 = arg10_1 = None
        permute_156 = torch.ops.aten.permute.default(add_217, [0, 2, 1]);  add_217 = None
        mul_249 = torch.ops.aten.mul.Tensor(getitem_184, permute_156);  getitem_184 = permute_156 = None
        view_186 = torch.ops.aten.view.default(mul_249, [1568, 768]);  mul_249 = None
        permute_157 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg12_1, view_186, permute_157);  arg12_1 = view_186 = permute_157 = None
        view_187 = torch.ops.aten.view.default(addmm_62, [8, 196, 256]);  addmm_62 = None
        add_218 = torch.ops.aten.add.Tensor(permute_152, view_187);  permute_152 = view_187 = None
        clone_157 = torch.ops.aten.clone.default(add_218, memory_format = torch.contiguous_format)
        var_mean_63 = torch.ops.aten.var_mean.correction(clone_157, [2], correction = 0, keepdim = True)
        getitem_188 = var_mean_63[0]
        getitem_189 = var_mean_63[1];  var_mean_63 = None
        add_219 = torch.ops.aten.add.Tensor(getitem_188, 1e-06);  getitem_188 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        sub_63 = torch.ops.aten.sub.Tensor(clone_157, getitem_189);  clone_157 = getitem_189 = None
        mul_250 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, arg13_1);  mul_250 = arg13_1 = None
        add_220 = torch.ops.aten.add.Tensor(mul_251, arg14_1);  mul_251 = arg14_1 = None
        view_188 = torch.ops.aten.view.default(add_220, [1568, 256]);  add_220 = None
        permute_158 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg16_1, view_188, permute_158);  arg16_1 = view_188 = permute_158 = None
        view_189 = torch.ops.aten.view.default(addmm_63, [8, 196, 1536]);  addmm_63 = None
        mul_252 = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_253 = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_31 = torch.ops.aten.erf.default(mul_253);  mul_253 = None
        add_221 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_252, add_221);  mul_252 = add_221 = None
        split_31 = torch.ops.aten.split.Tensor(mul_254, 768, -1);  mul_254 = None
        getitem_190 = split_31[0]
        getitem_191 = split_31[1];  split_31 = None
        clone_159 = torch.ops.aten.clone.default(getitem_191, memory_format = torch.contiguous_format);  getitem_191 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(clone_159, [2], correction = 0, keepdim = True)
        getitem_192 = var_mean_64[0]
        getitem_193 = var_mean_64[1];  var_mean_64 = None
        add_222 = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_64 = torch.ops.aten.sub.Tensor(clone_159, getitem_193);  clone_159 = getitem_193 = None
        mul_255 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, arg17_1);  mul_255 = arg17_1 = None
        add_223 = torch.ops.aten.add.Tensor(mul_256, arg18_1);  mul_256 = arg18_1 = None
        permute_159 = torch.ops.aten.permute.default(add_223, [0, 2, 1]);  add_223 = None
        permute_160 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        clone_160 = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_190 = torch.ops.aten.view.default(clone_160, [6144, 196]);  clone_160 = None
        mm_31 = torch.ops.aten.mm.default(view_190, permute_160);  view_190 = permute_160 = None
        view_191 = torch.ops.aten.view.default(mm_31, [8, 768, 196]);  mm_31 = None
        add_224 = torch.ops.aten.add.Tensor(view_191, arg20_1);  view_191 = arg20_1 = None
        permute_161 = torch.ops.aten.permute.default(add_224, [0, 2, 1]);  add_224 = None
        mul_257 = torch.ops.aten.mul.Tensor(getitem_190, permute_161);  getitem_190 = permute_161 = None
        view_192 = torch.ops.aten.view.default(mul_257, [1568, 768]);  mul_257 = None
        permute_162 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg22_1, view_192, permute_162);  arg22_1 = view_192 = permute_162 = None
        view_193 = torch.ops.aten.view.default(addmm_64, [8, 196, 256]);  addmm_64 = None
        add_225 = torch.ops.aten.add.Tensor(add_218, view_193);  add_218 = view_193 = None
        clone_162 = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_162, [2], correction = 0, keepdim = True)
        getitem_194 = var_mean_65[0]
        getitem_195 = var_mean_65[1];  var_mean_65 = None
        add_226 = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_65 = torch.ops.aten.sub.Tensor(clone_162, getitem_195);  clone_162 = getitem_195 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, arg23_1);  mul_258 = arg23_1 = None
        add_227 = torch.ops.aten.add.Tensor(mul_259, arg24_1);  mul_259 = arg24_1 = None
        view_194 = torch.ops.aten.view.default(add_227, [1568, 256]);  add_227 = None
        permute_163 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg26_1, view_194, permute_163);  arg26_1 = view_194 = permute_163 = None
        view_195 = torch.ops.aten.view.default(addmm_65, [8, 196, 1536]);  addmm_65 = None
        mul_260 = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_261 = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_32 = torch.ops.aten.erf.default(mul_261);  mul_261 = None
        add_228 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_262 = torch.ops.aten.mul.Tensor(mul_260, add_228);  mul_260 = add_228 = None
        split_32 = torch.ops.aten.split.Tensor(mul_262, 768, -1);  mul_262 = None
        getitem_196 = split_32[0]
        getitem_197 = split_32[1];  split_32 = None
        clone_164 = torch.ops.aten.clone.default(getitem_197, memory_format = torch.contiguous_format);  getitem_197 = None
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_164, [2], correction = 0, keepdim = True)
        getitem_198 = var_mean_66[0]
        getitem_199 = var_mean_66[1];  var_mean_66 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_66 = torch.ops.aten.sub.Tensor(clone_164, getitem_199);  clone_164 = getitem_199 = None
        mul_263 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_263, arg27_1);  mul_263 = arg27_1 = None
        add_230 = torch.ops.aten.add.Tensor(mul_264, arg28_1);  mul_264 = arg28_1 = None
        permute_164 = torch.ops.aten.permute.default(add_230, [0, 2, 1]);  add_230 = None
        permute_165 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        clone_165 = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_196 = torch.ops.aten.view.default(clone_165, [6144, 196]);  clone_165 = None
        mm_32 = torch.ops.aten.mm.default(view_196, permute_165);  view_196 = permute_165 = None
        view_197 = torch.ops.aten.view.default(mm_32, [8, 768, 196]);  mm_32 = None
        add_231 = torch.ops.aten.add.Tensor(view_197, arg30_1);  view_197 = arg30_1 = None
        permute_166 = torch.ops.aten.permute.default(add_231, [0, 2, 1]);  add_231 = None
        mul_265 = torch.ops.aten.mul.Tensor(getitem_196, permute_166);  getitem_196 = permute_166 = None
        view_198 = torch.ops.aten.view.default(mul_265, [1568, 768]);  mul_265 = None
        permute_167 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg32_1, view_198, permute_167);  arg32_1 = view_198 = permute_167 = None
        view_199 = torch.ops.aten.view.default(addmm_66, [8, 196, 256]);  addmm_66 = None
        add_232 = torch.ops.aten.add.Tensor(add_225, view_199);  add_225 = view_199 = None
        clone_167 = torch.ops.aten.clone.default(add_232, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_167, [2], correction = 0, keepdim = True)
        getitem_200 = var_mean_67[0]
        getitem_201 = var_mean_67[1];  var_mean_67 = None
        add_233 = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        sub_67 = torch.ops.aten.sub.Tensor(clone_167, getitem_201);  clone_167 = getitem_201 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, arg33_1);  mul_266 = arg33_1 = None
        add_234 = torch.ops.aten.add.Tensor(mul_267, arg34_1);  mul_267 = arg34_1 = None
        view_200 = torch.ops.aten.view.default(add_234, [1568, 256]);  add_234 = None
        permute_168 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg36_1, view_200, permute_168);  arg36_1 = view_200 = permute_168 = None
        view_201 = torch.ops.aten.view.default(addmm_67, [8, 196, 1536]);  addmm_67 = None
        mul_268 = torch.ops.aten.mul.Tensor(view_201, 0.5)
        mul_269 = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476);  view_201 = None
        erf_33 = torch.ops.aten.erf.default(mul_269);  mul_269 = None
        add_235 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_268, add_235);  mul_268 = add_235 = None
        split_33 = torch.ops.aten.split.Tensor(mul_270, 768, -1);  mul_270 = None
        getitem_202 = split_33[0]
        getitem_203 = split_33[1];  split_33 = None
        clone_169 = torch.ops.aten.clone.default(getitem_203, memory_format = torch.contiguous_format);  getitem_203 = None
        var_mean_68 = torch.ops.aten.var_mean.correction(clone_169, [2], correction = 0, keepdim = True)
        getitem_204 = var_mean_68[0]
        getitem_205 = var_mean_68[1];  var_mean_68 = None
        add_236 = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_68 = torch.ops.aten.sub.Tensor(clone_169, getitem_205);  clone_169 = getitem_205 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, arg37_1);  mul_271 = arg37_1 = None
        add_237 = torch.ops.aten.add.Tensor(mul_272, arg38_1);  mul_272 = arg38_1 = None
        permute_169 = torch.ops.aten.permute.default(add_237, [0, 2, 1]);  add_237 = None
        permute_170 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        clone_170 = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_202 = torch.ops.aten.view.default(clone_170, [6144, 196]);  clone_170 = None
        mm_33 = torch.ops.aten.mm.default(view_202, permute_170);  view_202 = permute_170 = None
        view_203 = torch.ops.aten.view.default(mm_33, [8, 768, 196]);  mm_33 = None
        add_238 = torch.ops.aten.add.Tensor(view_203, arg40_1);  view_203 = arg40_1 = None
        permute_171 = torch.ops.aten.permute.default(add_238, [0, 2, 1]);  add_238 = None
        mul_273 = torch.ops.aten.mul.Tensor(getitem_202, permute_171);  getitem_202 = permute_171 = None
        view_204 = torch.ops.aten.view.default(mul_273, [1568, 768]);  mul_273 = None
        permute_172 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg42_1, view_204, permute_172);  arg42_1 = view_204 = permute_172 = None
        view_205 = torch.ops.aten.view.default(addmm_68, [8, 196, 256]);  addmm_68 = None
        add_239 = torch.ops.aten.add.Tensor(add_232, view_205);  add_232 = view_205 = None
        clone_172 = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
        var_mean_69 = torch.ops.aten.var_mean.correction(clone_172, [2], correction = 0, keepdim = True)
        getitem_206 = var_mean_69[0]
        getitem_207 = var_mean_69[1];  var_mean_69 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_206, 1e-06);  getitem_206 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_69 = torch.ops.aten.sub.Tensor(clone_172, getitem_207);  clone_172 = getitem_207 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, arg43_1);  mul_274 = arg43_1 = None
        add_241 = torch.ops.aten.add.Tensor(mul_275, arg44_1);  mul_275 = arg44_1 = None
        view_206 = torch.ops.aten.view.default(add_241, [1568, 256]);  add_241 = None
        permute_173 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg46_1, view_206, permute_173);  arg46_1 = view_206 = permute_173 = None
        view_207 = torch.ops.aten.view.default(addmm_69, [8, 196, 1536]);  addmm_69 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_207, 0.5)
        mul_277 = torch.ops.aten.mul.Tensor(view_207, 0.7071067811865476);  view_207 = None
        erf_34 = torch.ops.aten.erf.default(mul_277);  mul_277 = None
        add_242 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_276, add_242);  mul_276 = add_242 = None
        split_34 = torch.ops.aten.split.Tensor(mul_278, 768, -1);  mul_278 = None
        getitem_208 = split_34[0]
        getitem_209 = split_34[1];  split_34 = None
        clone_174 = torch.ops.aten.clone.default(getitem_209, memory_format = torch.contiguous_format);  getitem_209 = None
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
        getitem_210 = var_mean_70[0]
        getitem_211 = var_mean_70[1];  var_mean_70 = None
        add_243 = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        sub_70 = torch.ops.aten.sub.Tensor(clone_174, getitem_211);  clone_174 = getitem_211 = None
        mul_279 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        mul_280 = torch.ops.aten.mul.Tensor(mul_279, arg47_1);  mul_279 = arg47_1 = None
        add_244 = torch.ops.aten.add.Tensor(mul_280, arg48_1);  mul_280 = arg48_1 = None
        permute_174 = torch.ops.aten.permute.default(add_244, [0, 2, 1]);  add_244 = None
        permute_175 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        clone_175 = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
        view_208 = torch.ops.aten.view.default(clone_175, [6144, 196]);  clone_175 = None
        mm_34 = torch.ops.aten.mm.default(view_208, permute_175);  view_208 = permute_175 = None
        view_209 = torch.ops.aten.view.default(mm_34, [8, 768, 196]);  mm_34 = None
        add_245 = torch.ops.aten.add.Tensor(view_209, arg50_1);  view_209 = arg50_1 = None
        permute_176 = torch.ops.aten.permute.default(add_245, [0, 2, 1]);  add_245 = None
        mul_281 = torch.ops.aten.mul.Tensor(getitem_208, permute_176);  getitem_208 = permute_176 = None
        view_210 = torch.ops.aten.view.default(mul_281, [1568, 768]);  mul_281 = None
        permute_177 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg52_1, view_210, permute_177);  arg52_1 = view_210 = permute_177 = None
        view_211 = torch.ops.aten.view.default(addmm_70, [8, 196, 256]);  addmm_70 = None
        add_246 = torch.ops.aten.add.Tensor(add_239, view_211);  add_239 = view_211 = None
        clone_177 = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
        getitem_212 = var_mean_71[0]
        getitem_213 = var_mean_71[1];  var_mean_71 = None
        add_247 = torch.ops.aten.add.Tensor(getitem_212, 1e-06);  getitem_212 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_71 = torch.ops.aten.sub.Tensor(clone_177, getitem_213);  clone_177 = getitem_213 = None
        mul_282 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        mul_283 = torch.ops.aten.mul.Tensor(mul_282, arg53_1);  mul_282 = arg53_1 = None
        add_248 = torch.ops.aten.add.Tensor(mul_283, arg54_1);  mul_283 = arg54_1 = None
        view_212 = torch.ops.aten.view.default(add_248, [1568, 256]);  add_248 = None
        permute_178 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg56_1, view_212, permute_178);  arg56_1 = view_212 = permute_178 = None
        view_213 = torch.ops.aten.view.default(addmm_71, [8, 196, 1536]);  addmm_71 = None
        mul_284 = torch.ops.aten.mul.Tensor(view_213, 0.5)
        mul_285 = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
        erf_35 = torch.ops.aten.erf.default(mul_285);  mul_285 = None
        add_249 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_284, add_249);  mul_284 = add_249 = None
        split_35 = torch.ops.aten.split.Tensor(mul_286, 768, -1);  mul_286 = None
        getitem_214 = split_35[0]
        getitem_215 = split_35[1];  split_35 = None
        clone_179 = torch.ops.aten.clone.default(getitem_215, memory_format = torch.contiguous_format);  getitem_215 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_179, [2], correction = 0, keepdim = True)
        getitem_216 = var_mean_72[0]
        getitem_217 = var_mean_72[1];  var_mean_72 = None
        add_250 = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_72 = torch.ops.aten.sub.Tensor(clone_179, getitem_217);  clone_179 = getitem_217 = None
        mul_287 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_287, arg57_1);  mul_287 = arg57_1 = None
        add_251 = torch.ops.aten.add.Tensor(mul_288, arg58_1);  mul_288 = arg58_1 = None
        permute_179 = torch.ops.aten.permute.default(add_251, [0, 2, 1]);  add_251 = None
        permute_180 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        clone_180 = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_214 = torch.ops.aten.view.default(clone_180, [6144, 196]);  clone_180 = None
        mm_35 = torch.ops.aten.mm.default(view_214, permute_180);  view_214 = permute_180 = None
        view_215 = torch.ops.aten.view.default(mm_35, [8, 768, 196]);  mm_35 = None
        add_252 = torch.ops.aten.add.Tensor(view_215, arg60_1);  view_215 = arg60_1 = None
        permute_181 = torch.ops.aten.permute.default(add_252, [0, 2, 1]);  add_252 = None
        mul_289 = torch.ops.aten.mul.Tensor(getitem_214, permute_181);  getitem_214 = permute_181 = None
        view_216 = torch.ops.aten.view.default(mul_289, [1568, 768]);  mul_289 = None
        permute_182 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg62_1, view_216, permute_182);  arg62_1 = view_216 = permute_182 = None
        view_217 = torch.ops.aten.view.default(addmm_72, [8, 196, 256]);  addmm_72 = None
        add_253 = torch.ops.aten.add.Tensor(add_246, view_217);  add_246 = view_217 = None
        clone_182 = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format)
        var_mean_73 = torch.ops.aten.var_mean.correction(clone_182, [2], correction = 0, keepdim = True)
        getitem_218 = var_mean_73[0]
        getitem_219 = var_mean_73[1];  var_mean_73 = None
        add_254 = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_73 = torch.ops.aten.sub.Tensor(clone_182, getitem_219);  clone_182 = getitem_219 = None
        mul_290 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, arg63_1);  mul_290 = arg63_1 = None
        add_255 = torch.ops.aten.add.Tensor(mul_291, arg64_1);  mul_291 = arg64_1 = None
        view_218 = torch.ops.aten.view.default(add_255, [1568, 256]);  add_255 = None
        permute_183 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg66_1, view_218, permute_183);  arg66_1 = view_218 = permute_183 = None
        view_219 = torch.ops.aten.view.default(addmm_73, [8, 196, 1536]);  addmm_73 = None
        mul_292 = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_293 = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_36 = torch.ops.aten.erf.default(mul_293);  mul_293 = None
        add_256 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_292, add_256);  mul_292 = add_256 = None
        split_36 = torch.ops.aten.split.Tensor(mul_294, 768, -1);  mul_294 = None
        getitem_220 = split_36[0]
        getitem_221 = split_36[1];  split_36 = None
        clone_184 = torch.ops.aten.clone.default(getitem_221, memory_format = torch.contiguous_format);  getitem_221 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(clone_184, [2], correction = 0, keepdim = True)
        getitem_222 = var_mean_74[0]
        getitem_223 = var_mean_74[1];  var_mean_74 = None
        add_257 = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_74 = torch.ops.aten.sub.Tensor(clone_184, getitem_223);  clone_184 = getitem_223 = None
        mul_295 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, arg67_1);  mul_295 = arg67_1 = None
        add_258 = torch.ops.aten.add.Tensor(mul_296, arg68_1);  mul_296 = arg68_1 = None
        permute_184 = torch.ops.aten.permute.default(add_258, [0, 2, 1]);  add_258 = None
        permute_185 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        clone_185 = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_220 = torch.ops.aten.view.default(clone_185, [6144, 196]);  clone_185 = None
        mm_36 = torch.ops.aten.mm.default(view_220, permute_185);  view_220 = permute_185 = None
        view_221 = torch.ops.aten.view.default(mm_36, [8, 768, 196]);  mm_36 = None
        add_259 = torch.ops.aten.add.Tensor(view_221, arg70_1);  view_221 = arg70_1 = None
        permute_186 = torch.ops.aten.permute.default(add_259, [0, 2, 1]);  add_259 = None
        mul_297 = torch.ops.aten.mul.Tensor(getitem_220, permute_186);  getitem_220 = permute_186 = None
        view_222 = torch.ops.aten.view.default(mul_297, [1568, 768]);  mul_297 = None
        permute_187 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg72_1, view_222, permute_187);  arg72_1 = view_222 = permute_187 = None
        view_223 = torch.ops.aten.view.default(addmm_74, [8, 196, 256]);  addmm_74 = None
        add_260 = torch.ops.aten.add.Tensor(add_253, view_223);  add_253 = view_223 = None
        clone_187 = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_187, [2], correction = 0, keepdim = True)
        getitem_224 = var_mean_75[0]
        getitem_225 = var_mean_75[1];  var_mean_75 = None
        add_261 = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_75 = torch.ops.aten.sub.Tensor(clone_187, getitem_225);  clone_187 = getitem_225 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, arg73_1);  mul_298 = arg73_1 = None
        add_262 = torch.ops.aten.add.Tensor(mul_299, arg74_1);  mul_299 = arg74_1 = None
        view_224 = torch.ops.aten.view.default(add_262, [1568, 256]);  add_262 = None
        permute_188 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg76_1, view_224, permute_188);  arg76_1 = view_224 = permute_188 = None
        view_225 = torch.ops.aten.view.default(addmm_75, [8, 196, 1536]);  addmm_75 = None
        mul_300 = torch.ops.aten.mul.Tensor(view_225, 0.5)
        mul_301 = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476);  view_225 = None
        erf_37 = torch.ops.aten.erf.default(mul_301);  mul_301 = None
        add_263 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_300, add_263);  mul_300 = add_263 = None
        split_37 = torch.ops.aten.split.Tensor(mul_302, 768, -1);  mul_302 = None
        getitem_226 = split_37[0]
        getitem_227 = split_37[1];  split_37 = None
        clone_189 = torch.ops.aten.clone.default(getitem_227, memory_format = torch.contiguous_format);  getitem_227 = None
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_189, [2], correction = 0, keepdim = True)
        getitem_228 = var_mean_76[0]
        getitem_229 = var_mean_76[1];  var_mean_76 = None
        add_264 = torch.ops.aten.add.Tensor(getitem_228, 1e-05);  getitem_228 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_76 = torch.ops.aten.sub.Tensor(clone_189, getitem_229);  clone_189 = getitem_229 = None
        mul_303 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_303, arg77_1);  mul_303 = arg77_1 = None
        add_265 = torch.ops.aten.add.Tensor(mul_304, arg78_1);  mul_304 = arg78_1 = None
        permute_189 = torch.ops.aten.permute.default(add_265, [0, 2, 1]);  add_265 = None
        permute_190 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        clone_190 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_226 = torch.ops.aten.view.default(clone_190, [6144, 196]);  clone_190 = None
        mm_37 = torch.ops.aten.mm.default(view_226, permute_190);  view_226 = permute_190 = None
        view_227 = torch.ops.aten.view.default(mm_37, [8, 768, 196]);  mm_37 = None
        add_266 = torch.ops.aten.add.Tensor(view_227, arg80_1);  view_227 = arg80_1 = None
        permute_191 = torch.ops.aten.permute.default(add_266, [0, 2, 1]);  add_266 = None
        mul_305 = torch.ops.aten.mul.Tensor(getitem_226, permute_191);  getitem_226 = permute_191 = None
        view_228 = torch.ops.aten.view.default(mul_305, [1568, 768]);  mul_305 = None
        permute_192 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg82_1, view_228, permute_192);  arg82_1 = view_228 = permute_192 = None
        view_229 = torch.ops.aten.view.default(addmm_76, [8, 196, 256]);  addmm_76 = None
        add_267 = torch.ops.aten.add.Tensor(add_260, view_229);  add_260 = view_229 = None
        clone_192 = torch.ops.aten.clone.default(add_267, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_192, [2], correction = 0, keepdim = True)
        getitem_230 = var_mean_77[0]
        getitem_231 = var_mean_77[1];  var_mean_77 = None
        add_268 = torch.ops.aten.add.Tensor(getitem_230, 1e-06);  getitem_230 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_77 = torch.ops.aten.sub.Tensor(clone_192, getitem_231);  clone_192 = getitem_231 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_306, arg83_1);  mul_306 = arg83_1 = None
        add_269 = torch.ops.aten.add.Tensor(mul_307, arg84_1);  mul_307 = arg84_1 = None
        view_230 = torch.ops.aten.view.default(add_269, [1568, 256]);  add_269 = None
        permute_193 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg86_1, view_230, permute_193);  arg86_1 = view_230 = permute_193 = None
        view_231 = torch.ops.aten.view.default(addmm_77, [8, 196, 1536]);  addmm_77 = None
        mul_308 = torch.ops.aten.mul.Tensor(view_231, 0.5)
        mul_309 = torch.ops.aten.mul.Tensor(view_231, 0.7071067811865476);  view_231 = None
        erf_38 = torch.ops.aten.erf.default(mul_309);  mul_309 = None
        add_270 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_308, add_270);  mul_308 = add_270 = None
        split_38 = torch.ops.aten.split.Tensor(mul_310, 768, -1);  mul_310 = None
        getitem_232 = split_38[0]
        getitem_233 = split_38[1];  split_38 = None
        clone_194 = torch.ops.aten.clone.default(getitem_233, memory_format = torch.contiguous_format);  getitem_233 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(clone_194, [2], correction = 0, keepdim = True)
        getitem_234 = var_mean_78[0]
        getitem_235 = var_mean_78[1];  var_mean_78 = None
        add_271 = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_78 = torch.ops.aten.sub.Tensor(clone_194, getitem_235);  clone_194 = getitem_235 = None
        mul_311 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_311, arg87_1);  mul_311 = arg87_1 = None
        add_272 = torch.ops.aten.add.Tensor(mul_312, arg88_1);  mul_312 = arg88_1 = None
        permute_194 = torch.ops.aten.permute.default(add_272, [0, 2, 1]);  add_272 = None
        permute_195 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        clone_195 = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_232 = torch.ops.aten.view.default(clone_195, [6144, 196]);  clone_195 = None
        mm_38 = torch.ops.aten.mm.default(view_232, permute_195);  view_232 = permute_195 = None
        view_233 = torch.ops.aten.view.default(mm_38, [8, 768, 196]);  mm_38 = None
        add_273 = torch.ops.aten.add.Tensor(view_233, arg90_1);  view_233 = arg90_1 = None
        permute_196 = torch.ops.aten.permute.default(add_273, [0, 2, 1]);  add_273 = None
        mul_313 = torch.ops.aten.mul.Tensor(getitem_232, permute_196);  getitem_232 = permute_196 = None
        view_234 = torch.ops.aten.view.default(mul_313, [1568, 768]);  mul_313 = None
        permute_197 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg92_1, view_234, permute_197);  arg92_1 = view_234 = permute_197 = None
        view_235 = torch.ops.aten.view.default(addmm_78, [8, 196, 256]);  addmm_78 = None
        add_274 = torch.ops.aten.add.Tensor(add_267, view_235);  add_267 = view_235 = None
        clone_197 = torch.ops.aten.clone.default(add_274, memory_format = torch.contiguous_format)
        var_mean_79 = torch.ops.aten.var_mean.correction(clone_197, [2], correction = 0, keepdim = True)
        getitem_236 = var_mean_79[0]
        getitem_237 = var_mean_79[1];  var_mean_79 = None
        add_275 = torch.ops.aten.add.Tensor(getitem_236, 1e-06);  getitem_236 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        sub_79 = torch.ops.aten.sub.Tensor(clone_197, getitem_237);  clone_197 = getitem_237 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, arg93_1);  mul_314 = arg93_1 = None
        add_276 = torch.ops.aten.add.Tensor(mul_315, arg94_1);  mul_315 = arg94_1 = None
        view_236 = torch.ops.aten.view.default(add_276, [1568, 256]);  add_276 = None
        permute_198 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg96_1, view_236, permute_198);  arg96_1 = view_236 = permute_198 = None
        view_237 = torch.ops.aten.view.default(addmm_79, [8, 196, 1536]);  addmm_79 = None
        mul_316 = torch.ops.aten.mul.Tensor(view_237, 0.5)
        mul_317 = torch.ops.aten.mul.Tensor(view_237, 0.7071067811865476);  view_237 = None
        erf_39 = torch.ops.aten.erf.default(mul_317);  mul_317 = None
        add_277 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_316, add_277);  mul_316 = add_277 = None
        split_39 = torch.ops.aten.split.Tensor(mul_318, 768, -1);  mul_318 = None
        getitem_238 = split_39[0]
        getitem_239 = split_39[1];  split_39 = None
        clone_199 = torch.ops.aten.clone.default(getitem_239, memory_format = torch.contiguous_format);  getitem_239 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_199, [2], correction = 0, keepdim = True)
        getitem_240 = var_mean_80[0]
        getitem_241 = var_mean_80[1];  var_mean_80 = None
        add_278 = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        sub_80 = torch.ops.aten.sub.Tensor(clone_199, getitem_241);  clone_199 = getitem_241 = None
        mul_319 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, arg97_1);  mul_319 = arg97_1 = None
        add_279 = torch.ops.aten.add.Tensor(mul_320, arg98_1);  mul_320 = arg98_1 = None
        permute_199 = torch.ops.aten.permute.default(add_279, [0, 2, 1]);  add_279 = None
        permute_200 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        clone_200 = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_238 = torch.ops.aten.view.default(clone_200, [6144, 196]);  clone_200 = None
        mm_39 = torch.ops.aten.mm.default(view_238, permute_200);  view_238 = permute_200 = None
        view_239 = torch.ops.aten.view.default(mm_39, [8, 768, 196]);  mm_39 = None
        add_280 = torch.ops.aten.add.Tensor(view_239, arg100_1);  view_239 = arg100_1 = None
        permute_201 = torch.ops.aten.permute.default(add_280, [0, 2, 1]);  add_280 = None
        mul_321 = torch.ops.aten.mul.Tensor(getitem_238, permute_201);  getitem_238 = permute_201 = None
        view_240 = torch.ops.aten.view.default(mul_321, [1568, 768]);  mul_321 = None
        permute_202 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg102_1, view_240, permute_202);  arg102_1 = view_240 = permute_202 = None
        view_241 = torch.ops.aten.view.default(addmm_80, [8, 196, 256]);  addmm_80 = None
        add_281 = torch.ops.aten.add.Tensor(add_274, view_241);  add_274 = view_241 = None
        clone_202 = torch.ops.aten.clone.default(add_281, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_202, [2], correction = 0, keepdim = True)
        getitem_242 = var_mean_81[0]
        getitem_243 = var_mean_81[1];  var_mean_81 = None
        add_282 = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_81 = torch.ops.aten.sub.Tensor(clone_202, getitem_243);  clone_202 = getitem_243 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, arg103_1);  mul_322 = arg103_1 = None
        add_283 = torch.ops.aten.add.Tensor(mul_323, arg104_1);  mul_323 = arg104_1 = None
        view_242 = torch.ops.aten.view.default(add_283, [1568, 256]);  add_283 = None
        permute_203 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg106_1, view_242, permute_203);  arg106_1 = view_242 = permute_203 = None
        view_243 = torch.ops.aten.view.default(addmm_81, [8, 196, 1536]);  addmm_81 = None
        mul_324 = torch.ops.aten.mul.Tensor(view_243, 0.5)
        mul_325 = torch.ops.aten.mul.Tensor(view_243, 0.7071067811865476);  view_243 = None
        erf_40 = torch.ops.aten.erf.default(mul_325);  mul_325 = None
        add_284 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_324, add_284);  mul_324 = add_284 = None
        split_40 = torch.ops.aten.split.Tensor(mul_326, 768, -1);  mul_326 = None
        getitem_244 = split_40[0]
        getitem_245 = split_40[1];  split_40 = None
        clone_204 = torch.ops.aten.clone.default(getitem_245, memory_format = torch.contiguous_format);  getitem_245 = None
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_204, [2], correction = 0, keepdim = True)
        getitem_246 = var_mean_82[0]
        getitem_247 = var_mean_82[1];  var_mean_82 = None
        add_285 = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        sub_82 = torch.ops.aten.sub.Tensor(clone_204, getitem_247);  clone_204 = getitem_247 = None
        mul_327 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_327, arg107_1);  mul_327 = arg107_1 = None
        add_286 = torch.ops.aten.add.Tensor(mul_328, arg108_1);  mul_328 = arg108_1 = None
        permute_204 = torch.ops.aten.permute.default(add_286, [0, 2, 1]);  add_286 = None
        permute_205 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        clone_205 = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_244 = torch.ops.aten.view.default(clone_205, [6144, 196]);  clone_205 = None
        mm_40 = torch.ops.aten.mm.default(view_244, permute_205);  view_244 = permute_205 = None
        view_245 = torch.ops.aten.view.default(mm_40, [8, 768, 196]);  mm_40 = None
        add_287 = torch.ops.aten.add.Tensor(view_245, arg110_1);  view_245 = arg110_1 = None
        permute_206 = torch.ops.aten.permute.default(add_287, [0, 2, 1]);  add_287 = None
        mul_329 = torch.ops.aten.mul.Tensor(getitem_244, permute_206);  getitem_244 = permute_206 = None
        view_246 = torch.ops.aten.view.default(mul_329, [1568, 768]);  mul_329 = None
        permute_207 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg112_1, view_246, permute_207);  arg112_1 = view_246 = permute_207 = None
        view_247 = torch.ops.aten.view.default(addmm_82, [8, 196, 256]);  addmm_82 = None
        add_288 = torch.ops.aten.add.Tensor(add_281, view_247);  add_281 = view_247 = None
        clone_207 = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
        var_mean_83 = torch.ops.aten.var_mean.correction(clone_207, [2], correction = 0, keepdim = True)
        getitem_248 = var_mean_83[0]
        getitem_249 = var_mean_83[1];  var_mean_83 = None
        add_289 = torch.ops.aten.add.Tensor(getitem_248, 1e-06);  getitem_248 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        sub_83 = torch.ops.aten.sub.Tensor(clone_207, getitem_249);  clone_207 = getitem_249 = None
        mul_330 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
        mul_331 = torch.ops.aten.mul.Tensor(mul_330, arg113_1);  mul_330 = arg113_1 = None
        add_290 = torch.ops.aten.add.Tensor(mul_331, arg114_1);  mul_331 = arg114_1 = None
        view_248 = torch.ops.aten.view.default(add_290, [1568, 256]);  add_290 = None
        permute_208 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg116_1, view_248, permute_208);  arg116_1 = view_248 = permute_208 = None
        view_249 = torch.ops.aten.view.default(addmm_83, [8, 196, 1536]);  addmm_83 = None
        mul_332 = torch.ops.aten.mul.Tensor(view_249, 0.5)
        mul_333 = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476);  view_249 = None
        erf_41 = torch.ops.aten.erf.default(mul_333);  mul_333 = None
        add_291 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_334 = torch.ops.aten.mul.Tensor(mul_332, add_291);  mul_332 = add_291 = None
        split_41 = torch.ops.aten.split.Tensor(mul_334, 768, -1);  mul_334 = None
        getitem_250 = split_41[0]
        getitem_251 = split_41[1];  split_41 = None
        clone_209 = torch.ops.aten.clone.default(getitem_251, memory_format = torch.contiguous_format);  getitem_251 = None
        var_mean_84 = torch.ops.aten.var_mean.correction(clone_209, [2], correction = 0, keepdim = True)
        getitem_252 = var_mean_84[0]
        getitem_253 = var_mean_84[1];  var_mean_84 = None
        add_292 = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_84 = torch.ops.aten.sub.Tensor(clone_209, getitem_253);  clone_209 = getitem_253 = None
        mul_335 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_335, arg117_1);  mul_335 = arg117_1 = None
        add_293 = torch.ops.aten.add.Tensor(mul_336, arg118_1);  mul_336 = arg118_1 = None
        permute_209 = torch.ops.aten.permute.default(add_293, [0, 2, 1]);  add_293 = None
        permute_210 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        clone_210 = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_250 = torch.ops.aten.view.default(clone_210, [6144, 196]);  clone_210 = None
        mm_41 = torch.ops.aten.mm.default(view_250, permute_210);  view_250 = permute_210 = None
        view_251 = torch.ops.aten.view.default(mm_41, [8, 768, 196]);  mm_41 = None
        add_294 = torch.ops.aten.add.Tensor(view_251, arg120_1);  view_251 = arg120_1 = None
        permute_211 = torch.ops.aten.permute.default(add_294, [0, 2, 1]);  add_294 = None
        mul_337 = torch.ops.aten.mul.Tensor(getitem_250, permute_211);  getitem_250 = permute_211 = None
        view_252 = torch.ops.aten.view.default(mul_337, [1568, 768]);  mul_337 = None
        permute_212 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg122_1, view_252, permute_212);  arg122_1 = view_252 = permute_212 = None
        view_253 = torch.ops.aten.view.default(addmm_84, [8, 196, 256]);  addmm_84 = None
        add_295 = torch.ops.aten.add.Tensor(add_288, view_253);  add_288 = view_253 = None
        clone_212 = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_212, [2], correction = 0, keepdim = True)
        getitem_254 = var_mean_85[0]
        getitem_255 = var_mean_85[1];  var_mean_85 = None
        add_296 = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_85 = torch.ops.aten.sub.Tensor(clone_212, getitem_255);  clone_212 = getitem_255 = None
        mul_338 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, arg123_1);  mul_338 = arg123_1 = None
        add_297 = torch.ops.aten.add.Tensor(mul_339, arg124_1);  mul_339 = arg124_1 = None
        view_254 = torch.ops.aten.view.default(add_297, [1568, 256]);  add_297 = None
        permute_213 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg126_1, view_254, permute_213);  arg126_1 = view_254 = permute_213 = None
        view_255 = torch.ops.aten.view.default(addmm_85, [8, 196, 1536]);  addmm_85 = None
        mul_340 = torch.ops.aten.mul.Tensor(view_255, 0.5)
        mul_341 = torch.ops.aten.mul.Tensor(view_255, 0.7071067811865476);  view_255 = None
        erf_42 = torch.ops.aten.erf.default(mul_341);  mul_341 = None
        add_298 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_340, add_298);  mul_340 = add_298 = None
        split_42 = torch.ops.aten.split.Tensor(mul_342, 768, -1);  mul_342 = None
        getitem_256 = split_42[0]
        getitem_257 = split_42[1];  split_42 = None
        clone_214 = torch.ops.aten.clone.default(getitem_257, memory_format = torch.contiguous_format);  getitem_257 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_214, [2], correction = 0, keepdim = True)
        getitem_258 = var_mean_86[0]
        getitem_259 = var_mean_86[1];  var_mean_86 = None
        add_299 = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_86 = torch.ops.aten.sub.Tensor(clone_214, getitem_259);  clone_214 = getitem_259 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, arg127_1);  mul_343 = arg127_1 = None
        add_300 = torch.ops.aten.add.Tensor(mul_344, arg128_1);  mul_344 = arg128_1 = None
        permute_214 = torch.ops.aten.permute.default(add_300, [0, 2, 1]);  add_300 = None
        permute_215 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        clone_215 = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_256 = torch.ops.aten.view.default(clone_215, [6144, 196]);  clone_215 = None
        mm_42 = torch.ops.aten.mm.default(view_256, permute_215);  view_256 = permute_215 = None
        view_257 = torch.ops.aten.view.default(mm_42, [8, 768, 196]);  mm_42 = None
        add_301 = torch.ops.aten.add.Tensor(view_257, arg130_1);  view_257 = arg130_1 = None
        permute_216 = torch.ops.aten.permute.default(add_301, [0, 2, 1]);  add_301 = None
        mul_345 = torch.ops.aten.mul.Tensor(getitem_256, permute_216);  getitem_256 = permute_216 = None
        view_258 = torch.ops.aten.view.default(mul_345, [1568, 768]);  mul_345 = None
        permute_217 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg132_1, view_258, permute_217);  arg132_1 = view_258 = permute_217 = None
        view_259 = torch.ops.aten.view.default(addmm_86, [8, 196, 256]);  addmm_86 = None
        add_302 = torch.ops.aten.add.Tensor(add_295, view_259);  add_295 = view_259 = None
        clone_217 = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_217, [2], correction = 0, keepdim = True)
        getitem_260 = var_mean_87[0]
        getitem_261 = var_mean_87[1];  var_mean_87 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_260, 1e-06);  getitem_260 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_87 = torch.ops.aten.sub.Tensor(clone_217, getitem_261);  clone_217 = getitem_261 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, arg133_1);  mul_346 = arg133_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_347, arg134_1);  mul_347 = arg134_1 = None
        view_260 = torch.ops.aten.view.default(add_304, [1568, 256]);  add_304 = None
        permute_218 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg136_1, view_260, permute_218);  arg136_1 = view_260 = permute_218 = None
        view_261 = torch.ops.aten.view.default(addmm_87, [8, 196, 1536]);  addmm_87 = None
        mul_348 = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_349 = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_43 = torch.ops.aten.erf.default(mul_349);  mul_349 = None
        add_305 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_348, add_305);  mul_348 = add_305 = None
        split_43 = torch.ops.aten.split.Tensor(mul_350, 768, -1);  mul_350 = None
        getitem_262 = split_43[0]
        getitem_263 = split_43[1];  split_43 = None
        clone_219 = torch.ops.aten.clone.default(getitem_263, memory_format = torch.contiguous_format);  getitem_263 = None
        var_mean_88 = torch.ops.aten.var_mean.correction(clone_219, [2], correction = 0, keepdim = True)
        getitem_264 = var_mean_88[0]
        getitem_265 = var_mean_88[1];  var_mean_88 = None
        add_306 = torch.ops.aten.add.Tensor(getitem_264, 1e-05);  getitem_264 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_88 = torch.ops.aten.sub.Tensor(clone_219, getitem_265);  clone_219 = getitem_265 = None
        mul_351 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        mul_352 = torch.ops.aten.mul.Tensor(mul_351, arg137_1);  mul_351 = arg137_1 = None
        add_307 = torch.ops.aten.add.Tensor(mul_352, arg138_1);  mul_352 = arg138_1 = None
        permute_219 = torch.ops.aten.permute.default(add_307, [0, 2, 1]);  add_307 = None
        permute_220 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        clone_220 = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_262 = torch.ops.aten.view.default(clone_220, [6144, 196]);  clone_220 = None
        mm_43 = torch.ops.aten.mm.default(view_262, permute_220);  view_262 = permute_220 = None
        view_263 = torch.ops.aten.view.default(mm_43, [8, 768, 196]);  mm_43 = None
        add_308 = torch.ops.aten.add.Tensor(view_263, arg140_1);  view_263 = arg140_1 = None
        permute_221 = torch.ops.aten.permute.default(add_308, [0, 2, 1]);  add_308 = None
        mul_353 = torch.ops.aten.mul.Tensor(getitem_262, permute_221);  getitem_262 = permute_221 = None
        view_264 = torch.ops.aten.view.default(mul_353, [1568, 768]);  mul_353 = None
        permute_222 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg142_1, view_264, permute_222);  arg142_1 = view_264 = permute_222 = None
        view_265 = torch.ops.aten.view.default(addmm_88, [8, 196, 256]);  addmm_88 = None
        add_309 = torch.ops.aten.add.Tensor(add_302, view_265);  add_302 = view_265 = None
        clone_222 = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_89 = torch.ops.aten.var_mean.correction(clone_222, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_89[0]
        getitem_267 = var_mean_89[1];  var_mean_89 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_89 = torch.ops.aten.sub.Tensor(clone_222, getitem_267);  clone_222 = getitem_267 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, arg143_1);  mul_354 = arg143_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_355, arg144_1);  mul_355 = arg144_1 = None
        view_266 = torch.ops.aten.view.default(add_311, [1568, 256]);  add_311 = None
        permute_223 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg146_1, view_266, permute_223);  arg146_1 = view_266 = permute_223 = None
        view_267 = torch.ops.aten.view.default(addmm_89, [8, 196, 1536]);  addmm_89 = None
        mul_356 = torch.ops.aten.mul.Tensor(view_267, 0.5)
        mul_357 = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
        erf_44 = torch.ops.aten.erf.default(mul_357);  mul_357 = None
        add_312 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_358 = torch.ops.aten.mul.Tensor(mul_356, add_312);  mul_356 = add_312 = None
        split_44 = torch.ops.aten.split.Tensor(mul_358, 768, -1);  mul_358 = None
        getitem_268 = split_44[0]
        getitem_269 = split_44[1];  split_44 = None
        clone_224 = torch.ops.aten.clone.default(getitem_269, memory_format = torch.contiguous_format);  getitem_269 = None
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_224, [2], correction = 0, keepdim = True)
        getitem_270 = var_mean_90[0]
        getitem_271 = var_mean_90[1];  var_mean_90 = None
        add_313 = torch.ops.aten.add.Tensor(getitem_270, 1e-05);  getitem_270 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_90 = torch.ops.aten.sub.Tensor(clone_224, getitem_271);  clone_224 = getitem_271 = None
        mul_359 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        mul_360 = torch.ops.aten.mul.Tensor(mul_359, arg147_1);  mul_359 = arg147_1 = None
        add_314 = torch.ops.aten.add.Tensor(mul_360, arg148_1);  mul_360 = arg148_1 = None
        permute_224 = torch.ops.aten.permute.default(add_314, [0, 2, 1]);  add_314 = None
        permute_225 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        clone_225 = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_268 = torch.ops.aten.view.default(clone_225, [6144, 196]);  clone_225 = None
        mm_44 = torch.ops.aten.mm.default(view_268, permute_225);  view_268 = permute_225 = None
        view_269 = torch.ops.aten.view.default(mm_44, [8, 768, 196]);  mm_44 = None
        add_315 = torch.ops.aten.add.Tensor(view_269, arg150_1);  view_269 = arg150_1 = None
        permute_226 = torch.ops.aten.permute.default(add_315, [0, 2, 1]);  add_315 = None
        mul_361 = torch.ops.aten.mul.Tensor(getitem_268, permute_226);  getitem_268 = permute_226 = None
        view_270 = torch.ops.aten.view.default(mul_361, [1568, 768]);  mul_361 = None
        permute_227 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg152_1, view_270, permute_227);  arg152_1 = view_270 = permute_227 = None
        view_271 = torch.ops.aten.view.default(addmm_90, [8, 196, 256]);  addmm_90 = None
        add_316 = torch.ops.aten.add.Tensor(add_309, view_271);  add_309 = view_271 = None
        clone_227 = torch.ops.aten.clone.default(add_316, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_227, [2], correction = 0, keepdim = True)
        getitem_272 = var_mean_91[0]
        getitem_273 = var_mean_91[1];  var_mean_91 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_272, 1e-06);  getitem_272 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_91 = torch.ops.aten.sub.Tensor(clone_227, getitem_273);  clone_227 = getitem_273 = None
        mul_362 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_362, arg153_1);  mul_362 = arg153_1 = None
        add_318 = torch.ops.aten.add.Tensor(mul_363, arg154_1);  mul_363 = arg154_1 = None
        view_272 = torch.ops.aten.view.default(add_318, [1568, 256]);  add_318 = None
        permute_228 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg156_1, view_272, permute_228);  arg156_1 = view_272 = permute_228 = None
        view_273 = torch.ops.aten.view.default(addmm_91, [8, 196, 1536]);  addmm_91 = None
        mul_364 = torch.ops.aten.mul.Tensor(view_273, 0.5)
        mul_365 = torch.ops.aten.mul.Tensor(view_273, 0.7071067811865476);  view_273 = None
        erf_45 = torch.ops.aten.erf.default(mul_365);  mul_365 = None
        add_319 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_364, add_319);  mul_364 = add_319 = None
        split_45 = torch.ops.aten.split.Tensor(mul_366, 768, -1);  mul_366 = None
        getitem_274 = split_45[0]
        getitem_275 = split_45[1];  split_45 = None
        clone_229 = torch.ops.aten.clone.default(getitem_275, memory_format = torch.contiguous_format);  getitem_275 = None
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_229, [2], correction = 0, keepdim = True)
        getitem_276 = var_mean_92[0]
        getitem_277 = var_mean_92[1];  var_mean_92 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_276, 1e-05);  getitem_276 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_92 = torch.ops.aten.sub.Tensor(clone_229, getitem_277);  clone_229 = getitem_277 = None
        mul_367 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, arg157_1);  mul_367 = arg157_1 = None
        add_321 = torch.ops.aten.add.Tensor(mul_368, arg158_1);  mul_368 = arg158_1 = None
        permute_229 = torch.ops.aten.permute.default(add_321, [0, 2, 1]);  add_321 = None
        permute_230 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        clone_230 = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        view_274 = torch.ops.aten.view.default(clone_230, [6144, 196]);  clone_230 = None
        mm_45 = torch.ops.aten.mm.default(view_274, permute_230);  view_274 = permute_230 = None
        view_275 = torch.ops.aten.view.default(mm_45, [8, 768, 196]);  mm_45 = None
        add_322 = torch.ops.aten.add.Tensor(view_275, arg160_1);  view_275 = arg160_1 = None
        permute_231 = torch.ops.aten.permute.default(add_322, [0, 2, 1]);  add_322 = None
        mul_369 = torch.ops.aten.mul.Tensor(getitem_274, permute_231);  getitem_274 = permute_231 = None
        view_276 = torch.ops.aten.view.default(mul_369, [1568, 768]);  mul_369 = None
        permute_232 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg162_1, view_276, permute_232);  arg162_1 = view_276 = permute_232 = None
        view_277 = torch.ops.aten.view.default(addmm_92, [8, 196, 256]);  addmm_92 = None
        add_323 = torch.ops.aten.add.Tensor(add_316, view_277);  add_316 = view_277 = None
        clone_232 = torch.ops.aten.clone.default(add_323, memory_format = torch.contiguous_format)
        var_mean_93 = torch.ops.aten.var_mean.correction(clone_232, [2], correction = 0, keepdim = True)
        getitem_278 = var_mean_93[0]
        getitem_279 = var_mean_93[1];  var_mean_93 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_93 = torch.ops.aten.sub.Tensor(clone_232, getitem_279);  clone_232 = getitem_279 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, arg163_1);  mul_370 = arg163_1 = None
        add_325 = torch.ops.aten.add.Tensor(mul_371, arg164_1);  mul_371 = arg164_1 = None
        view_278 = torch.ops.aten.view.default(add_325, [1568, 256]);  add_325 = None
        permute_233 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg166_1, view_278, permute_233);  arg166_1 = view_278 = permute_233 = None
        view_279 = torch.ops.aten.view.default(addmm_93, [8, 196, 1536]);  addmm_93 = None
        mul_372 = torch.ops.aten.mul.Tensor(view_279, 0.5)
        mul_373 = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476);  view_279 = None
        erf_46 = torch.ops.aten.erf.default(mul_373);  mul_373 = None
        add_326 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_372, add_326);  mul_372 = add_326 = None
        split_46 = torch.ops.aten.split.Tensor(mul_374, 768, -1);  mul_374 = None
        getitem_280 = split_46[0]
        getitem_281 = split_46[1];  split_46 = None
        clone_234 = torch.ops.aten.clone.default(getitem_281, memory_format = torch.contiguous_format);  getitem_281 = None
        var_mean_94 = torch.ops.aten.var_mean.correction(clone_234, [2], correction = 0, keepdim = True)
        getitem_282 = var_mean_94[0]
        getitem_283 = var_mean_94[1];  var_mean_94 = None
        add_327 = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        sub_94 = torch.ops.aten.sub.Tensor(clone_234, getitem_283);  clone_234 = getitem_283 = None
        mul_375 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        mul_376 = torch.ops.aten.mul.Tensor(mul_375, arg167_1);  mul_375 = arg167_1 = None
        add_328 = torch.ops.aten.add.Tensor(mul_376, arg168_1);  mul_376 = arg168_1 = None
        permute_234 = torch.ops.aten.permute.default(add_328, [0, 2, 1]);  add_328 = None
        permute_235 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        clone_235 = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_280 = torch.ops.aten.view.default(clone_235, [6144, 196]);  clone_235 = None
        mm_46 = torch.ops.aten.mm.default(view_280, permute_235);  view_280 = permute_235 = None
        view_281 = torch.ops.aten.view.default(mm_46, [8, 768, 196]);  mm_46 = None
        add_329 = torch.ops.aten.add.Tensor(view_281, arg170_1);  view_281 = arg170_1 = None
        permute_236 = torch.ops.aten.permute.default(add_329, [0, 2, 1]);  add_329 = None
        mul_377 = torch.ops.aten.mul.Tensor(getitem_280, permute_236);  getitem_280 = permute_236 = None
        view_282 = torch.ops.aten.view.default(mul_377, [1568, 768]);  mul_377 = None
        permute_237 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg172_1, view_282, permute_237);  arg172_1 = view_282 = permute_237 = None
        view_283 = torch.ops.aten.view.default(addmm_94, [8, 196, 256]);  addmm_94 = None
        add_330 = torch.ops.aten.add.Tensor(add_323, view_283);  add_323 = view_283 = None
        clone_237 = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_237, [2], correction = 0, keepdim = True)
        getitem_284 = var_mean_95[0]
        getitem_285 = var_mean_95[1];  var_mean_95 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_95 = torch.ops.aten.sub.Tensor(clone_237, getitem_285);  clone_237 = getitem_285 = None
        mul_378 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        mul_379 = torch.ops.aten.mul.Tensor(mul_378, arg173_1);  mul_378 = arg173_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_379, arg174_1);  mul_379 = arg174_1 = None
        view_284 = torch.ops.aten.view.default(add_332, [1568, 256]);  add_332 = None
        permute_238 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg176_1, view_284, permute_238);  arg176_1 = view_284 = permute_238 = None
        view_285 = torch.ops.aten.view.default(addmm_95, [8, 196, 1536]);  addmm_95 = None
        mul_380 = torch.ops.aten.mul.Tensor(view_285, 0.5)
        mul_381 = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
        erf_47 = torch.ops.aten.erf.default(mul_381);  mul_381 = None
        add_333 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_382 = torch.ops.aten.mul.Tensor(mul_380, add_333);  mul_380 = add_333 = None
        split_47 = torch.ops.aten.split.Tensor(mul_382, 768, -1);  mul_382 = None
        getitem_286 = split_47[0]
        getitem_287 = split_47[1];  split_47 = None
        clone_239 = torch.ops.aten.clone.default(getitem_287, memory_format = torch.contiguous_format);  getitem_287 = None
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
        getitem_288 = var_mean_96[0]
        getitem_289 = var_mean_96[1];  var_mean_96 = None
        add_334 = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_96 = torch.ops.aten.sub.Tensor(clone_239, getitem_289);  clone_239 = getitem_289 = None
        mul_383 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_383, arg177_1);  mul_383 = arg177_1 = None
        add_335 = torch.ops.aten.add.Tensor(mul_384, arg178_1);  mul_384 = arg178_1 = None
        permute_239 = torch.ops.aten.permute.default(add_335, [0, 2, 1]);  add_335 = None
        permute_240 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        clone_240 = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_286 = torch.ops.aten.view.default(clone_240, [6144, 196]);  clone_240 = None
        mm_47 = torch.ops.aten.mm.default(view_286, permute_240);  view_286 = permute_240 = None
        view_287 = torch.ops.aten.view.default(mm_47, [8, 768, 196]);  mm_47 = None
        add_336 = torch.ops.aten.add.Tensor(view_287, arg180_1);  view_287 = arg180_1 = None
        permute_241 = torch.ops.aten.permute.default(add_336, [0, 2, 1]);  add_336 = None
        mul_385 = torch.ops.aten.mul.Tensor(getitem_286, permute_241);  getitem_286 = permute_241 = None
        view_288 = torch.ops.aten.view.default(mul_385, [1568, 768]);  mul_385 = None
        permute_242 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg182_1, view_288, permute_242);  arg182_1 = view_288 = permute_242 = None
        view_289 = torch.ops.aten.view.default(addmm_96, [8, 196, 256]);  addmm_96 = None
        add_337 = torch.ops.aten.add.Tensor(add_330, view_289);  add_330 = view_289 = None
        clone_242 = torch.ops.aten.clone.default(add_337, memory_format = torch.contiguous_format)
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_242, [2], correction = 0, keepdim = True)
        getitem_290 = var_mean_97[0]
        getitem_291 = var_mean_97[1];  var_mean_97 = None
        add_338 = torch.ops.aten.add.Tensor(getitem_290, 1e-06);  getitem_290 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        sub_97 = torch.ops.aten.sub.Tensor(clone_242, getitem_291);  clone_242 = getitem_291 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, arg183_1);  mul_386 = arg183_1 = None
        add_339 = torch.ops.aten.add.Tensor(mul_387, arg184_1);  mul_387 = arg184_1 = None
        view_290 = torch.ops.aten.view.default(add_339, [1568, 256]);  add_339 = None
        permute_243 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg186_1, view_290, permute_243);  arg186_1 = view_290 = permute_243 = None
        view_291 = torch.ops.aten.view.default(addmm_97, [8, 196, 1536]);  addmm_97 = None
        mul_388 = torch.ops.aten.mul.Tensor(view_291, 0.5)
        mul_389 = torch.ops.aten.mul.Tensor(view_291, 0.7071067811865476);  view_291 = None
        erf_48 = torch.ops.aten.erf.default(mul_389);  mul_389 = None
        add_340 = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_388, add_340);  mul_388 = add_340 = None
        split_48 = torch.ops.aten.split.Tensor(mul_390, 768, -1);  mul_390 = None
        getitem_292 = split_48[0]
        getitem_293 = split_48[1];  split_48 = None
        clone_244 = torch.ops.aten.clone.default(getitem_293, memory_format = torch.contiguous_format);  getitem_293 = None
        var_mean_98 = torch.ops.aten.var_mean.correction(clone_244, [2], correction = 0, keepdim = True)
        getitem_294 = var_mean_98[0]
        getitem_295 = var_mean_98[1];  var_mean_98 = None
        add_341 = torch.ops.aten.add.Tensor(getitem_294, 1e-05);  getitem_294 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_98 = torch.ops.aten.sub.Tensor(clone_244, getitem_295);  clone_244 = getitem_295 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = rsqrt_98 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, arg187_1);  mul_391 = arg187_1 = None
        add_342 = torch.ops.aten.add.Tensor(mul_392, arg188_1);  mul_392 = arg188_1 = None
        permute_244 = torch.ops.aten.permute.default(add_342, [0, 2, 1]);  add_342 = None
        permute_245 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        clone_245 = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_292 = torch.ops.aten.view.default(clone_245, [6144, 196]);  clone_245 = None
        mm_48 = torch.ops.aten.mm.default(view_292, permute_245);  view_292 = permute_245 = None
        view_293 = torch.ops.aten.view.default(mm_48, [8, 768, 196]);  mm_48 = None
        add_343 = torch.ops.aten.add.Tensor(view_293, arg190_1);  view_293 = arg190_1 = None
        permute_246 = torch.ops.aten.permute.default(add_343, [0, 2, 1]);  add_343 = None
        mul_393 = torch.ops.aten.mul.Tensor(getitem_292, permute_246);  getitem_292 = permute_246 = None
        view_294 = torch.ops.aten.view.default(mul_393, [1568, 768]);  mul_393 = None
        permute_247 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg192_1, view_294, permute_247);  arg192_1 = view_294 = permute_247 = None
        view_295 = torch.ops.aten.view.default(addmm_98, [8, 196, 256]);  addmm_98 = None
        add_344 = torch.ops.aten.add.Tensor(add_337, view_295);  add_337 = view_295 = None
        clone_247 = torch.ops.aten.clone.default(add_344, memory_format = torch.contiguous_format)
        var_mean_99 = torch.ops.aten.var_mean.correction(clone_247, [2], correction = 0, keepdim = True)
        getitem_296 = var_mean_99[0]
        getitem_297 = var_mean_99[1];  var_mean_99 = None
        add_345 = torch.ops.aten.add.Tensor(getitem_296, 1e-06);  getitem_296 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
        sub_99 = torch.ops.aten.sub.Tensor(clone_247, getitem_297);  clone_247 = getitem_297 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = rsqrt_99 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, arg193_1);  mul_394 = arg193_1 = None
        add_346 = torch.ops.aten.add.Tensor(mul_395, arg194_1);  mul_395 = arg194_1 = None
        view_296 = torch.ops.aten.view.default(add_346, [1568, 256]);  add_346 = None
        permute_248 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg196_1, view_296, permute_248);  arg196_1 = view_296 = permute_248 = None
        view_297 = torch.ops.aten.view.default(addmm_99, [8, 196, 1536]);  addmm_99 = None
        mul_396 = torch.ops.aten.mul.Tensor(view_297, 0.5)
        mul_397 = torch.ops.aten.mul.Tensor(view_297, 0.7071067811865476);  view_297 = None
        erf_49 = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_347 = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_396, add_347);  mul_396 = add_347 = None
        split_49 = torch.ops.aten.split.Tensor(mul_398, 768, -1);  mul_398 = None
        getitem_298 = split_49[0]
        getitem_299 = split_49[1];  split_49 = None
        clone_249 = torch.ops.aten.clone.default(getitem_299, memory_format = torch.contiguous_format);  getitem_299 = None
        var_mean_100 = torch.ops.aten.var_mean.correction(clone_249, [2], correction = 0, keepdim = True)
        getitem_300 = var_mean_100[0]
        getitem_301 = var_mean_100[1];  var_mean_100 = None
        add_348 = torch.ops.aten.add.Tensor(getitem_300, 1e-05);  getitem_300 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        sub_100 = torch.ops.aten.sub.Tensor(clone_249, getitem_301);  clone_249 = getitem_301 = None
        mul_399 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = rsqrt_100 = None
        mul_400 = torch.ops.aten.mul.Tensor(mul_399, arg197_1);  mul_399 = arg197_1 = None
        add_349 = torch.ops.aten.add.Tensor(mul_400, arg198_1);  mul_400 = arg198_1 = None
        permute_249 = torch.ops.aten.permute.default(add_349, [0, 2, 1]);  add_349 = None
        permute_250 = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        clone_250 = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_298 = torch.ops.aten.view.default(clone_250, [6144, 196]);  clone_250 = None
        mm_49 = torch.ops.aten.mm.default(view_298, permute_250);  view_298 = permute_250 = None
        view_299 = torch.ops.aten.view.default(mm_49, [8, 768, 196]);  mm_49 = None
        add_350 = torch.ops.aten.add.Tensor(view_299, arg200_1);  view_299 = arg200_1 = None
        permute_251 = torch.ops.aten.permute.default(add_350, [0, 2, 1]);  add_350 = None
        mul_401 = torch.ops.aten.mul.Tensor(getitem_298, permute_251);  getitem_298 = permute_251 = None
        view_300 = torch.ops.aten.view.default(mul_401, [1568, 768]);  mul_401 = None
        permute_252 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg202_1, view_300, permute_252);  arg202_1 = view_300 = permute_252 = None
        view_301 = torch.ops.aten.view.default(addmm_100, [8, 196, 256]);  addmm_100 = None
        add_351 = torch.ops.aten.add.Tensor(add_344, view_301);  add_344 = view_301 = None
        clone_252 = torch.ops.aten.clone.default(add_351, memory_format = torch.contiguous_format)
        var_mean_101 = torch.ops.aten.var_mean.correction(clone_252, [2], correction = 0, keepdim = True)
        getitem_302 = var_mean_101[0]
        getitem_303 = var_mean_101[1];  var_mean_101 = None
        add_352 = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        sub_101 = torch.ops.aten.sub.Tensor(clone_252, getitem_303);  clone_252 = getitem_303 = None
        mul_402 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = rsqrt_101 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_402, arg203_1);  mul_402 = arg203_1 = None
        add_353 = torch.ops.aten.add.Tensor(mul_403, arg204_1);  mul_403 = arg204_1 = None
        view_302 = torch.ops.aten.view.default(add_353, [1568, 256]);  add_353 = None
        permute_253 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg206_1, view_302, permute_253);  arg206_1 = view_302 = permute_253 = None
        view_303 = torch.ops.aten.view.default(addmm_101, [8, 196, 1536]);  addmm_101 = None
        mul_404 = torch.ops.aten.mul.Tensor(view_303, 0.5)
        mul_405 = torch.ops.aten.mul.Tensor(view_303, 0.7071067811865476);  view_303 = None
        erf_50 = torch.ops.aten.erf.default(mul_405);  mul_405 = None
        add_354 = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_406 = torch.ops.aten.mul.Tensor(mul_404, add_354);  mul_404 = add_354 = None
        split_50 = torch.ops.aten.split.Tensor(mul_406, 768, -1);  mul_406 = None
        getitem_304 = split_50[0]
        getitem_305 = split_50[1];  split_50 = None
        clone_254 = torch.ops.aten.clone.default(getitem_305, memory_format = torch.contiguous_format);  getitem_305 = None
        var_mean_102 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
        getitem_306 = var_mean_102[0]
        getitem_307 = var_mean_102[1];  var_mean_102 = None
        add_355 = torch.ops.aten.add.Tensor(getitem_306, 1e-05);  getitem_306 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        sub_102 = torch.ops.aten.sub.Tensor(clone_254, getitem_307);  clone_254 = getitem_307 = None
        mul_407 = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = rsqrt_102 = None
        mul_408 = torch.ops.aten.mul.Tensor(mul_407, arg207_1);  mul_407 = arg207_1 = None
        add_356 = torch.ops.aten.add.Tensor(mul_408, arg208_1);  mul_408 = arg208_1 = None
        permute_254 = torch.ops.aten.permute.default(add_356, [0, 2, 1]);  add_356 = None
        permute_255 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        clone_255 = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_304 = torch.ops.aten.view.default(clone_255, [6144, 196]);  clone_255 = None
        mm_50 = torch.ops.aten.mm.default(view_304, permute_255);  view_304 = permute_255 = None
        view_305 = torch.ops.aten.view.default(mm_50, [8, 768, 196]);  mm_50 = None
        add_357 = torch.ops.aten.add.Tensor(view_305, arg210_1);  view_305 = arg210_1 = None
        permute_256 = torch.ops.aten.permute.default(add_357, [0, 2, 1]);  add_357 = None
        mul_409 = torch.ops.aten.mul.Tensor(getitem_304, permute_256);  getitem_304 = permute_256 = None
        view_306 = torch.ops.aten.view.default(mul_409, [1568, 768]);  mul_409 = None
        permute_257 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg212_1, view_306, permute_257);  arg212_1 = view_306 = permute_257 = None
        view_307 = torch.ops.aten.view.default(addmm_102, [8, 196, 256]);  addmm_102 = None
        add_358 = torch.ops.aten.add.Tensor(add_351, view_307);  add_351 = view_307 = None
        clone_257 = torch.ops.aten.clone.default(add_358, memory_format = torch.contiguous_format)
        var_mean_103 = torch.ops.aten.var_mean.correction(clone_257, [2], correction = 0, keepdim = True)
        getitem_308 = var_mean_103[0]
        getitem_309 = var_mean_103[1];  var_mean_103 = None
        add_359 = torch.ops.aten.add.Tensor(getitem_308, 1e-06);  getitem_308 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        sub_103 = torch.ops.aten.sub.Tensor(clone_257, getitem_309);  clone_257 = getitem_309 = None
        mul_410 = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = rsqrt_103 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_410, arg213_1);  mul_410 = arg213_1 = None
        add_360 = torch.ops.aten.add.Tensor(mul_411, arg214_1);  mul_411 = arg214_1 = None
        view_308 = torch.ops.aten.view.default(add_360, [1568, 256]);  add_360 = None
        permute_258 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg216_1, view_308, permute_258);  arg216_1 = view_308 = permute_258 = None
        view_309 = torch.ops.aten.view.default(addmm_103, [8, 196, 1536]);  addmm_103 = None
        mul_412 = torch.ops.aten.mul.Tensor(view_309, 0.5)
        mul_413 = torch.ops.aten.mul.Tensor(view_309, 0.7071067811865476);  view_309 = None
        erf_51 = torch.ops.aten.erf.default(mul_413);  mul_413 = None
        add_361 = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_414 = torch.ops.aten.mul.Tensor(mul_412, add_361);  mul_412 = add_361 = None
        split_51 = torch.ops.aten.split.Tensor(mul_414, 768, -1);  mul_414 = None
        getitem_310 = split_51[0]
        getitem_311 = split_51[1];  split_51 = None
        clone_259 = torch.ops.aten.clone.default(getitem_311, memory_format = torch.contiguous_format);  getitem_311 = None
        var_mean_104 = torch.ops.aten.var_mean.correction(clone_259, [2], correction = 0, keepdim = True)
        getitem_312 = var_mean_104[0]
        getitem_313 = var_mean_104[1];  var_mean_104 = None
        add_362 = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
        sub_104 = torch.ops.aten.sub.Tensor(clone_259, getitem_313);  clone_259 = getitem_313 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = rsqrt_104 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, arg217_1);  mul_415 = arg217_1 = None
        add_363 = torch.ops.aten.add.Tensor(mul_416, arg218_1);  mul_416 = arg218_1 = None
        permute_259 = torch.ops.aten.permute.default(add_363, [0, 2, 1]);  add_363 = None
        permute_260 = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        clone_260 = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_310 = torch.ops.aten.view.default(clone_260, [6144, 196]);  clone_260 = None
        mm_51 = torch.ops.aten.mm.default(view_310, permute_260);  view_310 = permute_260 = None
        view_311 = torch.ops.aten.view.default(mm_51, [8, 768, 196]);  mm_51 = None
        add_364 = torch.ops.aten.add.Tensor(view_311, arg220_1);  view_311 = arg220_1 = None
        permute_261 = torch.ops.aten.permute.default(add_364, [0, 2, 1]);  add_364 = None
        mul_417 = torch.ops.aten.mul.Tensor(getitem_310, permute_261);  getitem_310 = permute_261 = None
        view_312 = torch.ops.aten.view.default(mul_417, [1568, 768]);  mul_417 = None
        permute_262 = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg222_1, view_312, permute_262);  arg222_1 = view_312 = permute_262 = None
        view_313 = torch.ops.aten.view.default(addmm_104, [8, 196, 256]);  addmm_104 = None
        add_365 = torch.ops.aten.add.Tensor(add_358, view_313);  add_358 = view_313 = None
        clone_262 = torch.ops.aten.clone.default(add_365, memory_format = torch.contiguous_format)
        var_mean_105 = torch.ops.aten.var_mean.correction(clone_262, [2], correction = 0, keepdim = True)
        getitem_314 = var_mean_105[0]
        getitem_315 = var_mean_105[1];  var_mean_105 = None
        add_366 = torch.ops.aten.add.Tensor(getitem_314, 1e-06);  getitem_314 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
        sub_105 = torch.ops.aten.sub.Tensor(clone_262, getitem_315);  clone_262 = getitem_315 = None
        mul_418 = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = rsqrt_105 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_418, arg223_1);  mul_418 = arg223_1 = None
        add_367 = torch.ops.aten.add.Tensor(mul_419, arg224_1);  mul_419 = arg224_1 = None
        view_314 = torch.ops.aten.view.default(add_367, [1568, 256]);  add_367 = None
        permute_263 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg226_1, view_314, permute_263);  arg226_1 = view_314 = permute_263 = None
        view_315 = torch.ops.aten.view.default(addmm_105, [8, 196, 1536]);  addmm_105 = None
        mul_420 = torch.ops.aten.mul.Tensor(view_315, 0.5)
        mul_421 = torch.ops.aten.mul.Tensor(view_315, 0.7071067811865476);  view_315 = None
        erf_52 = torch.ops.aten.erf.default(mul_421);  mul_421 = None
        add_368 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_422 = torch.ops.aten.mul.Tensor(mul_420, add_368);  mul_420 = add_368 = None
        split_52 = torch.ops.aten.split.Tensor(mul_422, 768, -1);  mul_422 = None
        getitem_316 = split_52[0]
        getitem_317 = split_52[1];  split_52 = None
        clone_264 = torch.ops.aten.clone.default(getitem_317, memory_format = torch.contiguous_format);  getitem_317 = None
        var_mean_106 = torch.ops.aten.var_mean.correction(clone_264, [2], correction = 0, keepdim = True)
        getitem_318 = var_mean_106[0]
        getitem_319 = var_mean_106[1];  var_mean_106 = None
        add_369 = torch.ops.aten.add.Tensor(getitem_318, 1e-05);  getitem_318 = None
        rsqrt_106 = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
        sub_106 = torch.ops.aten.sub.Tensor(clone_264, getitem_319);  clone_264 = getitem_319 = None
        mul_423 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = rsqrt_106 = None
        mul_424 = torch.ops.aten.mul.Tensor(mul_423, arg227_1);  mul_423 = arg227_1 = None
        add_370 = torch.ops.aten.add.Tensor(mul_424, arg228_1);  mul_424 = arg228_1 = None
        permute_264 = torch.ops.aten.permute.default(add_370, [0, 2, 1]);  add_370 = None
        permute_265 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        clone_265 = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
        view_316 = torch.ops.aten.view.default(clone_265, [6144, 196]);  clone_265 = None
        mm_52 = torch.ops.aten.mm.default(view_316, permute_265);  view_316 = permute_265 = None
        view_317 = torch.ops.aten.view.default(mm_52, [8, 768, 196]);  mm_52 = None
        add_371 = torch.ops.aten.add.Tensor(view_317, arg230_1);  view_317 = arg230_1 = None
        permute_266 = torch.ops.aten.permute.default(add_371, [0, 2, 1]);  add_371 = None
        mul_425 = torch.ops.aten.mul.Tensor(getitem_316, permute_266);  getitem_316 = permute_266 = None
        view_318 = torch.ops.aten.view.default(mul_425, [1568, 768]);  mul_425 = None
        permute_267 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg232_1, view_318, permute_267);  arg232_1 = view_318 = permute_267 = None
        view_319 = torch.ops.aten.view.default(addmm_106, [8, 196, 256]);  addmm_106 = None
        add_372 = torch.ops.aten.add.Tensor(add_365, view_319);  add_365 = view_319 = None
        clone_267 = torch.ops.aten.clone.default(add_372, memory_format = torch.contiguous_format)
        var_mean_107 = torch.ops.aten.var_mean.correction(clone_267, [2], correction = 0, keepdim = True)
        getitem_320 = var_mean_107[0]
        getitem_321 = var_mean_107[1];  var_mean_107 = None
        add_373 = torch.ops.aten.add.Tensor(getitem_320, 1e-06);  getitem_320 = None
        rsqrt_107 = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
        sub_107 = torch.ops.aten.sub.Tensor(clone_267, getitem_321);  clone_267 = getitem_321 = None
        mul_426 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = rsqrt_107 = None
        mul_427 = torch.ops.aten.mul.Tensor(mul_426, arg233_1);  mul_426 = arg233_1 = None
        add_374 = torch.ops.aten.add.Tensor(mul_427, arg234_1);  mul_427 = arg234_1 = None
        view_320 = torch.ops.aten.view.default(add_374, [1568, 256]);  add_374 = None
        permute_268 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg236_1, view_320, permute_268);  arg236_1 = view_320 = permute_268 = None
        view_321 = torch.ops.aten.view.default(addmm_107, [8, 196, 1536]);  addmm_107 = None
        mul_428 = torch.ops.aten.mul.Tensor(view_321, 0.5)
        mul_429 = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476);  view_321 = None
        erf_53 = torch.ops.aten.erf.default(mul_429);  mul_429 = None
        add_375 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_430 = torch.ops.aten.mul.Tensor(mul_428, add_375);  mul_428 = add_375 = None
        split_53 = torch.ops.aten.split.Tensor(mul_430, 768, -1);  mul_430 = None
        getitem_322 = split_53[0]
        getitem_323 = split_53[1];  split_53 = None
        clone_269 = torch.ops.aten.clone.default(getitem_323, memory_format = torch.contiguous_format);  getitem_323 = None
        var_mean_108 = torch.ops.aten.var_mean.correction(clone_269, [2], correction = 0, keepdim = True)
        getitem_324 = var_mean_108[0]
        getitem_325 = var_mean_108[1];  var_mean_108 = None
        add_376 = torch.ops.aten.add.Tensor(getitem_324, 1e-05);  getitem_324 = None
        rsqrt_108 = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_108 = torch.ops.aten.sub.Tensor(clone_269, getitem_325);  clone_269 = getitem_325 = None
        mul_431 = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = rsqrt_108 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_431, arg237_1);  mul_431 = arg237_1 = None
        add_377 = torch.ops.aten.add.Tensor(mul_432, arg238_1);  mul_432 = arg238_1 = None
        permute_269 = torch.ops.aten.permute.default(add_377, [0, 2, 1]);  add_377 = None
        permute_270 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        clone_270 = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_322 = torch.ops.aten.view.default(clone_270, [6144, 196]);  clone_270 = None
        mm_53 = torch.ops.aten.mm.default(view_322, permute_270);  view_322 = permute_270 = None
        view_323 = torch.ops.aten.view.default(mm_53, [8, 768, 196]);  mm_53 = None
        add_378 = torch.ops.aten.add.Tensor(view_323, arg240_1);  view_323 = arg240_1 = None
        permute_271 = torch.ops.aten.permute.default(add_378, [0, 2, 1]);  add_378 = None
        mul_433 = torch.ops.aten.mul.Tensor(getitem_322, permute_271);  getitem_322 = permute_271 = None
        view_324 = torch.ops.aten.view.default(mul_433, [1568, 768]);  mul_433 = None
        permute_272 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg242_1, view_324, permute_272);  arg242_1 = view_324 = permute_272 = None
        view_325 = torch.ops.aten.view.default(addmm_108, [8, 196, 256]);  addmm_108 = None
        add_379 = torch.ops.aten.add.Tensor(add_372, view_325);  add_372 = view_325 = None
        clone_272 = torch.ops.aten.clone.default(add_379, memory_format = torch.contiguous_format)
        var_mean_109 = torch.ops.aten.var_mean.correction(clone_272, [2], correction = 0, keepdim = True)
        getitem_326 = var_mean_109[0]
        getitem_327 = var_mean_109[1];  var_mean_109 = None
        add_380 = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
        rsqrt_109 = torch.ops.aten.rsqrt.default(add_380);  add_380 = None
        sub_109 = torch.ops.aten.sub.Tensor(clone_272, getitem_327);  clone_272 = getitem_327 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = rsqrt_109 = None
        mul_435 = torch.ops.aten.mul.Tensor(mul_434, arg243_1);  mul_434 = arg243_1 = None
        add_381 = torch.ops.aten.add.Tensor(mul_435, arg244_1);  mul_435 = arg244_1 = None
        view_326 = torch.ops.aten.view.default(add_381, [1568, 256]);  add_381 = None
        permute_273 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg246_1, view_326, permute_273);  arg246_1 = view_326 = permute_273 = None
        view_327 = torch.ops.aten.view.default(addmm_109, [8, 196, 1536]);  addmm_109 = None
        mul_436 = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_437 = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_54 = torch.ops.aten.erf.default(mul_437);  mul_437 = None
        add_382 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_438 = torch.ops.aten.mul.Tensor(mul_436, add_382);  mul_436 = add_382 = None
        split_54 = torch.ops.aten.split.Tensor(mul_438, 768, -1);  mul_438 = None
        getitem_328 = split_54[0]
        getitem_329 = split_54[1];  split_54 = None
        clone_274 = torch.ops.aten.clone.default(getitem_329, memory_format = torch.contiguous_format);  getitem_329 = None
        var_mean_110 = torch.ops.aten.var_mean.correction(clone_274, [2], correction = 0, keepdim = True)
        getitem_330 = var_mean_110[0]
        getitem_331 = var_mean_110[1];  var_mean_110 = None
        add_383 = torch.ops.aten.add.Tensor(getitem_330, 1e-05);  getitem_330 = None
        rsqrt_110 = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
        sub_110 = torch.ops.aten.sub.Tensor(clone_274, getitem_331);  clone_274 = getitem_331 = None
        mul_439 = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = rsqrt_110 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_439, arg247_1);  mul_439 = arg247_1 = None
        add_384 = torch.ops.aten.add.Tensor(mul_440, arg248_1);  mul_440 = arg248_1 = None
        permute_274 = torch.ops.aten.permute.default(add_384, [0, 2, 1]);  add_384 = None
        permute_275 = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        clone_275 = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
        view_328 = torch.ops.aten.view.default(clone_275, [6144, 196]);  clone_275 = None
        mm_54 = torch.ops.aten.mm.default(view_328, permute_275);  view_328 = permute_275 = None
        view_329 = torch.ops.aten.view.default(mm_54, [8, 768, 196]);  mm_54 = None
        add_385 = torch.ops.aten.add.Tensor(view_329, arg250_1);  view_329 = arg250_1 = None
        permute_276 = torch.ops.aten.permute.default(add_385, [0, 2, 1]);  add_385 = None
        mul_441 = torch.ops.aten.mul.Tensor(getitem_328, permute_276);  getitem_328 = permute_276 = None
        view_330 = torch.ops.aten.view.default(mul_441, [1568, 768]);  mul_441 = None
        permute_277 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg252_1, view_330, permute_277);  arg252_1 = view_330 = permute_277 = None
        view_331 = torch.ops.aten.view.default(addmm_110, [8, 196, 256]);  addmm_110 = None
        add_386 = torch.ops.aten.add.Tensor(add_379, view_331);  add_379 = view_331 = None
        clone_277 = torch.ops.aten.clone.default(add_386, memory_format = torch.contiguous_format)
        var_mean_111 = torch.ops.aten.var_mean.correction(clone_277, [2], correction = 0, keepdim = True)
        getitem_332 = var_mean_111[0]
        getitem_333 = var_mean_111[1];  var_mean_111 = None
        add_387 = torch.ops.aten.add.Tensor(getitem_332, 1e-06);  getitem_332 = None
        rsqrt_111 = torch.ops.aten.rsqrt.default(add_387);  add_387 = None
        sub_111 = torch.ops.aten.sub.Tensor(clone_277, getitem_333);  clone_277 = getitem_333 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = rsqrt_111 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, arg253_1);  mul_442 = arg253_1 = None
        add_388 = torch.ops.aten.add.Tensor(mul_443, arg254_1);  mul_443 = arg254_1 = None
        view_332 = torch.ops.aten.view.default(add_388, [1568, 256]);  add_388 = None
        permute_278 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg256_1, view_332, permute_278);  arg256_1 = view_332 = permute_278 = None
        view_333 = torch.ops.aten.view.default(addmm_111, [8, 196, 1536]);  addmm_111 = None
        mul_444 = torch.ops.aten.mul.Tensor(view_333, 0.5)
        mul_445 = torch.ops.aten.mul.Tensor(view_333, 0.7071067811865476);  view_333 = None
        erf_55 = torch.ops.aten.erf.default(mul_445);  mul_445 = None
        add_389 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_444, add_389);  mul_444 = add_389 = None
        split_55 = torch.ops.aten.split.Tensor(mul_446, 768, -1);  mul_446 = None
        getitem_334 = split_55[0]
        getitem_335 = split_55[1];  split_55 = None
        clone_279 = torch.ops.aten.clone.default(getitem_335, memory_format = torch.contiguous_format);  getitem_335 = None
        var_mean_112 = torch.ops.aten.var_mean.correction(clone_279, [2], correction = 0, keepdim = True)
        getitem_336 = var_mean_112[0]
        getitem_337 = var_mean_112[1];  var_mean_112 = None
        add_390 = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_112 = torch.ops.aten.rsqrt.default(add_390);  add_390 = None
        sub_112 = torch.ops.aten.sub.Tensor(clone_279, getitem_337);  clone_279 = getitem_337 = None
        mul_447 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = rsqrt_112 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_447, arg257_1);  mul_447 = arg257_1 = None
        add_391 = torch.ops.aten.add.Tensor(mul_448, arg258_1);  mul_448 = arg258_1 = None
        permute_279 = torch.ops.aten.permute.default(add_391, [0, 2, 1]);  add_391 = None
        permute_280 = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        clone_280 = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_334 = torch.ops.aten.view.default(clone_280, [6144, 196]);  clone_280 = None
        mm_55 = torch.ops.aten.mm.default(view_334, permute_280);  view_334 = permute_280 = None
        view_335 = torch.ops.aten.view.default(mm_55, [8, 768, 196]);  mm_55 = None
        add_392 = torch.ops.aten.add.Tensor(view_335, arg260_1);  view_335 = arg260_1 = None
        permute_281 = torch.ops.aten.permute.default(add_392, [0, 2, 1]);  add_392 = None
        mul_449 = torch.ops.aten.mul.Tensor(getitem_334, permute_281);  getitem_334 = permute_281 = None
        view_336 = torch.ops.aten.view.default(mul_449, [1568, 768]);  mul_449 = None
        permute_282 = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg262_1, view_336, permute_282);  arg262_1 = view_336 = permute_282 = None
        view_337 = torch.ops.aten.view.default(addmm_112, [8, 196, 256]);  addmm_112 = None
        add_393 = torch.ops.aten.add.Tensor(add_386, view_337);  add_386 = view_337 = None
        clone_282 = torch.ops.aten.clone.default(add_393, memory_format = torch.contiguous_format)
        var_mean_113 = torch.ops.aten.var_mean.correction(clone_282, [2], correction = 0, keepdim = True)
        getitem_338 = var_mean_113[0]
        getitem_339 = var_mean_113[1];  var_mean_113 = None
        add_394 = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_113 = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
        sub_113 = torch.ops.aten.sub.Tensor(clone_282, getitem_339);  clone_282 = getitem_339 = None
        mul_450 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = rsqrt_113 = None
        mul_451 = torch.ops.aten.mul.Tensor(mul_450, arg263_1);  mul_450 = arg263_1 = None
        add_395 = torch.ops.aten.add.Tensor(mul_451, arg264_1);  mul_451 = arg264_1 = None
        view_338 = torch.ops.aten.view.default(add_395, [1568, 256]);  add_395 = None
        permute_283 = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg266_1, view_338, permute_283);  arg266_1 = view_338 = permute_283 = None
        view_339 = torch.ops.aten.view.default(addmm_113, [8, 196, 1536]);  addmm_113 = None
        mul_452 = torch.ops.aten.mul.Tensor(view_339, 0.5)
        mul_453 = torch.ops.aten.mul.Tensor(view_339, 0.7071067811865476);  view_339 = None
        erf_56 = torch.ops.aten.erf.default(mul_453);  mul_453 = None
        add_396 = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_454 = torch.ops.aten.mul.Tensor(mul_452, add_396);  mul_452 = add_396 = None
        split_56 = torch.ops.aten.split.Tensor(mul_454, 768, -1);  mul_454 = None
        getitem_340 = split_56[0]
        getitem_341 = split_56[1];  split_56 = None
        clone_284 = torch.ops.aten.clone.default(getitem_341, memory_format = torch.contiguous_format);  getitem_341 = None
        var_mean_114 = torch.ops.aten.var_mean.correction(clone_284, [2], correction = 0, keepdim = True)
        getitem_342 = var_mean_114[0]
        getitem_343 = var_mean_114[1];  var_mean_114 = None
        add_397 = torch.ops.aten.add.Tensor(getitem_342, 1e-05);  getitem_342 = None
        rsqrt_114 = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
        sub_114 = torch.ops.aten.sub.Tensor(clone_284, getitem_343);  clone_284 = getitem_343 = None
        mul_455 = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = rsqrt_114 = None
        mul_456 = torch.ops.aten.mul.Tensor(mul_455, arg267_1);  mul_455 = arg267_1 = None
        add_398 = torch.ops.aten.add.Tensor(mul_456, arg268_1);  mul_456 = arg268_1 = None
        permute_284 = torch.ops.aten.permute.default(add_398, [0, 2, 1]);  add_398 = None
        permute_285 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        clone_285 = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
        view_340 = torch.ops.aten.view.default(clone_285, [6144, 196]);  clone_285 = None
        mm_56 = torch.ops.aten.mm.default(view_340, permute_285);  view_340 = permute_285 = None
        view_341 = torch.ops.aten.view.default(mm_56, [8, 768, 196]);  mm_56 = None
        add_399 = torch.ops.aten.add.Tensor(view_341, arg270_1);  view_341 = arg270_1 = None
        permute_286 = torch.ops.aten.permute.default(add_399, [0, 2, 1]);  add_399 = None
        mul_457 = torch.ops.aten.mul.Tensor(getitem_340, permute_286);  getitem_340 = permute_286 = None
        view_342 = torch.ops.aten.view.default(mul_457, [1568, 768]);  mul_457 = None
        permute_287 = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg272_1, view_342, permute_287);  arg272_1 = view_342 = permute_287 = None
        view_343 = torch.ops.aten.view.default(addmm_114, [8, 196, 256]);  addmm_114 = None
        add_400 = torch.ops.aten.add.Tensor(add_393, view_343);  add_393 = view_343 = None
        clone_287 = torch.ops.aten.clone.default(add_400, memory_format = torch.contiguous_format)
        var_mean_115 = torch.ops.aten.var_mean.correction(clone_287, [2], correction = 0, keepdim = True)
        getitem_344 = var_mean_115[0]
        getitem_345 = var_mean_115[1];  var_mean_115 = None
        add_401 = torch.ops.aten.add.Tensor(getitem_344, 1e-06);  getitem_344 = None
        rsqrt_115 = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
        sub_115 = torch.ops.aten.sub.Tensor(clone_287, getitem_345);  clone_287 = getitem_345 = None
        mul_458 = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = rsqrt_115 = None
        mul_459 = torch.ops.aten.mul.Tensor(mul_458, arg273_1);  mul_458 = arg273_1 = None
        add_402 = torch.ops.aten.add.Tensor(mul_459, arg274_1);  mul_459 = arg274_1 = None
        view_344 = torch.ops.aten.view.default(add_402, [1568, 256]);  add_402 = None
        permute_288 = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg276_1, view_344, permute_288);  arg276_1 = view_344 = permute_288 = None
        view_345 = torch.ops.aten.view.default(addmm_115, [8, 196, 1536]);  addmm_115 = None
        mul_460 = torch.ops.aten.mul.Tensor(view_345, 0.5)
        mul_461 = torch.ops.aten.mul.Tensor(view_345, 0.7071067811865476);  view_345 = None
        erf_57 = torch.ops.aten.erf.default(mul_461);  mul_461 = None
        add_403 = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_462 = torch.ops.aten.mul.Tensor(mul_460, add_403);  mul_460 = add_403 = None
        split_57 = torch.ops.aten.split.Tensor(mul_462, 768, -1);  mul_462 = None
        getitem_346 = split_57[0]
        getitem_347 = split_57[1];  split_57 = None
        clone_289 = torch.ops.aten.clone.default(getitem_347, memory_format = torch.contiguous_format);  getitem_347 = None
        var_mean_116 = torch.ops.aten.var_mean.correction(clone_289, [2], correction = 0, keepdim = True)
        getitem_348 = var_mean_116[0]
        getitem_349 = var_mean_116[1];  var_mean_116 = None
        add_404 = torch.ops.aten.add.Tensor(getitem_348, 1e-05);  getitem_348 = None
        rsqrt_116 = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        sub_116 = torch.ops.aten.sub.Tensor(clone_289, getitem_349);  clone_289 = getitem_349 = None
        mul_463 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = rsqrt_116 = None
        mul_464 = torch.ops.aten.mul.Tensor(mul_463, arg277_1);  mul_463 = arg277_1 = None
        add_405 = torch.ops.aten.add.Tensor(mul_464, arg278_1);  mul_464 = arg278_1 = None
        permute_289 = torch.ops.aten.permute.default(add_405, [0, 2, 1]);  add_405 = None
        permute_290 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        clone_290 = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
        view_346 = torch.ops.aten.view.default(clone_290, [6144, 196]);  clone_290 = None
        mm_57 = torch.ops.aten.mm.default(view_346, permute_290);  view_346 = permute_290 = None
        view_347 = torch.ops.aten.view.default(mm_57, [8, 768, 196]);  mm_57 = None
        add_406 = torch.ops.aten.add.Tensor(view_347, arg280_1);  view_347 = arg280_1 = None
        permute_291 = torch.ops.aten.permute.default(add_406, [0, 2, 1]);  add_406 = None
        mul_465 = torch.ops.aten.mul.Tensor(getitem_346, permute_291);  getitem_346 = permute_291 = None
        view_348 = torch.ops.aten.view.default(mul_465, [1568, 768]);  mul_465 = None
        permute_292 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg282_1, view_348, permute_292);  arg282_1 = view_348 = permute_292 = None
        view_349 = torch.ops.aten.view.default(addmm_116, [8, 196, 256]);  addmm_116 = None
        add_407 = torch.ops.aten.add.Tensor(add_400, view_349);  add_400 = view_349 = None
        clone_292 = torch.ops.aten.clone.default(add_407, memory_format = torch.contiguous_format)
        var_mean_117 = torch.ops.aten.var_mean.correction(clone_292, [2], correction = 0, keepdim = True)
        getitem_350 = var_mean_117[0]
        getitem_351 = var_mean_117[1];  var_mean_117 = None
        add_408 = torch.ops.aten.add.Tensor(getitem_350, 1e-06);  getitem_350 = None
        rsqrt_117 = torch.ops.aten.rsqrt.default(add_408);  add_408 = None
        sub_117 = torch.ops.aten.sub.Tensor(clone_292, getitem_351);  clone_292 = getitem_351 = None
        mul_466 = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = rsqrt_117 = None
        mul_467 = torch.ops.aten.mul.Tensor(mul_466, arg283_1);  mul_466 = arg283_1 = None
        add_409 = torch.ops.aten.add.Tensor(mul_467, arg284_1);  mul_467 = arg284_1 = None
        view_350 = torch.ops.aten.view.default(add_409, [1568, 256]);  add_409 = None
        permute_293 = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg286_1, view_350, permute_293);  arg286_1 = view_350 = permute_293 = None
        view_351 = torch.ops.aten.view.default(addmm_117, [8, 196, 1536]);  addmm_117 = None
        mul_468 = torch.ops.aten.mul.Tensor(view_351, 0.5)
        mul_469 = torch.ops.aten.mul.Tensor(view_351, 0.7071067811865476);  view_351 = None
        erf_58 = torch.ops.aten.erf.default(mul_469);  mul_469 = None
        add_410 = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_470 = torch.ops.aten.mul.Tensor(mul_468, add_410);  mul_468 = add_410 = None
        split_58 = torch.ops.aten.split.Tensor(mul_470, 768, -1);  mul_470 = None
        getitem_352 = split_58[0]
        getitem_353 = split_58[1];  split_58 = None
        clone_294 = torch.ops.aten.clone.default(getitem_353, memory_format = torch.contiguous_format);  getitem_353 = None
        var_mean_118 = torch.ops.aten.var_mean.correction(clone_294, [2], correction = 0, keepdim = True)
        getitem_354 = var_mean_118[0]
        getitem_355 = var_mean_118[1];  var_mean_118 = None
        add_411 = torch.ops.aten.add.Tensor(getitem_354, 1e-05);  getitem_354 = None
        rsqrt_118 = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
        sub_118 = torch.ops.aten.sub.Tensor(clone_294, getitem_355);  clone_294 = getitem_355 = None
        mul_471 = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = rsqrt_118 = None
        mul_472 = torch.ops.aten.mul.Tensor(mul_471, arg287_1);  mul_471 = arg287_1 = None
        add_412 = torch.ops.aten.add.Tensor(mul_472, arg288_1);  mul_472 = arg288_1 = None
        permute_294 = torch.ops.aten.permute.default(add_412, [0, 2, 1]);  add_412 = None
        permute_295 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        clone_295 = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        view_352 = torch.ops.aten.view.default(clone_295, [6144, 196]);  clone_295 = None
        mm_58 = torch.ops.aten.mm.default(view_352, permute_295);  view_352 = permute_295 = None
        view_353 = torch.ops.aten.view.default(mm_58, [8, 768, 196]);  mm_58 = None
        add_413 = torch.ops.aten.add.Tensor(view_353, arg290_1);  view_353 = arg290_1 = None
        permute_296 = torch.ops.aten.permute.default(add_413, [0, 2, 1]);  add_413 = None
        mul_473 = torch.ops.aten.mul.Tensor(getitem_352, permute_296);  getitem_352 = permute_296 = None
        view_354 = torch.ops.aten.view.default(mul_473, [1568, 768]);  mul_473 = None
        permute_297 = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg292_1, view_354, permute_297);  arg292_1 = view_354 = permute_297 = None
        view_355 = torch.ops.aten.view.default(addmm_118, [8, 196, 256]);  addmm_118 = None
        add_414 = torch.ops.aten.add.Tensor(add_407, view_355);  add_407 = view_355 = None
        clone_297 = torch.ops.aten.clone.default(add_414, memory_format = torch.contiguous_format)
        var_mean_119 = torch.ops.aten.var_mean.correction(clone_297, [2], correction = 0, keepdim = True)
        getitem_356 = var_mean_119[0]
        getitem_357 = var_mean_119[1];  var_mean_119 = None
        add_415 = torch.ops.aten.add.Tensor(getitem_356, 1e-06);  getitem_356 = None
        rsqrt_119 = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
        sub_119 = torch.ops.aten.sub.Tensor(clone_297, getitem_357);  clone_297 = getitem_357 = None
        mul_474 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = rsqrt_119 = None
        mul_475 = torch.ops.aten.mul.Tensor(mul_474, arg293_1);  mul_474 = arg293_1 = None
        add_416 = torch.ops.aten.add.Tensor(mul_475, arg294_1);  mul_475 = arg294_1 = None
        view_356 = torch.ops.aten.view.default(add_416, [1568, 256]);  add_416 = None
        permute_298 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg296_1, view_356, permute_298);  arg296_1 = view_356 = permute_298 = None
        view_357 = torch.ops.aten.view.default(addmm_119, [8, 196, 1536]);  addmm_119 = None
        mul_476 = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_477 = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_59 = torch.ops.aten.erf.default(mul_477);  mul_477 = None
        add_417 = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_478 = torch.ops.aten.mul.Tensor(mul_476, add_417);  mul_476 = add_417 = None
        split_59 = torch.ops.aten.split.Tensor(mul_478, 768, -1);  mul_478 = None
        getitem_358 = split_59[0]
        getitem_359 = split_59[1];  split_59 = None
        clone_299 = torch.ops.aten.clone.default(getitem_359, memory_format = torch.contiguous_format);  getitem_359 = None
        var_mean_120 = torch.ops.aten.var_mean.correction(clone_299, [2], correction = 0, keepdim = True)
        getitem_360 = var_mean_120[0]
        getitem_361 = var_mean_120[1];  var_mean_120 = None
        add_418 = torch.ops.aten.add.Tensor(getitem_360, 1e-05);  getitem_360 = None
        rsqrt_120 = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        sub_120 = torch.ops.aten.sub.Tensor(clone_299, getitem_361);  clone_299 = getitem_361 = None
        mul_479 = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = rsqrt_120 = None
        mul_480 = torch.ops.aten.mul.Tensor(mul_479, arg297_1);  mul_479 = arg297_1 = None
        add_419 = torch.ops.aten.add.Tensor(mul_480, arg298_1);  mul_480 = arg298_1 = None
        permute_299 = torch.ops.aten.permute.default(add_419, [0, 2, 1]);  add_419 = None
        permute_300 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        clone_300 = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
        view_358 = torch.ops.aten.view.default(clone_300, [6144, 196]);  clone_300 = None
        mm_59 = torch.ops.aten.mm.default(view_358, permute_300);  view_358 = permute_300 = None
        view_359 = torch.ops.aten.view.default(mm_59, [8, 768, 196]);  mm_59 = None
        add_420 = torch.ops.aten.add.Tensor(view_359, arg300_1);  view_359 = arg300_1 = None
        permute_301 = torch.ops.aten.permute.default(add_420, [0, 2, 1]);  add_420 = None
        mul_481 = torch.ops.aten.mul.Tensor(getitem_358, permute_301);  getitem_358 = permute_301 = None
        view_360 = torch.ops.aten.view.default(mul_481, [1568, 768]);  mul_481 = None
        permute_302 = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg302_1, view_360, permute_302);  arg302_1 = view_360 = permute_302 = None
        view_361 = torch.ops.aten.view.default(addmm_120, [8, 196, 256]);  addmm_120 = None
        add_421 = torch.ops.aten.add.Tensor(add_414, view_361);  add_414 = view_361 = None
        clone_302 = torch.ops.aten.clone.default(add_421, memory_format = torch.contiguous_format);  add_421 = None
        var_mean_121 = torch.ops.aten.var_mean.correction(clone_302, [2], correction = 0, keepdim = True)
        getitem_362 = var_mean_121[0]
        getitem_363 = var_mean_121[1];  var_mean_121 = None
        add_422 = torch.ops.aten.add.Tensor(getitem_362, 1e-06);  getitem_362 = None
        rsqrt_121 = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        sub_121 = torch.ops.aten.sub.Tensor(clone_302, getitem_363);  clone_302 = getitem_363 = None
        mul_482 = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = rsqrt_121 = None
        mul_483 = torch.ops.aten.mul.Tensor(mul_482, arg303_1);  mul_482 = arg303_1 = None
        add_423 = torch.ops.aten.add.Tensor(mul_483, arg304_1);  mul_483 = arg304_1 = None
        mean_1 = torch.ops.aten.mean.dim(add_423, [1]);  add_423 = None
        permute_303 = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg306_1, mean_1, permute_303);  arg306_1 = mean_1 = permute_303 = None
        return (addmm_121,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf1, (256, 3, 16, 16), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf2, (256,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf3, (256,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (256,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1536, 256), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1536,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf9, (196, 196), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf10, (196,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf11, (256, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf13, (256,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf14, (256,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1536, 256), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1536,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf19, (196, 196), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf20, (196,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 768), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1536, 256), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1536,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf29, (196, 196), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf30, (196,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256, 768), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1536, 256), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1536,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf39, (196, 196), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf40, (196,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1536, 256), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1536,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf49, (196, 196), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf50, (196,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256, 768), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf53, (256,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf55, (1536, 256), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf56, (1536,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf59, (196, 196), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf60, (196,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf61, (256, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf62, (256,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf63, (256,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf64, (256,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf65, (1536, 256), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf66, (1536,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf69, (196, 196), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf70, (196,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf71, (256, 768), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf72, (256,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf74, (256,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf75, (1536, 256), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf76, (1536,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf79, (196, 196), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf80, (196,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf81, (256, 768), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf82, (256,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf83, (256,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf84, (256,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1536, 256), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf86, (1536,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf89, (196, 196), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf90, (196,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256, 768), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf92, (256,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf94, (256,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1536, 256), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1536,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf99, (196, 196), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf100, (196,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf101, (256, 768), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf102, (256,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf103, (256,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf104, (256,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1536, 256), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1536,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf109, (196, 196), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf110, (196,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256, 768), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf113, (256,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf114, (256,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1536, 256), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1536,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf119, (196, 196), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf120, (196,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1536, 256), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1536,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf129, (196, 196), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf130, (196,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf131, (256, 768), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf132, (256,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf133, (256,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1536, 256), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1536,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf139, (196, 196), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf140, (196,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf141, (256, 768), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf142, (256,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf144, (256,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1536, 256), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1536,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf149, (196, 196), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf150, (196,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf151, (256, 768), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf152, (256,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf153, (256,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf154, (256,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1536, 256), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1536,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf159, (196, 196), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf160, (196,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf161, (256, 768), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf162, (256,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf163, (256,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf164, (256,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf165, (1536, 256), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf166, (1536,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf169, (196, 196), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf170, (196,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf171, (256, 768), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf172, (256,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf173, (256,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf174, (256,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1536, 256), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf176, (1536,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf179, (196, 196), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf180, (196,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf181, (256, 768), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf182, (256,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf183, (256,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf184, (256,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1536, 256), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf186, (1536,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf189, (196, 196), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf190, (196,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf191, (256, 768), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf192, (256,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf193, (256,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf194, (256,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1536, 256), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1536,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf199, (196, 196), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf200, (196,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf201, (256, 768), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf202, (256,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf203, (256,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf204, (256,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1536, 256), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1536,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf207, (768,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf208, (768,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf209, (196, 196), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf210, (196,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf211, (256, 768), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf212, (256,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf213, (256,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf214, (256,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1536, 256), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1536,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf217, (768,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf218, (768,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf219, (196, 196), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf220, (196,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf221, (256, 768), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf222, (256,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf223, (256,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf224, (256,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf225, (1536, 256), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf226, (1536,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf227, (768,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf228, (768,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf229, (196, 196), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf230, (196,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf231, (256, 768), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf232, (256,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf233, (256,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf234, (256,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf235, (1536, 256), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf236, (1536,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf237, (768,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf238, (768,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf239, (196, 196), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf240, (196,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf241, (256, 768), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf242, (256,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf243, (256,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf244, (256,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1536, 256), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf246, (1536,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf247, (768,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf248, (768,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf249, (196, 196), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf250, (196,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf251, (256, 768), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf252, (256,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf253, (256,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf254, (256,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf255, (1536, 256), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1536,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf257, (768,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf258, (768,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf259, (196, 196), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf260, (196,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf261, (256, 768), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf262, (256,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf263, (256,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf264, (256,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf265, (1536, 256), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf266, (1536,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf267, (768,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf268, (768,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf269, (196, 196), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf270, (196,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf271, (256, 768), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf272, (256,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf273, (256,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf274, (256,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1536, 256), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1536,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf277, (768,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf278, (768,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf279, (196, 196), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf280, (196,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf281, (256, 768), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf282, (256,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf283, (256,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf284, (256,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf285, (1536, 256), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf286, (1536,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf287, (768,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf288, (768,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf289, (196, 196), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf290, (196,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf291, (256, 768), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf292, (256,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf293, (256,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf294, (256,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf295, (1536, 256), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf296, (1536,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf297, (768,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf298, (768,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf299, (196, 196), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf300, (196,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf301, (256, 768), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf302, (256,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf303, (256,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf304, (256,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1000, 256), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1000,), is_leaf=True)  # arg306_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)