
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1):
        convolution_40 = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        permute_155 = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None
        clone_74 = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(clone_74, [3], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_41 = torch.ops.aten.sub.Tensor(clone_74, getitem_83);  clone_74 = getitem_83 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, arg3_1);  mul_226 = arg3_1 = None
        add_155 = torch.ops.aten.add.Tensor(mul_227, arg4_1);  mul_227 = arg4_1 = None
        permute_156 = torch.ops.aten.permute.default(add_155, [0, 3, 1, 2]);  add_155 = None
        convolution_41 = torch.ops.aten.convolution.default(permute_156, arg5_1, arg6_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg5_1 = arg6_1 = None
        permute_157 = torch.ops.aten.permute.default(convolution_41, [0, 2, 3, 1]);  convolution_41 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(permute_157, [3], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_156 = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_42 = torch.ops.aten.sub.Tensor(permute_157, getitem_85);  permute_157 = getitem_85 = None
        mul_228 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_228, arg7_1);  mul_228 = arg7_1 = None
        add_157 = torch.ops.aten.add.Tensor(mul_229, arg8_1);  mul_229 = arg8_1 = None
        view_181 = torch.ops.aten.view.default(add_157, [41472, 128]);  add_157 = None
        permute_158 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg10_1, view_181, permute_158);  arg10_1 = view_181 = permute_158 = None
        view_182 = torch.ops.aten.view.default(addmm_73, [8, 72, 72, 512]);  addmm_73 = None
        mul_230 = torch.ops.aten.mul.Tensor(view_182, 0.5)
        mul_231 = torch.ops.aten.mul.Tensor(view_182, 0.7071067811865476);  view_182 = None
        erf_36 = torch.ops.aten.erf.default(mul_231);  mul_231 = None
        add_158 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_230, add_158);  mul_230 = add_158 = None
        view_183 = torch.ops.aten.view.default(mul_232, [41472, 512]);  mul_232 = None
        permute_159 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg12_1, view_183, permute_159);  arg12_1 = view_183 = permute_159 = None
        view_184 = torch.ops.aten.view.default(addmm_74, [8, 72, 72, 128]);  addmm_74 = None
        permute_160 = torch.ops.aten.permute.default(view_184, [0, 3, 1, 2]);  view_184 = None
        view_185 = torch.ops.aten.view.default(arg13_1, [1, -1, 1, 1]);  arg13_1 = None
        mul_233 = torch.ops.aten.mul.Tensor(permute_160, view_185);  permute_160 = view_185 = None
        add_159 = torch.ops.aten.add.Tensor(mul_233, permute_156);  mul_233 = permute_156 = None
        convolution_42 = torch.ops.aten.convolution.default(add_159, arg14_1, arg15_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg14_1 = arg15_1 = None
        permute_161 = torch.ops.aten.permute.default(convolution_42, [0, 2, 3, 1]);  convolution_42 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(permute_161, [3], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_160 = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_43 = torch.ops.aten.sub.Tensor(permute_161, getitem_87);  permute_161 = getitem_87 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, arg16_1);  mul_234 = arg16_1 = None
        add_161 = torch.ops.aten.add.Tensor(mul_235, arg17_1);  mul_235 = arg17_1 = None
        view_186 = torch.ops.aten.view.default(add_161, [41472, 128]);  add_161 = None
        permute_162 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg19_1, view_186, permute_162);  arg19_1 = view_186 = permute_162 = None
        view_187 = torch.ops.aten.view.default(addmm_75, [8, 72, 72, 512]);  addmm_75 = None
        mul_236 = torch.ops.aten.mul.Tensor(view_187, 0.5)
        mul_237 = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476);  view_187 = None
        erf_37 = torch.ops.aten.erf.default(mul_237);  mul_237 = None
        add_162 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_236, add_162);  mul_236 = add_162 = None
        view_188 = torch.ops.aten.view.default(mul_238, [41472, 512]);  mul_238 = None
        permute_163 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg21_1, view_188, permute_163);  arg21_1 = view_188 = permute_163 = None
        view_189 = torch.ops.aten.view.default(addmm_76, [8, 72, 72, 128]);  addmm_76 = None
        permute_164 = torch.ops.aten.permute.default(view_189, [0, 3, 1, 2]);  view_189 = None
        view_190 = torch.ops.aten.view.default(arg22_1, [1, -1, 1, 1]);  arg22_1 = None
        mul_239 = torch.ops.aten.mul.Tensor(permute_164, view_190);  permute_164 = view_190 = None
        add_163 = torch.ops.aten.add.Tensor(mul_239, add_159);  mul_239 = add_159 = None
        convolution_43 = torch.ops.aten.convolution.default(add_163, arg23_1, arg24_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg23_1 = arg24_1 = None
        permute_165 = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(permute_165, [3], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_164 = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_44 = torch.ops.aten.sub.Tensor(permute_165, getitem_89);  permute_165 = getitem_89 = None
        mul_240 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, arg25_1);  mul_240 = arg25_1 = None
        add_165 = torch.ops.aten.add.Tensor(mul_241, arg26_1);  mul_241 = arg26_1 = None
        view_191 = torch.ops.aten.view.default(add_165, [41472, 128]);  add_165 = None
        permute_166 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg28_1, view_191, permute_166);  arg28_1 = view_191 = permute_166 = None
        view_192 = torch.ops.aten.view.default(addmm_77, [8, 72, 72, 512]);  addmm_77 = None
        mul_242 = torch.ops.aten.mul.Tensor(view_192, 0.5)
        mul_243 = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476);  view_192 = None
        erf_38 = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_166 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_242, add_166);  mul_242 = add_166 = None
        view_193 = torch.ops.aten.view.default(mul_244, [41472, 512]);  mul_244 = None
        permute_167 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg30_1, view_193, permute_167);  arg30_1 = view_193 = permute_167 = None
        view_194 = torch.ops.aten.view.default(addmm_78, [8, 72, 72, 128]);  addmm_78 = None
        permute_168 = torch.ops.aten.permute.default(view_194, [0, 3, 1, 2]);  view_194 = None
        view_195 = torch.ops.aten.view.default(arg31_1, [1, -1, 1, 1]);  arg31_1 = None
        mul_245 = torch.ops.aten.mul.Tensor(permute_168, view_195);  permute_168 = view_195 = None
        add_167 = torch.ops.aten.add.Tensor(mul_245, add_163);  mul_245 = add_163 = None
        permute_169 = torch.ops.aten.permute.default(add_167, [0, 2, 3, 1]);  add_167 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(permute_169, [3], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_168 = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_45 = torch.ops.aten.sub.Tensor(permute_169, getitem_91);  permute_169 = getitem_91 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_246, arg32_1);  mul_246 = arg32_1 = None
        add_169 = torch.ops.aten.add.Tensor(mul_247, arg33_1);  mul_247 = arg33_1 = None
        permute_170 = torch.ops.aten.permute.default(add_169, [0, 3, 1, 2]);  add_169 = None
        convolution_44 = torch.ops.aten.convolution.default(permute_170, arg34_1, arg35_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_170 = arg34_1 = arg35_1 = None
        convolution_45 = torch.ops.aten.convolution.default(convolution_44, arg36_1, arg37_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg36_1 = arg37_1 = None
        permute_171 = torch.ops.aten.permute.default(convolution_45, [0, 2, 3, 1]);  convolution_45 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(permute_171, [3], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_46 = torch.ops.aten.sub.Tensor(permute_171, getitem_93);  permute_171 = getitem_93 = None
        mul_248 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, arg38_1);  mul_248 = arg38_1 = None
        add_171 = torch.ops.aten.add.Tensor(mul_249, arg39_1);  mul_249 = arg39_1 = None
        view_196 = torch.ops.aten.view.default(add_171, [10368, 256]);  add_171 = None
        permute_172 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg41_1, view_196, permute_172);  arg41_1 = view_196 = permute_172 = None
        view_197 = torch.ops.aten.view.default(addmm_79, [8, 36, 36, 1024]);  addmm_79 = None
        mul_250 = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_251 = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_39 = torch.ops.aten.erf.default(mul_251);  mul_251 = None
        add_172 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_250, add_172);  mul_250 = add_172 = None
        view_198 = torch.ops.aten.view.default(mul_252, [10368, 1024]);  mul_252 = None
        permute_173 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg43_1, view_198, permute_173);  arg43_1 = view_198 = permute_173 = None
        view_199 = torch.ops.aten.view.default(addmm_80, [8, 36, 36, 256]);  addmm_80 = None
        permute_174 = torch.ops.aten.permute.default(view_199, [0, 3, 1, 2]);  view_199 = None
        view_200 = torch.ops.aten.view.default(arg44_1, [1, -1, 1, 1]);  arg44_1 = None
        mul_253 = torch.ops.aten.mul.Tensor(permute_174, view_200);  permute_174 = view_200 = None
        add_173 = torch.ops.aten.add.Tensor(mul_253, convolution_44);  mul_253 = convolution_44 = None
        convolution_46 = torch.ops.aten.convolution.default(add_173, arg45_1, arg46_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg45_1 = arg46_1 = None
        permute_175 = torch.ops.aten.permute.default(convolution_46, [0, 2, 3, 1]);  convolution_46 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(permute_175, [3], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_47 = torch.ops.aten.sub.Tensor(permute_175, getitem_95);  permute_175 = getitem_95 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, arg47_1);  mul_254 = arg47_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_255, arg48_1);  mul_255 = arg48_1 = None
        view_201 = torch.ops.aten.view.default(add_175, [10368, 256]);  add_175 = None
        permute_176 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg50_1, view_201, permute_176);  arg50_1 = view_201 = permute_176 = None
        view_202 = torch.ops.aten.view.default(addmm_81, [8, 36, 36, 1024]);  addmm_81 = None
        mul_256 = torch.ops.aten.mul.Tensor(view_202, 0.5)
        mul_257 = torch.ops.aten.mul.Tensor(view_202, 0.7071067811865476);  view_202 = None
        erf_40 = torch.ops.aten.erf.default(mul_257);  mul_257 = None
        add_176 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_256, add_176);  mul_256 = add_176 = None
        view_203 = torch.ops.aten.view.default(mul_258, [10368, 1024]);  mul_258 = None
        permute_177 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg52_1, view_203, permute_177);  arg52_1 = view_203 = permute_177 = None
        view_204 = torch.ops.aten.view.default(addmm_82, [8, 36, 36, 256]);  addmm_82 = None
        permute_178 = torch.ops.aten.permute.default(view_204, [0, 3, 1, 2]);  view_204 = None
        view_205 = torch.ops.aten.view.default(arg53_1, [1, -1, 1, 1]);  arg53_1 = None
        mul_259 = torch.ops.aten.mul.Tensor(permute_178, view_205);  permute_178 = view_205 = None
        add_177 = torch.ops.aten.add.Tensor(mul_259, add_173);  mul_259 = add_173 = None
        convolution_47 = torch.ops.aten.convolution.default(add_177, arg54_1, arg55_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg54_1 = arg55_1 = None
        permute_179 = torch.ops.aten.permute.default(convolution_47, [0, 2, 3, 1]);  convolution_47 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(permute_179, [3], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_178 = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_48 = torch.ops.aten.sub.Tensor(permute_179, getitem_97);  permute_179 = getitem_97 = None
        mul_260 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, arg56_1);  mul_260 = arg56_1 = None
        add_179 = torch.ops.aten.add.Tensor(mul_261, arg57_1);  mul_261 = arg57_1 = None
        view_206 = torch.ops.aten.view.default(add_179, [10368, 256]);  add_179 = None
        permute_180 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg59_1, view_206, permute_180);  arg59_1 = view_206 = permute_180 = None
        view_207 = torch.ops.aten.view.default(addmm_83, [8, 36, 36, 1024]);  addmm_83 = None
        mul_262 = torch.ops.aten.mul.Tensor(view_207, 0.5)
        mul_263 = torch.ops.aten.mul.Tensor(view_207, 0.7071067811865476);  view_207 = None
        erf_41 = torch.ops.aten.erf.default(mul_263);  mul_263 = None
        add_180 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_262, add_180);  mul_262 = add_180 = None
        view_208 = torch.ops.aten.view.default(mul_264, [10368, 1024]);  mul_264 = None
        permute_181 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg61_1, view_208, permute_181);  arg61_1 = view_208 = permute_181 = None
        view_209 = torch.ops.aten.view.default(addmm_84, [8, 36, 36, 256]);  addmm_84 = None
        permute_182 = torch.ops.aten.permute.default(view_209, [0, 3, 1, 2]);  view_209 = None
        view_210 = torch.ops.aten.view.default(arg62_1, [1, -1, 1, 1]);  arg62_1 = None
        mul_265 = torch.ops.aten.mul.Tensor(permute_182, view_210);  permute_182 = view_210 = None
        add_181 = torch.ops.aten.add.Tensor(mul_265, add_177);  mul_265 = add_177 = None
        permute_183 = torch.ops.aten.permute.default(add_181, [0, 2, 3, 1]);  add_181 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(permute_183, [3], correction = 0, keepdim = True)
        getitem_98 = var_mean_49[0]
        getitem_99 = var_mean_49[1];  var_mean_49 = None
        add_182 = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_49 = torch.ops.aten.sub.Tensor(permute_183, getitem_99);  permute_183 = getitem_99 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, arg63_1);  mul_266 = arg63_1 = None
        add_183 = torch.ops.aten.add.Tensor(mul_267, arg64_1);  mul_267 = arg64_1 = None
        permute_184 = torch.ops.aten.permute.default(add_183, [0, 3, 1, 2]);  add_183 = None
        convolution_48 = torch.ops.aten.convolution.default(permute_184, arg65_1, arg66_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_184 = arg65_1 = arg66_1 = None
        convolution_49 = torch.ops.aten.convolution.default(convolution_48, arg67_1, arg68_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg67_1 = arg68_1 = None
        permute_185 = torch.ops.aten.permute.default(convolution_49, [0, 2, 3, 1]);  convolution_49 = None
        var_mean_50 = torch.ops.aten.var_mean.correction(permute_185, [3], correction = 0, keepdim = True)
        getitem_100 = var_mean_50[0]
        getitem_101 = var_mean_50[1];  var_mean_50 = None
        add_184 = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        sub_50 = torch.ops.aten.sub.Tensor(permute_185, getitem_101);  permute_185 = getitem_101 = None
        mul_268 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, arg69_1);  mul_268 = arg69_1 = None
        add_185 = torch.ops.aten.add.Tensor(mul_269, arg70_1);  mul_269 = arg70_1 = None
        view_211 = torch.ops.aten.view.default(add_185, [2592, 512]);  add_185 = None
        permute_186 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg72_1, view_211, permute_186);  arg72_1 = view_211 = permute_186 = None
        view_212 = torch.ops.aten.view.default(addmm_85, [8, 18, 18, 2048]);  addmm_85 = None
        mul_270 = torch.ops.aten.mul.Tensor(view_212, 0.5)
        mul_271 = torch.ops.aten.mul.Tensor(view_212, 0.7071067811865476);  view_212 = None
        erf_42 = torch.ops.aten.erf.default(mul_271);  mul_271 = None
        add_186 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_270, add_186);  mul_270 = add_186 = None
        view_213 = torch.ops.aten.view.default(mul_272, [2592, 2048]);  mul_272 = None
        permute_187 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg74_1, view_213, permute_187);  arg74_1 = view_213 = permute_187 = None
        view_214 = torch.ops.aten.view.default(addmm_86, [8, 18, 18, 512]);  addmm_86 = None
        permute_188 = torch.ops.aten.permute.default(view_214, [0, 3, 1, 2]);  view_214 = None
        view_215 = torch.ops.aten.view.default(arg75_1, [1, -1, 1, 1]);  arg75_1 = None
        mul_273 = torch.ops.aten.mul.Tensor(permute_188, view_215);  permute_188 = view_215 = None
        add_187 = torch.ops.aten.add.Tensor(mul_273, convolution_48);  mul_273 = convolution_48 = None
        convolution_50 = torch.ops.aten.convolution.default(add_187, arg76_1, arg77_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg76_1 = arg77_1 = None
        permute_189 = torch.ops.aten.permute.default(convolution_50, [0, 2, 3, 1]);  convolution_50 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(permute_189, [3], correction = 0, keepdim = True)
        getitem_102 = var_mean_51[0]
        getitem_103 = var_mean_51[1];  var_mean_51 = None
        add_188 = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_51 = torch.ops.aten.sub.Tensor(permute_189, getitem_103);  permute_189 = getitem_103 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, arg78_1);  mul_274 = arg78_1 = None
        add_189 = torch.ops.aten.add.Tensor(mul_275, arg79_1);  mul_275 = arg79_1 = None
        view_216 = torch.ops.aten.view.default(add_189, [2592, 512]);  add_189 = None
        permute_190 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg81_1, view_216, permute_190);  arg81_1 = view_216 = permute_190 = None
        view_217 = torch.ops.aten.view.default(addmm_87, [8, 18, 18, 2048]);  addmm_87 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_277 = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_43 = torch.ops.aten.erf.default(mul_277);  mul_277 = None
        add_190 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_276, add_190);  mul_276 = add_190 = None
        view_218 = torch.ops.aten.view.default(mul_278, [2592, 2048]);  mul_278 = None
        permute_191 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg83_1, view_218, permute_191);  arg83_1 = view_218 = permute_191 = None
        view_219 = torch.ops.aten.view.default(addmm_88, [8, 18, 18, 512]);  addmm_88 = None
        permute_192 = torch.ops.aten.permute.default(view_219, [0, 3, 1, 2]);  view_219 = None
        view_220 = torch.ops.aten.view.default(arg84_1, [1, -1, 1, 1]);  arg84_1 = None
        mul_279 = torch.ops.aten.mul.Tensor(permute_192, view_220);  permute_192 = view_220 = None
        add_191 = torch.ops.aten.add.Tensor(mul_279, add_187);  mul_279 = add_187 = None
        convolution_51 = torch.ops.aten.convolution.default(add_191, arg85_1, arg86_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg85_1 = arg86_1 = None
        permute_193 = torch.ops.aten.permute.default(convolution_51, [0, 2, 3, 1]);  convolution_51 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(permute_193, [3], correction = 0, keepdim = True)
        getitem_104 = var_mean_52[0]
        getitem_105 = var_mean_52[1];  var_mean_52 = None
        add_192 = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_52 = torch.ops.aten.sub.Tensor(permute_193, getitem_105);  permute_193 = getitem_105 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, arg87_1);  mul_280 = arg87_1 = None
        add_193 = torch.ops.aten.add.Tensor(mul_281, arg88_1);  mul_281 = arg88_1 = None
        view_221 = torch.ops.aten.view.default(add_193, [2592, 512]);  add_193 = None
        permute_194 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg90_1, view_221, permute_194);  arg90_1 = view_221 = permute_194 = None
        view_222 = torch.ops.aten.view.default(addmm_89, [8, 18, 18, 2048]);  addmm_89 = None
        mul_282 = torch.ops.aten.mul.Tensor(view_222, 0.5)
        mul_283 = torch.ops.aten.mul.Tensor(view_222, 0.7071067811865476);  view_222 = None
        erf_44 = torch.ops.aten.erf.default(mul_283);  mul_283 = None
        add_194 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_282, add_194);  mul_282 = add_194 = None
        view_223 = torch.ops.aten.view.default(mul_284, [2592, 2048]);  mul_284 = None
        permute_195 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg92_1, view_223, permute_195);  arg92_1 = view_223 = permute_195 = None
        view_224 = torch.ops.aten.view.default(addmm_90, [8, 18, 18, 512]);  addmm_90 = None
        permute_196 = torch.ops.aten.permute.default(view_224, [0, 3, 1, 2]);  view_224 = None
        view_225 = torch.ops.aten.view.default(arg93_1, [1, -1, 1, 1]);  arg93_1 = None
        mul_285 = torch.ops.aten.mul.Tensor(permute_196, view_225);  permute_196 = view_225 = None
        add_195 = torch.ops.aten.add.Tensor(mul_285, add_191);  mul_285 = add_191 = None
        convolution_52 = torch.ops.aten.convolution.default(add_195, arg94_1, arg95_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg94_1 = arg95_1 = None
        permute_197 = torch.ops.aten.permute.default(convolution_52, [0, 2, 3, 1]);  convolution_52 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(permute_197, [3], correction = 0, keepdim = True)
        getitem_106 = var_mean_53[0]
        getitem_107 = var_mean_53[1];  var_mean_53 = None
        add_196 = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_53 = torch.ops.aten.sub.Tensor(permute_197, getitem_107);  permute_197 = getitem_107 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, arg96_1);  mul_286 = arg96_1 = None
        add_197 = torch.ops.aten.add.Tensor(mul_287, arg97_1);  mul_287 = arg97_1 = None
        view_226 = torch.ops.aten.view.default(add_197, [2592, 512]);  add_197 = None
        permute_198 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg99_1, view_226, permute_198);  arg99_1 = view_226 = permute_198 = None
        view_227 = torch.ops.aten.view.default(addmm_91, [8, 18, 18, 2048]);  addmm_91 = None
        mul_288 = torch.ops.aten.mul.Tensor(view_227, 0.5)
        mul_289 = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476);  view_227 = None
        erf_45 = torch.ops.aten.erf.default(mul_289);  mul_289 = None
        add_198 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_288, add_198);  mul_288 = add_198 = None
        view_228 = torch.ops.aten.view.default(mul_290, [2592, 2048]);  mul_290 = None
        permute_199 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg101_1, view_228, permute_199);  arg101_1 = view_228 = permute_199 = None
        view_229 = torch.ops.aten.view.default(addmm_92, [8, 18, 18, 512]);  addmm_92 = None
        permute_200 = torch.ops.aten.permute.default(view_229, [0, 3, 1, 2]);  view_229 = None
        view_230 = torch.ops.aten.view.default(arg102_1, [1, -1, 1, 1]);  arg102_1 = None
        mul_291 = torch.ops.aten.mul.Tensor(permute_200, view_230);  permute_200 = view_230 = None
        add_199 = torch.ops.aten.add.Tensor(mul_291, add_195);  mul_291 = add_195 = None
        convolution_53 = torch.ops.aten.convolution.default(add_199, arg103_1, arg104_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg103_1 = arg104_1 = None
        permute_201 = torch.ops.aten.permute.default(convolution_53, [0, 2, 3, 1]);  convolution_53 = None
        var_mean_54 = torch.ops.aten.var_mean.correction(permute_201, [3], correction = 0, keepdim = True)
        getitem_108 = var_mean_54[0]
        getitem_109 = var_mean_54[1];  var_mean_54 = None
        add_200 = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        sub_54 = torch.ops.aten.sub.Tensor(permute_201, getitem_109);  permute_201 = getitem_109 = None
        mul_292 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, arg105_1);  mul_292 = arg105_1 = None
        add_201 = torch.ops.aten.add.Tensor(mul_293, arg106_1);  mul_293 = arg106_1 = None
        view_231 = torch.ops.aten.view.default(add_201, [2592, 512]);  add_201 = None
        permute_202 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg108_1, view_231, permute_202);  arg108_1 = view_231 = permute_202 = None
        view_232 = torch.ops.aten.view.default(addmm_93, [8, 18, 18, 2048]);  addmm_93 = None
        mul_294 = torch.ops.aten.mul.Tensor(view_232, 0.5)
        mul_295 = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476);  view_232 = None
        erf_46 = torch.ops.aten.erf.default(mul_295);  mul_295 = None
        add_202 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_294, add_202);  mul_294 = add_202 = None
        view_233 = torch.ops.aten.view.default(mul_296, [2592, 2048]);  mul_296 = None
        permute_203 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg110_1, view_233, permute_203);  arg110_1 = view_233 = permute_203 = None
        view_234 = torch.ops.aten.view.default(addmm_94, [8, 18, 18, 512]);  addmm_94 = None
        permute_204 = torch.ops.aten.permute.default(view_234, [0, 3, 1, 2]);  view_234 = None
        view_235 = torch.ops.aten.view.default(arg111_1, [1, -1, 1, 1]);  arg111_1 = None
        mul_297 = torch.ops.aten.mul.Tensor(permute_204, view_235);  permute_204 = view_235 = None
        add_203 = torch.ops.aten.add.Tensor(mul_297, add_199);  mul_297 = add_199 = None
        convolution_54 = torch.ops.aten.convolution.default(add_203, arg112_1, arg113_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg112_1 = arg113_1 = None
        permute_205 = torch.ops.aten.permute.default(convolution_54, [0, 2, 3, 1]);  convolution_54 = None
        var_mean_55 = torch.ops.aten.var_mean.correction(permute_205, [3], correction = 0, keepdim = True)
        getitem_110 = var_mean_55[0]
        getitem_111 = var_mean_55[1];  var_mean_55 = None
        add_204 = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        sub_55 = torch.ops.aten.sub.Tensor(permute_205, getitem_111);  permute_205 = getitem_111 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, arg114_1);  mul_298 = arg114_1 = None
        add_205 = torch.ops.aten.add.Tensor(mul_299, arg115_1);  mul_299 = arg115_1 = None
        view_236 = torch.ops.aten.view.default(add_205, [2592, 512]);  add_205 = None
        permute_206 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg117_1, view_236, permute_206);  arg117_1 = view_236 = permute_206 = None
        view_237 = torch.ops.aten.view.default(addmm_95, [8, 18, 18, 2048]);  addmm_95 = None
        mul_300 = torch.ops.aten.mul.Tensor(view_237, 0.5)
        mul_301 = torch.ops.aten.mul.Tensor(view_237, 0.7071067811865476);  view_237 = None
        erf_47 = torch.ops.aten.erf.default(mul_301);  mul_301 = None
        add_206 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_300, add_206);  mul_300 = add_206 = None
        view_238 = torch.ops.aten.view.default(mul_302, [2592, 2048]);  mul_302 = None
        permute_207 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg119_1, view_238, permute_207);  arg119_1 = view_238 = permute_207 = None
        view_239 = torch.ops.aten.view.default(addmm_96, [8, 18, 18, 512]);  addmm_96 = None
        permute_208 = torch.ops.aten.permute.default(view_239, [0, 3, 1, 2]);  view_239 = None
        view_240 = torch.ops.aten.view.default(arg120_1, [1, -1, 1, 1]);  arg120_1 = None
        mul_303 = torch.ops.aten.mul.Tensor(permute_208, view_240);  permute_208 = view_240 = None
        add_207 = torch.ops.aten.add.Tensor(mul_303, add_203);  mul_303 = add_203 = None
        convolution_55 = torch.ops.aten.convolution.default(add_207, arg121_1, arg122_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg121_1 = arg122_1 = None
        permute_209 = torch.ops.aten.permute.default(convolution_55, [0, 2, 3, 1]);  convolution_55 = None
        var_mean_56 = torch.ops.aten.var_mean.correction(permute_209, [3], correction = 0, keepdim = True)
        getitem_112 = var_mean_56[0]
        getitem_113 = var_mean_56[1];  var_mean_56 = None
        add_208 = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        sub_56 = torch.ops.aten.sub.Tensor(permute_209, getitem_113);  permute_209 = getitem_113 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, arg123_1);  mul_304 = arg123_1 = None
        add_209 = torch.ops.aten.add.Tensor(mul_305, arg124_1);  mul_305 = arg124_1 = None
        view_241 = torch.ops.aten.view.default(add_209, [2592, 512]);  add_209 = None
        permute_210 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg126_1, view_241, permute_210);  arg126_1 = view_241 = permute_210 = None
        view_242 = torch.ops.aten.view.default(addmm_97, [8, 18, 18, 2048]);  addmm_97 = None
        mul_306 = torch.ops.aten.mul.Tensor(view_242, 0.5)
        mul_307 = torch.ops.aten.mul.Tensor(view_242, 0.7071067811865476);  view_242 = None
        erf_48 = torch.ops.aten.erf.default(mul_307);  mul_307 = None
        add_210 = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_306, add_210);  mul_306 = add_210 = None
        view_243 = torch.ops.aten.view.default(mul_308, [2592, 2048]);  mul_308 = None
        permute_211 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg128_1, view_243, permute_211);  arg128_1 = view_243 = permute_211 = None
        view_244 = torch.ops.aten.view.default(addmm_98, [8, 18, 18, 512]);  addmm_98 = None
        permute_212 = torch.ops.aten.permute.default(view_244, [0, 3, 1, 2]);  view_244 = None
        view_245 = torch.ops.aten.view.default(arg129_1, [1, -1, 1, 1]);  arg129_1 = None
        mul_309 = torch.ops.aten.mul.Tensor(permute_212, view_245);  permute_212 = view_245 = None
        add_211 = torch.ops.aten.add.Tensor(mul_309, add_207);  mul_309 = add_207 = None
        convolution_56 = torch.ops.aten.convolution.default(add_211, arg130_1, arg131_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg130_1 = arg131_1 = None
        permute_213 = torch.ops.aten.permute.default(convolution_56, [0, 2, 3, 1]);  convolution_56 = None
        var_mean_57 = torch.ops.aten.var_mean.correction(permute_213, [3], correction = 0, keepdim = True)
        getitem_114 = var_mean_57[0]
        getitem_115 = var_mean_57[1];  var_mean_57 = None
        add_212 = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_57 = torch.ops.aten.sub.Tensor(permute_213, getitem_115);  permute_213 = getitem_115 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, arg132_1);  mul_310 = arg132_1 = None
        add_213 = torch.ops.aten.add.Tensor(mul_311, arg133_1);  mul_311 = arg133_1 = None
        view_246 = torch.ops.aten.view.default(add_213, [2592, 512]);  add_213 = None
        permute_214 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg135_1, view_246, permute_214);  arg135_1 = view_246 = permute_214 = None
        view_247 = torch.ops.aten.view.default(addmm_99, [8, 18, 18, 2048]);  addmm_99 = None
        mul_312 = torch.ops.aten.mul.Tensor(view_247, 0.5)
        mul_313 = torch.ops.aten.mul.Tensor(view_247, 0.7071067811865476);  view_247 = None
        erf_49 = torch.ops.aten.erf.default(mul_313);  mul_313 = None
        add_214 = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_312, add_214);  mul_312 = add_214 = None
        view_248 = torch.ops.aten.view.default(mul_314, [2592, 2048]);  mul_314 = None
        permute_215 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg137_1, view_248, permute_215);  arg137_1 = view_248 = permute_215 = None
        view_249 = torch.ops.aten.view.default(addmm_100, [8, 18, 18, 512]);  addmm_100 = None
        permute_216 = torch.ops.aten.permute.default(view_249, [0, 3, 1, 2]);  view_249 = None
        view_250 = torch.ops.aten.view.default(arg138_1, [1, -1, 1, 1]);  arg138_1 = None
        mul_315 = torch.ops.aten.mul.Tensor(permute_216, view_250);  permute_216 = view_250 = None
        add_215 = torch.ops.aten.add.Tensor(mul_315, add_211);  mul_315 = add_211 = None
        convolution_57 = torch.ops.aten.convolution.default(add_215, arg139_1, arg140_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg139_1 = arg140_1 = None
        permute_217 = torch.ops.aten.permute.default(convolution_57, [0, 2, 3, 1]);  convolution_57 = None
        var_mean_58 = torch.ops.aten.var_mean.correction(permute_217, [3], correction = 0, keepdim = True)
        getitem_116 = var_mean_58[0]
        getitem_117 = var_mean_58[1];  var_mean_58 = None
        add_216 = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_58 = torch.ops.aten.sub.Tensor(permute_217, getitem_117);  permute_217 = getitem_117 = None
        mul_316 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, arg141_1);  mul_316 = arg141_1 = None
        add_217 = torch.ops.aten.add.Tensor(mul_317, arg142_1);  mul_317 = arg142_1 = None
        view_251 = torch.ops.aten.view.default(add_217, [2592, 512]);  add_217 = None
        permute_218 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg144_1, view_251, permute_218);  arg144_1 = view_251 = permute_218 = None
        view_252 = torch.ops.aten.view.default(addmm_101, [8, 18, 18, 2048]);  addmm_101 = None
        mul_318 = torch.ops.aten.mul.Tensor(view_252, 0.5)
        mul_319 = torch.ops.aten.mul.Tensor(view_252, 0.7071067811865476);  view_252 = None
        erf_50 = torch.ops.aten.erf.default(mul_319);  mul_319 = None
        add_218 = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_318, add_218);  mul_318 = add_218 = None
        view_253 = torch.ops.aten.view.default(mul_320, [2592, 2048]);  mul_320 = None
        permute_219 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg146_1, view_253, permute_219);  arg146_1 = view_253 = permute_219 = None
        view_254 = torch.ops.aten.view.default(addmm_102, [8, 18, 18, 512]);  addmm_102 = None
        permute_220 = torch.ops.aten.permute.default(view_254, [0, 3, 1, 2]);  view_254 = None
        view_255 = torch.ops.aten.view.default(arg147_1, [1, -1, 1, 1]);  arg147_1 = None
        mul_321 = torch.ops.aten.mul.Tensor(permute_220, view_255);  permute_220 = view_255 = None
        add_219 = torch.ops.aten.add.Tensor(mul_321, add_215);  mul_321 = add_215 = None
        convolution_58 = torch.ops.aten.convolution.default(add_219, arg148_1, arg149_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg148_1 = arg149_1 = None
        permute_221 = torch.ops.aten.permute.default(convolution_58, [0, 2, 3, 1]);  convolution_58 = None
        var_mean_59 = torch.ops.aten.var_mean.correction(permute_221, [3], correction = 0, keepdim = True)
        getitem_118 = var_mean_59[0]
        getitem_119 = var_mean_59[1];  var_mean_59 = None
        add_220 = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_59 = torch.ops.aten.sub.Tensor(permute_221, getitem_119);  permute_221 = getitem_119 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, arg150_1);  mul_322 = arg150_1 = None
        add_221 = torch.ops.aten.add.Tensor(mul_323, arg151_1);  mul_323 = arg151_1 = None
        view_256 = torch.ops.aten.view.default(add_221, [2592, 512]);  add_221 = None
        permute_222 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg153_1, view_256, permute_222);  arg153_1 = view_256 = permute_222 = None
        view_257 = torch.ops.aten.view.default(addmm_103, [8, 18, 18, 2048]);  addmm_103 = None
        mul_324 = torch.ops.aten.mul.Tensor(view_257, 0.5)
        mul_325 = torch.ops.aten.mul.Tensor(view_257, 0.7071067811865476);  view_257 = None
        erf_51 = torch.ops.aten.erf.default(mul_325);  mul_325 = None
        add_222 = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_324, add_222);  mul_324 = add_222 = None
        view_258 = torch.ops.aten.view.default(mul_326, [2592, 2048]);  mul_326 = None
        permute_223 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg155_1, view_258, permute_223);  arg155_1 = view_258 = permute_223 = None
        view_259 = torch.ops.aten.view.default(addmm_104, [8, 18, 18, 512]);  addmm_104 = None
        permute_224 = torch.ops.aten.permute.default(view_259, [0, 3, 1, 2]);  view_259 = None
        view_260 = torch.ops.aten.view.default(arg156_1, [1, -1, 1, 1]);  arg156_1 = None
        mul_327 = torch.ops.aten.mul.Tensor(permute_224, view_260);  permute_224 = view_260 = None
        add_223 = torch.ops.aten.add.Tensor(mul_327, add_219);  mul_327 = add_219 = None
        convolution_59 = torch.ops.aten.convolution.default(add_223, arg157_1, arg158_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg157_1 = arg158_1 = None
        permute_225 = torch.ops.aten.permute.default(convolution_59, [0, 2, 3, 1]);  convolution_59 = None
        var_mean_60 = torch.ops.aten.var_mean.correction(permute_225, [3], correction = 0, keepdim = True)
        getitem_120 = var_mean_60[0]
        getitem_121 = var_mean_60[1];  var_mean_60 = None
        add_224 = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        sub_60 = torch.ops.aten.sub.Tensor(permute_225, getitem_121);  permute_225 = getitem_121 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, arg159_1);  mul_328 = arg159_1 = None
        add_225 = torch.ops.aten.add.Tensor(mul_329, arg160_1);  mul_329 = arg160_1 = None
        view_261 = torch.ops.aten.view.default(add_225, [2592, 512]);  add_225 = None
        permute_226 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg162_1, view_261, permute_226);  arg162_1 = view_261 = permute_226 = None
        view_262 = torch.ops.aten.view.default(addmm_105, [8, 18, 18, 2048]);  addmm_105 = None
        mul_330 = torch.ops.aten.mul.Tensor(view_262, 0.5)
        mul_331 = torch.ops.aten.mul.Tensor(view_262, 0.7071067811865476);  view_262 = None
        erf_52 = torch.ops.aten.erf.default(mul_331);  mul_331 = None
        add_226 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_330, add_226);  mul_330 = add_226 = None
        view_263 = torch.ops.aten.view.default(mul_332, [2592, 2048]);  mul_332 = None
        permute_227 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg164_1, view_263, permute_227);  arg164_1 = view_263 = permute_227 = None
        view_264 = torch.ops.aten.view.default(addmm_106, [8, 18, 18, 512]);  addmm_106 = None
        permute_228 = torch.ops.aten.permute.default(view_264, [0, 3, 1, 2]);  view_264 = None
        view_265 = torch.ops.aten.view.default(arg165_1, [1, -1, 1, 1]);  arg165_1 = None
        mul_333 = torch.ops.aten.mul.Tensor(permute_228, view_265);  permute_228 = view_265 = None
        add_227 = torch.ops.aten.add.Tensor(mul_333, add_223);  mul_333 = add_223 = None
        convolution_60 = torch.ops.aten.convolution.default(add_227, arg166_1, arg167_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg166_1 = arg167_1 = None
        permute_229 = torch.ops.aten.permute.default(convolution_60, [0, 2, 3, 1]);  convolution_60 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(permute_229, [3], correction = 0, keepdim = True)
        getitem_122 = var_mean_61[0]
        getitem_123 = var_mean_61[1];  var_mean_61 = None
        add_228 = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        sub_61 = torch.ops.aten.sub.Tensor(permute_229, getitem_123);  permute_229 = getitem_123 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, arg168_1);  mul_334 = arg168_1 = None
        add_229 = torch.ops.aten.add.Tensor(mul_335, arg169_1);  mul_335 = arg169_1 = None
        view_266 = torch.ops.aten.view.default(add_229, [2592, 512]);  add_229 = None
        permute_230 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg171_1, view_266, permute_230);  arg171_1 = view_266 = permute_230 = None
        view_267 = torch.ops.aten.view.default(addmm_107, [8, 18, 18, 2048]);  addmm_107 = None
        mul_336 = torch.ops.aten.mul.Tensor(view_267, 0.5)
        mul_337 = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
        erf_53 = torch.ops.aten.erf.default(mul_337);  mul_337 = None
        add_230 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_336, add_230);  mul_336 = add_230 = None
        view_268 = torch.ops.aten.view.default(mul_338, [2592, 2048]);  mul_338 = None
        permute_231 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg173_1, view_268, permute_231);  arg173_1 = view_268 = permute_231 = None
        view_269 = torch.ops.aten.view.default(addmm_108, [8, 18, 18, 512]);  addmm_108 = None
        permute_232 = torch.ops.aten.permute.default(view_269, [0, 3, 1, 2]);  view_269 = None
        view_270 = torch.ops.aten.view.default(arg174_1, [1, -1, 1, 1]);  arg174_1 = None
        mul_339 = torch.ops.aten.mul.Tensor(permute_232, view_270);  permute_232 = view_270 = None
        add_231 = torch.ops.aten.add.Tensor(mul_339, add_227);  mul_339 = add_227 = None
        convolution_61 = torch.ops.aten.convolution.default(add_231, arg175_1, arg176_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg175_1 = arg176_1 = None
        permute_233 = torch.ops.aten.permute.default(convolution_61, [0, 2, 3, 1]);  convolution_61 = None
        var_mean_62 = torch.ops.aten.var_mean.correction(permute_233, [3], correction = 0, keepdim = True)
        getitem_124 = var_mean_62[0]
        getitem_125 = var_mean_62[1];  var_mean_62 = None
        add_232 = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_62 = torch.ops.aten.sub.Tensor(permute_233, getitem_125);  permute_233 = getitem_125 = None
        mul_340 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, arg177_1);  mul_340 = arg177_1 = None
        add_233 = torch.ops.aten.add.Tensor(mul_341, arg178_1);  mul_341 = arg178_1 = None
        view_271 = torch.ops.aten.view.default(add_233, [2592, 512]);  add_233 = None
        permute_234 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg180_1, view_271, permute_234);  arg180_1 = view_271 = permute_234 = None
        view_272 = torch.ops.aten.view.default(addmm_109, [8, 18, 18, 2048]);  addmm_109 = None
        mul_342 = torch.ops.aten.mul.Tensor(view_272, 0.5)
        mul_343 = torch.ops.aten.mul.Tensor(view_272, 0.7071067811865476);  view_272 = None
        erf_54 = torch.ops.aten.erf.default(mul_343);  mul_343 = None
        add_234 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_342, add_234);  mul_342 = add_234 = None
        view_273 = torch.ops.aten.view.default(mul_344, [2592, 2048]);  mul_344 = None
        permute_235 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg182_1, view_273, permute_235);  arg182_1 = view_273 = permute_235 = None
        view_274 = torch.ops.aten.view.default(addmm_110, [8, 18, 18, 512]);  addmm_110 = None
        permute_236 = torch.ops.aten.permute.default(view_274, [0, 3, 1, 2]);  view_274 = None
        view_275 = torch.ops.aten.view.default(arg183_1, [1, -1, 1, 1]);  arg183_1 = None
        mul_345 = torch.ops.aten.mul.Tensor(permute_236, view_275);  permute_236 = view_275 = None
        add_235 = torch.ops.aten.add.Tensor(mul_345, add_231);  mul_345 = add_231 = None
        convolution_62 = torch.ops.aten.convolution.default(add_235, arg184_1, arg185_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg184_1 = arg185_1 = None
        permute_237 = torch.ops.aten.permute.default(convolution_62, [0, 2, 3, 1]);  convolution_62 = None
        var_mean_63 = torch.ops.aten.var_mean.correction(permute_237, [3], correction = 0, keepdim = True)
        getitem_126 = var_mean_63[0]
        getitem_127 = var_mean_63[1];  var_mean_63 = None
        add_236 = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_63 = torch.ops.aten.sub.Tensor(permute_237, getitem_127);  permute_237 = getitem_127 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, arg186_1);  mul_346 = arg186_1 = None
        add_237 = torch.ops.aten.add.Tensor(mul_347, arg187_1);  mul_347 = arg187_1 = None
        view_276 = torch.ops.aten.view.default(add_237, [2592, 512]);  add_237 = None
        permute_238 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg189_1, view_276, permute_238);  arg189_1 = view_276 = permute_238 = None
        view_277 = torch.ops.aten.view.default(addmm_111, [8, 18, 18, 2048]);  addmm_111 = None
        mul_348 = torch.ops.aten.mul.Tensor(view_277, 0.5)
        mul_349 = torch.ops.aten.mul.Tensor(view_277, 0.7071067811865476);  view_277 = None
        erf_55 = torch.ops.aten.erf.default(mul_349);  mul_349 = None
        add_238 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_348, add_238);  mul_348 = add_238 = None
        view_278 = torch.ops.aten.view.default(mul_350, [2592, 2048]);  mul_350 = None
        permute_239 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg191_1, view_278, permute_239);  arg191_1 = view_278 = permute_239 = None
        view_279 = torch.ops.aten.view.default(addmm_112, [8, 18, 18, 512]);  addmm_112 = None
        permute_240 = torch.ops.aten.permute.default(view_279, [0, 3, 1, 2]);  view_279 = None
        view_280 = torch.ops.aten.view.default(arg192_1, [1, -1, 1, 1]);  arg192_1 = None
        mul_351 = torch.ops.aten.mul.Tensor(permute_240, view_280);  permute_240 = view_280 = None
        add_239 = torch.ops.aten.add.Tensor(mul_351, add_235);  mul_351 = add_235 = None
        convolution_63 = torch.ops.aten.convolution.default(add_239, arg193_1, arg194_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg193_1 = arg194_1 = None
        permute_241 = torch.ops.aten.permute.default(convolution_63, [0, 2, 3, 1]);  convolution_63 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(permute_241, [3], correction = 0, keepdim = True)
        getitem_128 = var_mean_64[0]
        getitem_129 = var_mean_64[1];  var_mean_64 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_64 = torch.ops.aten.sub.Tensor(permute_241, getitem_129);  permute_241 = getitem_129 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, arg195_1);  mul_352 = arg195_1 = None
        add_241 = torch.ops.aten.add.Tensor(mul_353, arg196_1);  mul_353 = arg196_1 = None
        view_281 = torch.ops.aten.view.default(add_241, [2592, 512]);  add_241 = None
        permute_242 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg198_1, view_281, permute_242);  arg198_1 = view_281 = permute_242 = None
        view_282 = torch.ops.aten.view.default(addmm_113, [8, 18, 18, 2048]);  addmm_113 = None
        mul_354 = torch.ops.aten.mul.Tensor(view_282, 0.5)
        mul_355 = torch.ops.aten.mul.Tensor(view_282, 0.7071067811865476);  view_282 = None
        erf_56 = torch.ops.aten.erf.default(mul_355);  mul_355 = None
        add_242 = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_354, add_242);  mul_354 = add_242 = None
        view_283 = torch.ops.aten.view.default(mul_356, [2592, 2048]);  mul_356 = None
        permute_243 = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg200_1, view_283, permute_243);  arg200_1 = view_283 = permute_243 = None
        view_284 = torch.ops.aten.view.default(addmm_114, [8, 18, 18, 512]);  addmm_114 = None
        permute_244 = torch.ops.aten.permute.default(view_284, [0, 3, 1, 2]);  view_284 = None
        view_285 = torch.ops.aten.view.default(arg201_1, [1, -1, 1, 1]);  arg201_1 = None
        mul_357 = torch.ops.aten.mul.Tensor(permute_244, view_285);  permute_244 = view_285 = None
        add_243 = torch.ops.aten.add.Tensor(mul_357, add_239);  mul_357 = add_239 = None
        convolution_64 = torch.ops.aten.convolution.default(add_243, arg202_1, arg203_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg202_1 = arg203_1 = None
        permute_245 = torch.ops.aten.permute.default(convolution_64, [0, 2, 3, 1]);  convolution_64 = None
        var_mean_65 = torch.ops.aten.var_mean.correction(permute_245, [3], correction = 0, keepdim = True)
        getitem_130 = var_mean_65[0]
        getitem_131 = var_mean_65[1];  var_mean_65 = None
        add_244 = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_65 = torch.ops.aten.sub.Tensor(permute_245, getitem_131);  permute_245 = getitem_131 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, arg204_1);  mul_358 = arg204_1 = None
        add_245 = torch.ops.aten.add.Tensor(mul_359, arg205_1);  mul_359 = arg205_1 = None
        view_286 = torch.ops.aten.view.default(add_245, [2592, 512]);  add_245 = None
        permute_246 = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg207_1, view_286, permute_246);  arg207_1 = view_286 = permute_246 = None
        view_287 = torch.ops.aten.view.default(addmm_115, [8, 18, 18, 2048]);  addmm_115 = None
        mul_360 = torch.ops.aten.mul.Tensor(view_287, 0.5)
        mul_361 = torch.ops.aten.mul.Tensor(view_287, 0.7071067811865476);  view_287 = None
        erf_57 = torch.ops.aten.erf.default(mul_361);  mul_361 = None
        add_246 = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_360, add_246);  mul_360 = add_246 = None
        view_288 = torch.ops.aten.view.default(mul_362, [2592, 2048]);  mul_362 = None
        permute_247 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg209_1, view_288, permute_247);  arg209_1 = view_288 = permute_247 = None
        view_289 = torch.ops.aten.view.default(addmm_116, [8, 18, 18, 512]);  addmm_116 = None
        permute_248 = torch.ops.aten.permute.default(view_289, [0, 3, 1, 2]);  view_289 = None
        view_290 = torch.ops.aten.view.default(arg210_1, [1, -1, 1, 1]);  arg210_1 = None
        mul_363 = torch.ops.aten.mul.Tensor(permute_248, view_290);  permute_248 = view_290 = None
        add_247 = torch.ops.aten.add.Tensor(mul_363, add_243);  mul_363 = add_243 = None
        convolution_65 = torch.ops.aten.convolution.default(add_247, arg211_1, arg212_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg211_1 = arg212_1 = None
        permute_249 = torch.ops.aten.permute.default(convolution_65, [0, 2, 3, 1]);  convolution_65 = None
        var_mean_66 = torch.ops.aten.var_mean.correction(permute_249, [3], correction = 0, keepdim = True)
        getitem_132 = var_mean_66[0]
        getitem_133 = var_mean_66[1];  var_mean_66 = None
        add_248 = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
        sub_66 = torch.ops.aten.sub.Tensor(permute_249, getitem_133);  permute_249 = getitem_133 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, arg213_1);  mul_364 = arg213_1 = None
        add_249 = torch.ops.aten.add.Tensor(mul_365, arg214_1);  mul_365 = arg214_1 = None
        view_291 = torch.ops.aten.view.default(add_249, [2592, 512]);  add_249 = None
        permute_250 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg216_1, view_291, permute_250);  arg216_1 = view_291 = permute_250 = None
        view_292 = torch.ops.aten.view.default(addmm_117, [8, 18, 18, 2048]);  addmm_117 = None
        mul_366 = torch.ops.aten.mul.Tensor(view_292, 0.5)
        mul_367 = torch.ops.aten.mul.Tensor(view_292, 0.7071067811865476);  view_292 = None
        erf_58 = torch.ops.aten.erf.default(mul_367);  mul_367 = None
        add_250 = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_366, add_250);  mul_366 = add_250 = None
        view_293 = torch.ops.aten.view.default(mul_368, [2592, 2048]);  mul_368 = None
        permute_251 = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg218_1, view_293, permute_251);  arg218_1 = view_293 = permute_251 = None
        view_294 = torch.ops.aten.view.default(addmm_118, [8, 18, 18, 512]);  addmm_118 = None
        permute_252 = torch.ops.aten.permute.default(view_294, [0, 3, 1, 2]);  view_294 = None
        view_295 = torch.ops.aten.view.default(arg219_1, [1, -1, 1, 1]);  arg219_1 = None
        mul_369 = torch.ops.aten.mul.Tensor(permute_252, view_295);  permute_252 = view_295 = None
        add_251 = torch.ops.aten.add.Tensor(mul_369, add_247);  mul_369 = add_247 = None
        convolution_66 = torch.ops.aten.convolution.default(add_251, arg220_1, arg221_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg220_1 = arg221_1 = None
        permute_253 = torch.ops.aten.permute.default(convolution_66, [0, 2, 3, 1]);  convolution_66 = None
        var_mean_67 = torch.ops.aten.var_mean.correction(permute_253, [3], correction = 0, keepdim = True)
        getitem_134 = var_mean_67[0]
        getitem_135 = var_mean_67[1];  var_mean_67 = None
        add_252 = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        sub_67 = torch.ops.aten.sub.Tensor(permute_253, getitem_135);  permute_253 = getitem_135 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, arg222_1);  mul_370 = arg222_1 = None
        add_253 = torch.ops.aten.add.Tensor(mul_371, arg223_1);  mul_371 = arg223_1 = None
        view_296 = torch.ops.aten.view.default(add_253, [2592, 512]);  add_253 = None
        permute_254 = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg225_1, view_296, permute_254);  arg225_1 = view_296 = permute_254 = None
        view_297 = torch.ops.aten.view.default(addmm_119, [8, 18, 18, 2048]);  addmm_119 = None
        mul_372 = torch.ops.aten.mul.Tensor(view_297, 0.5)
        mul_373 = torch.ops.aten.mul.Tensor(view_297, 0.7071067811865476);  view_297 = None
        erf_59 = torch.ops.aten.erf.default(mul_373);  mul_373 = None
        add_254 = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_372, add_254);  mul_372 = add_254 = None
        view_298 = torch.ops.aten.view.default(mul_374, [2592, 2048]);  mul_374 = None
        permute_255 = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg227_1, view_298, permute_255);  arg227_1 = view_298 = permute_255 = None
        view_299 = torch.ops.aten.view.default(addmm_120, [8, 18, 18, 512]);  addmm_120 = None
        permute_256 = torch.ops.aten.permute.default(view_299, [0, 3, 1, 2]);  view_299 = None
        view_300 = torch.ops.aten.view.default(arg228_1, [1, -1, 1, 1]);  arg228_1 = None
        mul_375 = torch.ops.aten.mul.Tensor(permute_256, view_300);  permute_256 = view_300 = None
        add_255 = torch.ops.aten.add.Tensor(mul_375, add_251);  mul_375 = add_251 = None
        convolution_67 = torch.ops.aten.convolution.default(add_255, arg229_1, arg230_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg229_1 = arg230_1 = None
        permute_257 = torch.ops.aten.permute.default(convolution_67, [0, 2, 3, 1]);  convolution_67 = None
        var_mean_68 = torch.ops.aten.var_mean.correction(permute_257, [3], correction = 0, keepdim = True)
        getitem_136 = var_mean_68[0]
        getitem_137 = var_mean_68[1];  var_mean_68 = None
        add_256 = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        sub_68 = torch.ops.aten.sub.Tensor(permute_257, getitem_137);  permute_257 = getitem_137 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, arg231_1);  mul_376 = arg231_1 = None
        add_257 = torch.ops.aten.add.Tensor(mul_377, arg232_1);  mul_377 = arg232_1 = None
        view_301 = torch.ops.aten.view.default(add_257, [2592, 512]);  add_257 = None
        permute_258 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg234_1, view_301, permute_258);  arg234_1 = view_301 = permute_258 = None
        view_302 = torch.ops.aten.view.default(addmm_121, [8, 18, 18, 2048]);  addmm_121 = None
        mul_378 = torch.ops.aten.mul.Tensor(view_302, 0.5)
        mul_379 = torch.ops.aten.mul.Tensor(view_302, 0.7071067811865476);  view_302 = None
        erf_60 = torch.ops.aten.erf.default(mul_379);  mul_379 = None
        add_258 = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_378, add_258);  mul_378 = add_258 = None
        view_303 = torch.ops.aten.view.default(mul_380, [2592, 2048]);  mul_380 = None
        permute_259 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg236_1, view_303, permute_259);  arg236_1 = view_303 = permute_259 = None
        view_304 = torch.ops.aten.view.default(addmm_122, [8, 18, 18, 512]);  addmm_122 = None
        permute_260 = torch.ops.aten.permute.default(view_304, [0, 3, 1, 2]);  view_304 = None
        view_305 = torch.ops.aten.view.default(arg237_1, [1, -1, 1, 1]);  arg237_1 = None
        mul_381 = torch.ops.aten.mul.Tensor(permute_260, view_305);  permute_260 = view_305 = None
        add_259 = torch.ops.aten.add.Tensor(mul_381, add_255);  mul_381 = add_255 = None
        convolution_68 = torch.ops.aten.convolution.default(add_259, arg238_1, arg239_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg238_1 = arg239_1 = None
        permute_261 = torch.ops.aten.permute.default(convolution_68, [0, 2, 3, 1]);  convolution_68 = None
        var_mean_69 = torch.ops.aten.var_mean.correction(permute_261, [3], correction = 0, keepdim = True)
        getitem_138 = var_mean_69[0]
        getitem_139 = var_mean_69[1];  var_mean_69 = None
        add_260 = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_69 = torch.ops.aten.sub.Tensor(permute_261, getitem_139);  permute_261 = getitem_139 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, arg240_1);  mul_382 = arg240_1 = None
        add_261 = torch.ops.aten.add.Tensor(mul_383, arg241_1);  mul_383 = arg241_1 = None
        view_306 = torch.ops.aten.view.default(add_261, [2592, 512]);  add_261 = None
        permute_262 = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg243_1, view_306, permute_262);  arg243_1 = view_306 = permute_262 = None
        view_307 = torch.ops.aten.view.default(addmm_123, [8, 18, 18, 2048]);  addmm_123 = None
        mul_384 = torch.ops.aten.mul.Tensor(view_307, 0.5)
        mul_385 = torch.ops.aten.mul.Tensor(view_307, 0.7071067811865476);  view_307 = None
        erf_61 = torch.ops.aten.erf.default(mul_385);  mul_385 = None
        add_262 = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_386 = torch.ops.aten.mul.Tensor(mul_384, add_262);  mul_384 = add_262 = None
        view_308 = torch.ops.aten.view.default(mul_386, [2592, 2048]);  mul_386 = None
        permute_263 = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg245_1, view_308, permute_263);  arg245_1 = view_308 = permute_263 = None
        view_309 = torch.ops.aten.view.default(addmm_124, [8, 18, 18, 512]);  addmm_124 = None
        permute_264 = torch.ops.aten.permute.default(view_309, [0, 3, 1, 2]);  view_309 = None
        view_310 = torch.ops.aten.view.default(arg246_1, [1, -1, 1, 1]);  arg246_1 = None
        mul_387 = torch.ops.aten.mul.Tensor(permute_264, view_310);  permute_264 = view_310 = None
        add_263 = torch.ops.aten.add.Tensor(mul_387, add_259);  mul_387 = add_259 = None
        convolution_69 = torch.ops.aten.convolution.default(add_263, arg247_1, arg248_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg247_1 = arg248_1 = None
        permute_265 = torch.ops.aten.permute.default(convolution_69, [0, 2, 3, 1]);  convolution_69 = None
        var_mean_70 = torch.ops.aten.var_mean.correction(permute_265, [3], correction = 0, keepdim = True)
        getitem_140 = var_mean_70[0]
        getitem_141 = var_mean_70[1];  var_mean_70 = None
        add_264 = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_70 = torch.ops.aten.sub.Tensor(permute_265, getitem_141);  permute_265 = getitem_141 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, arg249_1);  mul_388 = arg249_1 = None
        add_265 = torch.ops.aten.add.Tensor(mul_389, arg250_1);  mul_389 = arg250_1 = None
        view_311 = torch.ops.aten.view.default(add_265, [2592, 512]);  add_265 = None
        permute_266 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg252_1, view_311, permute_266);  arg252_1 = view_311 = permute_266 = None
        view_312 = torch.ops.aten.view.default(addmm_125, [8, 18, 18, 2048]);  addmm_125 = None
        mul_390 = torch.ops.aten.mul.Tensor(view_312, 0.5)
        mul_391 = torch.ops.aten.mul.Tensor(view_312, 0.7071067811865476);  view_312 = None
        erf_62 = torch.ops.aten.erf.default(mul_391);  mul_391 = None
        add_266 = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_390, add_266);  mul_390 = add_266 = None
        view_313 = torch.ops.aten.view.default(mul_392, [2592, 2048]);  mul_392 = None
        permute_267 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg254_1, view_313, permute_267);  arg254_1 = view_313 = permute_267 = None
        view_314 = torch.ops.aten.view.default(addmm_126, [8, 18, 18, 512]);  addmm_126 = None
        permute_268 = torch.ops.aten.permute.default(view_314, [0, 3, 1, 2]);  view_314 = None
        view_315 = torch.ops.aten.view.default(arg255_1, [1, -1, 1, 1]);  arg255_1 = None
        mul_393 = torch.ops.aten.mul.Tensor(permute_268, view_315);  permute_268 = view_315 = None
        add_267 = torch.ops.aten.add.Tensor(mul_393, add_263);  mul_393 = add_263 = None
        convolution_70 = torch.ops.aten.convolution.default(add_267, arg256_1, arg257_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg256_1 = arg257_1 = None
        permute_269 = torch.ops.aten.permute.default(convolution_70, [0, 2, 3, 1]);  convolution_70 = None
        var_mean_71 = torch.ops.aten.var_mean.correction(permute_269, [3], correction = 0, keepdim = True)
        getitem_142 = var_mean_71[0]
        getitem_143 = var_mean_71[1];  var_mean_71 = None
        add_268 = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_71 = torch.ops.aten.sub.Tensor(permute_269, getitem_143);  permute_269 = getitem_143 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, arg258_1);  mul_394 = arg258_1 = None
        add_269 = torch.ops.aten.add.Tensor(mul_395, arg259_1);  mul_395 = arg259_1 = None
        view_316 = torch.ops.aten.view.default(add_269, [2592, 512]);  add_269 = None
        permute_270 = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg261_1, view_316, permute_270);  arg261_1 = view_316 = permute_270 = None
        view_317 = torch.ops.aten.view.default(addmm_127, [8, 18, 18, 2048]);  addmm_127 = None
        mul_396 = torch.ops.aten.mul.Tensor(view_317, 0.5)
        mul_397 = torch.ops.aten.mul.Tensor(view_317, 0.7071067811865476);  view_317 = None
        erf_63 = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_270 = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_396, add_270);  mul_396 = add_270 = None
        view_318 = torch.ops.aten.view.default(mul_398, [2592, 2048]);  mul_398 = None
        permute_271 = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg263_1, view_318, permute_271);  arg263_1 = view_318 = permute_271 = None
        view_319 = torch.ops.aten.view.default(addmm_128, [8, 18, 18, 512]);  addmm_128 = None
        permute_272 = torch.ops.aten.permute.default(view_319, [0, 3, 1, 2]);  view_319 = None
        view_320 = torch.ops.aten.view.default(arg264_1, [1, -1, 1, 1]);  arg264_1 = None
        mul_399 = torch.ops.aten.mul.Tensor(permute_272, view_320);  permute_272 = view_320 = None
        add_271 = torch.ops.aten.add.Tensor(mul_399, add_267);  mul_399 = add_267 = None
        convolution_71 = torch.ops.aten.convolution.default(add_271, arg265_1, arg266_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg265_1 = arg266_1 = None
        permute_273 = torch.ops.aten.permute.default(convolution_71, [0, 2, 3, 1]);  convolution_71 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(permute_273, [3], correction = 0, keepdim = True)
        getitem_144 = var_mean_72[0]
        getitem_145 = var_mean_72[1];  var_mean_72 = None
        add_272 = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        sub_72 = torch.ops.aten.sub.Tensor(permute_273, getitem_145);  permute_273 = getitem_145 = None
        mul_400 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_400, arg267_1);  mul_400 = arg267_1 = None
        add_273 = torch.ops.aten.add.Tensor(mul_401, arg268_1);  mul_401 = arg268_1 = None
        view_321 = torch.ops.aten.view.default(add_273, [2592, 512]);  add_273 = None
        permute_274 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg270_1, view_321, permute_274);  arg270_1 = view_321 = permute_274 = None
        view_322 = torch.ops.aten.view.default(addmm_129, [8, 18, 18, 2048]);  addmm_129 = None
        mul_402 = torch.ops.aten.mul.Tensor(view_322, 0.5)
        mul_403 = torch.ops.aten.mul.Tensor(view_322, 0.7071067811865476);  view_322 = None
        erf_64 = torch.ops.aten.erf.default(mul_403);  mul_403 = None
        add_274 = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_404 = torch.ops.aten.mul.Tensor(mul_402, add_274);  mul_402 = add_274 = None
        view_323 = torch.ops.aten.view.default(mul_404, [2592, 2048]);  mul_404 = None
        permute_275 = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg272_1, view_323, permute_275);  arg272_1 = view_323 = permute_275 = None
        view_324 = torch.ops.aten.view.default(addmm_130, [8, 18, 18, 512]);  addmm_130 = None
        permute_276 = torch.ops.aten.permute.default(view_324, [0, 3, 1, 2]);  view_324 = None
        view_325 = torch.ops.aten.view.default(arg273_1, [1, -1, 1, 1]);  arg273_1 = None
        mul_405 = torch.ops.aten.mul.Tensor(permute_276, view_325);  permute_276 = view_325 = None
        add_275 = torch.ops.aten.add.Tensor(mul_405, add_271);  mul_405 = add_271 = None
        convolution_72 = torch.ops.aten.convolution.default(add_275, arg274_1, arg275_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg274_1 = arg275_1 = None
        permute_277 = torch.ops.aten.permute.default(convolution_72, [0, 2, 3, 1]);  convolution_72 = None
        var_mean_73 = torch.ops.aten.var_mean.correction(permute_277, [3], correction = 0, keepdim = True)
        getitem_146 = var_mean_73[0]
        getitem_147 = var_mean_73[1];  var_mean_73 = None
        add_276 = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_73 = torch.ops.aten.sub.Tensor(permute_277, getitem_147);  permute_277 = getitem_147 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, arg276_1);  mul_406 = arg276_1 = None
        add_277 = torch.ops.aten.add.Tensor(mul_407, arg277_1);  mul_407 = arg277_1 = None
        view_326 = torch.ops.aten.view.default(add_277, [2592, 512]);  add_277 = None
        permute_278 = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg279_1, view_326, permute_278);  arg279_1 = view_326 = permute_278 = None
        view_327 = torch.ops.aten.view.default(addmm_131, [8, 18, 18, 2048]);  addmm_131 = None
        mul_408 = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_409 = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_65 = torch.ops.aten.erf.default(mul_409);  mul_409 = None
        add_278 = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_408, add_278);  mul_408 = add_278 = None
        view_328 = torch.ops.aten.view.default(mul_410, [2592, 2048]);  mul_410 = None
        permute_279 = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg281_1, view_328, permute_279);  arg281_1 = view_328 = permute_279 = None
        view_329 = torch.ops.aten.view.default(addmm_132, [8, 18, 18, 512]);  addmm_132 = None
        permute_280 = torch.ops.aten.permute.default(view_329, [0, 3, 1, 2]);  view_329 = None
        view_330 = torch.ops.aten.view.default(arg282_1, [1, -1, 1, 1]);  arg282_1 = None
        mul_411 = torch.ops.aten.mul.Tensor(permute_280, view_330);  permute_280 = view_330 = None
        add_279 = torch.ops.aten.add.Tensor(mul_411, add_275);  mul_411 = add_275 = None
        convolution_73 = torch.ops.aten.convolution.default(add_279, arg283_1, arg284_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg283_1 = arg284_1 = None
        permute_281 = torch.ops.aten.permute.default(convolution_73, [0, 2, 3, 1]);  convolution_73 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(permute_281, [3], correction = 0, keepdim = True)
        getitem_148 = var_mean_74[0]
        getitem_149 = var_mean_74[1];  var_mean_74 = None
        add_280 = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        sub_74 = torch.ops.aten.sub.Tensor(permute_281, getitem_149);  permute_281 = getitem_149 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, arg285_1);  mul_412 = arg285_1 = None
        add_281 = torch.ops.aten.add.Tensor(mul_413, arg286_1);  mul_413 = arg286_1 = None
        view_331 = torch.ops.aten.view.default(add_281, [2592, 512]);  add_281 = None
        permute_282 = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg288_1, view_331, permute_282);  arg288_1 = view_331 = permute_282 = None
        view_332 = torch.ops.aten.view.default(addmm_133, [8, 18, 18, 2048]);  addmm_133 = None
        mul_414 = torch.ops.aten.mul.Tensor(view_332, 0.5)
        mul_415 = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
        erf_66 = torch.ops.aten.erf.default(mul_415);  mul_415 = None
        add_282 = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_414, add_282);  mul_414 = add_282 = None
        view_333 = torch.ops.aten.view.default(mul_416, [2592, 2048]);  mul_416 = None
        permute_283 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg290_1, view_333, permute_283);  arg290_1 = view_333 = permute_283 = None
        view_334 = torch.ops.aten.view.default(addmm_134, [8, 18, 18, 512]);  addmm_134 = None
        permute_284 = torch.ops.aten.permute.default(view_334, [0, 3, 1, 2]);  view_334 = None
        view_335 = torch.ops.aten.view.default(arg291_1, [1, -1, 1, 1]);  arg291_1 = None
        mul_417 = torch.ops.aten.mul.Tensor(permute_284, view_335);  permute_284 = view_335 = None
        add_283 = torch.ops.aten.add.Tensor(mul_417, add_279);  mul_417 = add_279 = None
        convolution_74 = torch.ops.aten.convolution.default(add_283, arg292_1, arg293_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg292_1 = arg293_1 = None
        permute_285 = torch.ops.aten.permute.default(convolution_74, [0, 2, 3, 1]);  convolution_74 = None
        var_mean_75 = torch.ops.aten.var_mean.correction(permute_285, [3], correction = 0, keepdim = True)
        getitem_150 = var_mean_75[0]
        getitem_151 = var_mean_75[1];  var_mean_75 = None
        add_284 = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
        sub_75 = torch.ops.aten.sub.Tensor(permute_285, getitem_151);  permute_285 = getitem_151 = None
        mul_418 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_418, arg294_1);  mul_418 = arg294_1 = None
        add_285 = torch.ops.aten.add.Tensor(mul_419, arg295_1);  mul_419 = arg295_1 = None
        view_336 = torch.ops.aten.view.default(add_285, [2592, 512]);  add_285 = None
        permute_286 = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg297_1, view_336, permute_286);  arg297_1 = view_336 = permute_286 = None
        view_337 = torch.ops.aten.view.default(addmm_135, [8, 18, 18, 2048]);  addmm_135 = None
        mul_420 = torch.ops.aten.mul.Tensor(view_337, 0.5)
        mul_421 = torch.ops.aten.mul.Tensor(view_337, 0.7071067811865476);  view_337 = None
        erf_67 = torch.ops.aten.erf.default(mul_421);  mul_421 = None
        add_286 = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_422 = torch.ops.aten.mul.Tensor(mul_420, add_286);  mul_420 = add_286 = None
        view_338 = torch.ops.aten.view.default(mul_422, [2592, 2048]);  mul_422 = None
        permute_287 = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg299_1, view_338, permute_287);  arg299_1 = view_338 = permute_287 = None
        view_339 = torch.ops.aten.view.default(addmm_136, [8, 18, 18, 512]);  addmm_136 = None
        permute_288 = torch.ops.aten.permute.default(view_339, [0, 3, 1, 2]);  view_339 = None
        view_340 = torch.ops.aten.view.default(arg300_1, [1, -1, 1, 1]);  arg300_1 = None
        mul_423 = torch.ops.aten.mul.Tensor(permute_288, view_340);  permute_288 = view_340 = None
        add_287 = torch.ops.aten.add.Tensor(mul_423, add_283);  mul_423 = add_283 = None
        convolution_75 = torch.ops.aten.convolution.default(add_287, arg301_1, arg302_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg301_1 = arg302_1 = None
        permute_289 = torch.ops.aten.permute.default(convolution_75, [0, 2, 3, 1]);  convolution_75 = None
        var_mean_76 = torch.ops.aten.var_mean.correction(permute_289, [3], correction = 0, keepdim = True)
        getitem_152 = var_mean_76[0]
        getitem_153 = var_mean_76[1];  var_mean_76 = None
        add_288 = torch.ops.aten.add.Tensor(getitem_152, 1e-06);  getitem_152 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
        sub_76 = torch.ops.aten.sub.Tensor(permute_289, getitem_153);  permute_289 = getitem_153 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, arg303_1);  mul_424 = arg303_1 = None
        add_289 = torch.ops.aten.add.Tensor(mul_425, arg304_1);  mul_425 = arg304_1 = None
        view_341 = torch.ops.aten.view.default(add_289, [2592, 512]);  add_289 = None
        permute_290 = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg306_1, view_341, permute_290);  arg306_1 = view_341 = permute_290 = None
        view_342 = torch.ops.aten.view.default(addmm_137, [8, 18, 18, 2048]);  addmm_137 = None
        mul_426 = torch.ops.aten.mul.Tensor(view_342, 0.5)
        mul_427 = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476);  view_342 = None
        erf_68 = torch.ops.aten.erf.default(mul_427);  mul_427 = None
        add_290 = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_426, add_290);  mul_426 = add_290 = None
        view_343 = torch.ops.aten.view.default(mul_428, [2592, 2048]);  mul_428 = None
        permute_291 = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg308_1, view_343, permute_291);  arg308_1 = view_343 = permute_291 = None
        view_344 = torch.ops.aten.view.default(addmm_138, [8, 18, 18, 512]);  addmm_138 = None
        permute_292 = torch.ops.aten.permute.default(view_344, [0, 3, 1, 2]);  view_344 = None
        view_345 = torch.ops.aten.view.default(arg309_1, [1, -1, 1, 1]);  arg309_1 = None
        mul_429 = torch.ops.aten.mul.Tensor(permute_292, view_345);  permute_292 = view_345 = None
        add_291 = torch.ops.aten.add.Tensor(mul_429, add_287);  mul_429 = add_287 = None
        permute_293 = torch.ops.aten.permute.default(add_291, [0, 2, 3, 1]);  add_291 = None
        var_mean_77 = torch.ops.aten.var_mean.correction(permute_293, [3], correction = 0, keepdim = True)
        getitem_154 = var_mean_77[0]
        getitem_155 = var_mean_77[1];  var_mean_77 = None
        add_292 = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_77 = torch.ops.aten.sub.Tensor(permute_293, getitem_155);  permute_293 = getitem_155 = None
        mul_430 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        mul_431 = torch.ops.aten.mul.Tensor(mul_430, arg310_1);  mul_430 = arg310_1 = None
        add_293 = torch.ops.aten.add.Tensor(mul_431, arg311_1);  mul_431 = arg311_1 = None
        permute_294 = torch.ops.aten.permute.default(add_293, [0, 3, 1, 2]);  add_293 = None
        convolution_76 = torch.ops.aten.convolution.default(permute_294, arg312_1, arg313_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_294 = arg312_1 = arg313_1 = None
        convolution_77 = torch.ops.aten.convolution.default(convolution_76, arg314_1, arg315_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg314_1 = arg315_1 = None
        permute_295 = torch.ops.aten.permute.default(convolution_77, [0, 2, 3, 1]);  convolution_77 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(permute_295, [3], correction = 0, keepdim = True)
        getitem_156 = var_mean_78[0]
        getitem_157 = var_mean_78[1];  var_mean_78 = None
        add_294 = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        sub_78 = torch.ops.aten.sub.Tensor(permute_295, getitem_157);  permute_295 = getitem_157 = None
        mul_432 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        mul_433 = torch.ops.aten.mul.Tensor(mul_432, arg316_1);  mul_432 = arg316_1 = None
        add_295 = torch.ops.aten.add.Tensor(mul_433, arg317_1);  mul_433 = arg317_1 = None
        view_346 = torch.ops.aten.view.default(add_295, [648, 1024]);  add_295 = None
        permute_296 = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg319_1, view_346, permute_296);  arg319_1 = view_346 = permute_296 = None
        view_347 = torch.ops.aten.view.default(addmm_139, [8, 9, 9, 4096]);  addmm_139 = None
        mul_434 = torch.ops.aten.mul.Tensor(view_347, 0.5)
        mul_435 = torch.ops.aten.mul.Tensor(view_347, 0.7071067811865476);  view_347 = None
        erf_69 = torch.ops.aten.erf.default(mul_435);  mul_435 = None
        add_296 = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_436 = torch.ops.aten.mul.Tensor(mul_434, add_296);  mul_434 = add_296 = None
        view_348 = torch.ops.aten.view.default(mul_436, [648, 4096]);  mul_436 = None
        permute_297 = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg321_1, view_348, permute_297);  arg321_1 = view_348 = permute_297 = None
        view_349 = torch.ops.aten.view.default(addmm_140, [8, 9, 9, 1024]);  addmm_140 = None
        permute_298 = torch.ops.aten.permute.default(view_349, [0, 3, 1, 2]);  view_349 = None
        view_350 = torch.ops.aten.view.default(arg322_1, [1, -1, 1, 1]);  arg322_1 = None
        mul_437 = torch.ops.aten.mul.Tensor(permute_298, view_350);  permute_298 = view_350 = None
        add_297 = torch.ops.aten.add.Tensor(mul_437, convolution_76);  mul_437 = convolution_76 = None
        convolution_78 = torch.ops.aten.convolution.default(add_297, arg323_1, arg324_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg323_1 = arg324_1 = None
        permute_299 = torch.ops.aten.permute.default(convolution_78, [0, 2, 3, 1]);  convolution_78 = None
        var_mean_79 = torch.ops.aten.var_mean.correction(permute_299, [3], correction = 0, keepdim = True)
        getitem_158 = var_mean_79[0]
        getitem_159 = var_mean_79[1];  var_mean_79 = None
        add_298 = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
        sub_79 = torch.ops.aten.sub.Tensor(permute_299, getitem_159);  permute_299 = getitem_159 = None
        mul_438 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        mul_439 = torch.ops.aten.mul.Tensor(mul_438, arg325_1);  mul_438 = arg325_1 = None
        add_299 = torch.ops.aten.add.Tensor(mul_439, arg326_1);  mul_439 = arg326_1 = None
        view_351 = torch.ops.aten.view.default(add_299, [648, 1024]);  add_299 = None
        permute_300 = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg328_1, view_351, permute_300);  arg328_1 = view_351 = permute_300 = None
        view_352 = torch.ops.aten.view.default(addmm_141, [8, 9, 9, 4096]);  addmm_141 = None
        mul_440 = torch.ops.aten.mul.Tensor(view_352, 0.5)
        mul_441 = torch.ops.aten.mul.Tensor(view_352, 0.7071067811865476);  view_352 = None
        erf_70 = torch.ops.aten.erf.default(mul_441);  mul_441 = None
        add_300 = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_442 = torch.ops.aten.mul.Tensor(mul_440, add_300);  mul_440 = add_300 = None
        view_353 = torch.ops.aten.view.default(mul_442, [648, 4096]);  mul_442 = None
        permute_301 = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg330_1, view_353, permute_301);  arg330_1 = view_353 = permute_301 = None
        view_354 = torch.ops.aten.view.default(addmm_142, [8, 9, 9, 1024]);  addmm_142 = None
        permute_302 = torch.ops.aten.permute.default(view_354, [0, 3, 1, 2]);  view_354 = None
        view_355 = torch.ops.aten.view.default(arg331_1, [1, -1, 1, 1]);  arg331_1 = None
        mul_443 = torch.ops.aten.mul.Tensor(permute_302, view_355);  permute_302 = view_355 = None
        add_301 = torch.ops.aten.add.Tensor(mul_443, add_297);  mul_443 = add_297 = None
        convolution_79 = torch.ops.aten.convolution.default(add_301, arg332_1, arg333_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg332_1 = arg333_1 = None
        permute_303 = torch.ops.aten.permute.default(convolution_79, [0, 2, 3, 1]);  convolution_79 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(permute_303, [3], correction = 0, keepdim = True)
        getitem_160 = var_mean_80[0]
        getitem_161 = var_mean_80[1];  var_mean_80 = None
        add_302 = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
        sub_80 = torch.ops.aten.sub.Tensor(permute_303, getitem_161);  permute_303 = getitem_161 = None
        mul_444 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, arg334_1);  mul_444 = arg334_1 = None
        add_303 = torch.ops.aten.add.Tensor(mul_445, arg335_1);  mul_445 = arg335_1 = None
        view_356 = torch.ops.aten.view.default(add_303, [648, 1024]);  add_303 = None
        permute_304 = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg337_1, view_356, permute_304);  arg337_1 = view_356 = permute_304 = None
        view_357 = torch.ops.aten.view.default(addmm_143, [8, 9, 9, 4096]);  addmm_143 = None
        mul_446 = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_447 = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_71 = torch.ops.aten.erf.default(mul_447);  mul_447 = None
        add_304 = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_446, add_304);  mul_446 = add_304 = None
        view_358 = torch.ops.aten.view.default(mul_448, [648, 4096]);  mul_448 = None
        permute_305 = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg339_1, view_358, permute_305);  arg339_1 = view_358 = permute_305 = None
        view_359 = torch.ops.aten.view.default(addmm_144, [8, 9, 9, 1024]);  addmm_144 = None
        permute_306 = torch.ops.aten.permute.default(view_359, [0, 3, 1, 2]);  view_359 = None
        view_360 = torch.ops.aten.view.default(arg340_1, [1, -1, 1, 1]);  arg340_1 = None
        mul_449 = torch.ops.aten.mul.Tensor(permute_306, view_360);  permute_306 = view_360 = None
        add_305 = torch.ops.aten.add.Tensor(mul_449, add_301);  mul_449 = add_301 = None
        mean_1 = torch.ops.aten.mean.dim(add_305, [-1, -2], True);  add_305 = None
        as_strided_1 = torch.ops.aten.as_strided.default(mean_1, [8, 1024, 1, 1], [1024, 1, 1024, 1024]);  mean_1 = None
        permute_307 = torch.ops.aten.permute.default(as_strided_1, [0, 2, 3, 1]);  as_strided_1 = None
        var_mean_81 = torch.ops.aten.var_mean.correction(permute_307, [3], correction = 0, keepdim = True)
        getitem_162 = var_mean_81[0]
        getitem_163 = var_mean_81[1];  var_mean_81 = None
        add_306 = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_81 = torch.ops.aten.sub.Tensor(permute_307, getitem_163);  permute_307 = getitem_163 = None
        mul_450 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        mul_451 = torch.ops.aten.mul.Tensor(mul_450, arg341_1);  mul_450 = arg341_1 = None
        add_307 = torch.ops.aten.add.Tensor(mul_451, arg342_1);  mul_451 = arg342_1 = None
        permute_308 = torch.ops.aten.permute.default(add_307, [0, 3, 1, 2]);  add_307 = None
        view_361 = torch.ops.aten.view.default(permute_308, [8, 1024]);  permute_308 = None
        permute_309 = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg344_1, view_361, permute_309);  arg344_1 = view_361 = permute_309 = None
        return (addmm_145,)
        
def load_args(reader):
    buf0 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf0, (128, 3, 4, 4), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 7962624, device=device(type='cuda', index=0))
    reader.tensor(buf2, (8, 3, 288, 288), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 25088, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128, 1, 7, 7), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf9, (512, 128), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf10, (512,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128, 512), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 25088, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128, 1, 7, 7), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512, 128), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128, 512), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 25088, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128, 1, 7, 7), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf24, (128,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512, 128), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf28, (512,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128, 512), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf31, (128,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256, 128, 2, 2), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 50176, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256, 1, 7, 7), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1024, 256), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1024,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256, 1024), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 50176, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256, 1, 7, 7), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf46, (256,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf47, (256,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf48, (256,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf49, (1024, 256), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf50, (1024,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256, 1024), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf53, (256,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 50176, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256, 1, 7, 7), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf58, (1024, 256), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1024,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256, 1024), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf61, (256,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf62, (256,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf63, (256,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf64, (256,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512, 256, 2, 2), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512, 1, 7, 7), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf71, (2048, 512), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf72, (2048,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512, 2048), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512, 1, 7, 7), is_leaf=True)  # arg76_1
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
    buf85 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512, 1, 7, 7), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf89, (2048, 512), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf90, (2048,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512, 2048), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf94, (512, 1, 7, 7), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf95, (512,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf96, (512,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf98, (2048, 512), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf99, (2048,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512, 2048), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf102, (512,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf103, (512, 1, 7, 7), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf107, (2048, 512), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf108, (2048,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512, 2048), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512, 1, 7, 7), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf116, (2048, 512), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf117, (2048,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512, 2048), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf121, (512, 1, 7, 7), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf122, (512,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf123, (512,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf124, (512,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf125, (2048, 512), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf126, (2048,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf127, (512, 2048), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf130, (512, 1, 7, 7), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf131, (512,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf133, (512,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf134, (2048, 512), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf135, (2048,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf136, (512, 2048), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512, 1, 7, 7), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf141, (512,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf142, (512,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf143, (2048, 512), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf144, (2048,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf145, (512, 2048), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf146, (512,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf147, (512,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf148, (512, 1, 7, 7), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf149, (512,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf150, (512,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf151, (512,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf152, (2048, 512), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf153, (2048,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf154, (512, 2048), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf155, (512,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf156, (512,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf157, (512, 1, 7, 7), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf158, (512,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf159, (512,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf160, (512,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf161, (2048, 512), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf162, (2048,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512, 2048), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf164, (512,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf165, (512,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf166, (512, 1, 7, 7), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf167, (512,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf168, (512,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf169, (512,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf170, (2048, 512), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf171, (2048,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf172, (512, 2048), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf173, (512,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512, 1, 7, 7), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf176, (512,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf177, (512,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf178, (512,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf179, (2048, 512), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf180, (2048,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512, 2048), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf183, (512,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf184, (512, 1, 7, 7), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf185, (512,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf187, (512,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf188, (2048, 512), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf189, (2048,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf190, (512, 2048), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf191, (512,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512, 1, 7, 7), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf194, (512,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf195, (512,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf196, (512,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf197, (2048, 512), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf198, (2048,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf199, (512, 2048), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf200, (512,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf201, (512,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf202, (512, 1, 7, 7), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf203, (512,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf205, (512,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf206, (2048, 512), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf207, (2048,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf208, (512, 2048), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf209, (512,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf210, (512,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf211, (512, 1, 7, 7), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf212, (512,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf213, (512,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf214, (512,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf215, (2048, 512), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf216, (2048,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf217, (512, 2048), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf218, (512,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf219, (512,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf220, (512, 1, 7, 7), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf222, (512,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf224, (2048, 512), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf225, (2048,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf226, (512, 2048), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf227, (512,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf228, (512,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf229, (512, 1, 7, 7), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf230, (512,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf231, (512,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf232, (512,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf233, (2048, 512), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf234, (2048,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf235, (512, 2048), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf236, (512,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf237, (512,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf238, (512, 1, 7, 7), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf239, (512,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf240, (512,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf241, (512,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf242, (2048, 512), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf243, (2048,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf244, (512, 2048), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf245, (512,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf246, (512,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf247, (512, 1, 7, 7), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf248, (512,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf249, (512,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf250, (512,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf251, (2048, 512), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf252, (2048,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf253, (512, 2048), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf254, (512,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf255, (512,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf256, (512, 1, 7, 7), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf257, (512,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf258, (512,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf259, (512,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf260, (2048, 512), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf261, (2048,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf262, (512, 2048), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf263, (512,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf264, (512,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf265, (512, 1, 7, 7), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf266, (512,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf267, (512,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf268, (512,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf269, (2048, 512), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf270, (2048,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf271, (512, 2048), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf272, (512,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf273, (512,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf274, (512, 1, 7, 7), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf275, (512,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf276, (512,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf277, (512,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf278, (2048, 512), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf279, (2048,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf280, (512, 2048), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf281, (512,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf282, (512,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf283, (512, 1, 7, 7), is_leaf=True)  # arg283_1
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
    buf292 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf292, (512, 1, 7, 7), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf293, (512,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf294, (512,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf295, (512,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf296, (2048, 512), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf297, (2048,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf298, (512, 2048), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf299, (512,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf300, (512,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 100352, device=device(type='cuda', index=0))
    reader.tensor(buf301, (512, 1, 7, 7), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf302, (512,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf303, (512,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf304, (512,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf305, (2048, 512), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf306, (2048,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf307, (512, 2048), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf308, (512,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf309, (512,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf310, (512,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf311, (512,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf312, (1024, 512, 2, 2), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf313, (1024,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf314, (1024, 1, 7, 7), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf315, (1024,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1024,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1024,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf318, (4096, 1024), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf319, (4096,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf320, (1024, 4096), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf321, (1024,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1024,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf323, (1024, 1, 7, 7), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1024,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1024,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf326, (1024,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf327, (4096, 1024), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf328, (4096,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf329, (1024, 4096), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf330, (1024,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf331, (1024,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf332, (1024, 1, 7, 7), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf333, (1024,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf334, (1024,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf335, (1024,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf336, (4096, 1024), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf337, (4096,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf338, (1024, 4096), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1024,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf340, (1024,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf341, (1024,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf342, (1024,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf343, (1000, 1024), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1000,), is_leaf=True)  # arg344_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)