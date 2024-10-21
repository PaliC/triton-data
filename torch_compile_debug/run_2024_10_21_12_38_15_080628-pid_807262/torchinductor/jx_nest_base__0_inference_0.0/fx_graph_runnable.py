
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1):
        convolution_3 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        permute_187 = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
        view_397 = torch.ops.aten.view.default(permute_187, [8, 4, 14, 4, 14, 128]);  permute_187 = None
        permute_188 = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2, 4, 5]);  view_397 = None
        clone_173 = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_398 = torch.ops.aten.view.default(clone_173, [8, 16, 196, 128]);  clone_173 = None
        add_177 = torch.ops.aten.add.Tensor(view_398, arg3_1);  view_398 = arg3_1 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(add_177, [3], correction = 0, keepdim = True)
        getitem_178 = var_mean_51[0]
        getitem_179 = var_mean_51[1];  var_mean_51 = None
        add_178 = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_75 = torch.ops.aten.sub.Tensor(add_177, getitem_179);  getitem_179 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = rsqrt_51 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, arg4_1);  mul_222 = arg4_1 = None
        add_179 = torch.ops.aten.add.Tensor(mul_223, arg5_1);  mul_223 = arg5_1 = None
        view_399 = torch.ops.aten.view.default(add_179, [25088, 128]);  add_179 = None
        permute_189 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg7_1, view_399, permute_189);  arg7_1 = view_399 = permute_189 = None
        view_400 = torch.ops.aten.view.default(addmm_97, [8, 16, 196, 384]);  addmm_97 = None
        view_401 = torch.ops.aten.view.default(view_400, [8, 16, 196, 3, 4, 32]);  view_400 = None
        permute_190 = torch.ops.aten.permute.default(view_401, [3, 0, 4, 1, 2, 5]);  view_401 = None
        unbind_24 = torch.ops.aten.unbind.int(permute_190);  permute_190 = None
        getitem_180 = unbind_24[0]
        getitem_181 = unbind_24[1]
        getitem_182 = unbind_24[2];  unbind_24 = None
        mul_224 = torch.ops.aten.mul.Scalar(getitem_180, 0.42044820762685725);  getitem_180 = None
        permute_191 = torch.ops.aten.permute.default(getitem_181, [0, 1, 2, 4, 3]);  getitem_181 = None
        mul_225 = torch.ops.aten.mul.Scalar(permute_191, 0.42044820762685725);  permute_191 = None
        expand_96 = torch.ops.aten.expand.default(mul_224, [8, 4, 16, 196, 32]);  mul_224 = None
        clone_174 = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_402 = torch.ops.aten.view.default(clone_174, [512, 196, 32]);  clone_174 = None
        expand_97 = torch.ops.aten.expand.default(mul_225, [8, 4, 16, 32, 196]);  mul_225 = None
        clone_175 = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_403 = torch.ops.aten.view.default(clone_175, [512, 32, 196]);  clone_175 = None
        bmm_48 = torch.ops.aten.bmm.default(view_402, view_403);  view_402 = view_403 = None
        view_404 = torch.ops.aten.view.default(bmm_48, [8, 4, 16, 196, 196]);  bmm_48 = None
        amax_24 = torch.ops.aten.amax.default(view_404, [-1], True)
        sub_76 = torch.ops.aten.sub.Tensor(view_404, amax_24);  amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24 = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        eq_24 = torch.ops.aten.eq.Scalar(view_404, -inf);  view_404 = None
        logical_not_48 = torch.ops.aten.logical_not.default(eq_24);  eq_24 = None
        any_25 = torch.ops.aten.any.dim(logical_not_48, -1, True);  logical_not_48 = None
        logical_not_49 = torch.ops.aten.logical_not.default(any_25);  any_25 = None
        full_default = torch.ops.aten.full.default([8, 4, 16, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24 = torch.ops.aten.where.self(logical_not_49, full_default, div_24);  logical_not_49 = full_default = div_24 = None
        expand_98 = torch.ops.aten.expand.default(where_24, [8, 4, 16, 196, 196]);  where_24 = None
        view_405 = torch.ops.aten.view.default(expand_98, [512, 196, 196]);  expand_98 = None
        expand_99 = torch.ops.aten.expand.default(getitem_182, [8, 4, 16, 196, 32]);  getitem_182 = None
        clone_176 = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_406 = torch.ops.aten.view.default(clone_176, [512, 196, 32]);  clone_176 = None
        bmm_49 = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407 = torch.ops.aten.view.default(bmm_49, [8, 4, 16, 196, 32]);  bmm_49 = None
        permute_192 = torch.ops.aten.permute.default(view_407, [0, 2, 3, 4, 1]);  view_407 = None
        clone_177 = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_408 = torch.ops.aten.view.default(clone_177, [8, 16, 196, 128]);  clone_177 = None
        view_409 = torch.ops.aten.view.default(view_408, [25088, 128]);  view_408 = None
        permute_193 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg9_1, view_409, permute_193);  arg9_1 = view_409 = permute_193 = None
        view_410 = torch.ops.aten.view.default(addmm_98, [8, 16, 196, 128]);  addmm_98 = None
        add_180 = torch.ops.aten.add.Tensor(add_177, view_410);  add_177 = view_410 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(add_180, [3], correction = 0, keepdim = True)
        getitem_183 = var_mean_52[0]
        getitem_184 = var_mean_52[1];  var_mean_52 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_183, 1e-06);  getitem_183 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_77 = torch.ops.aten.sub.Tensor(add_180, getitem_184);  getitem_184 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_52);  sub_77 = rsqrt_52 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, arg10_1);  mul_226 = arg10_1 = None
        add_182 = torch.ops.aten.add.Tensor(mul_227, arg11_1);  mul_227 = arg11_1 = None
        view_411 = torch.ops.aten.view.default(add_182, [25088, 128]);  add_182 = None
        permute_194 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg13_1, view_411, permute_194);  arg13_1 = view_411 = permute_194 = None
        view_412 = torch.ops.aten.view.default(addmm_99, [8, 16, 196, 512]);  addmm_99 = None
        mul_228 = torch.ops.aten.mul.Tensor(view_412, 0.5)
        mul_229 = torch.ops.aten.mul.Tensor(view_412, 0.7071067811865476);  view_412 = None
        erf_24 = torch.ops.aten.erf.default(mul_229);  mul_229 = None
        add_183 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_228, add_183);  mul_228 = add_183 = None
        view_413 = torch.ops.aten.view.default(mul_230, [25088, 512]);  mul_230 = None
        permute_195 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg15_1, view_413, permute_195);  arg15_1 = view_413 = permute_195 = None
        view_414 = torch.ops.aten.view.default(addmm_100, [8, 16, 196, 128]);  addmm_100 = None
        add_184 = torch.ops.aten.add.Tensor(add_180, view_414);  add_180 = view_414 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(add_184, [3], correction = 0, keepdim = True)
        getitem_185 = var_mean_53[0]
        getitem_186 = var_mean_53[1];  var_mean_53 = None
        add_185 = torch.ops.aten.add.Tensor(getitem_185, 1e-06);  getitem_185 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_78 = torch.ops.aten.sub.Tensor(add_184, getitem_186);  getitem_186 = None
        mul_231 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_53);  sub_78 = rsqrt_53 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_231, arg16_1);  mul_231 = arg16_1 = None
        add_186 = torch.ops.aten.add.Tensor(mul_232, arg17_1);  mul_232 = arg17_1 = None
        view_415 = torch.ops.aten.view.default(add_186, [25088, 128]);  add_186 = None
        permute_196 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg19_1, view_415, permute_196);  arg19_1 = view_415 = permute_196 = None
        view_416 = torch.ops.aten.view.default(addmm_101, [8, 16, 196, 384]);  addmm_101 = None
        view_417 = torch.ops.aten.view.default(view_416, [8, 16, 196, 3, 4, 32]);  view_416 = None
        permute_197 = torch.ops.aten.permute.default(view_417, [3, 0, 4, 1, 2, 5]);  view_417 = None
        unbind_25 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_187 = unbind_25[0]
        getitem_188 = unbind_25[1]
        getitem_189 = unbind_25[2];  unbind_25 = None
        mul_233 = torch.ops.aten.mul.Scalar(getitem_187, 0.42044820762685725);  getitem_187 = None
        permute_198 = torch.ops.aten.permute.default(getitem_188, [0, 1, 2, 4, 3]);  getitem_188 = None
        mul_234 = torch.ops.aten.mul.Scalar(permute_198, 0.42044820762685725);  permute_198 = None
        expand_100 = torch.ops.aten.expand.default(mul_233, [8, 4, 16, 196, 32]);  mul_233 = None
        clone_181 = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_418 = torch.ops.aten.view.default(clone_181, [512, 196, 32]);  clone_181 = None
        expand_101 = torch.ops.aten.expand.default(mul_234, [8, 4, 16, 32, 196]);  mul_234 = None
        clone_182 = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_419 = torch.ops.aten.view.default(clone_182, [512, 32, 196]);  clone_182 = None
        bmm_50 = torch.ops.aten.bmm.default(view_418, view_419);  view_418 = view_419 = None
        view_420 = torch.ops.aten.view.default(bmm_50, [8, 4, 16, 196, 196]);  bmm_50 = None
        amax_25 = torch.ops.aten.amax.default(view_420, [-1], True)
        sub_79 = torch.ops.aten.sub.Tensor(view_420, amax_25);  amax_25 = None
        exp_25 = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25 = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        eq_25 = torch.ops.aten.eq.Scalar(view_420, -inf);  view_420 = None
        logical_not_50 = torch.ops.aten.logical_not.default(eq_25);  eq_25 = None
        any_26 = torch.ops.aten.any.dim(logical_not_50, -1, True);  logical_not_50 = None
        logical_not_51 = torch.ops.aten.logical_not.default(any_26);  any_26 = None
        full_default_1 = torch.ops.aten.full.default([8, 4, 16, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25 = torch.ops.aten.where.self(logical_not_51, full_default_1, div_25);  logical_not_51 = full_default_1 = div_25 = None
        expand_102 = torch.ops.aten.expand.default(where_25, [8, 4, 16, 196, 196]);  where_25 = None
        view_421 = torch.ops.aten.view.default(expand_102, [512, 196, 196]);  expand_102 = None
        expand_103 = torch.ops.aten.expand.default(getitem_189, [8, 4, 16, 196, 32]);  getitem_189 = None
        clone_183 = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_422 = torch.ops.aten.view.default(clone_183, [512, 196, 32]);  clone_183 = None
        bmm_51 = torch.ops.aten.bmm.default(view_421, view_422);  view_421 = view_422 = None
        view_423 = torch.ops.aten.view.default(bmm_51, [8, 4, 16, 196, 32]);  bmm_51 = None
        permute_199 = torch.ops.aten.permute.default(view_423, [0, 2, 3, 4, 1]);  view_423 = None
        clone_184 = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_424 = torch.ops.aten.view.default(clone_184, [8, 16, 196, 128]);  clone_184 = None
        view_425 = torch.ops.aten.view.default(view_424, [25088, 128]);  view_424 = None
        permute_200 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg21_1, view_425, permute_200);  arg21_1 = view_425 = permute_200 = None
        view_426 = torch.ops.aten.view.default(addmm_102, [8, 16, 196, 128]);  addmm_102 = None
        add_187 = torch.ops.aten.add.Tensor(add_184, view_426);  add_184 = view_426 = None
        var_mean_54 = torch.ops.aten.var_mean.correction(add_187, [3], correction = 0, keepdim = True)
        getitem_190 = var_mean_54[0]
        getitem_191 = var_mean_54[1];  var_mean_54 = None
        add_188 = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_80 = torch.ops.aten.sub.Tensor(add_187, getitem_191);  getitem_191 = None
        mul_235 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_54);  sub_80 = rsqrt_54 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, arg22_1);  mul_235 = arg22_1 = None
        add_189 = torch.ops.aten.add.Tensor(mul_236, arg23_1);  mul_236 = arg23_1 = None
        view_427 = torch.ops.aten.view.default(add_189, [25088, 128]);  add_189 = None
        permute_201 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg25_1, view_427, permute_201);  arg25_1 = view_427 = permute_201 = None
        view_428 = torch.ops.aten.view.default(addmm_103, [8, 16, 196, 512]);  addmm_103 = None
        mul_237 = torch.ops.aten.mul.Tensor(view_428, 0.5)
        mul_238 = torch.ops.aten.mul.Tensor(view_428, 0.7071067811865476);  view_428 = None
        erf_25 = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_190 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_237, add_190);  mul_237 = add_190 = None
        view_429 = torch.ops.aten.view.default(mul_239, [25088, 512]);  mul_239 = None
        permute_202 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg27_1, view_429, permute_202);  arg27_1 = view_429 = permute_202 = None
        view_430 = torch.ops.aten.view.default(addmm_104, [8, 16, 196, 128]);  addmm_104 = None
        add_191 = torch.ops.aten.add.Tensor(add_187, view_430);  add_187 = view_430 = None
        view_431 = torch.ops.aten.view.default(add_191, [8, 4, 4, 14, 14, 128]);  add_191 = None
        permute_203 = torch.ops.aten.permute.default(view_431, [0, 1, 3, 2, 4, 5]);  view_431 = None
        clone_188 = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_432 = torch.ops.aten.view.default(clone_188, [8, 56, 56, 128]);  clone_188 = None
        permute_204 = torch.ops.aten.permute.default(view_432, [0, 3, 1, 2]);  view_432 = None
        convolution_4 = torch.ops.aten.convolution.default(permute_204, arg28_1, arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_204 = arg28_1 = arg29_1 = None
        permute_205 = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None
        var_mean_55 = torch.ops.aten.var_mean.correction(permute_205, [3], correction = 0, keepdim = True)
        getitem_192 = var_mean_55[0]
        getitem_193 = var_mean_55[1];  var_mean_55 = None
        add_192 = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_81 = torch.ops.aten.sub.Tensor(permute_205, getitem_193);  permute_205 = getitem_193 = None
        mul_240 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_55);  sub_81 = rsqrt_55 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, arg30_1);  mul_240 = arg30_1 = None
        add_193 = torch.ops.aten.add.Tensor(mul_241, arg31_1);  mul_241 = arg31_1 = None
        permute_206 = torch.ops.aten.permute.default(add_193, [0, 3, 1, 2]);  add_193 = None
        constant_pad_nd_2 = torch.ops.aten.constant_pad_nd.default(permute_206, [0, 1, 0, 1], -inf);  permute_206 = None
        _low_memory_max_pool2d_with_offsets_2 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_2, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_2 = None
        getitem_194 = _low_memory_max_pool2d_with_offsets_2[0];  _low_memory_max_pool2d_with_offsets_2 = None
        permute_207 = torch.ops.aten.permute.default(getitem_194, [0, 2, 3, 1]);  getitem_194 = None
        view_433 = torch.ops.aten.view.default(permute_207, [8, 2, 14, 2, 14, 256]);  permute_207 = None
        permute_208 = torch.ops.aten.permute.default(view_433, [0, 1, 3, 2, 4, 5]);  view_433 = None
        clone_189 = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
        view_434 = torch.ops.aten.view.default(clone_189, [8, 4, 196, 256]);  clone_189 = None
        add_194 = torch.ops.aten.add.Tensor(view_434, arg32_1);  view_434 = arg32_1 = None
        var_mean_56 = torch.ops.aten.var_mean.correction(add_194, [3], correction = 0, keepdim = True)
        getitem_196 = var_mean_56[0]
        getitem_197 = var_mean_56[1];  var_mean_56 = None
        add_195 = torch.ops.aten.add.Tensor(getitem_196, 1e-06);  getitem_196 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_82 = torch.ops.aten.sub.Tensor(add_194, getitem_197);  getitem_197 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_56);  sub_82 = rsqrt_56 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, arg33_1);  mul_242 = arg33_1 = None
        add_196 = torch.ops.aten.add.Tensor(mul_243, arg34_1);  mul_243 = arg34_1 = None
        view_435 = torch.ops.aten.view.default(add_196, [6272, 256]);  add_196 = None
        permute_209 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg36_1, view_435, permute_209);  arg36_1 = view_435 = permute_209 = None
        view_436 = torch.ops.aten.view.default(addmm_105, [8, 4, 196, 768]);  addmm_105 = None
        view_437 = torch.ops.aten.view.default(view_436, [8, 4, 196, 3, 8, 32]);  view_436 = None
        permute_210 = torch.ops.aten.permute.default(view_437, [3, 0, 4, 1, 2, 5]);  view_437 = None
        unbind_26 = torch.ops.aten.unbind.int(permute_210);  permute_210 = None
        getitem_198 = unbind_26[0]
        getitem_199 = unbind_26[1]
        getitem_200 = unbind_26[2];  unbind_26 = None
        mul_244 = torch.ops.aten.mul.Scalar(getitem_198, 0.42044820762685725);  getitem_198 = None
        permute_211 = torch.ops.aten.permute.default(getitem_199, [0, 1, 2, 4, 3]);  getitem_199 = None
        mul_245 = torch.ops.aten.mul.Scalar(permute_211, 0.42044820762685725);  permute_211 = None
        expand_104 = torch.ops.aten.expand.default(mul_244, [8, 8, 4, 196, 32]);  mul_244 = None
        clone_190 = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_438 = torch.ops.aten.view.default(clone_190, [256, 196, 32]);  clone_190 = None
        expand_105 = torch.ops.aten.expand.default(mul_245, [8, 8, 4, 32, 196]);  mul_245 = None
        clone_191 = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_439 = torch.ops.aten.view.default(clone_191, [256, 32, 196]);  clone_191 = None
        bmm_52 = torch.ops.aten.bmm.default(view_438, view_439);  view_438 = view_439 = None
        view_440 = torch.ops.aten.view.default(bmm_52, [8, 8, 4, 196, 196]);  bmm_52 = None
        amax_26 = torch.ops.aten.amax.default(view_440, [-1], True)
        sub_83 = torch.ops.aten.sub.Tensor(view_440, amax_26);  amax_26 = None
        exp_26 = torch.ops.aten.exp.default(sub_83);  sub_83 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_26 = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        eq_26 = torch.ops.aten.eq.Scalar(view_440, -inf);  view_440 = None
        logical_not_52 = torch.ops.aten.logical_not.default(eq_26);  eq_26 = None
        any_27 = torch.ops.aten.any.dim(logical_not_52, -1, True);  logical_not_52 = None
        logical_not_53 = torch.ops.aten.logical_not.default(any_27);  any_27 = None
        full_default_2 = torch.ops.aten.full.default([8, 8, 4, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26 = torch.ops.aten.where.self(logical_not_53, full_default_2, div_26);  logical_not_53 = full_default_2 = div_26 = None
        expand_106 = torch.ops.aten.expand.default(where_26, [8, 8, 4, 196, 196]);  where_26 = None
        view_441 = torch.ops.aten.view.default(expand_106, [256, 196, 196]);  expand_106 = None
        expand_107 = torch.ops.aten.expand.default(getitem_200, [8, 8, 4, 196, 32]);  getitem_200 = None
        clone_192 = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_442 = torch.ops.aten.view.default(clone_192, [256, 196, 32]);  clone_192 = None
        bmm_53 = torch.ops.aten.bmm.default(view_441, view_442);  view_441 = view_442 = None
        view_443 = torch.ops.aten.view.default(bmm_53, [8, 8, 4, 196, 32]);  bmm_53 = None
        permute_212 = torch.ops.aten.permute.default(view_443, [0, 2, 3, 4, 1]);  view_443 = None
        clone_193 = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_444 = torch.ops.aten.view.default(clone_193, [8, 4, 196, 256]);  clone_193 = None
        view_445 = torch.ops.aten.view.default(view_444, [6272, 256]);  view_444 = None
        permute_213 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg38_1, view_445, permute_213);  arg38_1 = view_445 = permute_213 = None
        view_446 = torch.ops.aten.view.default(addmm_106, [8, 4, 196, 256]);  addmm_106 = None
        add_197 = torch.ops.aten.add.Tensor(add_194, view_446);  add_194 = view_446 = None
        var_mean_57 = torch.ops.aten.var_mean.correction(add_197, [3], correction = 0, keepdim = True)
        getitem_201 = var_mean_57[0]
        getitem_202 = var_mean_57[1];  var_mean_57 = None
        add_198 = torch.ops.aten.add.Tensor(getitem_201, 1e-06);  getitem_201 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        sub_84 = torch.ops.aten.sub.Tensor(add_197, getitem_202);  getitem_202 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_57);  sub_84 = rsqrt_57 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_246, arg39_1);  mul_246 = arg39_1 = None
        add_199 = torch.ops.aten.add.Tensor(mul_247, arg40_1);  mul_247 = arg40_1 = None
        view_447 = torch.ops.aten.view.default(add_199, [6272, 256]);  add_199 = None
        permute_214 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg42_1, view_447, permute_214);  arg42_1 = view_447 = permute_214 = None
        view_448 = torch.ops.aten.view.default(addmm_107, [8, 4, 196, 1024]);  addmm_107 = None
        mul_248 = torch.ops.aten.mul.Tensor(view_448, 0.5)
        mul_249 = torch.ops.aten.mul.Tensor(view_448, 0.7071067811865476);  view_448 = None
        erf_26 = torch.ops.aten.erf.default(mul_249);  mul_249 = None
        add_200 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_250 = torch.ops.aten.mul.Tensor(mul_248, add_200);  mul_248 = add_200 = None
        view_449 = torch.ops.aten.view.default(mul_250, [6272, 1024]);  mul_250 = None
        permute_215 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg44_1, view_449, permute_215);  arg44_1 = view_449 = permute_215 = None
        view_450 = torch.ops.aten.view.default(addmm_108, [8, 4, 196, 256]);  addmm_108 = None
        add_201 = torch.ops.aten.add.Tensor(add_197, view_450);  add_197 = view_450 = None
        var_mean_58 = torch.ops.aten.var_mean.correction(add_201, [3], correction = 0, keepdim = True)
        getitem_203 = var_mean_58[0]
        getitem_204 = var_mean_58[1];  var_mean_58 = None
        add_202 = torch.ops.aten.add.Tensor(getitem_203, 1e-06);  getitem_203 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_85 = torch.ops.aten.sub.Tensor(add_201, getitem_204);  getitem_204 = None
        mul_251 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_58);  sub_85 = rsqrt_58 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, arg45_1);  mul_251 = arg45_1 = None
        add_203 = torch.ops.aten.add.Tensor(mul_252, arg46_1);  mul_252 = arg46_1 = None
        view_451 = torch.ops.aten.view.default(add_203, [6272, 256]);  add_203 = None
        permute_216 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg48_1, view_451, permute_216);  arg48_1 = view_451 = permute_216 = None
        view_452 = torch.ops.aten.view.default(addmm_109, [8, 4, 196, 768]);  addmm_109 = None
        view_453 = torch.ops.aten.view.default(view_452, [8, 4, 196, 3, 8, 32]);  view_452 = None
        permute_217 = torch.ops.aten.permute.default(view_453, [3, 0, 4, 1, 2, 5]);  view_453 = None
        unbind_27 = torch.ops.aten.unbind.int(permute_217);  permute_217 = None
        getitem_205 = unbind_27[0]
        getitem_206 = unbind_27[1]
        getitem_207 = unbind_27[2];  unbind_27 = None
        mul_253 = torch.ops.aten.mul.Scalar(getitem_205, 0.42044820762685725);  getitem_205 = None
        permute_218 = torch.ops.aten.permute.default(getitem_206, [0, 1, 2, 4, 3]);  getitem_206 = None
        mul_254 = torch.ops.aten.mul.Scalar(permute_218, 0.42044820762685725);  permute_218 = None
        expand_108 = torch.ops.aten.expand.default(mul_253, [8, 8, 4, 196, 32]);  mul_253 = None
        clone_197 = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_454 = torch.ops.aten.view.default(clone_197, [256, 196, 32]);  clone_197 = None
        expand_109 = torch.ops.aten.expand.default(mul_254, [8, 8, 4, 32, 196]);  mul_254 = None
        clone_198 = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_455 = torch.ops.aten.view.default(clone_198, [256, 32, 196]);  clone_198 = None
        bmm_54 = torch.ops.aten.bmm.default(view_454, view_455);  view_454 = view_455 = None
        view_456 = torch.ops.aten.view.default(bmm_54, [8, 8, 4, 196, 196]);  bmm_54 = None
        amax_27 = torch.ops.aten.amax.default(view_456, [-1], True)
        sub_86 = torch.ops.aten.sub.Tensor(view_456, amax_27);  amax_27 = None
        exp_27 = torch.ops.aten.exp.default(sub_86);  sub_86 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        eq_27 = torch.ops.aten.eq.Scalar(view_456, -inf);  view_456 = None
        logical_not_54 = torch.ops.aten.logical_not.default(eq_27);  eq_27 = None
        any_28 = torch.ops.aten.any.dim(logical_not_54, -1, True);  logical_not_54 = None
        logical_not_55 = torch.ops.aten.logical_not.default(any_28);  any_28 = None
        full_default_3 = torch.ops.aten.full.default([8, 8, 4, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27 = torch.ops.aten.where.self(logical_not_55, full_default_3, div_27);  logical_not_55 = full_default_3 = div_27 = None
        expand_110 = torch.ops.aten.expand.default(where_27, [8, 8, 4, 196, 196]);  where_27 = None
        view_457 = torch.ops.aten.view.default(expand_110, [256, 196, 196]);  expand_110 = None
        expand_111 = torch.ops.aten.expand.default(getitem_207, [8, 8, 4, 196, 32]);  getitem_207 = None
        clone_199 = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_458 = torch.ops.aten.view.default(clone_199, [256, 196, 32]);  clone_199 = None
        bmm_55 = torch.ops.aten.bmm.default(view_457, view_458);  view_457 = view_458 = None
        view_459 = torch.ops.aten.view.default(bmm_55, [8, 8, 4, 196, 32]);  bmm_55 = None
        permute_219 = torch.ops.aten.permute.default(view_459, [0, 2, 3, 4, 1]);  view_459 = None
        clone_200 = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_460 = torch.ops.aten.view.default(clone_200, [8, 4, 196, 256]);  clone_200 = None
        view_461 = torch.ops.aten.view.default(view_460, [6272, 256]);  view_460 = None
        permute_220 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg50_1, view_461, permute_220);  arg50_1 = view_461 = permute_220 = None
        view_462 = torch.ops.aten.view.default(addmm_110, [8, 4, 196, 256]);  addmm_110 = None
        add_204 = torch.ops.aten.add.Tensor(add_201, view_462);  add_201 = view_462 = None
        var_mean_59 = torch.ops.aten.var_mean.correction(add_204, [3], correction = 0, keepdim = True)
        getitem_208 = var_mean_59[0]
        getitem_209 = var_mean_59[1];  var_mean_59 = None
        add_205 = torch.ops.aten.add.Tensor(getitem_208, 1e-06);  getitem_208 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        sub_87 = torch.ops.aten.sub.Tensor(add_204, getitem_209);  getitem_209 = None
        mul_255 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_59);  sub_87 = rsqrt_59 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, arg51_1);  mul_255 = arg51_1 = None
        add_206 = torch.ops.aten.add.Tensor(mul_256, arg52_1);  mul_256 = arg52_1 = None
        view_463 = torch.ops.aten.view.default(add_206, [6272, 256]);  add_206 = None
        permute_221 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg54_1, view_463, permute_221);  arg54_1 = view_463 = permute_221 = None
        view_464 = torch.ops.aten.view.default(addmm_111, [8, 4, 196, 1024]);  addmm_111 = None
        mul_257 = torch.ops.aten.mul.Tensor(view_464, 0.5)
        mul_258 = torch.ops.aten.mul.Tensor(view_464, 0.7071067811865476);  view_464 = None
        erf_27 = torch.ops.aten.erf.default(mul_258);  mul_258 = None
        add_207 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_257, add_207);  mul_257 = add_207 = None
        view_465 = torch.ops.aten.view.default(mul_259, [6272, 1024]);  mul_259 = None
        permute_222 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg56_1, view_465, permute_222);  arg56_1 = view_465 = permute_222 = None
        view_466 = torch.ops.aten.view.default(addmm_112, [8, 4, 196, 256]);  addmm_112 = None
        add_208 = torch.ops.aten.add.Tensor(add_204, view_466);  add_204 = view_466 = None
        view_467 = torch.ops.aten.view.default(add_208, [8, 2, 2, 14, 14, 256]);  add_208 = None
        permute_223 = torch.ops.aten.permute.default(view_467, [0, 1, 3, 2, 4, 5]);  view_467 = None
        clone_204 = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_468 = torch.ops.aten.view.default(clone_204, [8, 28, 28, 256]);  clone_204 = None
        permute_224 = torch.ops.aten.permute.default(view_468, [0, 3, 1, 2]);  view_468 = None
        convolution_5 = torch.ops.aten.convolution.default(permute_224, arg57_1, arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_224 = arg57_1 = arg58_1 = None
        permute_225 = torch.ops.aten.permute.default(convolution_5, [0, 2, 3, 1]);  convolution_5 = None
        var_mean_60 = torch.ops.aten.var_mean.correction(permute_225, [3], correction = 0, keepdim = True)
        getitem_210 = var_mean_60[0]
        getitem_211 = var_mean_60[1];  var_mean_60 = None
        add_209 = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_88 = torch.ops.aten.sub.Tensor(permute_225, getitem_211);  permute_225 = getitem_211 = None
        mul_260 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_60);  sub_88 = rsqrt_60 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, arg59_1);  mul_260 = arg59_1 = None
        add_210 = torch.ops.aten.add.Tensor(mul_261, arg60_1);  mul_261 = arg60_1 = None
        permute_226 = torch.ops.aten.permute.default(add_210, [0, 3, 1, 2]);  add_210 = None
        constant_pad_nd_3 = torch.ops.aten.constant_pad_nd.default(permute_226, [0, 1, 0, 1], -inf);  permute_226 = None
        _low_memory_max_pool2d_with_offsets_3 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_3, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_3 = None
        getitem_212 = _low_memory_max_pool2d_with_offsets_3[0];  _low_memory_max_pool2d_with_offsets_3 = None
        permute_227 = torch.ops.aten.permute.default(getitem_212, [0, 2, 3, 1]);  getitem_212 = None
        view_469 = torch.ops.aten.view.default(permute_227, [8, 1, 14, 1, 14, 512]);  permute_227 = None
        permute_228 = torch.ops.aten.permute.default(view_469, [0, 1, 3, 2, 4, 5]);  view_469 = None
        view_470 = torch.ops.aten.view.default(permute_228, [8, 1, -1, 512]);  permute_228 = None
        add_211 = torch.ops.aten.add.Tensor(view_470, arg61_1);  view_470 = arg61_1 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(add_211, [3], correction = 0, keepdim = True)
        getitem_214 = var_mean_61[0]
        getitem_215 = var_mean_61[1];  var_mean_61 = None
        add_212 = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_89 = torch.ops.aten.sub.Tensor(add_211, getitem_215);  getitem_215 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_61);  sub_89 = rsqrt_61 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, arg62_1);  mul_262 = arg62_1 = None
        add_213 = torch.ops.aten.add.Tensor(mul_263, arg63_1);  mul_263 = arg63_1 = None
        view_471 = torch.ops.aten.view.default(add_213, [1568, 512]);  add_213 = None
        permute_229 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg65_1, view_471, permute_229);  arg65_1 = view_471 = permute_229 = None
        view_472 = torch.ops.aten.view.default(addmm_113, [8, 1, 196, 1536]);  addmm_113 = None
        view_473 = torch.ops.aten.view.default(view_472, [8, 1, 196, 3, 16, 32]);  view_472 = None
        permute_230 = torch.ops.aten.permute.default(view_473, [3, 0, 4, 1, 2, 5]);  view_473 = None
        unbind_28 = torch.ops.aten.unbind.int(permute_230);  permute_230 = None
        getitem_216 = unbind_28[0]
        getitem_217 = unbind_28[1]
        getitem_218 = unbind_28[2];  unbind_28 = None
        mul_264 = torch.ops.aten.mul.Scalar(getitem_216, 0.42044820762685725);  getitem_216 = None
        permute_231 = torch.ops.aten.permute.default(getitem_217, [0, 1, 2, 4, 3]);  getitem_217 = None
        mul_265 = torch.ops.aten.mul.Scalar(permute_231, 0.42044820762685725);  permute_231 = None
        expand_112 = torch.ops.aten.expand.default(mul_264, [8, 16, 1, 196, 32]);  mul_264 = None
        clone_205 = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_474 = torch.ops.aten.view.default(clone_205, [128, 196, 32]);  clone_205 = None
        expand_113 = torch.ops.aten.expand.default(mul_265, [8, 16, 1, 32, 196]);  mul_265 = None
        clone_206 = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_475 = torch.ops.aten.view.default(clone_206, [128, 32, 196]);  clone_206 = None
        bmm_56 = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
        view_476 = torch.ops.aten.view.default(bmm_56, [8, 16, 1, 196, 196]);  bmm_56 = None
        amax_28 = torch.ops.aten.amax.default(view_476, [-1], True)
        sub_90 = torch.ops.aten.sub.Tensor(view_476, amax_28);  amax_28 = None
        exp_28 = torch.ops.aten.exp.default(sub_90);  sub_90 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_28 = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        eq_28 = torch.ops.aten.eq.Scalar(view_476, -inf);  view_476 = None
        logical_not_56 = torch.ops.aten.logical_not.default(eq_28);  eq_28 = None
        any_29 = torch.ops.aten.any.dim(logical_not_56, -1, True);  logical_not_56 = None
        logical_not_57 = torch.ops.aten.logical_not.default(any_29);  any_29 = None
        full_default_4 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28 = torch.ops.aten.where.self(logical_not_57, full_default_4, div_28);  logical_not_57 = full_default_4 = div_28 = None
        expand_114 = torch.ops.aten.expand.default(where_28, [8, 16, 1, 196, 196]);  where_28 = None
        view_477 = torch.ops.aten.view.default(expand_114, [128, 196, 196]);  expand_114 = None
        expand_115 = torch.ops.aten.expand.default(getitem_218, [8, 16, 1, 196, 32]);  getitem_218 = None
        clone_207 = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_478 = torch.ops.aten.view.default(clone_207, [128, 196, 32]);  clone_207 = None
        bmm_57 = torch.ops.aten.bmm.default(view_477, view_478);  view_477 = view_478 = None
        view_479 = torch.ops.aten.view.default(bmm_57, [8, 16, 1, 196, 32]);  bmm_57 = None
        permute_232 = torch.ops.aten.permute.default(view_479, [0, 2, 3, 4, 1]);  view_479 = None
        clone_208 = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_480 = torch.ops.aten.view.default(clone_208, [8, 1, 196, 512]);  clone_208 = None
        view_481 = torch.ops.aten.view.default(view_480, [1568, 512]);  view_480 = None
        permute_233 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg67_1, view_481, permute_233);  arg67_1 = view_481 = permute_233 = None
        view_482 = torch.ops.aten.view.default(addmm_114, [8, 1, 196, 512]);  addmm_114 = None
        add_214 = torch.ops.aten.add.Tensor(add_211, view_482);  add_211 = view_482 = None
        var_mean_62 = torch.ops.aten.var_mean.correction(add_214, [3], correction = 0, keepdim = True)
        getitem_219 = var_mean_62[0]
        getitem_220 = var_mean_62[1];  var_mean_62 = None
        add_215 = torch.ops.aten.add.Tensor(getitem_219, 1e-06);  getitem_219 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_91 = torch.ops.aten.sub.Tensor(add_214, getitem_220);  getitem_220 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_62);  sub_91 = rsqrt_62 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, arg68_1);  mul_266 = arg68_1 = None
        add_216 = torch.ops.aten.add.Tensor(mul_267, arg69_1);  mul_267 = arg69_1 = None
        view_483 = torch.ops.aten.view.default(add_216, [1568, 512]);  add_216 = None
        permute_234 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg71_1, view_483, permute_234);  arg71_1 = view_483 = permute_234 = None
        view_484 = torch.ops.aten.view.default(addmm_115, [8, 1, 196, 2048]);  addmm_115 = None
        mul_268 = torch.ops.aten.mul.Tensor(view_484, 0.5)
        mul_269 = torch.ops.aten.mul.Tensor(view_484, 0.7071067811865476);  view_484 = None
        erf_28 = torch.ops.aten.erf.default(mul_269);  mul_269 = None
        add_217 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_268, add_217);  mul_268 = add_217 = None
        view_485 = torch.ops.aten.view.default(mul_270, [1568, 2048]);  mul_270 = None
        permute_235 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg73_1, view_485, permute_235);  arg73_1 = view_485 = permute_235 = None
        view_486 = torch.ops.aten.view.default(addmm_116, [8, 1, 196, 512]);  addmm_116 = None
        add_218 = torch.ops.aten.add.Tensor(add_214, view_486);  add_214 = view_486 = None
        var_mean_63 = torch.ops.aten.var_mean.correction(add_218, [3], correction = 0, keepdim = True)
        getitem_221 = var_mean_63[0]
        getitem_222 = var_mean_63[1];  var_mean_63 = None
        add_219 = torch.ops.aten.add.Tensor(getitem_221, 1e-06);  getitem_221 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        sub_92 = torch.ops.aten.sub.Tensor(add_218, getitem_222);  getitem_222 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_63);  sub_92 = rsqrt_63 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, arg74_1);  mul_271 = arg74_1 = None
        add_220 = torch.ops.aten.add.Tensor(mul_272, arg75_1);  mul_272 = arg75_1 = None
        view_487 = torch.ops.aten.view.default(add_220, [1568, 512]);  add_220 = None
        permute_236 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg77_1, view_487, permute_236);  arg77_1 = view_487 = permute_236 = None
        view_488 = torch.ops.aten.view.default(addmm_117, [8, 1, 196, 1536]);  addmm_117 = None
        view_489 = torch.ops.aten.view.default(view_488, [8, 1, 196, 3, 16, 32]);  view_488 = None
        permute_237 = torch.ops.aten.permute.default(view_489, [3, 0, 4, 1, 2, 5]);  view_489 = None
        unbind_29 = torch.ops.aten.unbind.int(permute_237);  permute_237 = None
        getitem_223 = unbind_29[0]
        getitem_224 = unbind_29[1]
        getitem_225 = unbind_29[2];  unbind_29 = None
        mul_273 = torch.ops.aten.mul.Scalar(getitem_223, 0.42044820762685725);  getitem_223 = None
        permute_238 = torch.ops.aten.permute.default(getitem_224, [0, 1, 2, 4, 3]);  getitem_224 = None
        mul_274 = torch.ops.aten.mul.Scalar(permute_238, 0.42044820762685725);  permute_238 = None
        expand_116 = torch.ops.aten.expand.default(mul_273, [8, 16, 1, 196, 32]);  mul_273 = None
        clone_212 = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_490 = torch.ops.aten.view.default(clone_212, [128, 196, 32]);  clone_212 = None
        expand_117 = torch.ops.aten.expand.default(mul_274, [8, 16, 1, 32, 196]);  mul_274 = None
        clone_213 = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_491 = torch.ops.aten.view.default(clone_213, [128, 32, 196]);  clone_213 = None
        bmm_58 = torch.ops.aten.bmm.default(view_490, view_491);  view_490 = view_491 = None
        view_492 = torch.ops.aten.view.default(bmm_58, [8, 16, 1, 196, 196]);  bmm_58 = None
        amax_29 = torch.ops.aten.amax.default(view_492, [-1], True)
        sub_93 = torch.ops.aten.sub.Tensor(view_492, amax_29);  amax_29 = None
        exp_29 = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_29 = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
        eq_29 = torch.ops.aten.eq.Scalar(view_492, -inf);  view_492 = None
        logical_not_58 = torch.ops.aten.logical_not.default(eq_29);  eq_29 = None
        any_30 = torch.ops.aten.any.dim(logical_not_58, -1, True);  logical_not_58 = None
        logical_not_59 = torch.ops.aten.logical_not.default(any_30);  any_30 = None
        full_default_5 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29 = torch.ops.aten.where.self(logical_not_59, full_default_5, div_29);  logical_not_59 = full_default_5 = div_29 = None
        expand_118 = torch.ops.aten.expand.default(where_29, [8, 16, 1, 196, 196]);  where_29 = None
        view_493 = torch.ops.aten.view.default(expand_118, [128, 196, 196]);  expand_118 = None
        expand_119 = torch.ops.aten.expand.default(getitem_225, [8, 16, 1, 196, 32]);  getitem_225 = None
        clone_214 = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_494 = torch.ops.aten.view.default(clone_214, [128, 196, 32]);  clone_214 = None
        bmm_59 = torch.ops.aten.bmm.default(view_493, view_494);  view_493 = view_494 = None
        view_495 = torch.ops.aten.view.default(bmm_59, [8, 16, 1, 196, 32]);  bmm_59 = None
        permute_239 = torch.ops.aten.permute.default(view_495, [0, 2, 3, 4, 1]);  view_495 = None
        clone_215 = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_496 = torch.ops.aten.view.default(clone_215, [8, 1, 196, 512]);  clone_215 = None
        view_497 = torch.ops.aten.view.default(view_496, [1568, 512]);  view_496 = None
        permute_240 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg79_1, view_497, permute_240);  arg79_1 = view_497 = permute_240 = None
        view_498 = torch.ops.aten.view.default(addmm_118, [8, 1, 196, 512]);  addmm_118 = None
        add_221 = torch.ops.aten.add.Tensor(add_218, view_498);  add_218 = view_498 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(add_221, [3], correction = 0, keepdim = True)
        getitem_226 = var_mean_64[0]
        getitem_227 = var_mean_64[1];  var_mean_64 = None
        add_222 = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_94 = torch.ops.aten.sub.Tensor(add_221, getitem_227);  getitem_227 = None
        mul_275 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_64);  sub_94 = rsqrt_64 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, arg80_1);  mul_275 = arg80_1 = None
        add_223 = torch.ops.aten.add.Tensor(mul_276, arg81_1);  mul_276 = arg81_1 = None
        view_499 = torch.ops.aten.view.default(add_223, [1568, 512]);  add_223 = None
        permute_241 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg83_1, view_499, permute_241);  arg83_1 = view_499 = permute_241 = None
        view_500 = torch.ops.aten.view.default(addmm_119, [8, 1, 196, 2048]);  addmm_119 = None
        mul_277 = torch.ops.aten.mul.Tensor(view_500, 0.5)
        mul_278 = torch.ops.aten.mul.Tensor(view_500, 0.7071067811865476);  view_500 = None
        erf_29 = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_224 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_277, add_224);  mul_277 = add_224 = None
        view_501 = torch.ops.aten.view.default(mul_279, [1568, 2048]);  mul_279 = None
        permute_242 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg85_1, view_501, permute_242);  arg85_1 = view_501 = permute_242 = None
        view_502 = torch.ops.aten.view.default(addmm_120, [8, 1, 196, 512]);  addmm_120 = None
        add_225 = torch.ops.aten.add.Tensor(add_221, view_502);  add_221 = view_502 = None
        var_mean_65 = torch.ops.aten.var_mean.correction(add_225, [3], correction = 0, keepdim = True)
        getitem_228 = var_mean_65[0]
        getitem_229 = var_mean_65[1];  var_mean_65 = None
        add_226 = torch.ops.aten.add.Tensor(getitem_228, 1e-06);  getitem_228 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_95 = torch.ops.aten.sub.Tensor(add_225, getitem_229);  getitem_229 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_65);  sub_95 = rsqrt_65 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, arg86_1);  mul_280 = arg86_1 = None
        add_227 = torch.ops.aten.add.Tensor(mul_281, arg87_1);  mul_281 = arg87_1 = None
        view_503 = torch.ops.aten.view.default(add_227, [1568, 512]);  add_227 = None
        permute_243 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg89_1, view_503, permute_243);  arg89_1 = view_503 = permute_243 = None
        view_504 = torch.ops.aten.view.default(addmm_121, [8, 1, 196, 1536]);  addmm_121 = None
        view_505 = torch.ops.aten.view.default(view_504, [8, 1, 196, 3, 16, 32]);  view_504 = None
        permute_244 = torch.ops.aten.permute.default(view_505, [3, 0, 4, 1, 2, 5]);  view_505 = None
        unbind_30 = torch.ops.aten.unbind.int(permute_244);  permute_244 = None
        getitem_230 = unbind_30[0]
        getitem_231 = unbind_30[1]
        getitem_232 = unbind_30[2];  unbind_30 = None
        mul_282 = torch.ops.aten.mul.Scalar(getitem_230, 0.42044820762685725);  getitem_230 = None
        permute_245 = torch.ops.aten.permute.default(getitem_231, [0, 1, 2, 4, 3]);  getitem_231 = None
        mul_283 = torch.ops.aten.mul.Scalar(permute_245, 0.42044820762685725);  permute_245 = None
        expand_120 = torch.ops.aten.expand.default(mul_282, [8, 16, 1, 196, 32]);  mul_282 = None
        clone_219 = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_506 = torch.ops.aten.view.default(clone_219, [128, 196, 32]);  clone_219 = None
        expand_121 = torch.ops.aten.expand.default(mul_283, [8, 16, 1, 32, 196]);  mul_283 = None
        clone_220 = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_507 = torch.ops.aten.view.default(clone_220, [128, 32, 196]);  clone_220 = None
        bmm_60 = torch.ops.aten.bmm.default(view_506, view_507);  view_506 = view_507 = None
        view_508 = torch.ops.aten.view.default(bmm_60, [8, 16, 1, 196, 196]);  bmm_60 = None
        amax_30 = torch.ops.aten.amax.default(view_508, [-1], True)
        sub_96 = torch.ops.aten.sub.Tensor(view_508, amax_30);  amax_30 = None
        exp_30 = torch.ops.aten.exp.default(sub_96);  sub_96 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_30 = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        eq_30 = torch.ops.aten.eq.Scalar(view_508, -inf);  view_508 = None
        logical_not_60 = torch.ops.aten.logical_not.default(eq_30);  eq_30 = None
        any_31 = torch.ops.aten.any.dim(logical_not_60, -1, True);  logical_not_60 = None
        logical_not_61 = torch.ops.aten.logical_not.default(any_31);  any_31 = None
        full_default_6 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30 = torch.ops.aten.where.self(logical_not_61, full_default_6, div_30);  logical_not_61 = full_default_6 = div_30 = None
        expand_122 = torch.ops.aten.expand.default(where_30, [8, 16, 1, 196, 196]);  where_30 = None
        view_509 = torch.ops.aten.view.default(expand_122, [128, 196, 196]);  expand_122 = None
        expand_123 = torch.ops.aten.expand.default(getitem_232, [8, 16, 1, 196, 32]);  getitem_232 = None
        clone_221 = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_510 = torch.ops.aten.view.default(clone_221, [128, 196, 32]);  clone_221 = None
        bmm_61 = torch.ops.aten.bmm.default(view_509, view_510);  view_509 = view_510 = None
        view_511 = torch.ops.aten.view.default(bmm_61, [8, 16, 1, 196, 32]);  bmm_61 = None
        permute_246 = torch.ops.aten.permute.default(view_511, [0, 2, 3, 4, 1]);  view_511 = None
        clone_222 = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_512 = torch.ops.aten.view.default(clone_222, [8, 1, 196, 512]);  clone_222 = None
        view_513 = torch.ops.aten.view.default(view_512, [1568, 512]);  view_512 = None
        permute_247 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg91_1, view_513, permute_247);  arg91_1 = view_513 = permute_247 = None
        view_514 = torch.ops.aten.view.default(addmm_122, [8, 1, 196, 512]);  addmm_122 = None
        add_228 = torch.ops.aten.add.Tensor(add_225, view_514);  add_225 = view_514 = None
        var_mean_66 = torch.ops.aten.var_mean.correction(add_228, [3], correction = 0, keepdim = True)
        getitem_233 = var_mean_66[0]
        getitem_234 = var_mean_66[1];  var_mean_66 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_233, 1e-06);  getitem_233 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_97 = torch.ops.aten.sub.Tensor(add_228, getitem_234);  getitem_234 = None
        mul_284 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_66);  sub_97 = rsqrt_66 = None
        mul_285 = torch.ops.aten.mul.Tensor(mul_284, arg92_1);  mul_284 = arg92_1 = None
        add_230 = torch.ops.aten.add.Tensor(mul_285, arg93_1);  mul_285 = arg93_1 = None
        view_515 = torch.ops.aten.view.default(add_230, [1568, 512]);  add_230 = None
        permute_248 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg95_1, view_515, permute_248);  arg95_1 = view_515 = permute_248 = None
        view_516 = torch.ops.aten.view.default(addmm_123, [8, 1, 196, 2048]);  addmm_123 = None
        mul_286 = torch.ops.aten.mul.Tensor(view_516, 0.5)
        mul_287 = torch.ops.aten.mul.Tensor(view_516, 0.7071067811865476);  view_516 = None
        erf_30 = torch.ops.aten.erf.default(mul_287);  mul_287 = None
        add_231 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_286, add_231);  mul_286 = add_231 = None
        view_517 = torch.ops.aten.view.default(mul_288, [1568, 2048]);  mul_288 = None
        permute_249 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg97_1, view_517, permute_249);  arg97_1 = view_517 = permute_249 = None
        view_518 = torch.ops.aten.view.default(addmm_124, [8, 1, 196, 512]);  addmm_124 = None
        add_232 = torch.ops.aten.add.Tensor(add_228, view_518);  add_228 = view_518 = None
        var_mean_67 = torch.ops.aten.var_mean.correction(add_232, [3], correction = 0, keepdim = True)
        getitem_235 = var_mean_67[0]
        getitem_236 = var_mean_67[1];  var_mean_67 = None
        add_233 = torch.ops.aten.add.Tensor(getitem_235, 1e-06);  getitem_235 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        sub_98 = torch.ops.aten.sub.Tensor(add_232, getitem_236);  getitem_236 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_67);  sub_98 = rsqrt_67 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, arg98_1);  mul_289 = arg98_1 = None
        add_234 = torch.ops.aten.add.Tensor(mul_290, arg99_1);  mul_290 = arg99_1 = None
        view_519 = torch.ops.aten.view.default(add_234, [1568, 512]);  add_234 = None
        permute_250 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg101_1, view_519, permute_250);  arg101_1 = view_519 = permute_250 = None
        view_520 = torch.ops.aten.view.default(addmm_125, [8, 1, 196, 1536]);  addmm_125 = None
        view_521 = torch.ops.aten.view.default(view_520, [8, 1, 196, 3, 16, 32]);  view_520 = None
        permute_251 = torch.ops.aten.permute.default(view_521, [3, 0, 4, 1, 2, 5]);  view_521 = None
        unbind_31 = torch.ops.aten.unbind.int(permute_251);  permute_251 = None
        getitem_237 = unbind_31[0]
        getitem_238 = unbind_31[1]
        getitem_239 = unbind_31[2];  unbind_31 = None
        mul_291 = torch.ops.aten.mul.Scalar(getitem_237, 0.42044820762685725);  getitem_237 = None
        permute_252 = torch.ops.aten.permute.default(getitem_238, [0, 1, 2, 4, 3]);  getitem_238 = None
        mul_292 = torch.ops.aten.mul.Scalar(permute_252, 0.42044820762685725);  permute_252 = None
        expand_124 = torch.ops.aten.expand.default(mul_291, [8, 16, 1, 196, 32]);  mul_291 = None
        clone_226 = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_522 = torch.ops.aten.view.default(clone_226, [128, 196, 32]);  clone_226 = None
        expand_125 = torch.ops.aten.expand.default(mul_292, [8, 16, 1, 32, 196]);  mul_292 = None
        clone_227 = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_523 = torch.ops.aten.view.default(clone_227, [128, 32, 196]);  clone_227 = None
        bmm_62 = torch.ops.aten.bmm.default(view_522, view_523);  view_522 = view_523 = None
        view_524 = torch.ops.aten.view.default(bmm_62, [8, 16, 1, 196, 196]);  bmm_62 = None
        amax_31 = torch.ops.aten.amax.default(view_524, [-1], True)
        sub_99 = torch.ops.aten.sub.Tensor(view_524, amax_31);  amax_31 = None
        exp_31 = torch.ops.aten.exp.default(sub_99);  sub_99 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_31 = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
        eq_31 = torch.ops.aten.eq.Scalar(view_524, -inf);  view_524 = None
        logical_not_62 = torch.ops.aten.logical_not.default(eq_31);  eq_31 = None
        any_32 = torch.ops.aten.any.dim(logical_not_62, -1, True);  logical_not_62 = None
        logical_not_63 = torch.ops.aten.logical_not.default(any_32);  any_32 = None
        full_default_7 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31 = torch.ops.aten.where.self(logical_not_63, full_default_7, div_31);  logical_not_63 = full_default_7 = div_31 = None
        expand_126 = torch.ops.aten.expand.default(where_31, [8, 16, 1, 196, 196]);  where_31 = None
        view_525 = torch.ops.aten.view.default(expand_126, [128, 196, 196]);  expand_126 = None
        expand_127 = torch.ops.aten.expand.default(getitem_239, [8, 16, 1, 196, 32]);  getitem_239 = None
        clone_228 = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_526 = torch.ops.aten.view.default(clone_228, [128, 196, 32]);  clone_228 = None
        bmm_63 = torch.ops.aten.bmm.default(view_525, view_526);  view_525 = view_526 = None
        view_527 = torch.ops.aten.view.default(bmm_63, [8, 16, 1, 196, 32]);  bmm_63 = None
        permute_253 = torch.ops.aten.permute.default(view_527, [0, 2, 3, 4, 1]);  view_527 = None
        clone_229 = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_528 = torch.ops.aten.view.default(clone_229, [8, 1, 196, 512]);  clone_229 = None
        view_529 = torch.ops.aten.view.default(view_528, [1568, 512]);  view_528 = None
        permute_254 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg103_1, view_529, permute_254);  arg103_1 = view_529 = permute_254 = None
        view_530 = torch.ops.aten.view.default(addmm_126, [8, 1, 196, 512]);  addmm_126 = None
        add_235 = torch.ops.aten.add.Tensor(add_232, view_530);  add_232 = view_530 = None
        var_mean_68 = torch.ops.aten.var_mean.correction(add_235, [3], correction = 0, keepdim = True)
        getitem_240 = var_mean_68[0]
        getitem_241 = var_mean_68[1];  var_mean_68 = None
        add_236 = torch.ops.aten.add.Tensor(getitem_240, 1e-06);  getitem_240 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_100 = torch.ops.aten.sub.Tensor(add_235, getitem_241);  getitem_241 = None
        mul_293 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_68);  sub_100 = rsqrt_68 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_293, arg104_1);  mul_293 = arg104_1 = None
        add_237 = torch.ops.aten.add.Tensor(mul_294, arg105_1);  mul_294 = arg105_1 = None
        view_531 = torch.ops.aten.view.default(add_237, [1568, 512]);  add_237 = None
        permute_255 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg107_1, view_531, permute_255);  arg107_1 = view_531 = permute_255 = None
        view_532 = torch.ops.aten.view.default(addmm_127, [8, 1, 196, 2048]);  addmm_127 = None
        mul_295 = torch.ops.aten.mul.Tensor(view_532, 0.5)
        mul_296 = torch.ops.aten.mul.Tensor(view_532, 0.7071067811865476);  view_532 = None
        erf_31 = torch.ops.aten.erf.default(mul_296);  mul_296 = None
        add_238 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_295, add_238);  mul_295 = add_238 = None
        view_533 = torch.ops.aten.view.default(mul_297, [1568, 2048]);  mul_297 = None
        permute_256 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg109_1, view_533, permute_256);  arg109_1 = view_533 = permute_256 = None
        view_534 = torch.ops.aten.view.default(addmm_128, [8, 1, 196, 512]);  addmm_128 = None
        add_239 = torch.ops.aten.add.Tensor(add_235, view_534);  add_235 = view_534 = None
        var_mean_69 = torch.ops.aten.var_mean.correction(add_239, [3], correction = 0, keepdim = True)
        getitem_242 = var_mean_69[0]
        getitem_243 = var_mean_69[1];  var_mean_69 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_101 = torch.ops.aten.sub.Tensor(add_239, getitem_243);  getitem_243 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_69);  sub_101 = rsqrt_69 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, arg110_1);  mul_298 = arg110_1 = None
        add_241 = torch.ops.aten.add.Tensor(mul_299, arg111_1);  mul_299 = arg111_1 = None
        view_535 = torch.ops.aten.view.default(add_241, [1568, 512]);  add_241 = None
        permute_257 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg113_1, view_535, permute_257);  arg113_1 = view_535 = permute_257 = None
        view_536 = torch.ops.aten.view.default(addmm_129, [8, 1, 196, 1536]);  addmm_129 = None
        view_537 = torch.ops.aten.view.default(view_536, [8, 1, 196, 3, 16, 32]);  view_536 = None
        permute_258 = torch.ops.aten.permute.default(view_537, [3, 0, 4, 1, 2, 5]);  view_537 = None
        unbind_32 = torch.ops.aten.unbind.int(permute_258);  permute_258 = None
        getitem_244 = unbind_32[0]
        getitem_245 = unbind_32[1]
        getitem_246 = unbind_32[2];  unbind_32 = None
        mul_300 = torch.ops.aten.mul.Scalar(getitem_244, 0.42044820762685725);  getitem_244 = None
        permute_259 = torch.ops.aten.permute.default(getitem_245, [0, 1, 2, 4, 3]);  getitem_245 = None
        mul_301 = torch.ops.aten.mul.Scalar(permute_259, 0.42044820762685725);  permute_259 = None
        expand_128 = torch.ops.aten.expand.default(mul_300, [8, 16, 1, 196, 32]);  mul_300 = None
        clone_233 = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_538 = torch.ops.aten.view.default(clone_233, [128, 196, 32]);  clone_233 = None
        expand_129 = torch.ops.aten.expand.default(mul_301, [8, 16, 1, 32, 196]);  mul_301 = None
        clone_234 = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_539 = torch.ops.aten.view.default(clone_234, [128, 32, 196]);  clone_234 = None
        bmm_64 = torch.ops.aten.bmm.default(view_538, view_539);  view_538 = view_539 = None
        view_540 = torch.ops.aten.view.default(bmm_64, [8, 16, 1, 196, 196]);  bmm_64 = None
        amax_32 = torch.ops.aten.amax.default(view_540, [-1], True)
        sub_102 = torch.ops.aten.sub.Tensor(view_540, amax_32);  amax_32 = None
        exp_32 = torch.ops.aten.exp.default(sub_102);  sub_102 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_32 = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        eq_32 = torch.ops.aten.eq.Scalar(view_540, -inf);  view_540 = None
        logical_not_64 = torch.ops.aten.logical_not.default(eq_32);  eq_32 = None
        any_33 = torch.ops.aten.any.dim(logical_not_64, -1, True);  logical_not_64 = None
        logical_not_65 = torch.ops.aten.logical_not.default(any_33);  any_33 = None
        full_default_8 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32 = torch.ops.aten.where.self(logical_not_65, full_default_8, div_32);  logical_not_65 = full_default_8 = div_32 = None
        expand_130 = torch.ops.aten.expand.default(where_32, [8, 16, 1, 196, 196]);  where_32 = None
        view_541 = torch.ops.aten.view.default(expand_130, [128, 196, 196]);  expand_130 = None
        expand_131 = torch.ops.aten.expand.default(getitem_246, [8, 16, 1, 196, 32]);  getitem_246 = None
        clone_235 = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_542 = torch.ops.aten.view.default(clone_235, [128, 196, 32]);  clone_235 = None
        bmm_65 = torch.ops.aten.bmm.default(view_541, view_542);  view_541 = view_542 = None
        view_543 = torch.ops.aten.view.default(bmm_65, [8, 16, 1, 196, 32]);  bmm_65 = None
        permute_260 = torch.ops.aten.permute.default(view_543, [0, 2, 3, 4, 1]);  view_543 = None
        clone_236 = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_544 = torch.ops.aten.view.default(clone_236, [8, 1, 196, 512]);  clone_236 = None
        view_545 = torch.ops.aten.view.default(view_544, [1568, 512]);  view_544 = None
        permute_261 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg115_1, view_545, permute_261);  arg115_1 = view_545 = permute_261 = None
        view_546 = torch.ops.aten.view.default(addmm_130, [8, 1, 196, 512]);  addmm_130 = None
        add_242 = torch.ops.aten.add.Tensor(add_239, view_546);  add_239 = view_546 = None
        var_mean_70 = torch.ops.aten.var_mean.correction(add_242, [3], correction = 0, keepdim = True)
        getitem_247 = var_mean_70[0]
        getitem_248 = var_mean_70[1];  var_mean_70 = None
        add_243 = torch.ops.aten.add.Tensor(getitem_247, 1e-06);  getitem_247 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        sub_103 = torch.ops.aten.sub.Tensor(add_242, getitem_248);  getitem_248 = None
        mul_302 = torch.ops.aten.mul.Tensor(sub_103, rsqrt_70);  sub_103 = rsqrt_70 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_302, arg116_1);  mul_302 = arg116_1 = None
        add_244 = torch.ops.aten.add.Tensor(mul_303, arg117_1);  mul_303 = arg117_1 = None
        view_547 = torch.ops.aten.view.default(add_244, [1568, 512]);  add_244 = None
        permute_262 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg119_1, view_547, permute_262);  arg119_1 = view_547 = permute_262 = None
        view_548 = torch.ops.aten.view.default(addmm_131, [8, 1, 196, 2048]);  addmm_131 = None
        mul_304 = torch.ops.aten.mul.Tensor(view_548, 0.5)
        mul_305 = torch.ops.aten.mul.Tensor(view_548, 0.7071067811865476);  view_548 = None
        erf_32 = torch.ops.aten.erf.default(mul_305);  mul_305 = None
        add_245 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_306 = torch.ops.aten.mul.Tensor(mul_304, add_245);  mul_304 = add_245 = None
        view_549 = torch.ops.aten.view.default(mul_306, [1568, 2048]);  mul_306 = None
        permute_263 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg121_1, view_549, permute_263);  arg121_1 = view_549 = permute_263 = None
        view_550 = torch.ops.aten.view.default(addmm_132, [8, 1, 196, 512]);  addmm_132 = None
        add_246 = torch.ops.aten.add.Tensor(add_242, view_550);  add_242 = view_550 = None
        var_mean_71 = torch.ops.aten.var_mean.correction(add_246, [3], correction = 0, keepdim = True)
        getitem_249 = var_mean_71[0]
        getitem_250 = var_mean_71[1];  var_mean_71 = None
        add_247 = torch.ops.aten.add.Tensor(getitem_249, 1e-06);  getitem_249 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_104 = torch.ops.aten.sub.Tensor(add_246, getitem_250);  getitem_250 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_104, rsqrt_71);  sub_104 = rsqrt_71 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, arg122_1);  mul_307 = arg122_1 = None
        add_248 = torch.ops.aten.add.Tensor(mul_308, arg123_1);  mul_308 = arg123_1 = None
        view_551 = torch.ops.aten.view.default(add_248, [1568, 512]);  add_248 = None
        permute_264 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg125_1, view_551, permute_264);  arg125_1 = view_551 = permute_264 = None
        view_552 = torch.ops.aten.view.default(addmm_133, [8, 1, 196, 1536]);  addmm_133 = None
        view_553 = torch.ops.aten.view.default(view_552, [8, 1, 196, 3, 16, 32]);  view_552 = None
        permute_265 = torch.ops.aten.permute.default(view_553, [3, 0, 4, 1, 2, 5]);  view_553 = None
        unbind_33 = torch.ops.aten.unbind.int(permute_265);  permute_265 = None
        getitem_251 = unbind_33[0]
        getitem_252 = unbind_33[1]
        getitem_253 = unbind_33[2];  unbind_33 = None
        mul_309 = torch.ops.aten.mul.Scalar(getitem_251, 0.42044820762685725);  getitem_251 = None
        permute_266 = torch.ops.aten.permute.default(getitem_252, [0, 1, 2, 4, 3]);  getitem_252 = None
        mul_310 = torch.ops.aten.mul.Scalar(permute_266, 0.42044820762685725);  permute_266 = None
        expand_132 = torch.ops.aten.expand.default(mul_309, [8, 16, 1, 196, 32]);  mul_309 = None
        clone_240 = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
        view_554 = torch.ops.aten.view.default(clone_240, [128, 196, 32]);  clone_240 = None
        expand_133 = torch.ops.aten.expand.default(mul_310, [8, 16, 1, 32, 196]);  mul_310 = None
        clone_241 = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_555 = torch.ops.aten.view.default(clone_241, [128, 32, 196]);  clone_241 = None
        bmm_66 = torch.ops.aten.bmm.default(view_554, view_555);  view_554 = view_555 = None
        view_556 = torch.ops.aten.view.default(bmm_66, [8, 16, 1, 196, 196]);  bmm_66 = None
        amax_33 = torch.ops.aten.amax.default(view_556, [-1], True)
        sub_105 = torch.ops.aten.sub.Tensor(view_556, amax_33);  amax_33 = None
        exp_33 = torch.ops.aten.exp.default(sub_105);  sub_105 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
        eq_33 = torch.ops.aten.eq.Scalar(view_556, -inf);  view_556 = None
        logical_not_66 = torch.ops.aten.logical_not.default(eq_33);  eq_33 = None
        any_34 = torch.ops.aten.any.dim(logical_not_66, -1, True);  logical_not_66 = None
        logical_not_67 = torch.ops.aten.logical_not.default(any_34);  any_34 = None
        full_default_9 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33 = torch.ops.aten.where.self(logical_not_67, full_default_9, div_33);  logical_not_67 = full_default_9 = div_33 = None
        expand_134 = torch.ops.aten.expand.default(where_33, [8, 16, 1, 196, 196]);  where_33 = None
        view_557 = torch.ops.aten.view.default(expand_134, [128, 196, 196]);  expand_134 = None
        expand_135 = torch.ops.aten.expand.default(getitem_253, [8, 16, 1, 196, 32]);  getitem_253 = None
        clone_242 = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_558 = torch.ops.aten.view.default(clone_242, [128, 196, 32]);  clone_242 = None
        bmm_67 = torch.ops.aten.bmm.default(view_557, view_558);  view_557 = view_558 = None
        view_559 = torch.ops.aten.view.default(bmm_67, [8, 16, 1, 196, 32]);  bmm_67 = None
        permute_267 = torch.ops.aten.permute.default(view_559, [0, 2, 3, 4, 1]);  view_559 = None
        clone_243 = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_560 = torch.ops.aten.view.default(clone_243, [8, 1, 196, 512]);  clone_243 = None
        view_561 = torch.ops.aten.view.default(view_560, [1568, 512]);  view_560 = None
        permute_268 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg127_1, view_561, permute_268);  arg127_1 = view_561 = permute_268 = None
        view_562 = torch.ops.aten.view.default(addmm_134, [8, 1, 196, 512]);  addmm_134 = None
        add_249 = torch.ops.aten.add.Tensor(add_246, view_562);  add_246 = view_562 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(add_249, [3], correction = 0, keepdim = True)
        getitem_254 = var_mean_72[0]
        getitem_255 = var_mean_72[1];  var_mean_72 = None
        add_250 = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_106 = torch.ops.aten.sub.Tensor(add_249, getitem_255);  getitem_255 = None
        mul_311 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_72);  sub_106 = rsqrt_72 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_311, arg128_1);  mul_311 = arg128_1 = None
        add_251 = torch.ops.aten.add.Tensor(mul_312, arg129_1);  mul_312 = arg129_1 = None
        view_563 = torch.ops.aten.view.default(add_251, [1568, 512]);  add_251 = None
        permute_269 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg131_1, view_563, permute_269);  arg131_1 = view_563 = permute_269 = None
        view_564 = torch.ops.aten.view.default(addmm_135, [8, 1, 196, 2048]);  addmm_135 = None
        mul_313 = torch.ops.aten.mul.Tensor(view_564, 0.5)
        mul_314 = torch.ops.aten.mul.Tensor(view_564, 0.7071067811865476);  view_564 = None
        erf_33 = torch.ops.aten.erf.default(mul_314);  mul_314 = None
        add_252 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_313, add_252);  mul_313 = add_252 = None
        view_565 = torch.ops.aten.view.default(mul_315, [1568, 2048]);  mul_315 = None
        permute_270 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg133_1, view_565, permute_270);  arg133_1 = view_565 = permute_270 = None
        view_566 = torch.ops.aten.view.default(addmm_136, [8, 1, 196, 512]);  addmm_136 = None
        add_253 = torch.ops.aten.add.Tensor(add_249, view_566);  add_249 = view_566 = None
        var_mean_73 = torch.ops.aten.var_mean.correction(add_253, [3], correction = 0, keepdim = True)
        getitem_256 = var_mean_73[0]
        getitem_257 = var_mean_73[1];  var_mean_73 = None
        add_254 = torch.ops.aten.add.Tensor(getitem_256, 1e-06);  getitem_256 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_107 = torch.ops.aten.sub.Tensor(add_253, getitem_257);  getitem_257 = None
        mul_316 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_73);  sub_107 = rsqrt_73 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, arg134_1);  mul_316 = arg134_1 = None
        add_255 = torch.ops.aten.add.Tensor(mul_317, arg135_1);  mul_317 = arg135_1 = None
        view_567 = torch.ops.aten.view.default(add_255, [1568, 512]);  add_255 = None
        permute_271 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg137_1, view_567, permute_271);  arg137_1 = view_567 = permute_271 = None
        view_568 = torch.ops.aten.view.default(addmm_137, [8, 1, 196, 1536]);  addmm_137 = None
        view_569 = torch.ops.aten.view.default(view_568, [8, 1, 196, 3, 16, 32]);  view_568 = None
        permute_272 = torch.ops.aten.permute.default(view_569, [3, 0, 4, 1, 2, 5]);  view_569 = None
        unbind_34 = torch.ops.aten.unbind.int(permute_272);  permute_272 = None
        getitem_258 = unbind_34[0]
        getitem_259 = unbind_34[1]
        getitem_260 = unbind_34[2];  unbind_34 = None
        mul_318 = torch.ops.aten.mul.Scalar(getitem_258, 0.42044820762685725);  getitem_258 = None
        permute_273 = torch.ops.aten.permute.default(getitem_259, [0, 1, 2, 4, 3]);  getitem_259 = None
        mul_319 = torch.ops.aten.mul.Scalar(permute_273, 0.42044820762685725);  permute_273 = None
        expand_136 = torch.ops.aten.expand.default(mul_318, [8, 16, 1, 196, 32]);  mul_318 = None
        clone_247 = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
        view_570 = torch.ops.aten.view.default(clone_247, [128, 196, 32]);  clone_247 = None
        expand_137 = torch.ops.aten.expand.default(mul_319, [8, 16, 1, 32, 196]);  mul_319 = None
        clone_248 = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_571 = torch.ops.aten.view.default(clone_248, [128, 32, 196]);  clone_248 = None
        bmm_68 = torch.ops.aten.bmm.default(view_570, view_571);  view_570 = view_571 = None
        view_572 = torch.ops.aten.view.default(bmm_68, [8, 16, 1, 196, 196]);  bmm_68 = None
        amax_34 = torch.ops.aten.amax.default(view_572, [-1], True)
        sub_108 = torch.ops.aten.sub.Tensor(view_572, amax_34);  amax_34 = None
        exp_34 = torch.ops.aten.exp.default(sub_108);  sub_108 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_34 = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        eq_34 = torch.ops.aten.eq.Scalar(view_572, -inf);  view_572 = None
        logical_not_68 = torch.ops.aten.logical_not.default(eq_34);  eq_34 = None
        any_35 = torch.ops.aten.any.dim(logical_not_68, -1, True);  logical_not_68 = None
        logical_not_69 = torch.ops.aten.logical_not.default(any_35);  any_35 = None
        full_default_10 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34 = torch.ops.aten.where.self(logical_not_69, full_default_10, div_34);  logical_not_69 = full_default_10 = div_34 = None
        expand_138 = torch.ops.aten.expand.default(where_34, [8, 16, 1, 196, 196]);  where_34 = None
        view_573 = torch.ops.aten.view.default(expand_138, [128, 196, 196]);  expand_138 = None
        expand_139 = torch.ops.aten.expand.default(getitem_260, [8, 16, 1, 196, 32]);  getitem_260 = None
        clone_249 = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_574 = torch.ops.aten.view.default(clone_249, [128, 196, 32]);  clone_249 = None
        bmm_69 = torch.ops.aten.bmm.default(view_573, view_574);  view_573 = view_574 = None
        view_575 = torch.ops.aten.view.default(bmm_69, [8, 16, 1, 196, 32]);  bmm_69 = None
        permute_274 = torch.ops.aten.permute.default(view_575, [0, 2, 3, 4, 1]);  view_575 = None
        clone_250 = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
        view_576 = torch.ops.aten.view.default(clone_250, [8, 1, 196, 512]);  clone_250 = None
        view_577 = torch.ops.aten.view.default(view_576, [1568, 512]);  view_576 = None
        permute_275 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg139_1, view_577, permute_275);  arg139_1 = view_577 = permute_275 = None
        view_578 = torch.ops.aten.view.default(addmm_138, [8, 1, 196, 512]);  addmm_138 = None
        add_256 = torch.ops.aten.add.Tensor(add_253, view_578);  add_253 = view_578 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(add_256, [3], correction = 0, keepdim = True)
        getitem_261 = var_mean_74[0]
        getitem_262 = var_mean_74[1];  var_mean_74 = None
        add_257 = torch.ops.aten.add.Tensor(getitem_261, 1e-06);  getitem_261 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_109 = torch.ops.aten.sub.Tensor(add_256, getitem_262);  getitem_262 = None
        mul_320 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_74);  sub_109 = rsqrt_74 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_320, arg140_1);  mul_320 = arg140_1 = None
        add_258 = torch.ops.aten.add.Tensor(mul_321, arg141_1);  mul_321 = arg141_1 = None
        view_579 = torch.ops.aten.view.default(add_258, [1568, 512]);  add_258 = None
        permute_276 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg143_1, view_579, permute_276);  arg143_1 = view_579 = permute_276 = None
        view_580 = torch.ops.aten.view.default(addmm_139, [8, 1, 196, 2048]);  addmm_139 = None
        mul_322 = torch.ops.aten.mul.Tensor(view_580, 0.5)
        mul_323 = torch.ops.aten.mul.Tensor(view_580, 0.7071067811865476);  view_580 = None
        erf_34 = torch.ops.aten.erf.default(mul_323);  mul_323 = None
        add_259 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_324 = torch.ops.aten.mul.Tensor(mul_322, add_259);  mul_322 = add_259 = None
        view_581 = torch.ops.aten.view.default(mul_324, [1568, 2048]);  mul_324 = None
        permute_277 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg145_1, view_581, permute_277);  arg145_1 = view_581 = permute_277 = None
        view_582 = torch.ops.aten.view.default(addmm_140, [8, 1, 196, 512]);  addmm_140 = None
        add_260 = torch.ops.aten.add.Tensor(add_256, view_582);  add_256 = view_582 = None
        var_mean_75 = torch.ops.aten.var_mean.correction(add_260, [3], correction = 0, keepdim = True)
        getitem_263 = var_mean_75[0]
        getitem_264 = var_mean_75[1];  var_mean_75 = None
        add_261 = torch.ops.aten.add.Tensor(getitem_263, 1e-06);  getitem_263 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_110 = torch.ops.aten.sub.Tensor(add_260, getitem_264);  getitem_264 = None
        mul_325 = torch.ops.aten.mul.Tensor(sub_110, rsqrt_75);  sub_110 = rsqrt_75 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, arg146_1);  mul_325 = arg146_1 = None
        add_262 = torch.ops.aten.add.Tensor(mul_326, arg147_1);  mul_326 = arg147_1 = None
        view_583 = torch.ops.aten.view.default(add_262, [1568, 512]);  add_262 = None
        permute_278 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg149_1, view_583, permute_278);  arg149_1 = view_583 = permute_278 = None
        view_584 = torch.ops.aten.view.default(addmm_141, [8, 1, 196, 1536]);  addmm_141 = None
        view_585 = torch.ops.aten.view.default(view_584, [8, 1, 196, 3, 16, 32]);  view_584 = None
        permute_279 = torch.ops.aten.permute.default(view_585, [3, 0, 4, 1, 2, 5]);  view_585 = None
        unbind_35 = torch.ops.aten.unbind.int(permute_279);  permute_279 = None
        getitem_265 = unbind_35[0]
        getitem_266 = unbind_35[1]
        getitem_267 = unbind_35[2];  unbind_35 = None
        mul_327 = torch.ops.aten.mul.Scalar(getitem_265, 0.42044820762685725);  getitem_265 = None
        permute_280 = torch.ops.aten.permute.default(getitem_266, [0, 1, 2, 4, 3]);  getitem_266 = None
        mul_328 = torch.ops.aten.mul.Scalar(permute_280, 0.42044820762685725);  permute_280 = None
        expand_140 = torch.ops.aten.expand.default(mul_327, [8, 16, 1, 196, 32]);  mul_327 = None
        clone_254 = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_586 = torch.ops.aten.view.default(clone_254, [128, 196, 32]);  clone_254 = None
        expand_141 = torch.ops.aten.expand.default(mul_328, [8, 16, 1, 32, 196]);  mul_328 = None
        clone_255 = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_587 = torch.ops.aten.view.default(clone_255, [128, 32, 196]);  clone_255 = None
        bmm_70 = torch.ops.aten.bmm.default(view_586, view_587);  view_586 = view_587 = None
        view_588 = torch.ops.aten.view.default(bmm_70, [8, 16, 1, 196, 196]);  bmm_70 = None
        amax_35 = torch.ops.aten.amax.default(view_588, [-1], True)
        sub_111 = torch.ops.aten.sub.Tensor(view_588, amax_35);  amax_35 = None
        exp_35 = torch.ops.aten.exp.default(sub_111);  sub_111 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_35 = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
        eq_35 = torch.ops.aten.eq.Scalar(view_588, -inf);  view_588 = None
        logical_not_70 = torch.ops.aten.logical_not.default(eq_35);  eq_35 = None
        any_36 = torch.ops.aten.any.dim(logical_not_70, -1, True);  logical_not_70 = None
        logical_not_71 = torch.ops.aten.logical_not.default(any_36);  any_36 = None
        full_default_11 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35 = torch.ops.aten.where.self(logical_not_71, full_default_11, div_35);  logical_not_71 = full_default_11 = div_35 = None
        expand_142 = torch.ops.aten.expand.default(where_35, [8, 16, 1, 196, 196]);  where_35 = None
        view_589 = torch.ops.aten.view.default(expand_142, [128, 196, 196]);  expand_142 = None
        expand_143 = torch.ops.aten.expand.default(getitem_267, [8, 16, 1, 196, 32]);  getitem_267 = None
        clone_256 = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
        view_590 = torch.ops.aten.view.default(clone_256, [128, 196, 32]);  clone_256 = None
        bmm_71 = torch.ops.aten.bmm.default(view_589, view_590);  view_589 = view_590 = None
        view_591 = torch.ops.aten.view.default(bmm_71, [8, 16, 1, 196, 32]);  bmm_71 = None
        permute_281 = torch.ops.aten.permute.default(view_591, [0, 2, 3, 4, 1]);  view_591 = None
        clone_257 = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_592 = torch.ops.aten.view.default(clone_257, [8, 1, 196, 512]);  clone_257 = None
        view_593 = torch.ops.aten.view.default(view_592, [1568, 512]);  view_592 = None
        permute_282 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg151_1, view_593, permute_282);  arg151_1 = view_593 = permute_282 = None
        view_594 = torch.ops.aten.view.default(addmm_142, [8, 1, 196, 512]);  addmm_142 = None
        add_263 = torch.ops.aten.add.Tensor(add_260, view_594);  add_260 = view_594 = None
        var_mean_76 = torch.ops.aten.var_mean.correction(add_263, [3], correction = 0, keepdim = True)
        getitem_268 = var_mean_76[0]
        getitem_269 = var_mean_76[1];  var_mean_76 = None
        add_264 = torch.ops.aten.add.Tensor(getitem_268, 1e-06);  getitem_268 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_112 = torch.ops.aten.sub.Tensor(add_263, getitem_269);  getitem_269 = None
        mul_329 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_76);  sub_112 = rsqrt_76 = None
        mul_330 = torch.ops.aten.mul.Tensor(mul_329, arg152_1);  mul_329 = arg152_1 = None
        add_265 = torch.ops.aten.add.Tensor(mul_330, arg153_1);  mul_330 = arg153_1 = None
        view_595 = torch.ops.aten.view.default(add_265, [1568, 512]);  add_265 = None
        permute_283 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg155_1, view_595, permute_283);  arg155_1 = view_595 = permute_283 = None
        view_596 = torch.ops.aten.view.default(addmm_143, [8, 1, 196, 2048]);  addmm_143 = None
        mul_331 = torch.ops.aten.mul.Tensor(view_596, 0.5)
        mul_332 = torch.ops.aten.mul.Tensor(view_596, 0.7071067811865476);  view_596 = None
        erf_35 = torch.ops.aten.erf.default(mul_332);  mul_332 = None
        add_266 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_331, add_266);  mul_331 = add_266 = None
        view_597 = torch.ops.aten.view.default(mul_333, [1568, 2048]);  mul_333 = None
        permute_284 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg157_1, view_597, permute_284);  arg157_1 = view_597 = permute_284 = None
        view_598 = torch.ops.aten.view.default(addmm_144, [8, 1, 196, 512]);  addmm_144 = None
        add_267 = torch.ops.aten.add.Tensor(add_263, view_598);  add_263 = view_598 = None
        var_mean_77 = torch.ops.aten.var_mean.correction(add_267, [3], correction = 0, keepdim = True)
        getitem_270 = var_mean_77[0]
        getitem_271 = var_mean_77[1];  var_mean_77 = None
        add_268 = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_113 = torch.ops.aten.sub.Tensor(add_267, getitem_271);  getitem_271 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_77);  sub_113 = rsqrt_77 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, arg158_1);  mul_334 = arg158_1 = None
        add_269 = torch.ops.aten.add.Tensor(mul_335, arg159_1);  mul_335 = arg159_1 = None
        view_599 = torch.ops.aten.view.default(add_269, [1568, 512]);  add_269 = None
        permute_285 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg161_1, view_599, permute_285);  arg161_1 = view_599 = permute_285 = None
        view_600 = torch.ops.aten.view.default(addmm_145, [8, 1, 196, 1536]);  addmm_145 = None
        view_601 = torch.ops.aten.view.default(view_600, [8, 1, 196, 3, 16, 32]);  view_600 = None
        permute_286 = torch.ops.aten.permute.default(view_601, [3, 0, 4, 1, 2, 5]);  view_601 = None
        unbind_36 = torch.ops.aten.unbind.int(permute_286);  permute_286 = None
        getitem_272 = unbind_36[0]
        getitem_273 = unbind_36[1]
        getitem_274 = unbind_36[2];  unbind_36 = None
        mul_336 = torch.ops.aten.mul.Scalar(getitem_272, 0.42044820762685725);  getitem_272 = None
        permute_287 = torch.ops.aten.permute.default(getitem_273, [0, 1, 2, 4, 3]);  getitem_273 = None
        mul_337 = torch.ops.aten.mul.Scalar(permute_287, 0.42044820762685725);  permute_287 = None
        expand_144 = torch.ops.aten.expand.default(mul_336, [8, 16, 1, 196, 32]);  mul_336 = None
        clone_261 = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
        view_602 = torch.ops.aten.view.default(clone_261, [128, 196, 32]);  clone_261 = None
        expand_145 = torch.ops.aten.expand.default(mul_337, [8, 16, 1, 32, 196]);  mul_337 = None
        clone_262 = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_603 = torch.ops.aten.view.default(clone_262, [128, 32, 196]);  clone_262 = None
        bmm_72 = torch.ops.aten.bmm.default(view_602, view_603);  view_602 = view_603 = None
        view_604 = torch.ops.aten.view.default(bmm_72, [8, 16, 1, 196, 196]);  bmm_72 = None
        amax_36 = torch.ops.aten.amax.default(view_604, [-1], True)
        sub_114 = torch.ops.aten.sub.Tensor(view_604, amax_36);  amax_36 = None
        exp_36 = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36 = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        eq_36 = torch.ops.aten.eq.Scalar(view_604, -inf);  view_604 = None
        logical_not_72 = torch.ops.aten.logical_not.default(eq_36);  eq_36 = None
        any_37 = torch.ops.aten.any.dim(logical_not_72, -1, True);  logical_not_72 = None
        logical_not_73 = torch.ops.aten.logical_not.default(any_37);  any_37 = None
        full_default_12 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36 = torch.ops.aten.where.self(logical_not_73, full_default_12, div_36);  logical_not_73 = full_default_12 = div_36 = None
        expand_146 = torch.ops.aten.expand.default(where_36, [8, 16, 1, 196, 196]);  where_36 = None
        view_605 = torch.ops.aten.view.default(expand_146, [128, 196, 196]);  expand_146 = None
        expand_147 = torch.ops.aten.expand.default(getitem_274, [8, 16, 1, 196, 32]);  getitem_274 = None
        clone_263 = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_606 = torch.ops.aten.view.default(clone_263, [128, 196, 32]);  clone_263 = None
        bmm_73 = torch.ops.aten.bmm.default(view_605, view_606);  view_605 = view_606 = None
        view_607 = torch.ops.aten.view.default(bmm_73, [8, 16, 1, 196, 32]);  bmm_73 = None
        permute_288 = torch.ops.aten.permute.default(view_607, [0, 2, 3, 4, 1]);  view_607 = None
        clone_264 = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_608 = torch.ops.aten.view.default(clone_264, [8, 1, 196, 512]);  clone_264 = None
        view_609 = torch.ops.aten.view.default(view_608, [1568, 512]);  view_608 = None
        permute_289 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg163_1, view_609, permute_289);  arg163_1 = view_609 = permute_289 = None
        view_610 = torch.ops.aten.view.default(addmm_146, [8, 1, 196, 512]);  addmm_146 = None
        add_270 = torch.ops.aten.add.Tensor(add_267, view_610);  add_267 = view_610 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(add_270, [3], correction = 0, keepdim = True)
        getitem_275 = var_mean_78[0]
        getitem_276 = var_mean_78[1];  var_mean_78 = None
        add_271 = torch.ops.aten.add.Tensor(getitem_275, 1e-06);  getitem_275 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_115 = torch.ops.aten.sub.Tensor(add_270, getitem_276);  getitem_276 = None
        mul_338 = torch.ops.aten.mul.Tensor(sub_115, rsqrt_78);  sub_115 = rsqrt_78 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, arg164_1);  mul_338 = arg164_1 = None
        add_272 = torch.ops.aten.add.Tensor(mul_339, arg165_1);  mul_339 = arg165_1 = None
        view_611 = torch.ops.aten.view.default(add_272, [1568, 512]);  add_272 = None
        permute_290 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg167_1, view_611, permute_290);  arg167_1 = view_611 = permute_290 = None
        view_612 = torch.ops.aten.view.default(addmm_147, [8, 1, 196, 2048]);  addmm_147 = None
        mul_340 = torch.ops.aten.mul.Tensor(view_612, 0.5)
        mul_341 = torch.ops.aten.mul.Tensor(view_612, 0.7071067811865476);  view_612 = None
        erf_36 = torch.ops.aten.erf.default(mul_341);  mul_341 = None
        add_273 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_340, add_273);  mul_340 = add_273 = None
        view_613 = torch.ops.aten.view.default(mul_342, [1568, 2048]);  mul_342 = None
        permute_291 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg169_1, view_613, permute_291);  arg169_1 = view_613 = permute_291 = None
        view_614 = torch.ops.aten.view.default(addmm_148, [8, 1, 196, 512]);  addmm_148 = None
        add_274 = torch.ops.aten.add.Tensor(add_270, view_614);  add_270 = view_614 = None
        var_mean_79 = torch.ops.aten.var_mean.correction(add_274, [3], correction = 0, keepdim = True)
        getitem_277 = var_mean_79[0]
        getitem_278 = var_mean_79[1];  var_mean_79 = None
        add_275 = torch.ops.aten.add.Tensor(getitem_277, 1e-06);  getitem_277 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        sub_116 = torch.ops.aten.sub.Tensor(add_274, getitem_278);  getitem_278 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_79);  sub_116 = rsqrt_79 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, arg170_1);  mul_343 = arg170_1 = None
        add_276 = torch.ops.aten.add.Tensor(mul_344, arg171_1);  mul_344 = arg171_1 = None
        view_615 = torch.ops.aten.view.default(add_276, [1568, 512]);  add_276 = None
        permute_292 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg173_1, view_615, permute_292);  arg173_1 = view_615 = permute_292 = None
        view_616 = torch.ops.aten.view.default(addmm_149, [8, 1, 196, 1536]);  addmm_149 = None
        view_617 = torch.ops.aten.view.default(view_616, [8, 1, 196, 3, 16, 32]);  view_616 = None
        permute_293 = torch.ops.aten.permute.default(view_617, [3, 0, 4, 1, 2, 5]);  view_617 = None
        unbind_37 = torch.ops.aten.unbind.int(permute_293);  permute_293 = None
        getitem_279 = unbind_37[0]
        getitem_280 = unbind_37[1]
        getitem_281 = unbind_37[2];  unbind_37 = None
        mul_345 = torch.ops.aten.mul.Scalar(getitem_279, 0.42044820762685725);  getitem_279 = None
        permute_294 = torch.ops.aten.permute.default(getitem_280, [0, 1, 2, 4, 3]);  getitem_280 = None
        mul_346 = torch.ops.aten.mul.Scalar(permute_294, 0.42044820762685725);  permute_294 = None
        expand_148 = torch.ops.aten.expand.default(mul_345, [8, 16, 1, 196, 32]);  mul_345 = None
        clone_268 = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_618 = torch.ops.aten.view.default(clone_268, [128, 196, 32]);  clone_268 = None
        expand_149 = torch.ops.aten.expand.default(mul_346, [8, 16, 1, 32, 196]);  mul_346 = None
        clone_269 = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_619 = torch.ops.aten.view.default(clone_269, [128, 32, 196]);  clone_269 = None
        bmm_74 = torch.ops.aten.bmm.default(view_618, view_619);  view_618 = view_619 = None
        view_620 = torch.ops.aten.view.default(bmm_74, [8, 16, 1, 196, 196]);  bmm_74 = None
        amax_37 = torch.ops.aten.amax.default(view_620, [-1], True)
        sub_117 = torch.ops.aten.sub.Tensor(view_620, amax_37);  amax_37 = None
        exp_37 = torch.ops.aten.exp.default(sub_117);  sub_117 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37 = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        eq_37 = torch.ops.aten.eq.Scalar(view_620, -inf);  view_620 = None
        logical_not_74 = torch.ops.aten.logical_not.default(eq_37);  eq_37 = None
        any_38 = torch.ops.aten.any.dim(logical_not_74, -1, True);  logical_not_74 = None
        logical_not_75 = torch.ops.aten.logical_not.default(any_38);  any_38 = None
        full_default_13 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_37 = torch.ops.aten.where.self(logical_not_75, full_default_13, div_37);  logical_not_75 = full_default_13 = div_37 = None
        expand_150 = torch.ops.aten.expand.default(where_37, [8, 16, 1, 196, 196]);  where_37 = None
        view_621 = torch.ops.aten.view.default(expand_150, [128, 196, 196]);  expand_150 = None
        expand_151 = torch.ops.aten.expand.default(getitem_281, [8, 16, 1, 196, 32]);  getitem_281 = None
        clone_270 = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_622 = torch.ops.aten.view.default(clone_270, [128, 196, 32]);  clone_270 = None
        bmm_75 = torch.ops.aten.bmm.default(view_621, view_622);  view_621 = view_622 = None
        view_623 = torch.ops.aten.view.default(bmm_75, [8, 16, 1, 196, 32]);  bmm_75 = None
        permute_295 = torch.ops.aten.permute.default(view_623, [0, 2, 3, 4, 1]);  view_623 = None
        clone_271 = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
        view_624 = torch.ops.aten.view.default(clone_271, [8, 1, 196, 512]);  clone_271 = None
        view_625 = torch.ops.aten.view.default(view_624, [1568, 512]);  view_624 = None
        permute_296 = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg175_1, view_625, permute_296);  arg175_1 = view_625 = permute_296 = None
        view_626 = torch.ops.aten.view.default(addmm_150, [8, 1, 196, 512]);  addmm_150 = None
        add_277 = torch.ops.aten.add.Tensor(add_274, view_626);  add_274 = view_626 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(add_277, [3], correction = 0, keepdim = True)
        getitem_282 = var_mean_80[0]
        getitem_283 = var_mean_80[1];  var_mean_80 = None
        add_278 = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        sub_118 = torch.ops.aten.sub.Tensor(add_277, getitem_283);  getitem_283 = None
        mul_347 = torch.ops.aten.mul.Tensor(sub_118, rsqrt_80);  sub_118 = rsqrt_80 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_347, arg176_1);  mul_347 = arg176_1 = None
        add_279 = torch.ops.aten.add.Tensor(mul_348, arg177_1);  mul_348 = arg177_1 = None
        view_627 = torch.ops.aten.view.default(add_279, [1568, 512]);  add_279 = None
        permute_297 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg179_1, view_627, permute_297);  arg179_1 = view_627 = permute_297 = None
        view_628 = torch.ops.aten.view.default(addmm_151, [8, 1, 196, 2048]);  addmm_151 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_628, 0.5)
        mul_350 = torch.ops.aten.mul.Tensor(view_628, 0.7071067811865476);  view_628 = None
        erf_37 = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_280 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_349, add_280);  mul_349 = add_280 = None
        view_629 = torch.ops.aten.view.default(mul_351, [1568, 2048]);  mul_351 = None
        permute_298 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg181_1, view_629, permute_298);  arg181_1 = view_629 = permute_298 = None
        view_630 = torch.ops.aten.view.default(addmm_152, [8, 1, 196, 512]);  addmm_152 = None
        add_281 = torch.ops.aten.add.Tensor(add_277, view_630);  add_277 = view_630 = None
        var_mean_81 = torch.ops.aten.var_mean.correction(add_281, [3], correction = 0, keepdim = True)
        getitem_284 = var_mean_81[0]
        getitem_285 = var_mean_81[1];  var_mean_81 = None
        add_282 = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_119 = torch.ops.aten.sub.Tensor(add_281, getitem_285);  getitem_285 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_81);  sub_119 = rsqrt_81 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, arg182_1);  mul_352 = arg182_1 = None
        add_283 = torch.ops.aten.add.Tensor(mul_353, arg183_1);  mul_353 = arg183_1 = None
        view_631 = torch.ops.aten.view.default(add_283, [1568, 512]);  add_283 = None
        permute_299 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg185_1, view_631, permute_299);  arg185_1 = view_631 = permute_299 = None
        view_632 = torch.ops.aten.view.default(addmm_153, [8, 1, 196, 1536]);  addmm_153 = None
        view_633 = torch.ops.aten.view.default(view_632, [8, 1, 196, 3, 16, 32]);  view_632 = None
        permute_300 = torch.ops.aten.permute.default(view_633, [3, 0, 4, 1, 2, 5]);  view_633 = None
        unbind_38 = torch.ops.aten.unbind.int(permute_300);  permute_300 = None
        getitem_286 = unbind_38[0]
        getitem_287 = unbind_38[1]
        getitem_288 = unbind_38[2];  unbind_38 = None
        mul_354 = torch.ops.aten.mul.Scalar(getitem_286, 0.42044820762685725);  getitem_286 = None
        permute_301 = torch.ops.aten.permute.default(getitem_287, [0, 1, 2, 4, 3]);  getitem_287 = None
        mul_355 = torch.ops.aten.mul.Scalar(permute_301, 0.42044820762685725);  permute_301 = None
        expand_152 = torch.ops.aten.expand.default(mul_354, [8, 16, 1, 196, 32]);  mul_354 = None
        clone_275 = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
        view_634 = torch.ops.aten.view.default(clone_275, [128, 196, 32]);  clone_275 = None
        expand_153 = torch.ops.aten.expand.default(mul_355, [8, 16, 1, 32, 196]);  mul_355 = None
        clone_276 = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_635 = torch.ops.aten.view.default(clone_276, [128, 32, 196]);  clone_276 = None
        bmm_76 = torch.ops.aten.bmm.default(view_634, view_635);  view_634 = view_635 = None
        view_636 = torch.ops.aten.view.default(bmm_76, [8, 16, 1, 196, 196]);  bmm_76 = None
        amax_38 = torch.ops.aten.amax.default(view_636, [-1], True)
        sub_120 = torch.ops.aten.sub.Tensor(view_636, amax_38);  amax_38 = None
        exp_38 = torch.ops.aten.exp.default(sub_120);  sub_120 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38 = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        eq_38 = torch.ops.aten.eq.Scalar(view_636, -inf);  view_636 = None
        logical_not_76 = torch.ops.aten.logical_not.default(eq_38);  eq_38 = None
        any_39 = torch.ops.aten.any.dim(logical_not_76, -1, True);  logical_not_76 = None
        logical_not_77 = torch.ops.aten.logical_not.default(any_39);  any_39 = None
        full_default_14 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_38 = torch.ops.aten.where.self(logical_not_77, full_default_14, div_38);  logical_not_77 = full_default_14 = div_38 = None
        expand_154 = torch.ops.aten.expand.default(where_38, [8, 16, 1, 196, 196]);  where_38 = None
        view_637 = torch.ops.aten.view.default(expand_154, [128, 196, 196]);  expand_154 = None
        expand_155 = torch.ops.aten.expand.default(getitem_288, [8, 16, 1, 196, 32]);  getitem_288 = None
        clone_277 = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_638 = torch.ops.aten.view.default(clone_277, [128, 196, 32]);  clone_277 = None
        bmm_77 = torch.ops.aten.bmm.default(view_637, view_638);  view_637 = view_638 = None
        view_639 = torch.ops.aten.view.default(bmm_77, [8, 16, 1, 196, 32]);  bmm_77 = None
        permute_302 = torch.ops.aten.permute.default(view_639, [0, 2, 3, 4, 1]);  view_639 = None
        clone_278 = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_640 = torch.ops.aten.view.default(clone_278, [8, 1, 196, 512]);  clone_278 = None
        view_641 = torch.ops.aten.view.default(view_640, [1568, 512]);  view_640 = None
        permute_303 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg187_1, view_641, permute_303);  arg187_1 = view_641 = permute_303 = None
        view_642 = torch.ops.aten.view.default(addmm_154, [8, 1, 196, 512]);  addmm_154 = None
        add_284 = torch.ops.aten.add.Tensor(add_281, view_642);  add_281 = view_642 = None
        var_mean_82 = torch.ops.aten.var_mean.correction(add_284, [3], correction = 0, keepdim = True)
        getitem_289 = var_mean_82[0]
        getitem_290 = var_mean_82[1];  var_mean_82 = None
        add_285 = torch.ops.aten.add.Tensor(getitem_289, 1e-06);  getitem_289 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        sub_121 = torch.ops.aten.sub.Tensor(add_284, getitem_290);  getitem_290 = None
        mul_356 = torch.ops.aten.mul.Tensor(sub_121, rsqrt_82);  sub_121 = rsqrt_82 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_356, arg188_1);  mul_356 = arg188_1 = None
        add_286 = torch.ops.aten.add.Tensor(mul_357, arg189_1);  mul_357 = arg189_1 = None
        view_643 = torch.ops.aten.view.default(add_286, [1568, 512]);  add_286 = None
        permute_304 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg191_1, view_643, permute_304);  arg191_1 = view_643 = permute_304 = None
        view_644 = torch.ops.aten.view.default(addmm_155, [8, 1, 196, 2048]);  addmm_155 = None
        mul_358 = torch.ops.aten.mul.Tensor(view_644, 0.5)
        mul_359 = torch.ops.aten.mul.Tensor(view_644, 0.7071067811865476);  view_644 = None
        erf_38 = torch.ops.aten.erf.default(mul_359);  mul_359 = None
        add_287 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_360 = torch.ops.aten.mul.Tensor(mul_358, add_287);  mul_358 = add_287 = None
        view_645 = torch.ops.aten.view.default(mul_360, [1568, 2048]);  mul_360 = None
        permute_305 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg193_1, view_645, permute_305);  arg193_1 = view_645 = permute_305 = None
        view_646 = torch.ops.aten.view.default(addmm_156, [8, 1, 196, 512]);  addmm_156 = None
        add_288 = torch.ops.aten.add.Tensor(add_284, view_646);  add_284 = view_646 = None
        var_mean_83 = torch.ops.aten.var_mean.correction(add_288, [3], correction = 0, keepdim = True)
        getitem_291 = var_mean_83[0]
        getitem_292 = var_mean_83[1];  var_mean_83 = None
        add_289 = torch.ops.aten.add.Tensor(getitem_291, 1e-06);  getitem_291 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        sub_122 = torch.ops.aten.sub.Tensor(add_288, getitem_292);  getitem_292 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_122, rsqrt_83);  sub_122 = rsqrt_83 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, arg194_1);  mul_361 = arg194_1 = None
        add_290 = torch.ops.aten.add.Tensor(mul_362, arg195_1);  mul_362 = arg195_1 = None
        view_647 = torch.ops.aten.view.default(add_290, [1568, 512]);  add_290 = None
        permute_306 = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg197_1, view_647, permute_306);  arg197_1 = view_647 = permute_306 = None
        view_648 = torch.ops.aten.view.default(addmm_157, [8, 1, 196, 1536]);  addmm_157 = None
        view_649 = torch.ops.aten.view.default(view_648, [8, 1, 196, 3, 16, 32]);  view_648 = None
        permute_307 = torch.ops.aten.permute.default(view_649, [3, 0, 4, 1, 2, 5]);  view_649 = None
        unbind_39 = torch.ops.aten.unbind.int(permute_307);  permute_307 = None
        getitem_293 = unbind_39[0]
        getitem_294 = unbind_39[1]
        getitem_295 = unbind_39[2];  unbind_39 = None
        mul_363 = torch.ops.aten.mul.Scalar(getitem_293, 0.42044820762685725);  getitem_293 = None
        permute_308 = torch.ops.aten.permute.default(getitem_294, [0, 1, 2, 4, 3]);  getitem_294 = None
        mul_364 = torch.ops.aten.mul.Scalar(permute_308, 0.42044820762685725);  permute_308 = None
        expand_156 = torch.ops.aten.expand.default(mul_363, [8, 16, 1, 196, 32]);  mul_363 = None
        clone_282 = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_650 = torch.ops.aten.view.default(clone_282, [128, 196, 32]);  clone_282 = None
        expand_157 = torch.ops.aten.expand.default(mul_364, [8, 16, 1, 32, 196]);  mul_364 = None
        clone_283 = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_651 = torch.ops.aten.view.default(clone_283, [128, 32, 196]);  clone_283 = None
        bmm_78 = torch.ops.aten.bmm.default(view_650, view_651);  view_650 = view_651 = None
        view_652 = torch.ops.aten.view.default(bmm_78, [8, 16, 1, 196, 196]);  bmm_78 = None
        amax_39 = torch.ops.aten.amax.default(view_652, [-1], True)
        sub_123 = torch.ops.aten.sub.Tensor(view_652, amax_39);  amax_39 = None
        exp_39 = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39 = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        eq_39 = torch.ops.aten.eq.Scalar(view_652, -inf);  view_652 = None
        logical_not_78 = torch.ops.aten.logical_not.default(eq_39);  eq_39 = None
        any_40 = torch.ops.aten.any.dim(logical_not_78, -1, True);  logical_not_78 = None
        logical_not_79 = torch.ops.aten.logical_not.default(any_40);  any_40 = None
        full_default_15 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_39 = torch.ops.aten.where.self(logical_not_79, full_default_15, div_39);  logical_not_79 = full_default_15 = div_39 = None
        expand_158 = torch.ops.aten.expand.default(where_39, [8, 16, 1, 196, 196]);  where_39 = None
        view_653 = torch.ops.aten.view.default(expand_158, [128, 196, 196]);  expand_158 = None
        expand_159 = torch.ops.aten.expand.default(getitem_295, [8, 16, 1, 196, 32]);  getitem_295 = None
        clone_284 = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_654 = torch.ops.aten.view.default(clone_284, [128, 196, 32]);  clone_284 = None
        bmm_79 = torch.ops.aten.bmm.default(view_653, view_654);  view_653 = view_654 = None
        view_655 = torch.ops.aten.view.default(bmm_79, [8, 16, 1, 196, 32]);  bmm_79 = None
        permute_309 = torch.ops.aten.permute.default(view_655, [0, 2, 3, 4, 1]);  view_655 = None
        clone_285 = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
        view_656 = torch.ops.aten.view.default(clone_285, [8, 1, 196, 512]);  clone_285 = None
        view_657 = torch.ops.aten.view.default(view_656, [1568, 512]);  view_656 = None
        permute_310 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg199_1, view_657, permute_310);  arg199_1 = view_657 = permute_310 = None
        view_658 = torch.ops.aten.view.default(addmm_158, [8, 1, 196, 512]);  addmm_158 = None
        add_291 = torch.ops.aten.add.Tensor(add_288, view_658);  add_288 = view_658 = None
        var_mean_84 = torch.ops.aten.var_mean.correction(add_291, [3], correction = 0, keepdim = True)
        getitem_296 = var_mean_84[0]
        getitem_297 = var_mean_84[1];  var_mean_84 = None
        add_292 = torch.ops.aten.add.Tensor(getitem_296, 1e-06);  getitem_296 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_124 = torch.ops.aten.sub.Tensor(add_291, getitem_297);  getitem_297 = None
        mul_365 = torch.ops.aten.mul.Tensor(sub_124, rsqrt_84);  sub_124 = rsqrt_84 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_365, arg200_1);  mul_365 = arg200_1 = None
        add_293 = torch.ops.aten.add.Tensor(mul_366, arg201_1);  mul_366 = arg201_1 = None
        view_659 = torch.ops.aten.view.default(add_293, [1568, 512]);  add_293 = None
        permute_311 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg203_1, view_659, permute_311);  arg203_1 = view_659 = permute_311 = None
        view_660 = torch.ops.aten.view.default(addmm_159, [8, 1, 196, 2048]);  addmm_159 = None
        mul_367 = torch.ops.aten.mul.Tensor(view_660, 0.5)
        mul_368 = torch.ops.aten.mul.Tensor(view_660, 0.7071067811865476);  view_660 = None
        erf_39 = torch.ops.aten.erf.default(mul_368);  mul_368 = None
        add_294 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_367, add_294);  mul_367 = add_294 = None
        view_661 = torch.ops.aten.view.default(mul_369, [1568, 2048]);  mul_369 = None
        permute_312 = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg205_1, view_661, permute_312);  arg205_1 = view_661 = permute_312 = None
        view_662 = torch.ops.aten.view.default(addmm_160, [8, 1, 196, 512]);  addmm_160 = None
        add_295 = torch.ops.aten.add.Tensor(add_291, view_662);  add_291 = view_662 = None
        var_mean_85 = torch.ops.aten.var_mean.correction(add_295, [3], correction = 0, keepdim = True)
        getitem_298 = var_mean_85[0]
        getitem_299 = var_mean_85[1];  var_mean_85 = None
        add_296 = torch.ops.aten.add.Tensor(getitem_298, 1e-06);  getitem_298 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_125 = torch.ops.aten.sub.Tensor(add_295, getitem_299);  getitem_299 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_125, rsqrt_85);  sub_125 = rsqrt_85 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, arg206_1);  mul_370 = arg206_1 = None
        add_297 = torch.ops.aten.add.Tensor(mul_371, arg207_1);  mul_371 = arg207_1 = None
        view_663 = torch.ops.aten.view.default(add_297, [1568, 512]);  add_297 = None
        permute_313 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg209_1, view_663, permute_313);  arg209_1 = view_663 = permute_313 = None
        view_664 = torch.ops.aten.view.default(addmm_161, [8, 1, 196, 1536]);  addmm_161 = None
        view_665 = torch.ops.aten.view.default(view_664, [8, 1, 196, 3, 16, 32]);  view_664 = None
        permute_314 = torch.ops.aten.permute.default(view_665, [3, 0, 4, 1, 2, 5]);  view_665 = None
        unbind_40 = torch.ops.aten.unbind.int(permute_314);  permute_314 = None
        getitem_300 = unbind_40[0]
        getitem_301 = unbind_40[1]
        getitem_302 = unbind_40[2];  unbind_40 = None
        mul_372 = torch.ops.aten.mul.Scalar(getitem_300, 0.42044820762685725);  getitem_300 = None
        permute_315 = torch.ops.aten.permute.default(getitem_301, [0, 1, 2, 4, 3]);  getitem_301 = None
        mul_373 = torch.ops.aten.mul.Scalar(permute_315, 0.42044820762685725);  permute_315 = None
        expand_160 = torch.ops.aten.expand.default(mul_372, [8, 16, 1, 196, 32]);  mul_372 = None
        clone_289 = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
        view_666 = torch.ops.aten.view.default(clone_289, [128, 196, 32]);  clone_289 = None
        expand_161 = torch.ops.aten.expand.default(mul_373, [8, 16, 1, 32, 196]);  mul_373 = None
        clone_290 = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_667 = torch.ops.aten.view.default(clone_290, [128, 32, 196]);  clone_290 = None
        bmm_80 = torch.ops.aten.bmm.default(view_666, view_667);  view_666 = view_667 = None
        view_668 = torch.ops.aten.view.default(bmm_80, [8, 16, 1, 196, 196]);  bmm_80 = None
        amax_40 = torch.ops.aten.amax.default(view_668, [-1], True)
        sub_126 = torch.ops.aten.sub.Tensor(view_668, amax_40);  amax_40 = None
        exp_40 = torch.ops.aten.exp.default(sub_126);  sub_126 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40 = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        eq_40 = torch.ops.aten.eq.Scalar(view_668, -inf);  view_668 = None
        logical_not_80 = torch.ops.aten.logical_not.default(eq_40);  eq_40 = None
        any_41 = torch.ops.aten.any.dim(logical_not_80, -1, True);  logical_not_80 = None
        logical_not_81 = torch.ops.aten.logical_not.default(any_41);  any_41 = None
        full_default_16 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40 = torch.ops.aten.where.self(logical_not_81, full_default_16, div_40);  logical_not_81 = full_default_16 = div_40 = None
        expand_162 = torch.ops.aten.expand.default(where_40, [8, 16, 1, 196, 196]);  where_40 = None
        view_669 = torch.ops.aten.view.default(expand_162, [128, 196, 196]);  expand_162 = None
        expand_163 = torch.ops.aten.expand.default(getitem_302, [8, 16, 1, 196, 32]);  getitem_302 = None
        clone_291 = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_670 = torch.ops.aten.view.default(clone_291, [128, 196, 32]);  clone_291 = None
        bmm_81 = torch.ops.aten.bmm.default(view_669, view_670);  view_669 = view_670 = None
        view_671 = torch.ops.aten.view.default(bmm_81, [8, 16, 1, 196, 32]);  bmm_81 = None
        permute_316 = torch.ops.aten.permute.default(view_671, [0, 2, 3, 4, 1]);  view_671 = None
        clone_292 = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
        view_672 = torch.ops.aten.view.default(clone_292, [8, 1, 196, 512]);  clone_292 = None
        view_673 = torch.ops.aten.view.default(view_672, [1568, 512]);  view_672 = None
        permute_317 = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg211_1, view_673, permute_317);  arg211_1 = view_673 = permute_317 = None
        view_674 = torch.ops.aten.view.default(addmm_162, [8, 1, 196, 512]);  addmm_162 = None
        add_298 = torch.ops.aten.add.Tensor(add_295, view_674);  add_295 = view_674 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(add_298, [3], correction = 0, keepdim = True)
        getitem_303 = var_mean_86[0]
        getitem_304 = var_mean_86[1];  var_mean_86 = None
        add_299 = torch.ops.aten.add.Tensor(getitem_303, 1e-06);  getitem_303 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_127 = torch.ops.aten.sub.Tensor(add_298, getitem_304);  getitem_304 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_127, rsqrt_86);  sub_127 = rsqrt_86 = None
        mul_375 = torch.ops.aten.mul.Tensor(mul_374, arg212_1);  mul_374 = arg212_1 = None
        add_300 = torch.ops.aten.add.Tensor(mul_375, arg213_1);  mul_375 = arg213_1 = None
        view_675 = torch.ops.aten.view.default(add_300, [1568, 512]);  add_300 = None
        permute_318 = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg215_1, view_675, permute_318);  arg215_1 = view_675 = permute_318 = None
        view_676 = torch.ops.aten.view.default(addmm_163, [8, 1, 196, 2048]);  addmm_163 = None
        mul_376 = torch.ops.aten.mul.Tensor(view_676, 0.5)
        mul_377 = torch.ops.aten.mul.Tensor(view_676, 0.7071067811865476);  view_676 = None
        erf_40 = torch.ops.aten.erf.default(mul_377);  mul_377 = None
        add_301 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_378 = torch.ops.aten.mul.Tensor(mul_376, add_301);  mul_376 = add_301 = None
        view_677 = torch.ops.aten.view.default(mul_378, [1568, 2048]);  mul_378 = None
        permute_319 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg217_1, view_677, permute_319);  arg217_1 = view_677 = permute_319 = None
        view_678 = torch.ops.aten.view.default(addmm_164, [8, 1, 196, 512]);  addmm_164 = None
        add_302 = torch.ops.aten.add.Tensor(add_298, view_678);  add_298 = view_678 = None
        var_mean_87 = torch.ops.aten.var_mean.correction(add_302, [3], correction = 0, keepdim = True)
        getitem_305 = var_mean_87[0]
        getitem_306 = var_mean_87[1];  var_mean_87 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_305, 1e-06);  getitem_305 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_128 = torch.ops.aten.sub.Tensor(add_302, getitem_306);  getitem_306 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_128, rsqrt_87);  sub_128 = rsqrt_87 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, arg218_1);  mul_379 = arg218_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_380, arg219_1);  mul_380 = arg219_1 = None
        view_679 = torch.ops.aten.view.default(add_304, [1568, 512]);  add_304 = None
        permute_320 = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg221_1, view_679, permute_320);  arg221_1 = view_679 = permute_320 = None
        view_680 = torch.ops.aten.view.default(addmm_165, [8, 1, 196, 1536]);  addmm_165 = None
        view_681 = torch.ops.aten.view.default(view_680, [8, 1, 196, 3, 16, 32]);  view_680 = None
        permute_321 = torch.ops.aten.permute.default(view_681, [3, 0, 4, 1, 2, 5]);  view_681 = None
        unbind_41 = torch.ops.aten.unbind.int(permute_321);  permute_321 = None
        getitem_307 = unbind_41[0]
        getitem_308 = unbind_41[1]
        getitem_309 = unbind_41[2];  unbind_41 = None
        mul_381 = torch.ops.aten.mul.Scalar(getitem_307, 0.42044820762685725);  getitem_307 = None
        permute_322 = torch.ops.aten.permute.default(getitem_308, [0, 1, 2, 4, 3]);  getitem_308 = None
        mul_382 = torch.ops.aten.mul.Scalar(permute_322, 0.42044820762685725);  permute_322 = None
        expand_164 = torch.ops.aten.expand.default(mul_381, [8, 16, 1, 196, 32]);  mul_381 = None
        clone_296 = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_682 = torch.ops.aten.view.default(clone_296, [128, 196, 32]);  clone_296 = None
        expand_165 = torch.ops.aten.expand.default(mul_382, [8, 16, 1, 32, 196]);  mul_382 = None
        clone_297 = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_683 = torch.ops.aten.view.default(clone_297, [128, 32, 196]);  clone_297 = None
        bmm_82 = torch.ops.aten.bmm.default(view_682, view_683);  view_682 = view_683 = None
        view_684 = torch.ops.aten.view.default(bmm_82, [8, 16, 1, 196, 196]);  bmm_82 = None
        amax_41 = torch.ops.aten.amax.default(view_684, [-1], True)
        sub_129 = torch.ops.aten.sub.Tensor(view_684, amax_41);  amax_41 = None
        exp_41 = torch.ops.aten.exp.default(sub_129);  sub_129 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41 = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        eq_41 = torch.ops.aten.eq.Scalar(view_684, -inf);  view_684 = None
        logical_not_82 = torch.ops.aten.logical_not.default(eq_41);  eq_41 = None
        any_42 = torch.ops.aten.any.dim(logical_not_82, -1, True);  logical_not_82 = None
        logical_not_83 = torch.ops.aten.logical_not.default(any_42);  any_42 = None
        full_default_17 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_41 = torch.ops.aten.where.self(logical_not_83, full_default_17, div_41);  logical_not_83 = full_default_17 = div_41 = None
        expand_166 = torch.ops.aten.expand.default(where_41, [8, 16, 1, 196, 196]);  where_41 = None
        view_685 = torch.ops.aten.view.default(expand_166, [128, 196, 196]);  expand_166 = None
        expand_167 = torch.ops.aten.expand.default(getitem_309, [8, 16, 1, 196, 32]);  getitem_309 = None
        clone_298 = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_686 = torch.ops.aten.view.default(clone_298, [128, 196, 32]);  clone_298 = None
        bmm_83 = torch.ops.aten.bmm.default(view_685, view_686);  view_685 = view_686 = None
        view_687 = torch.ops.aten.view.default(bmm_83, [8, 16, 1, 196, 32]);  bmm_83 = None
        permute_323 = torch.ops.aten.permute.default(view_687, [0, 2, 3, 4, 1]);  view_687 = None
        clone_299 = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        view_688 = torch.ops.aten.view.default(clone_299, [8, 1, 196, 512]);  clone_299 = None
        view_689 = torch.ops.aten.view.default(view_688, [1568, 512]);  view_688 = None
        permute_324 = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg223_1, view_689, permute_324);  arg223_1 = view_689 = permute_324 = None
        view_690 = torch.ops.aten.view.default(addmm_166, [8, 1, 196, 512]);  addmm_166 = None
        add_305 = torch.ops.aten.add.Tensor(add_302, view_690);  add_302 = view_690 = None
        var_mean_88 = torch.ops.aten.var_mean.correction(add_305, [3], correction = 0, keepdim = True)
        getitem_310 = var_mean_88[0]
        getitem_311 = var_mean_88[1];  var_mean_88 = None
        add_306 = torch.ops.aten.add.Tensor(getitem_310, 1e-06);  getitem_310 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_130 = torch.ops.aten.sub.Tensor(add_305, getitem_311);  getitem_311 = None
        mul_383 = torch.ops.aten.mul.Tensor(sub_130, rsqrt_88);  sub_130 = rsqrt_88 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_383, arg224_1);  mul_383 = arg224_1 = None
        add_307 = torch.ops.aten.add.Tensor(mul_384, arg225_1);  mul_384 = arg225_1 = None
        view_691 = torch.ops.aten.view.default(add_307, [1568, 512]);  add_307 = None
        permute_325 = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg227_1, view_691, permute_325);  arg227_1 = view_691 = permute_325 = None
        view_692 = torch.ops.aten.view.default(addmm_167, [8, 1, 196, 2048]);  addmm_167 = None
        mul_385 = torch.ops.aten.mul.Tensor(view_692, 0.5)
        mul_386 = torch.ops.aten.mul.Tensor(view_692, 0.7071067811865476);  view_692 = None
        erf_41 = torch.ops.aten.erf.default(mul_386);  mul_386 = None
        add_308 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_385, add_308);  mul_385 = add_308 = None
        view_693 = torch.ops.aten.view.default(mul_387, [1568, 2048]);  mul_387 = None
        permute_326 = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg229_1, view_693, permute_326);  arg229_1 = view_693 = permute_326 = None
        view_694 = torch.ops.aten.view.default(addmm_168, [8, 1, 196, 512]);  addmm_168 = None
        add_309 = torch.ops.aten.add.Tensor(add_305, view_694);  add_305 = view_694 = None
        var_mean_89 = torch.ops.aten.var_mean.correction(add_309, [3], correction = 0, keepdim = True)
        getitem_312 = var_mean_89[0]
        getitem_313 = var_mean_89[1];  var_mean_89 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_312, 1e-06);  getitem_312 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_131 = torch.ops.aten.sub.Tensor(add_309, getitem_313);  getitem_313 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_131, rsqrt_89);  sub_131 = rsqrt_89 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, arg230_1);  mul_388 = arg230_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_389, arg231_1);  mul_389 = arg231_1 = None
        view_695 = torch.ops.aten.view.default(add_311, [1568, 512]);  add_311 = None
        permute_327 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg233_1, view_695, permute_327);  arg233_1 = view_695 = permute_327 = None
        view_696 = torch.ops.aten.view.default(addmm_169, [8, 1, 196, 1536]);  addmm_169 = None
        view_697 = torch.ops.aten.view.default(view_696, [8, 1, 196, 3, 16, 32]);  view_696 = None
        permute_328 = torch.ops.aten.permute.default(view_697, [3, 0, 4, 1, 2, 5]);  view_697 = None
        unbind_42 = torch.ops.aten.unbind.int(permute_328);  permute_328 = None
        getitem_314 = unbind_42[0]
        getitem_315 = unbind_42[1]
        getitem_316 = unbind_42[2];  unbind_42 = None
        mul_390 = torch.ops.aten.mul.Scalar(getitem_314, 0.42044820762685725);  getitem_314 = None
        permute_329 = torch.ops.aten.permute.default(getitem_315, [0, 1, 2, 4, 3]);  getitem_315 = None
        mul_391 = torch.ops.aten.mul.Scalar(permute_329, 0.42044820762685725);  permute_329 = None
        expand_168 = torch.ops.aten.expand.default(mul_390, [8, 16, 1, 196, 32]);  mul_390 = None
        clone_303 = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
        view_698 = torch.ops.aten.view.default(clone_303, [128, 196, 32]);  clone_303 = None
        expand_169 = torch.ops.aten.expand.default(mul_391, [8, 16, 1, 32, 196]);  mul_391 = None
        clone_304 = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_699 = torch.ops.aten.view.default(clone_304, [128, 32, 196]);  clone_304 = None
        bmm_84 = torch.ops.aten.bmm.default(view_698, view_699);  view_698 = view_699 = None
        view_700 = torch.ops.aten.view.default(bmm_84, [8, 16, 1, 196, 196]);  bmm_84 = None
        amax_42 = torch.ops.aten.amax.default(view_700, [-1], True)
        sub_132 = torch.ops.aten.sub.Tensor(view_700, amax_42);  amax_42 = None
        exp_42 = torch.ops.aten.exp.default(sub_132);  sub_132 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42 = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        eq_42 = torch.ops.aten.eq.Scalar(view_700, -inf);  view_700 = None
        logical_not_84 = torch.ops.aten.logical_not.default(eq_42);  eq_42 = None
        any_43 = torch.ops.aten.any.dim(logical_not_84, -1, True);  logical_not_84 = None
        logical_not_85 = torch.ops.aten.logical_not.default(any_43);  any_43 = None
        full_default_18 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_42 = torch.ops.aten.where.self(logical_not_85, full_default_18, div_42);  logical_not_85 = full_default_18 = div_42 = None
        expand_170 = torch.ops.aten.expand.default(where_42, [8, 16, 1, 196, 196]);  where_42 = None
        view_701 = torch.ops.aten.view.default(expand_170, [128, 196, 196]);  expand_170 = None
        expand_171 = torch.ops.aten.expand.default(getitem_316, [8, 16, 1, 196, 32]);  getitem_316 = None
        clone_305 = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_702 = torch.ops.aten.view.default(clone_305, [128, 196, 32]);  clone_305 = None
        bmm_85 = torch.ops.aten.bmm.default(view_701, view_702);  view_701 = view_702 = None
        view_703 = torch.ops.aten.view.default(bmm_85, [8, 16, 1, 196, 32]);  bmm_85 = None
        permute_330 = torch.ops.aten.permute.default(view_703, [0, 2, 3, 4, 1]);  view_703 = None
        clone_306 = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_704 = torch.ops.aten.view.default(clone_306, [8, 1, 196, 512]);  clone_306 = None
        view_705 = torch.ops.aten.view.default(view_704, [1568, 512]);  view_704 = None
        permute_331 = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg235_1, view_705, permute_331);  arg235_1 = view_705 = permute_331 = None
        view_706 = torch.ops.aten.view.default(addmm_170, [8, 1, 196, 512]);  addmm_170 = None
        add_312 = torch.ops.aten.add.Tensor(add_309, view_706);  add_309 = view_706 = None
        var_mean_90 = torch.ops.aten.var_mean.correction(add_312, [3], correction = 0, keepdim = True)
        getitem_317 = var_mean_90[0]
        getitem_318 = var_mean_90[1];  var_mean_90 = None
        add_313 = torch.ops.aten.add.Tensor(getitem_317, 1e-06);  getitem_317 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_133 = torch.ops.aten.sub.Tensor(add_312, getitem_318);  getitem_318 = None
        mul_392 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_90);  sub_133 = rsqrt_90 = None
        mul_393 = torch.ops.aten.mul.Tensor(mul_392, arg236_1);  mul_392 = arg236_1 = None
        add_314 = torch.ops.aten.add.Tensor(mul_393, arg237_1);  mul_393 = arg237_1 = None
        view_707 = torch.ops.aten.view.default(add_314, [1568, 512]);  add_314 = None
        permute_332 = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg239_1, view_707, permute_332);  arg239_1 = view_707 = permute_332 = None
        view_708 = torch.ops.aten.view.default(addmm_171, [8, 1, 196, 2048]);  addmm_171 = None
        mul_394 = torch.ops.aten.mul.Tensor(view_708, 0.5)
        mul_395 = torch.ops.aten.mul.Tensor(view_708, 0.7071067811865476);  view_708 = None
        erf_42 = torch.ops.aten.erf.default(mul_395);  mul_395 = None
        add_315 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_394, add_315);  mul_394 = add_315 = None
        view_709 = torch.ops.aten.view.default(mul_396, [1568, 2048]);  mul_396 = None
        permute_333 = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_172 = torch.ops.aten.addmm.default(arg241_1, view_709, permute_333);  arg241_1 = view_709 = permute_333 = None
        view_710 = torch.ops.aten.view.default(addmm_172, [8, 1, 196, 512]);  addmm_172 = None
        add_316 = torch.ops.aten.add.Tensor(add_312, view_710);  add_312 = view_710 = None
        var_mean_91 = torch.ops.aten.var_mean.correction(add_316, [3], correction = 0, keepdim = True)
        getitem_319 = var_mean_91[0]
        getitem_320 = var_mean_91[1];  var_mean_91 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_319, 1e-06);  getitem_319 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_134 = torch.ops.aten.sub.Tensor(add_316, getitem_320);  getitem_320 = None
        mul_397 = torch.ops.aten.mul.Tensor(sub_134, rsqrt_91);  sub_134 = rsqrt_91 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_397, arg242_1);  mul_397 = arg242_1 = None
        add_318 = torch.ops.aten.add.Tensor(mul_398, arg243_1);  mul_398 = arg243_1 = None
        view_711 = torch.ops.aten.view.default(add_318, [1568, 512]);  add_318 = None
        permute_334 = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_173 = torch.ops.aten.addmm.default(arg245_1, view_711, permute_334);  arg245_1 = view_711 = permute_334 = None
        view_712 = torch.ops.aten.view.default(addmm_173, [8, 1, 196, 1536]);  addmm_173 = None
        view_713 = torch.ops.aten.view.default(view_712, [8, 1, 196, 3, 16, 32]);  view_712 = None
        permute_335 = torch.ops.aten.permute.default(view_713, [3, 0, 4, 1, 2, 5]);  view_713 = None
        unbind_43 = torch.ops.aten.unbind.int(permute_335);  permute_335 = None
        getitem_321 = unbind_43[0]
        getitem_322 = unbind_43[1]
        getitem_323 = unbind_43[2];  unbind_43 = None
        mul_399 = torch.ops.aten.mul.Scalar(getitem_321, 0.42044820762685725);  getitem_321 = None
        permute_336 = torch.ops.aten.permute.default(getitem_322, [0, 1, 2, 4, 3]);  getitem_322 = None
        mul_400 = torch.ops.aten.mul.Scalar(permute_336, 0.42044820762685725);  permute_336 = None
        expand_172 = torch.ops.aten.expand.default(mul_399, [8, 16, 1, 196, 32]);  mul_399 = None
        clone_310 = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_714 = torch.ops.aten.view.default(clone_310, [128, 196, 32]);  clone_310 = None
        expand_173 = torch.ops.aten.expand.default(mul_400, [8, 16, 1, 32, 196]);  mul_400 = None
        clone_311 = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_715 = torch.ops.aten.view.default(clone_311, [128, 32, 196]);  clone_311 = None
        bmm_86 = torch.ops.aten.bmm.default(view_714, view_715);  view_714 = view_715 = None
        view_716 = torch.ops.aten.view.default(bmm_86, [8, 16, 1, 196, 196]);  bmm_86 = None
        amax_43 = torch.ops.aten.amax.default(view_716, [-1], True)
        sub_135 = torch.ops.aten.sub.Tensor(view_716, amax_43);  amax_43 = None
        exp_43 = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43 = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        eq_43 = torch.ops.aten.eq.Scalar(view_716, -inf);  view_716 = None
        logical_not_86 = torch.ops.aten.logical_not.default(eq_43);  eq_43 = None
        any_44 = torch.ops.aten.any.dim(logical_not_86, -1, True);  logical_not_86 = None
        logical_not_87 = torch.ops.aten.logical_not.default(any_44);  any_44 = None
        full_default_19 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43 = torch.ops.aten.where.self(logical_not_87, full_default_19, div_43);  logical_not_87 = full_default_19 = div_43 = None
        expand_174 = torch.ops.aten.expand.default(where_43, [8, 16, 1, 196, 196]);  where_43 = None
        view_717 = torch.ops.aten.view.default(expand_174, [128, 196, 196]);  expand_174 = None
        expand_175 = torch.ops.aten.expand.default(getitem_323, [8, 16, 1, 196, 32]);  getitem_323 = None
        clone_312 = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_718 = torch.ops.aten.view.default(clone_312, [128, 196, 32]);  clone_312 = None
        bmm_87 = torch.ops.aten.bmm.default(view_717, view_718);  view_717 = view_718 = None
        view_719 = torch.ops.aten.view.default(bmm_87, [8, 16, 1, 196, 32]);  bmm_87 = None
        permute_337 = torch.ops.aten.permute.default(view_719, [0, 2, 3, 4, 1]);  view_719 = None
        clone_313 = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_720 = torch.ops.aten.view.default(clone_313, [8, 1, 196, 512]);  clone_313 = None
        view_721 = torch.ops.aten.view.default(view_720, [1568, 512]);  view_720 = None
        permute_338 = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        addmm_174 = torch.ops.aten.addmm.default(arg247_1, view_721, permute_338);  arg247_1 = view_721 = permute_338 = None
        view_722 = torch.ops.aten.view.default(addmm_174, [8, 1, 196, 512]);  addmm_174 = None
        add_319 = torch.ops.aten.add.Tensor(add_316, view_722);  add_316 = view_722 = None
        var_mean_92 = torch.ops.aten.var_mean.correction(add_319, [3], correction = 0, keepdim = True)
        getitem_324 = var_mean_92[0]
        getitem_325 = var_mean_92[1];  var_mean_92 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_324, 1e-06);  getitem_324 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_136 = torch.ops.aten.sub.Tensor(add_319, getitem_325);  getitem_325 = None
        mul_401 = torch.ops.aten.mul.Tensor(sub_136, rsqrt_92);  sub_136 = rsqrt_92 = None
        mul_402 = torch.ops.aten.mul.Tensor(mul_401, arg248_1);  mul_401 = arg248_1 = None
        add_321 = torch.ops.aten.add.Tensor(mul_402, arg249_1);  mul_402 = arg249_1 = None
        view_723 = torch.ops.aten.view.default(add_321, [1568, 512]);  add_321 = None
        permute_339 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_175 = torch.ops.aten.addmm.default(arg251_1, view_723, permute_339);  arg251_1 = view_723 = permute_339 = None
        view_724 = torch.ops.aten.view.default(addmm_175, [8, 1, 196, 2048]);  addmm_175 = None
        mul_403 = torch.ops.aten.mul.Tensor(view_724, 0.5)
        mul_404 = torch.ops.aten.mul.Tensor(view_724, 0.7071067811865476);  view_724 = None
        erf_43 = torch.ops.aten.erf.default(mul_404);  mul_404 = None
        add_322 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_403, add_322);  mul_403 = add_322 = None
        view_725 = torch.ops.aten.view.default(mul_405, [1568, 2048]);  mul_405 = None
        permute_340 = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        addmm_176 = torch.ops.aten.addmm.default(arg253_1, view_725, permute_340);  arg253_1 = view_725 = permute_340 = None
        view_726 = torch.ops.aten.view.default(addmm_176, [8, 1, 196, 512]);  addmm_176 = None
        add_323 = torch.ops.aten.add.Tensor(add_319, view_726);  add_319 = view_726 = None
        var_mean_93 = torch.ops.aten.var_mean.correction(add_323, [3], correction = 0, keepdim = True)
        getitem_326 = var_mean_93[0]
        getitem_327 = var_mean_93[1];  var_mean_93 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_137 = torch.ops.aten.sub.Tensor(add_323, getitem_327);  getitem_327 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_137, rsqrt_93);  sub_137 = rsqrt_93 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, arg254_1);  mul_406 = arg254_1 = None
        add_325 = torch.ops.aten.add.Tensor(mul_407, arg255_1);  mul_407 = arg255_1 = None
        view_727 = torch.ops.aten.view.default(add_325, [1568, 512]);  add_325 = None
        permute_341 = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        addmm_177 = torch.ops.aten.addmm.default(arg257_1, view_727, permute_341);  arg257_1 = view_727 = permute_341 = None
        view_728 = torch.ops.aten.view.default(addmm_177, [8, 1, 196, 1536]);  addmm_177 = None
        view_729 = torch.ops.aten.view.default(view_728, [8, 1, 196, 3, 16, 32]);  view_728 = None
        permute_342 = torch.ops.aten.permute.default(view_729, [3, 0, 4, 1, 2, 5]);  view_729 = None
        unbind_44 = torch.ops.aten.unbind.int(permute_342);  permute_342 = None
        getitem_328 = unbind_44[0]
        getitem_329 = unbind_44[1]
        getitem_330 = unbind_44[2];  unbind_44 = None
        mul_408 = torch.ops.aten.mul.Scalar(getitem_328, 0.42044820762685725);  getitem_328 = None
        permute_343 = torch.ops.aten.permute.default(getitem_329, [0, 1, 2, 4, 3]);  getitem_329 = None
        mul_409 = torch.ops.aten.mul.Scalar(permute_343, 0.42044820762685725);  permute_343 = None
        expand_176 = torch.ops.aten.expand.default(mul_408, [8, 16, 1, 196, 32]);  mul_408 = None
        clone_317 = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
        view_730 = torch.ops.aten.view.default(clone_317, [128, 196, 32]);  clone_317 = None
        expand_177 = torch.ops.aten.expand.default(mul_409, [8, 16, 1, 32, 196]);  mul_409 = None
        clone_318 = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_731 = torch.ops.aten.view.default(clone_318, [128, 32, 196]);  clone_318 = None
        bmm_88 = torch.ops.aten.bmm.default(view_730, view_731);  view_730 = view_731 = None
        view_732 = torch.ops.aten.view.default(bmm_88, [8, 16, 1, 196, 196]);  bmm_88 = None
        amax_44 = torch.ops.aten.amax.default(view_732, [-1], True)
        sub_138 = torch.ops.aten.sub.Tensor(view_732, amax_44);  amax_44 = None
        exp_44 = torch.ops.aten.exp.default(sub_138);  sub_138 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44 = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        eq_44 = torch.ops.aten.eq.Scalar(view_732, -inf);  view_732 = None
        logical_not_88 = torch.ops.aten.logical_not.default(eq_44);  eq_44 = None
        any_45 = torch.ops.aten.any.dim(logical_not_88, -1, True);  logical_not_88 = None
        logical_not_89 = torch.ops.aten.logical_not.default(any_45);  any_45 = None
        full_default_20 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44 = torch.ops.aten.where.self(logical_not_89, full_default_20, div_44);  logical_not_89 = full_default_20 = div_44 = None
        expand_178 = torch.ops.aten.expand.default(where_44, [8, 16, 1, 196, 196]);  where_44 = None
        view_733 = torch.ops.aten.view.default(expand_178, [128, 196, 196]);  expand_178 = None
        expand_179 = torch.ops.aten.expand.default(getitem_330, [8, 16, 1, 196, 32]);  getitem_330 = None
        clone_319 = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_734 = torch.ops.aten.view.default(clone_319, [128, 196, 32]);  clone_319 = None
        bmm_89 = torch.ops.aten.bmm.default(view_733, view_734);  view_733 = view_734 = None
        view_735 = torch.ops.aten.view.default(bmm_89, [8, 16, 1, 196, 32]);  bmm_89 = None
        permute_344 = torch.ops.aten.permute.default(view_735, [0, 2, 3, 4, 1]);  view_735 = None
        clone_320 = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
        view_736 = torch.ops.aten.view.default(clone_320, [8, 1, 196, 512]);  clone_320 = None
        view_737 = torch.ops.aten.view.default(view_736, [1568, 512]);  view_736 = None
        permute_345 = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        addmm_178 = torch.ops.aten.addmm.default(arg259_1, view_737, permute_345);  arg259_1 = view_737 = permute_345 = None
        view_738 = torch.ops.aten.view.default(addmm_178, [8, 1, 196, 512]);  addmm_178 = None
        add_326 = torch.ops.aten.add.Tensor(add_323, view_738);  add_323 = view_738 = None
        var_mean_94 = torch.ops.aten.var_mean.correction(add_326, [3], correction = 0, keepdim = True)
        getitem_331 = var_mean_94[0]
        getitem_332 = var_mean_94[1];  var_mean_94 = None
        add_327 = torch.ops.aten.add.Tensor(getitem_331, 1e-06);  getitem_331 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        sub_139 = torch.ops.aten.sub.Tensor(add_326, getitem_332);  getitem_332 = None
        mul_410 = torch.ops.aten.mul.Tensor(sub_139, rsqrt_94);  sub_139 = rsqrt_94 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_410, arg260_1);  mul_410 = arg260_1 = None
        add_328 = torch.ops.aten.add.Tensor(mul_411, arg261_1);  mul_411 = arg261_1 = None
        view_739 = torch.ops.aten.view.default(add_328, [1568, 512]);  add_328 = None
        permute_346 = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_179 = torch.ops.aten.addmm.default(arg263_1, view_739, permute_346);  arg263_1 = view_739 = permute_346 = None
        view_740 = torch.ops.aten.view.default(addmm_179, [8, 1, 196, 2048]);  addmm_179 = None
        mul_412 = torch.ops.aten.mul.Tensor(view_740, 0.5)
        mul_413 = torch.ops.aten.mul.Tensor(view_740, 0.7071067811865476);  view_740 = None
        erf_44 = torch.ops.aten.erf.default(mul_413);  mul_413 = None
        add_329 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_414 = torch.ops.aten.mul.Tensor(mul_412, add_329);  mul_412 = add_329 = None
        view_741 = torch.ops.aten.view.default(mul_414, [1568, 2048]);  mul_414 = None
        permute_347 = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_180 = torch.ops.aten.addmm.default(arg265_1, view_741, permute_347);  arg265_1 = view_741 = permute_347 = None
        view_742 = torch.ops.aten.view.default(addmm_180, [8, 1, 196, 512]);  addmm_180 = None
        add_330 = torch.ops.aten.add.Tensor(add_326, view_742);  add_326 = view_742 = None
        var_mean_95 = torch.ops.aten.var_mean.correction(add_330, [3], correction = 0, keepdim = True)
        getitem_333 = var_mean_95[0]
        getitem_334 = var_mean_95[1];  var_mean_95 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_333, 1e-06);  getitem_333 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_140 = torch.ops.aten.sub.Tensor(add_330, getitem_334);  getitem_334 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_140, rsqrt_95);  sub_140 = rsqrt_95 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, arg266_1);  mul_415 = arg266_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_416, arg267_1);  mul_416 = arg267_1 = None
        view_743 = torch.ops.aten.view.default(add_332, [1568, 512]);  add_332 = None
        permute_348 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg269_1, view_743, permute_348);  arg269_1 = view_743 = permute_348 = None
        view_744 = torch.ops.aten.view.default(addmm_181, [8, 1, 196, 1536]);  addmm_181 = None
        view_745 = torch.ops.aten.view.default(view_744, [8, 1, 196, 3, 16, 32]);  view_744 = None
        permute_349 = torch.ops.aten.permute.default(view_745, [3, 0, 4, 1, 2, 5]);  view_745 = None
        unbind_45 = torch.ops.aten.unbind.int(permute_349);  permute_349 = None
        getitem_335 = unbind_45[0]
        getitem_336 = unbind_45[1]
        getitem_337 = unbind_45[2];  unbind_45 = None
        mul_417 = torch.ops.aten.mul.Scalar(getitem_335, 0.42044820762685725);  getitem_335 = None
        permute_350 = torch.ops.aten.permute.default(getitem_336, [0, 1, 2, 4, 3]);  getitem_336 = None
        mul_418 = torch.ops.aten.mul.Scalar(permute_350, 0.42044820762685725);  permute_350 = None
        expand_180 = torch.ops.aten.expand.default(mul_417, [8, 16, 1, 196, 32]);  mul_417 = None
        clone_324 = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_746 = torch.ops.aten.view.default(clone_324, [128, 196, 32]);  clone_324 = None
        expand_181 = torch.ops.aten.expand.default(mul_418, [8, 16, 1, 32, 196]);  mul_418 = None
        clone_325 = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_747 = torch.ops.aten.view.default(clone_325, [128, 32, 196]);  clone_325 = None
        bmm_90 = torch.ops.aten.bmm.default(view_746, view_747);  view_746 = view_747 = None
        view_748 = torch.ops.aten.view.default(bmm_90, [8, 16, 1, 196, 196]);  bmm_90 = None
        amax_45 = torch.ops.aten.amax.default(view_748, [-1], True)
        sub_141 = torch.ops.aten.sub.Tensor(view_748, amax_45);  amax_45 = None
        exp_45 = torch.ops.aten.exp.default(sub_141);  sub_141 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45 = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        eq_45 = torch.ops.aten.eq.Scalar(view_748, -inf);  view_748 = None
        logical_not_90 = torch.ops.aten.logical_not.default(eq_45);  eq_45 = None
        any_46 = torch.ops.aten.any.dim(logical_not_90, -1, True);  logical_not_90 = None
        logical_not_91 = torch.ops.aten.logical_not.default(any_46);  any_46 = None
        full_default_21 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_45 = torch.ops.aten.where.self(logical_not_91, full_default_21, div_45);  logical_not_91 = full_default_21 = div_45 = None
        expand_182 = torch.ops.aten.expand.default(where_45, [8, 16, 1, 196, 196]);  where_45 = None
        view_749 = torch.ops.aten.view.default(expand_182, [128, 196, 196]);  expand_182 = None
        expand_183 = torch.ops.aten.expand.default(getitem_337, [8, 16, 1, 196, 32]);  getitem_337 = None
        clone_326 = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_750 = torch.ops.aten.view.default(clone_326, [128, 196, 32]);  clone_326 = None
        bmm_91 = torch.ops.aten.bmm.default(view_749, view_750);  view_749 = view_750 = None
        view_751 = torch.ops.aten.view.default(bmm_91, [8, 16, 1, 196, 32]);  bmm_91 = None
        permute_351 = torch.ops.aten.permute.default(view_751, [0, 2, 3, 4, 1]);  view_751 = None
        clone_327 = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
        view_752 = torch.ops.aten.view.default(clone_327, [8, 1, 196, 512]);  clone_327 = None
        view_753 = torch.ops.aten.view.default(view_752, [1568, 512]);  view_752 = None
        permute_352 = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg271_1, view_753, permute_352);  arg271_1 = view_753 = permute_352 = None
        view_754 = torch.ops.aten.view.default(addmm_182, [8, 1, 196, 512]);  addmm_182 = None
        add_333 = torch.ops.aten.add.Tensor(add_330, view_754);  add_330 = view_754 = None
        var_mean_96 = torch.ops.aten.var_mean.correction(add_333, [3], correction = 0, keepdim = True)
        getitem_338 = var_mean_96[0]
        getitem_339 = var_mean_96[1];  var_mean_96 = None
        add_334 = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_142 = torch.ops.aten.sub.Tensor(add_333, getitem_339);  getitem_339 = None
        mul_419 = torch.ops.aten.mul.Tensor(sub_142, rsqrt_96);  sub_142 = rsqrt_96 = None
        mul_420 = torch.ops.aten.mul.Tensor(mul_419, arg272_1);  mul_419 = arg272_1 = None
        add_335 = torch.ops.aten.add.Tensor(mul_420, arg273_1);  mul_420 = arg273_1 = None
        view_755 = torch.ops.aten.view.default(add_335, [1568, 512]);  add_335 = None
        permute_353 = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg275_1, view_755, permute_353);  arg275_1 = view_755 = permute_353 = None
        view_756 = torch.ops.aten.view.default(addmm_183, [8, 1, 196, 2048]);  addmm_183 = None
        mul_421 = torch.ops.aten.mul.Tensor(view_756, 0.5)
        mul_422 = torch.ops.aten.mul.Tensor(view_756, 0.7071067811865476);  view_756 = None
        erf_45 = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_336 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_421, add_336);  mul_421 = add_336 = None
        view_757 = torch.ops.aten.view.default(mul_423, [1568, 2048]);  mul_423 = None
        permute_354 = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        addmm_184 = torch.ops.aten.addmm.default(arg277_1, view_757, permute_354);  arg277_1 = view_757 = permute_354 = None
        view_758 = torch.ops.aten.view.default(addmm_184, [8, 1, 196, 512]);  addmm_184 = None
        add_337 = torch.ops.aten.add.Tensor(add_333, view_758);  add_333 = view_758 = None
        var_mean_97 = torch.ops.aten.var_mean.correction(add_337, [3], correction = 0, keepdim = True)
        getitem_340 = var_mean_97[0]
        getitem_341 = var_mean_97[1];  var_mean_97 = None
        add_338 = torch.ops.aten.add.Tensor(getitem_340, 1e-06);  getitem_340 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        sub_143 = torch.ops.aten.sub.Tensor(add_337, getitem_341);  getitem_341 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_143, rsqrt_97);  sub_143 = rsqrt_97 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, arg278_1);  mul_424 = arg278_1 = None
        add_339 = torch.ops.aten.add.Tensor(mul_425, arg279_1);  mul_425 = arg279_1 = None
        view_759 = torch.ops.aten.view.default(add_339, [1568, 512]);  add_339 = None
        permute_355 = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg281_1, view_759, permute_355);  arg281_1 = view_759 = permute_355 = None
        view_760 = torch.ops.aten.view.default(addmm_185, [8, 1, 196, 1536]);  addmm_185 = None
        view_761 = torch.ops.aten.view.default(view_760, [8, 1, 196, 3, 16, 32]);  view_760 = None
        permute_356 = torch.ops.aten.permute.default(view_761, [3, 0, 4, 1, 2, 5]);  view_761 = None
        unbind_46 = torch.ops.aten.unbind.int(permute_356);  permute_356 = None
        getitem_342 = unbind_46[0]
        getitem_343 = unbind_46[1]
        getitem_344 = unbind_46[2];  unbind_46 = None
        mul_426 = torch.ops.aten.mul.Scalar(getitem_342, 0.42044820762685725);  getitem_342 = None
        permute_357 = torch.ops.aten.permute.default(getitem_343, [0, 1, 2, 4, 3]);  getitem_343 = None
        mul_427 = torch.ops.aten.mul.Scalar(permute_357, 0.42044820762685725);  permute_357 = None
        expand_184 = torch.ops.aten.expand.default(mul_426, [8, 16, 1, 196, 32]);  mul_426 = None
        clone_331 = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
        view_762 = torch.ops.aten.view.default(clone_331, [128, 196, 32]);  clone_331 = None
        expand_185 = torch.ops.aten.expand.default(mul_427, [8, 16, 1, 32, 196]);  mul_427 = None
        clone_332 = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_763 = torch.ops.aten.view.default(clone_332, [128, 32, 196]);  clone_332 = None
        bmm_92 = torch.ops.aten.bmm.default(view_762, view_763);  view_762 = view_763 = None
        view_764 = torch.ops.aten.view.default(bmm_92, [8, 16, 1, 196, 196]);  bmm_92 = None
        amax_46 = torch.ops.aten.amax.default(view_764, [-1], True)
        sub_144 = torch.ops.aten.sub.Tensor(view_764, amax_46);  amax_46 = None
        exp_46 = torch.ops.aten.exp.default(sub_144);  sub_144 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46 = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        eq_46 = torch.ops.aten.eq.Scalar(view_764, -inf);  view_764 = None
        logical_not_92 = torch.ops.aten.logical_not.default(eq_46);  eq_46 = None
        any_47 = torch.ops.aten.any.dim(logical_not_92, -1, True);  logical_not_92 = None
        logical_not_93 = torch.ops.aten.logical_not.default(any_47);  any_47 = None
        full_default_22 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46 = torch.ops.aten.where.self(logical_not_93, full_default_22, div_46);  logical_not_93 = full_default_22 = div_46 = None
        expand_186 = torch.ops.aten.expand.default(where_46, [8, 16, 1, 196, 196]);  where_46 = None
        view_765 = torch.ops.aten.view.default(expand_186, [128, 196, 196]);  expand_186 = None
        expand_187 = torch.ops.aten.expand.default(getitem_344, [8, 16, 1, 196, 32]);  getitem_344 = None
        clone_333 = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_766 = torch.ops.aten.view.default(clone_333, [128, 196, 32]);  clone_333 = None
        bmm_93 = torch.ops.aten.bmm.default(view_765, view_766);  view_765 = view_766 = None
        view_767 = torch.ops.aten.view.default(bmm_93, [8, 16, 1, 196, 32]);  bmm_93 = None
        permute_358 = torch.ops.aten.permute.default(view_767, [0, 2, 3, 4, 1]);  view_767 = None
        clone_334 = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_768 = torch.ops.aten.view.default(clone_334, [8, 1, 196, 512]);  clone_334 = None
        view_769 = torch.ops.aten.view.default(view_768, [1568, 512]);  view_768 = None
        permute_359 = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_186 = torch.ops.aten.addmm.default(arg283_1, view_769, permute_359);  arg283_1 = view_769 = permute_359 = None
        view_770 = torch.ops.aten.view.default(addmm_186, [8, 1, 196, 512]);  addmm_186 = None
        add_340 = torch.ops.aten.add.Tensor(add_337, view_770);  add_337 = view_770 = None
        var_mean_98 = torch.ops.aten.var_mean.correction(add_340, [3], correction = 0, keepdim = True)
        getitem_345 = var_mean_98[0]
        getitem_346 = var_mean_98[1];  var_mean_98 = None
        add_341 = torch.ops.aten.add.Tensor(getitem_345, 1e-06);  getitem_345 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_145 = torch.ops.aten.sub.Tensor(add_340, getitem_346);  getitem_346 = None
        mul_428 = torch.ops.aten.mul.Tensor(sub_145, rsqrt_98);  sub_145 = rsqrt_98 = None
        mul_429 = torch.ops.aten.mul.Tensor(mul_428, arg284_1);  mul_428 = arg284_1 = None
        add_342 = torch.ops.aten.add.Tensor(mul_429, arg285_1);  mul_429 = arg285_1 = None
        view_771 = torch.ops.aten.view.default(add_342, [1568, 512]);  add_342 = None
        permute_360 = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg287_1, view_771, permute_360);  arg287_1 = view_771 = permute_360 = None
        view_772 = torch.ops.aten.view.default(addmm_187, [8, 1, 196, 2048]);  addmm_187 = None
        mul_430 = torch.ops.aten.mul.Tensor(view_772, 0.5)
        mul_431 = torch.ops.aten.mul.Tensor(view_772, 0.7071067811865476);  view_772 = None
        erf_46 = torch.ops.aten.erf.default(mul_431);  mul_431 = None
        add_343 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_430, add_343);  mul_430 = add_343 = None
        view_773 = torch.ops.aten.view.default(mul_432, [1568, 2048]);  mul_432 = None
        permute_361 = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg289_1, view_773, permute_361);  arg289_1 = view_773 = permute_361 = None
        view_774 = torch.ops.aten.view.default(addmm_188, [8, 1, 196, 512]);  addmm_188 = None
        add_344 = torch.ops.aten.add.Tensor(add_340, view_774);  add_340 = view_774 = None
        var_mean_99 = torch.ops.aten.var_mean.correction(add_344, [3], correction = 0, keepdim = True)
        getitem_347 = var_mean_99[0]
        getitem_348 = var_mean_99[1];  var_mean_99 = None
        add_345 = torch.ops.aten.add.Tensor(getitem_347, 1e-06);  getitem_347 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
        sub_146 = torch.ops.aten.sub.Tensor(add_344, getitem_348);  getitem_348 = None
        mul_433 = torch.ops.aten.mul.Tensor(sub_146, rsqrt_99);  sub_146 = rsqrt_99 = None
        mul_434 = torch.ops.aten.mul.Tensor(mul_433, arg290_1);  mul_433 = arg290_1 = None
        add_346 = torch.ops.aten.add.Tensor(mul_434, arg291_1);  mul_434 = arg291_1 = None
        view_775 = torch.ops.aten.view.default(add_346, [1568, 512]);  add_346 = None
        permute_362 = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg293_1, view_775, permute_362);  arg293_1 = view_775 = permute_362 = None
        view_776 = torch.ops.aten.view.default(addmm_189, [8, 1, 196, 1536]);  addmm_189 = None
        view_777 = torch.ops.aten.view.default(view_776, [8, 1, 196, 3, 16, 32]);  view_776 = None
        permute_363 = torch.ops.aten.permute.default(view_777, [3, 0, 4, 1, 2, 5]);  view_777 = None
        unbind_47 = torch.ops.aten.unbind.int(permute_363);  permute_363 = None
        getitem_349 = unbind_47[0]
        getitem_350 = unbind_47[1]
        getitem_351 = unbind_47[2];  unbind_47 = None
        mul_435 = torch.ops.aten.mul.Scalar(getitem_349, 0.42044820762685725);  getitem_349 = None
        permute_364 = torch.ops.aten.permute.default(getitem_350, [0, 1, 2, 4, 3]);  getitem_350 = None
        mul_436 = torch.ops.aten.mul.Scalar(permute_364, 0.42044820762685725);  permute_364 = None
        expand_188 = torch.ops.aten.expand.default(mul_435, [8, 16, 1, 196, 32]);  mul_435 = None
        clone_338 = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_778 = torch.ops.aten.view.default(clone_338, [128, 196, 32]);  clone_338 = None
        expand_189 = torch.ops.aten.expand.default(mul_436, [8, 16, 1, 32, 196]);  mul_436 = None
        clone_339 = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_779 = torch.ops.aten.view.default(clone_339, [128, 32, 196]);  clone_339 = None
        bmm_94 = torch.ops.aten.bmm.default(view_778, view_779);  view_778 = view_779 = None
        view_780 = torch.ops.aten.view.default(bmm_94, [8, 16, 1, 196, 196]);  bmm_94 = None
        amax_47 = torch.ops.aten.amax.default(view_780, [-1], True)
        sub_147 = torch.ops.aten.sub.Tensor(view_780, amax_47);  amax_47 = None
        exp_47 = torch.ops.aten.exp.default(sub_147);  sub_147 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        eq_47 = torch.ops.aten.eq.Scalar(view_780, -inf);  view_780 = None
        logical_not_94 = torch.ops.aten.logical_not.default(eq_47);  eq_47 = None
        any_48 = torch.ops.aten.any.dim(logical_not_94, -1, True);  logical_not_94 = None
        logical_not_95 = torch.ops.aten.logical_not.default(any_48);  any_48 = None
        full_default_23 = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_47 = torch.ops.aten.where.self(logical_not_95, full_default_23, div_47);  logical_not_95 = full_default_23 = div_47 = None
        expand_190 = torch.ops.aten.expand.default(where_47, [8, 16, 1, 196, 196]);  where_47 = None
        view_781 = torch.ops.aten.view.default(expand_190, [128, 196, 196]);  expand_190 = None
        expand_191 = torch.ops.aten.expand.default(getitem_351, [8, 16, 1, 196, 32]);  getitem_351 = None
        clone_340 = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_782 = torch.ops.aten.view.default(clone_340, [128, 196, 32]);  clone_340 = None
        bmm_95 = torch.ops.aten.bmm.default(view_781, view_782);  view_781 = view_782 = None
        view_783 = torch.ops.aten.view.default(bmm_95, [8, 16, 1, 196, 32]);  bmm_95 = None
        permute_365 = torch.ops.aten.permute.default(view_783, [0, 2, 3, 4, 1]);  view_783 = None
        clone_341 = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_784 = torch.ops.aten.view.default(clone_341, [8, 1, 196, 512]);  clone_341 = None
        view_785 = torch.ops.aten.view.default(view_784, [1568, 512]);  view_784 = None
        permute_366 = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg295_1, view_785, permute_366);  arg295_1 = view_785 = permute_366 = None
        view_786 = torch.ops.aten.view.default(addmm_190, [8, 1, 196, 512]);  addmm_190 = None
        add_347 = torch.ops.aten.add.Tensor(add_344, view_786);  add_344 = view_786 = None
        var_mean_100 = torch.ops.aten.var_mean.correction(add_347, [3], correction = 0, keepdim = True)
        getitem_352 = var_mean_100[0]
        getitem_353 = var_mean_100[1];  var_mean_100 = None
        add_348 = torch.ops.aten.add.Tensor(getitem_352, 1e-06);  getitem_352 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        sub_148 = torch.ops.aten.sub.Tensor(add_347, getitem_353);  getitem_353 = None
        mul_437 = torch.ops.aten.mul.Tensor(sub_148, rsqrt_100);  sub_148 = rsqrt_100 = None
        mul_438 = torch.ops.aten.mul.Tensor(mul_437, arg296_1);  mul_437 = arg296_1 = None
        add_349 = torch.ops.aten.add.Tensor(mul_438, arg297_1);  mul_438 = arg297_1 = None
        view_787 = torch.ops.aten.view.default(add_349, [1568, 512]);  add_349 = None
        permute_367 = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_191 = torch.ops.aten.addmm.default(arg299_1, view_787, permute_367);  arg299_1 = view_787 = permute_367 = None
        view_788 = torch.ops.aten.view.default(addmm_191, [8, 1, 196, 2048]);  addmm_191 = None
        mul_439 = torch.ops.aten.mul.Tensor(view_788, 0.5)
        mul_440 = torch.ops.aten.mul.Tensor(view_788, 0.7071067811865476);  view_788 = None
        erf_47 = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_350 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_439, add_350);  mul_439 = add_350 = None
        view_789 = torch.ops.aten.view.default(mul_441, [1568, 2048]);  mul_441 = None
        permute_368 = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_192 = torch.ops.aten.addmm.default(arg301_1, view_789, permute_368);  arg301_1 = view_789 = permute_368 = None
        view_790 = torch.ops.aten.view.default(addmm_192, [8, 1, 196, 512]);  addmm_192 = None
        add_351 = torch.ops.aten.add.Tensor(add_347, view_790);  add_347 = view_790 = None
        view_791 = torch.ops.aten.view.default(add_351, [8, 1, 1, 14, 14, 512]);  add_351 = None
        permute_369 = torch.ops.aten.permute.default(view_791, [0, 1, 3, 2, 4, 5]);  view_791 = None
        view_792 = torch.ops.aten.view.default(permute_369, [8, 14, 14, 512]);  permute_369 = None
        permute_370 = torch.ops.aten.permute.default(view_792, [0, 3, 1, 2]);  view_792 = None
        permute_371 = torch.ops.aten.permute.default(permute_370, [0, 2, 3, 1]);  permute_370 = None
        var_mean_101 = torch.ops.aten.var_mean.correction(permute_371, [3], correction = 0, keepdim = True)
        getitem_354 = var_mean_101[0]
        getitem_355 = var_mean_101[1];  var_mean_101 = None
        add_352 = torch.ops.aten.add.Tensor(getitem_354, 1e-06);  getitem_354 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        sub_149 = torch.ops.aten.sub.Tensor(permute_371, getitem_355);  permute_371 = getitem_355 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_149, rsqrt_101);  sub_149 = rsqrt_101 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, arg302_1);  mul_442 = arg302_1 = None
        add_353 = torch.ops.aten.add.Tensor(mul_443, arg303_1);  mul_443 = arg303_1 = None
        permute_372 = torch.ops.aten.permute.default(add_353, [0, 3, 1, 2]);  add_353 = None
        mean_1 = torch.ops.aten.mean.dim(permute_372, [-1, -2], True);  permute_372 = None
        as_strided_1 = torch.ops.aten.as_strided.default(mean_1, [8, 512, 1, 1], [512, 1, 512, 512]);  mean_1 = None
        view_793 = torch.ops.aten.view.default(as_strided_1, [8, 512]);  as_strided_1 = None
        permute_373 = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        addmm_193 = torch.ops.aten.addmm.default(arg305_1, view_793, permute_373);  arg305_1 = view_793 = permute_373 = None
        return (addmm_193,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 3, 4, 4), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1605632, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 16, 196, 128), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384, 128), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf7, (384,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128, 128), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512, 128), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128, 512), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 128), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf19, (384,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128, 128), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512, 128), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128, 512), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256, 128, 3, 3), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 802816, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1, 4, 196, 256), is_leaf=True)  # arg32_1
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
    buf41 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1024, 256), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1024,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256, 1024), is_leaf=True)  # arg43_1
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
    buf53 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf53, (1024, 256), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf54, (1024,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256, 1024), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512, 256, 3, 3), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf59, (512,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf60, (512,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 401408, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1, 1, 196, 512), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf64, (1536, 512), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf65, (1536,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512, 512), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf70, (2048, 512), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf71, (2048,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512, 2048), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf76, (1536, 512), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf77, (1536,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf78, (512, 512), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf82, (2048, 512), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf83, (2048,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf84, (512, 2048), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1536, 512), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf89, (1536,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512, 512), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf94, (2048, 512), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf95, (2048,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf96, (512, 2048), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1536, 512), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1536,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf102, (512, 512), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf103, (512,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf106, (2048, 512), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf107, (2048,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf108, (512, 2048), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf112, (1536, 512), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf113, (1536,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512, 512), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf118, (2048, 512), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf119, (2048,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512, 2048), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf121, (512,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf122, (512,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf123, (512,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1536, 512), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1536,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf126, (512, 512), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf127, (512,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf130, (2048, 512), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf131, (2048,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512, 2048), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf133, (512,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf134, (512,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf135, (512,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1536, 512), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1536,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512, 512), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf141, (512,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf142, (2048, 512), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf143, (2048,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf144, (512, 2048), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf145, (512,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf146, (512,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf147, (512,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1536, 512), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1536,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf150, (512, 512), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf151, (512,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf152, (512,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf153, (512,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf154, (2048, 512), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf155, (2048,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf156, (512, 2048), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf157, (512,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf158, (512,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf159, (512,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf160, (1536, 512), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf161, (1536,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512, 512), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf164, (512,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf165, (512,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf166, (2048, 512), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf167, (2048,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf168, (512, 2048), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf169, (512,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf170, (512,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf171, (512,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1536, 512), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1536,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512, 512), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf176, (512,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf177, (512,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf178, (2048, 512), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf179, (2048,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512, 2048), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf183, (512,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1536, 512), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1536,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512, 512), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf187, (512,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf188, (512,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf189, (512,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf190, (2048, 512), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf191, (2048,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512, 2048), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf194, (512,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf195, (512,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1536, 512), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1536,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf198, (512, 512), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf199, (512,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf200, (512,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf201, (512,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf202, (2048, 512), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf203, (2048,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512, 2048), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf205, (512,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf206, (512,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf207, (512,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf208, (1536, 512), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf209, (1536,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf210, (512, 512), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf211, (512,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf212, (512,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf213, (512,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf214, (2048, 512), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf215, (2048,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf216, (512, 2048), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf217, (512,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf218, (512,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf219, (512,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf220, (1536, 512), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf221, (1536,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf222, (512, 512), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf224, (512,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf225, (512,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf226, (2048, 512), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf227, (2048,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf228, (512, 2048), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf229, (512,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf230, (512,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf231, (512,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1536, 512), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1536,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf234, (512, 512), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf235, (512,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf236, (512,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf237, (512,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf238, (2048, 512), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf239, (2048,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf240, (512, 2048), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf241, (512,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf242, (512,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf243, (512,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf244, (1536, 512), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1536,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf246, (512, 512), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf247, (512,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf248, (512,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf249, (512,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf250, (2048, 512), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf251, (2048,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf252, (512, 2048), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf253, (512,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf254, (512,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf255, (512,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1536, 512), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf257, (1536,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf258, (512, 512), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf259, (512,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf260, (512,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf261, (512,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf262, (2048, 512), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf263, (2048,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf264, (512, 2048), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf265, (512,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf266, (512,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf267, (512,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1536, 512), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf269, (1536,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf270, (512, 512), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf271, (512,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf272, (512,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf273, (512,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf274, (2048, 512), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf275, (2048,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf276, (512, 2048), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf277, (512,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf278, (512,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf279, (512,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf280, (1536, 512), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf281, (1536,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf282, (512, 512), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf283, (512,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf284, (512,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf285, (512,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf286, (2048, 512), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf287, (2048,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf288, (512, 2048), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf289, (512,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf290, (512,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf291, (512,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1536, 512), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1536,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf294, (512, 512), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf295, (512,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf296, (512,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf297, (512,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf298, (2048, 512), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf299, (2048,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf300, (512, 2048), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf301, (512,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf302, (512,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf303, (512,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf304, (1000, 512), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1000,), is_leaf=True)  # arg305_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)