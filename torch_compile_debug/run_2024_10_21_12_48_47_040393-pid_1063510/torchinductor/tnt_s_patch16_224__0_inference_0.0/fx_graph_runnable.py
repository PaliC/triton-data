
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1):
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg2_1, arg3_1, [4, 4], [3, 3], [1, 1], False, [0, 0], 1);  arg0_1 = arg2_1 = arg3_1 = None
        iota_4 = torch.ops.prims.iota.default(14, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        iota_5 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        add_214 = torch.ops.aten.add.Tensor(unsqueeze_6, unsqueeze_7);  unsqueeze_6 = unsqueeze_7 = None
        iota_6 = torch.ops.prims.iota.default(14, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None
        iota_7 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        add_215 = torch.ops.aten.add.Tensor(unsqueeze_8, unsqueeze_9);  unsqueeze_8 = unsqueeze_9 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(add_214, -1);  add_214 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        index_1 = torch.ops.aten.index.Tensor(convolution_1, [None, None, unsqueeze_11, add_215]);  convolution_1 = unsqueeze_11 = add_215 = None
        permute_233 = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
        clone_185 = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
        view_498 = torch.ops.aten.view.default(clone_185, [8, 384, 196]);  clone_185 = None
        permute_234 = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
        clone_186 = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_499 = torch.ops.aten.view.default(clone_186, [1568, 24, 4, 4]);  clone_186 = None
        add_216 = torch.ops.aten.add.Tensor(view_499, arg1_1);  view_499 = arg1_1 = None
        view_500 = torch.ops.aten.view.default(add_216, [1568, 24, -1]);  add_216 = None
        permute_235 = torch.ops.aten.permute.default(view_500, [0, 2, 1]);  view_500 = None
        clone_187 = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format)
        view_501 = torch.ops.aten.view.default(clone_187, [8, 196, 384]);  clone_187 = None
        var_mean_63 = torch.ops.aten.var_mean.correction(view_501, [2], correction = 0, keepdim = True)
        getitem_174 = var_mean_63[0]
        getitem_175 = var_mean_63[1];  var_mean_63 = None
        add_217 = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        sub_87 = torch.ops.aten.sub.Tensor(view_501, getitem_175);  view_501 = getitem_175 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_63);  sub_87 = rsqrt_63 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, arg4_1);  mul_222 = arg4_1 = None
        add_218 = torch.ops.aten.add.Tensor(mul_223, arg5_1);  mul_223 = arg5_1 = None
        view_502 = torch.ops.aten.view.default(add_218, [1568, 384]);  add_218 = None
        permute_236 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg7_1, view_502, permute_236);  arg7_1 = view_502 = permute_236 = None
        view_503 = torch.ops.aten.view.default(addmm_86, [8, 196, 384]);  addmm_86 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(view_503, [2], correction = 0, keepdim = True)
        getitem_176 = var_mean_64[0]
        getitem_177 = var_mean_64[1];  var_mean_64 = None
        add_219 = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        sub_88 = torch.ops.aten.sub.Tensor(view_503, getitem_177);  view_503 = getitem_177 = None
        mul_224 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_64);  sub_88 = rsqrt_64 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, arg8_1);  mul_224 = arg8_1 = None
        add_220 = torch.ops.aten.add.Tensor(mul_225, arg9_1);  mul_225 = arg9_1 = None
        expand_97 = torch.ops.aten.expand.default(arg10_1, [8, -1, -1]);  arg10_1 = None
        cat_13 = torch.ops.aten.cat.default([expand_97, add_220], 1);  expand_97 = add_220 = None
        add_221 = torch.ops.aten.add.Tensor(cat_13, arg11_1);  cat_13 = arg11_1 = None
        clone_189 = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_189, [2], correction = 0, keepdim = True)
        getitem_178 = var_mean_65[0]
        getitem_179 = var_mean_65[1];  var_mean_65 = None
        add_222 = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_89 = torch.ops.aten.sub.Tensor(clone_189, getitem_179);  clone_189 = getitem_179 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_65);  sub_89 = rsqrt_65 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, arg12_1);  mul_226 = arg12_1 = None
        add_223 = torch.ops.aten.add.Tensor(mul_227, arg13_1);  mul_227 = arg13_1 = None
        permute_237 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        view_504 = torch.ops.aten.view.default(add_223, [25088, 24])
        mm_48 = torch.ops.aten.mm.default(view_504, permute_237);  view_504 = permute_237 = None
        view_505 = torch.ops.aten.view.default(mm_48, [1568, 16, 48]);  mm_48 = None
        view_506 = torch.ops.aten.view.default(view_505, [1568, 16, 2, 4, 6]);  view_505 = None
        permute_238 = torch.ops.aten.permute.default(view_506, [2, 0, 3, 1, 4]);  view_506 = None
        unbind_24 = torch.ops.aten.unbind.int(permute_238);  permute_238 = None
        getitem_180 = unbind_24[0]
        getitem_181 = unbind_24[1];  unbind_24 = None
        permute_239 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        view_507 = torch.ops.aten.view.default(add_223, [25088, 24]);  add_223 = None
        mm_49 = torch.ops.aten.mm.default(view_507, permute_239);  view_507 = permute_239 = None
        view_508 = torch.ops.aten.view.default(mm_49, [1568, 16, 24]);  mm_49 = None
        view_509 = torch.ops.aten.view.default(view_508, [1568, 16, 4, -1]);  view_508 = None
        permute_240 = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
        permute_241 = torch.ops.aten.permute.default(getitem_181, [0, 1, 3, 2]);  getitem_181 = None
        expand_98 = torch.ops.aten.expand.default(getitem_180, [1568, 4, 16, 6]);  getitem_180 = None
        clone_190 = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_510 = torch.ops.aten.view.default(clone_190, [6272, 16, 6]);  clone_190 = None
        expand_99 = torch.ops.aten.expand.default(permute_241, [1568, 4, 6, 16]);  permute_241 = None
        clone_191 = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_511 = torch.ops.aten.view.default(clone_191, [6272, 6, 16]);  clone_191 = None
        bmm_48 = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
        view_512 = torch.ops.aten.view.default(bmm_48, [1568, 4, 16, 16]);  bmm_48 = None
        mul_tensor_46 = torch.ops.aten.mul.Tensor(view_512, 1);  view_512 = None
        amax_default_23 = torch.ops.aten.amax.default(mul_tensor_46, [-1], True)
        sub_tensor_23 = torch.ops.aten.sub.Tensor(mul_tensor_46, amax_default_23);  mul_tensor_46 = amax_default_23 = None
        mul_tensor_47 = torch.ops.aten.mul.Tensor(sub_tensor_23, 0.408248290463863);  sub_tensor_23 = None
        exp_24 = torch.ops.aten.exp.default(mul_tensor_47);  mul_tensor_47 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24 = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        expand_100 = torch.ops.aten.expand.default(div_24, [1568, 4, 16, 16]);  div_24 = None
        view_513 = torch.ops.aten.view.default(expand_100, [6272, 16, 16]);  expand_100 = None
        expand_101 = torch.ops.aten.expand.default(permute_240, [1568, 4, 16, 6]);  permute_240 = None
        clone_192 = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_514 = torch.ops.aten.view.default(clone_192, [6272, 16, 6]);  clone_192 = None
        bmm_49 = torch.ops.aten.bmm.default(view_513, view_514);  view_513 = view_514 = None
        view_515 = torch.ops.aten.view.default(bmm_49, [1568, 4, 16, 6]);  bmm_49 = None
        permute_242 = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
        clone_193 = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_516 = torch.ops.aten.view.default(clone_193, [1568, 16, 24]);  clone_193 = None
        view_517 = torch.ops.aten.view.default(view_516, [25088, 24]);  view_516 = None
        permute_243 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg17_1, view_517, permute_243);  arg17_1 = view_517 = permute_243 = None
        view_518 = torch.ops.aten.view.default(addmm_87, [1568, 16, 24]);  addmm_87 = None
        add_224 = torch.ops.aten.add.Tensor(permute_235, view_518);  permute_235 = view_518 = None
        clone_194 = torch.ops.aten.clone.default(add_224, memory_format = torch.contiguous_format)
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_194, [2], correction = 0, keepdim = True)
        getitem_182 = var_mean_66[0]
        getitem_183 = var_mean_66[1];  var_mean_66 = None
        add_225 = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_91 = torch.ops.aten.sub.Tensor(clone_194, getitem_183);  clone_194 = getitem_183 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_66);  sub_91 = rsqrt_66 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, arg18_1);  mul_229 = arg18_1 = None
        add_226 = torch.ops.aten.add.Tensor(mul_230, arg19_1);  mul_230 = arg19_1 = None
        view_519 = torch.ops.aten.view.default(add_226, [25088, 24]);  add_226 = None
        permute_244 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg21_1, view_519, permute_244);  arg21_1 = view_519 = permute_244 = None
        view_520 = torch.ops.aten.view.default(addmm_88, [1568, 16, 96]);  addmm_88 = None
        mul_231 = torch.ops.aten.mul.Tensor(view_520, 0.5)
        mul_232 = torch.ops.aten.mul.Tensor(view_520, 0.7071067811865476);  view_520 = None
        erf_24 = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_227 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_231, add_227);  mul_231 = add_227 = None
        view_521 = torch.ops.aten.view.default(mul_233, [25088, 96]);  mul_233 = None
        permute_245 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg23_1, view_521, permute_245);  arg23_1 = view_521 = permute_245 = None
        view_522 = torch.ops.aten.view.default(addmm_89, [1568, 16, 24]);  addmm_89 = None
        add_228 = torch.ops.aten.add.Tensor(add_224, view_522);  add_224 = view_522 = None
        slice_55 = torch.ops.aten.slice.Tensor(add_221, 1, 0, 1)
        slice_57 = torch.ops.aten.slice.Tensor(add_221, 1, 1, 9223372036854775807);  add_221 = None
        clone_197 = torch.ops.aten.clone.default(add_228, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_197, [2], correction = 0, keepdim = True)
        getitem_184 = var_mean_67[0]
        getitem_185 = var_mean_67[1];  var_mean_67 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_92 = torch.ops.aten.sub.Tensor(clone_197, getitem_185);  clone_197 = getitem_185 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_67);  sub_92 = rsqrt_67 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, arg24_1);  mul_234 = arg24_1 = None
        add_230 = torch.ops.aten.add.Tensor(mul_235, arg25_1);  mul_235 = arg25_1 = None
        view_523 = torch.ops.aten.view.default(add_230, [8, 196, -1]);  add_230 = None
        view_524 = torch.ops.aten.view.default(view_523, [1568, 384]);  view_523 = None
        permute_246 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg27_1, view_524, permute_246);  arg27_1 = view_524 = permute_246 = None
        view_525 = torch.ops.aten.view.default(addmm_90, [8, 196, 384]);  addmm_90 = None
        add_231 = torch.ops.aten.add.Tensor(slice_57, view_525);  slice_57 = view_525 = None
        cat_14 = torch.ops.aten.cat.default([slice_55, add_231], 1);  slice_55 = add_231 = None
        var_mean_68 = torch.ops.aten.var_mean.correction(cat_14, [2], correction = 0, keepdim = True)
        getitem_186 = var_mean_68[0]
        getitem_187 = var_mean_68[1];  var_mean_68 = None
        add_232 = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_93 = torch.ops.aten.sub.Tensor(cat_14, getitem_187);  getitem_187 = None
        mul_236 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_68);  sub_93 = rsqrt_68 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, arg28_1);  mul_236 = arg28_1 = None
        add_233 = torch.ops.aten.add.Tensor(mul_237, arg29_1);  mul_237 = arg29_1 = None
        permute_247 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        view_526 = torch.ops.aten.view.default(add_233, [1576, 384])
        mm_50 = torch.ops.aten.mm.default(view_526, permute_247);  view_526 = permute_247 = None
        view_527 = torch.ops.aten.view.default(mm_50, [8, 197, 768]);  mm_50 = None
        view_528 = torch.ops.aten.view.default(view_527, [8, 197, 2, 6, 64]);  view_527 = None
        permute_248 = torch.ops.aten.permute.default(view_528, [2, 0, 3, 1, 4]);  view_528 = None
        unbind_25 = torch.ops.aten.unbind.int(permute_248);  permute_248 = None
        getitem_188 = unbind_25[0]
        getitem_189 = unbind_25[1];  unbind_25 = None
        permute_249 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        view_529 = torch.ops.aten.view.default(add_233, [1576, 384]);  add_233 = None
        mm_51 = torch.ops.aten.mm.default(view_529, permute_249);  view_529 = permute_249 = None
        view_530 = torch.ops.aten.view.default(mm_51, [8, 197, 384]);  mm_51 = None
        view_531 = torch.ops.aten.view.default(view_530, [8, 197, 6, -1]);  view_530 = None
        permute_250 = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
        permute_251 = torch.ops.aten.permute.default(getitem_189, [0, 1, 3, 2]);  getitem_189 = None
        expand_102 = torch.ops.aten.expand.default(getitem_188, [8, 6, 197, 64]);  getitem_188 = None
        clone_198 = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
        view_532 = torch.ops.aten.view.default(clone_198, [48, 197, 64]);  clone_198 = None
        expand_103 = torch.ops.aten.expand.default(permute_251, [8, 6, 64, 197]);  permute_251 = None
        clone_199 = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_533 = torch.ops.aten.view.default(clone_199, [48, 64, 197]);  clone_199 = None
        bmm_50 = torch.ops.aten.bmm.default(view_532, view_533);  view_532 = view_533 = None
        view_534 = torch.ops.aten.view.default(bmm_50, [8, 6, 197, 197]);  bmm_50 = None
        mul_tensor_44 = torch.ops.aten.mul.Tensor(view_534, 1);  view_534 = None
        amax_default_22 = torch.ops.aten.amax.default(mul_tensor_44, [-1], True)
        sub_tensor_22 = torch.ops.aten.sub.Tensor(mul_tensor_44, amax_default_22);  mul_tensor_44 = amax_default_22 = None
        mul_tensor_45 = torch.ops.aten.mul.Tensor(sub_tensor_22, 0.125);  sub_tensor_22 = None
        exp_25 = torch.ops.aten.exp.default(mul_tensor_45);  mul_tensor_45 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25 = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        expand_104 = torch.ops.aten.expand.default(div_25, [8, 6, 197, 197]);  div_25 = None
        view_535 = torch.ops.aten.view.default(expand_104, [48, 197, 197]);  expand_104 = None
        expand_105 = torch.ops.aten.expand.default(permute_250, [8, 6, 197, 64]);  permute_250 = None
        clone_200 = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_536 = torch.ops.aten.view.default(clone_200, [48, 197, 64]);  clone_200 = None
        bmm_51 = torch.ops.aten.bmm.default(view_535, view_536);  view_535 = view_536 = None
        view_537 = torch.ops.aten.view.default(bmm_51, [8, 6, 197, 64]);  bmm_51 = None
        permute_252 = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
        clone_201 = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_538 = torch.ops.aten.view.default(clone_201, [8, 197, 384]);  clone_201 = None
        view_539 = torch.ops.aten.view.default(view_538, [1576, 384]);  view_538 = None
        permute_253 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg33_1, view_539, permute_253);  arg33_1 = view_539 = permute_253 = None
        view_540 = torch.ops.aten.view.default(addmm_91, [8, 197, 384]);  addmm_91 = None
        add_234 = torch.ops.aten.add.Tensor(cat_14, view_540);  cat_14 = view_540 = None
        var_mean_69 = torch.ops.aten.var_mean.correction(add_234, [2], correction = 0, keepdim = True)
        getitem_190 = var_mean_69[0]
        getitem_191 = var_mean_69[1];  var_mean_69 = None
        add_235 = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        sub_95 = torch.ops.aten.sub.Tensor(add_234, getitem_191);  getitem_191 = None
        mul_239 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_69);  sub_95 = rsqrt_69 = None
        mul_240 = torch.ops.aten.mul.Tensor(mul_239, arg34_1);  mul_239 = arg34_1 = None
        add_236 = torch.ops.aten.add.Tensor(mul_240, arg35_1);  mul_240 = arg35_1 = None
        view_541 = torch.ops.aten.view.default(add_236, [1576, 384]);  add_236 = None
        permute_254 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg37_1, view_541, permute_254);  arg37_1 = view_541 = permute_254 = None
        view_542 = torch.ops.aten.view.default(addmm_92, [8, 197, 1536]);  addmm_92 = None
        mul_241 = torch.ops.aten.mul.Tensor(view_542, 0.5)
        mul_242 = torch.ops.aten.mul.Tensor(view_542, 0.7071067811865476);  view_542 = None
        erf_25 = torch.ops.aten.erf.default(mul_242);  mul_242 = None
        add_237 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_241, add_237);  mul_241 = add_237 = None
        view_543 = torch.ops.aten.view.default(mul_243, [1576, 1536]);  mul_243 = None
        permute_255 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg39_1, view_543, permute_255);  arg39_1 = view_543 = permute_255 = None
        view_544 = torch.ops.aten.view.default(addmm_93, [8, 197, 384]);  addmm_93 = None
        add_238 = torch.ops.aten.add.Tensor(add_234, view_544);  add_234 = view_544 = None
        clone_204 = torch.ops.aten.clone.default(add_228, memory_format = torch.contiguous_format)
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_204, [2], correction = 0, keepdim = True)
        getitem_192 = var_mean_70[0]
        getitem_193 = var_mean_70[1];  var_mean_70 = None
        add_239 = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_96 = torch.ops.aten.sub.Tensor(clone_204, getitem_193);  clone_204 = getitem_193 = None
        mul_244 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_70);  sub_96 = rsqrt_70 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, arg40_1);  mul_244 = arg40_1 = None
        add_240 = torch.ops.aten.add.Tensor(mul_245, arg41_1);  mul_245 = arg41_1 = None
        permute_256 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        view_545 = torch.ops.aten.view.default(add_240, [25088, 24])
        mm_52 = torch.ops.aten.mm.default(view_545, permute_256);  view_545 = permute_256 = None
        view_546 = torch.ops.aten.view.default(mm_52, [1568, 16, 48]);  mm_52 = None
        view_547 = torch.ops.aten.view.default(view_546, [1568, 16, 2, 4, 6]);  view_546 = None
        permute_257 = torch.ops.aten.permute.default(view_547, [2, 0, 3, 1, 4]);  view_547 = None
        unbind_26 = torch.ops.aten.unbind.int(permute_257);  permute_257 = None
        getitem_194 = unbind_26[0]
        getitem_195 = unbind_26[1];  unbind_26 = None
        permute_258 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        view_548 = torch.ops.aten.view.default(add_240, [25088, 24]);  add_240 = None
        mm_53 = torch.ops.aten.mm.default(view_548, permute_258);  view_548 = permute_258 = None
        view_549 = torch.ops.aten.view.default(mm_53, [1568, 16, 24]);  mm_53 = None
        view_550 = torch.ops.aten.view.default(view_549, [1568, 16, 4, -1]);  view_549 = None
        permute_259 = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
        permute_260 = torch.ops.aten.permute.default(getitem_195, [0, 1, 3, 2]);  getitem_195 = None
        expand_106 = torch.ops.aten.expand.default(getitem_194, [1568, 4, 16, 6]);  getitem_194 = None
        clone_205 = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_551 = torch.ops.aten.view.default(clone_205, [6272, 16, 6]);  clone_205 = None
        expand_107 = torch.ops.aten.expand.default(permute_260, [1568, 4, 6, 16]);  permute_260 = None
        clone_206 = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_552 = torch.ops.aten.view.default(clone_206, [6272, 6, 16]);  clone_206 = None
        bmm_52 = torch.ops.aten.bmm.default(view_551, view_552);  view_551 = view_552 = None
        view_553 = torch.ops.aten.view.default(bmm_52, [1568, 4, 16, 16]);  bmm_52 = None
        mul_tensor_42 = torch.ops.aten.mul.Tensor(view_553, 1);  view_553 = None
        amax_default_21 = torch.ops.aten.amax.default(mul_tensor_42, [-1], True)
        sub_tensor_21 = torch.ops.aten.sub.Tensor(mul_tensor_42, amax_default_21);  mul_tensor_42 = amax_default_21 = None
        mul_tensor_43 = torch.ops.aten.mul.Tensor(sub_tensor_21, 0.408248290463863);  sub_tensor_21 = None
        exp_26 = torch.ops.aten.exp.default(mul_tensor_43);  mul_tensor_43 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_26 = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        expand_108 = torch.ops.aten.expand.default(div_26, [1568, 4, 16, 16]);  div_26 = None
        view_554 = torch.ops.aten.view.default(expand_108, [6272, 16, 16]);  expand_108 = None
        expand_109 = torch.ops.aten.expand.default(permute_259, [1568, 4, 16, 6]);  permute_259 = None
        clone_207 = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_555 = torch.ops.aten.view.default(clone_207, [6272, 16, 6]);  clone_207 = None
        bmm_53 = torch.ops.aten.bmm.default(view_554, view_555);  view_554 = view_555 = None
        view_556 = torch.ops.aten.view.default(bmm_53, [1568, 4, 16, 6]);  bmm_53 = None
        permute_261 = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        clone_208 = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_557 = torch.ops.aten.view.default(clone_208, [1568, 16, 24]);  clone_208 = None
        view_558 = torch.ops.aten.view.default(view_557, [25088, 24]);  view_557 = None
        permute_262 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg45_1, view_558, permute_262);  arg45_1 = view_558 = permute_262 = None
        view_559 = torch.ops.aten.view.default(addmm_94, [1568, 16, 24]);  addmm_94 = None
        add_241 = torch.ops.aten.add.Tensor(add_228, view_559);  add_228 = view_559 = None
        clone_209 = torch.ops.aten.clone.default(add_241, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_209, [2], correction = 0, keepdim = True)
        getitem_196 = var_mean_71[0]
        getitem_197 = var_mean_71[1];  var_mean_71 = None
        add_242 = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
        sub_98 = torch.ops.aten.sub.Tensor(clone_209, getitem_197);  clone_209 = getitem_197 = None
        mul_247 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_71);  sub_98 = rsqrt_71 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_247, arg46_1);  mul_247 = arg46_1 = None
        add_243 = torch.ops.aten.add.Tensor(mul_248, arg47_1);  mul_248 = arg47_1 = None
        view_560 = torch.ops.aten.view.default(add_243, [25088, 24]);  add_243 = None
        permute_263 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg49_1, view_560, permute_263);  arg49_1 = view_560 = permute_263 = None
        view_561 = torch.ops.aten.view.default(addmm_95, [1568, 16, 96]);  addmm_95 = None
        mul_249 = torch.ops.aten.mul.Tensor(view_561, 0.5)
        mul_250 = torch.ops.aten.mul.Tensor(view_561, 0.7071067811865476);  view_561 = None
        erf_26 = torch.ops.aten.erf.default(mul_250);  mul_250 = None
        add_244 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_249, add_244);  mul_249 = add_244 = None
        view_562 = torch.ops.aten.view.default(mul_251, [25088, 96]);  mul_251 = None
        permute_264 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg51_1, view_562, permute_264);  arg51_1 = view_562 = permute_264 = None
        view_563 = torch.ops.aten.view.default(addmm_96, [1568, 16, 24]);  addmm_96 = None
        add_245 = torch.ops.aten.add.Tensor(add_241, view_563);  add_241 = view_563 = None
        slice_59 = torch.ops.aten.slice.Tensor(add_238, 1, 0, 1)
        slice_61 = torch.ops.aten.slice.Tensor(add_238, 1, 1, 9223372036854775807);  add_238 = None
        clone_212 = torch.ops.aten.clone.default(add_245, memory_format = torch.contiguous_format)
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_212, [2], correction = 0, keepdim = True)
        getitem_198 = var_mean_72[0]
        getitem_199 = var_mean_72[1];  var_mean_72 = None
        add_246 = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_99 = torch.ops.aten.sub.Tensor(clone_212, getitem_199);  clone_212 = getitem_199 = None
        mul_252 = torch.ops.aten.mul.Tensor(sub_99, rsqrt_72);  sub_99 = rsqrt_72 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, arg52_1);  mul_252 = arg52_1 = None
        add_247 = torch.ops.aten.add.Tensor(mul_253, arg53_1);  mul_253 = arg53_1 = None
        view_564 = torch.ops.aten.view.default(add_247, [8, 196, -1]);  add_247 = None
        view_565 = torch.ops.aten.view.default(view_564, [1568, 384]);  view_564 = None
        permute_265 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg55_1, view_565, permute_265);  arg55_1 = view_565 = permute_265 = None
        view_566 = torch.ops.aten.view.default(addmm_97, [8, 196, 384]);  addmm_97 = None
        add_248 = torch.ops.aten.add.Tensor(slice_61, view_566);  slice_61 = view_566 = None
        cat_15 = torch.ops.aten.cat.default([slice_59, add_248], 1);  slice_59 = add_248 = None
        var_mean_73 = torch.ops.aten.var_mean.correction(cat_15, [2], correction = 0, keepdim = True)
        getitem_200 = var_mean_73[0]
        getitem_201 = var_mean_73[1];  var_mean_73 = None
        add_249 = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
        sub_100 = torch.ops.aten.sub.Tensor(cat_15, getitem_201);  getitem_201 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_73);  sub_100 = rsqrt_73 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, arg56_1);  mul_254 = arg56_1 = None
        add_250 = torch.ops.aten.add.Tensor(mul_255, arg57_1);  mul_255 = arg57_1 = None
        permute_266 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        view_567 = torch.ops.aten.view.default(add_250, [1576, 384])
        mm_54 = torch.ops.aten.mm.default(view_567, permute_266);  view_567 = permute_266 = None
        view_568 = torch.ops.aten.view.default(mm_54, [8, 197, 768]);  mm_54 = None
        view_569 = torch.ops.aten.view.default(view_568, [8, 197, 2, 6, 64]);  view_568 = None
        permute_267 = torch.ops.aten.permute.default(view_569, [2, 0, 3, 1, 4]);  view_569 = None
        unbind_27 = torch.ops.aten.unbind.int(permute_267);  permute_267 = None
        getitem_202 = unbind_27[0]
        getitem_203 = unbind_27[1];  unbind_27 = None
        permute_268 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        view_570 = torch.ops.aten.view.default(add_250, [1576, 384]);  add_250 = None
        mm_55 = torch.ops.aten.mm.default(view_570, permute_268);  view_570 = permute_268 = None
        view_571 = torch.ops.aten.view.default(mm_55, [8, 197, 384]);  mm_55 = None
        view_572 = torch.ops.aten.view.default(view_571, [8, 197, 6, -1]);  view_571 = None
        permute_269 = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        permute_270 = torch.ops.aten.permute.default(getitem_203, [0, 1, 3, 2]);  getitem_203 = None
        expand_110 = torch.ops.aten.expand.default(getitem_202, [8, 6, 197, 64]);  getitem_202 = None
        clone_213 = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
        view_573 = torch.ops.aten.view.default(clone_213, [48, 197, 64]);  clone_213 = None
        expand_111 = torch.ops.aten.expand.default(permute_270, [8, 6, 64, 197]);  permute_270 = None
        clone_214 = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_574 = torch.ops.aten.view.default(clone_214, [48, 64, 197]);  clone_214 = None
        bmm_54 = torch.ops.aten.bmm.default(view_573, view_574);  view_573 = view_574 = None
        view_575 = torch.ops.aten.view.default(bmm_54, [8, 6, 197, 197]);  bmm_54 = None
        mul_tensor_40 = torch.ops.aten.mul.Tensor(view_575, 1);  view_575 = None
        amax_default_20 = torch.ops.aten.amax.default(mul_tensor_40, [-1], True)
        sub_tensor_20 = torch.ops.aten.sub.Tensor(mul_tensor_40, amax_default_20);  mul_tensor_40 = amax_default_20 = None
        mul_tensor_41 = torch.ops.aten.mul.Tensor(sub_tensor_20, 0.125);  sub_tensor_20 = None
        exp_27 = torch.ops.aten.exp.default(mul_tensor_41);  mul_tensor_41 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        expand_112 = torch.ops.aten.expand.default(div_27, [8, 6, 197, 197]);  div_27 = None
        view_576 = torch.ops.aten.view.default(expand_112, [48, 197, 197]);  expand_112 = None
        expand_113 = torch.ops.aten.expand.default(permute_269, [8, 6, 197, 64]);  permute_269 = None
        clone_215 = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_577 = torch.ops.aten.view.default(clone_215, [48, 197, 64]);  clone_215 = None
        bmm_55 = torch.ops.aten.bmm.default(view_576, view_577);  view_576 = view_577 = None
        view_578 = torch.ops.aten.view.default(bmm_55, [8, 6, 197, 64]);  bmm_55 = None
        permute_271 = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        clone_216 = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_579 = torch.ops.aten.view.default(clone_216, [8, 197, 384]);  clone_216 = None
        view_580 = torch.ops.aten.view.default(view_579, [1576, 384]);  view_579 = None
        permute_272 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg61_1, view_580, permute_272);  arg61_1 = view_580 = permute_272 = None
        view_581 = torch.ops.aten.view.default(addmm_98, [8, 197, 384]);  addmm_98 = None
        add_251 = torch.ops.aten.add.Tensor(cat_15, view_581);  cat_15 = view_581 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(add_251, [2], correction = 0, keepdim = True)
        getitem_204 = var_mean_74[0]
        getitem_205 = var_mean_74[1];  var_mean_74 = None
        add_252 = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        sub_102 = torch.ops.aten.sub.Tensor(add_251, getitem_205);  getitem_205 = None
        mul_257 = torch.ops.aten.mul.Tensor(sub_102, rsqrt_74);  sub_102 = rsqrt_74 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_257, arg62_1);  mul_257 = arg62_1 = None
        add_253 = torch.ops.aten.add.Tensor(mul_258, arg63_1);  mul_258 = arg63_1 = None
        view_582 = torch.ops.aten.view.default(add_253, [1576, 384]);  add_253 = None
        permute_273 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg65_1, view_582, permute_273);  arg65_1 = view_582 = permute_273 = None
        view_583 = torch.ops.aten.view.default(addmm_99, [8, 197, 1536]);  addmm_99 = None
        mul_259 = torch.ops.aten.mul.Tensor(view_583, 0.5)
        mul_260 = torch.ops.aten.mul.Tensor(view_583, 0.7071067811865476);  view_583 = None
        erf_27 = torch.ops.aten.erf.default(mul_260);  mul_260 = None
        add_254 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_259, add_254);  mul_259 = add_254 = None
        view_584 = torch.ops.aten.view.default(mul_261, [1576, 1536]);  mul_261 = None
        permute_274 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg67_1, view_584, permute_274);  arg67_1 = view_584 = permute_274 = None
        view_585 = torch.ops.aten.view.default(addmm_100, [8, 197, 384]);  addmm_100 = None
        add_255 = torch.ops.aten.add.Tensor(add_251, view_585);  add_251 = view_585 = None
        clone_219 = torch.ops.aten.clone.default(add_245, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_219, [2], correction = 0, keepdim = True)
        getitem_206 = var_mean_75[0]
        getitem_207 = var_mean_75[1];  var_mean_75 = None
        add_256 = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        sub_103 = torch.ops.aten.sub.Tensor(clone_219, getitem_207);  clone_219 = getitem_207 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_103, rsqrt_75);  sub_103 = rsqrt_75 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, arg68_1);  mul_262 = arg68_1 = None
        add_257 = torch.ops.aten.add.Tensor(mul_263, arg69_1);  mul_263 = arg69_1 = None
        permute_275 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        view_586 = torch.ops.aten.view.default(add_257, [25088, 24])
        mm_56 = torch.ops.aten.mm.default(view_586, permute_275);  view_586 = permute_275 = None
        view_587 = torch.ops.aten.view.default(mm_56, [1568, 16, 48]);  mm_56 = None
        view_588 = torch.ops.aten.view.default(view_587, [1568, 16, 2, 4, 6]);  view_587 = None
        permute_276 = torch.ops.aten.permute.default(view_588, [2, 0, 3, 1, 4]);  view_588 = None
        unbind_28 = torch.ops.aten.unbind.int(permute_276);  permute_276 = None
        getitem_208 = unbind_28[0]
        getitem_209 = unbind_28[1];  unbind_28 = None
        permute_277 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        view_589 = torch.ops.aten.view.default(add_257, [25088, 24]);  add_257 = None
        mm_57 = torch.ops.aten.mm.default(view_589, permute_277);  view_589 = permute_277 = None
        view_590 = torch.ops.aten.view.default(mm_57, [1568, 16, 24]);  mm_57 = None
        view_591 = torch.ops.aten.view.default(view_590, [1568, 16, 4, -1]);  view_590 = None
        permute_278 = torch.ops.aten.permute.default(view_591, [0, 2, 1, 3]);  view_591 = None
        permute_279 = torch.ops.aten.permute.default(getitem_209, [0, 1, 3, 2]);  getitem_209 = None
        expand_114 = torch.ops.aten.expand.default(getitem_208, [1568, 4, 16, 6]);  getitem_208 = None
        clone_220 = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_592 = torch.ops.aten.view.default(clone_220, [6272, 16, 6]);  clone_220 = None
        expand_115 = torch.ops.aten.expand.default(permute_279, [1568, 4, 6, 16]);  permute_279 = None
        clone_221 = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_593 = torch.ops.aten.view.default(clone_221, [6272, 6, 16]);  clone_221 = None
        bmm_56 = torch.ops.aten.bmm.default(view_592, view_593);  view_592 = view_593 = None
        view_594 = torch.ops.aten.view.default(bmm_56, [1568, 4, 16, 16]);  bmm_56 = None
        mul_tensor_38 = torch.ops.aten.mul.Tensor(view_594, 1);  view_594 = None
        amax_default_19 = torch.ops.aten.amax.default(mul_tensor_38, [-1], True)
        sub_tensor_19 = torch.ops.aten.sub.Tensor(mul_tensor_38, amax_default_19);  mul_tensor_38 = amax_default_19 = None
        mul_tensor_39 = torch.ops.aten.mul.Tensor(sub_tensor_19, 0.408248290463863);  sub_tensor_19 = None
        exp_28 = torch.ops.aten.exp.default(mul_tensor_39);  mul_tensor_39 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_28 = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        expand_116 = torch.ops.aten.expand.default(div_28, [1568, 4, 16, 16]);  div_28 = None
        view_595 = torch.ops.aten.view.default(expand_116, [6272, 16, 16]);  expand_116 = None
        expand_117 = torch.ops.aten.expand.default(permute_278, [1568, 4, 16, 6]);  permute_278 = None
        clone_222 = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_596 = torch.ops.aten.view.default(clone_222, [6272, 16, 6]);  clone_222 = None
        bmm_57 = torch.ops.aten.bmm.default(view_595, view_596);  view_595 = view_596 = None
        view_597 = torch.ops.aten.view.default(bmm_57, [1568, 4, 16, 6]);  bmm_57 = None
        permute_280 = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
        clone_223 = torch.ops.aten.clone.default(permute_280, memory_format = torch.contiguous_format);  permute_280 = None
        view_598 = torch.ops.aten.view.default(clone_223, [1568, 16, 24]);  clone_223 = None
        view_599 = torch.ops.aten.view.default(view_598, [25088, 24]);  view_598 = None
        permute_281 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg73_1, view_599, permute_281);  arg73_1 = view_599 = permute_281 = None
        view_600 = torch.ops.aten.view.default(addmm_101, [1568, 16, 24]);  addmm_101 = None
        add_258 = torch.ops.aten.add.Tensor(add_245, view_600);  add_245 = view_600 = None
        clone_224 = torch.ops.aten.clone.default(add_258, memory_format = torch.contiguous_format)
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_224, [2], correction = 0, keepdim = True)
        getitem_210 = var_mean_76[0]
        getitem_211 = var_mean_76[1];  var_mean_76 = None
        add_259 = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
        sub_105 = torch.ops.aten.sub.Tensor(clone_224, getitem_211);  clone_224 = getitem_211 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_105, rsqrt_76);  sub_105 = rsqrt_76 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, arg74_1);  mul_265 = arg74_1 = None
        add_260 = torch.ops.aten.add.Tensor(mul_266, arg75_1);  mul_266 = arg75_1 = None
        view_601 = torch.ops.aten.view.default(add_260, [25088, 24]);  add_260 = None
        permute_282 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg77_1, view_601, permute_282);  arg77_1 = view_601 = permute_282 = None
        view_602 = torch.ops.aten.view.default(addmm_102, [1568, 16, 96]);  addmm_102 = None
        mul_267 = torch.ops.aten.mul.Tensor(view_602, 0.5)
        mul_268 = torch.ops.aten.mul.Tensor(view_602, 0.7071067811865476);  view_602 = None
        erf_28 = torch.ops.aten.erf.default(mul_268);  mul_268 = None
        add_261 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_267, add_261);  mul_267 = add_261 = None
        view_603 = torch.ops.aten.view.default(mul_269, [25088, 96]);  mul_269 = None
        permute_283 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg79_1, view_603, permute_283);  arg79_1 = view_603 = permute_283 = None
        view_604 = torch.ops.aten.view.default(addmm_103, [1568, 16, 24]);  addmm_103 = None
        add_262 = torch.ops.aten.add.Tensor(add_258, view_604);  add_258 = view_604 = None
        slice_63 = torch.ops.aten.slice.Tensor(add_255, 1, 0, 1)
        slice_65 = torch.ops.aten.slice.Tensor(add_255, 1, 1, 9223372036854775807);  add_255 = None
        clone_227 = torch.ops.aten.clone.default(add_262, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_227, [2], correction = 0, keepdim = True)
        getitem_212 = var_mean_77[0]
        getitem_213 = var_mean_77[1];  var_mean_77 = None
        add_263 = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        sub_106 = torch.ops.aten.sub.Tensor(clone_227, getitem_213);  clone_227 = getitem_213 = None
        mul_270 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_77);  sub_106 = rsqrt_77 = None
        mul_271 = torch.ops.aten.mul.Tensor(mul_270, arg80_1);  mul_270 = arg80_1 = None
        add_264 = torch.ops.aten.add.Tensor(mul_271, arg81_1);  mul_271 = arg81_1 = None
        view_605 = torch.ops.aten.view.default(add_264, [8, 196, -1]);  add_264 = None
        view_606 = torch.ops.aten.view.default(view_605, [1568, 384]);  view_605 = None
        permute_284 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg83_1, view_606, permute_284);  arg83_1 = view_606 = permute_284 = None
        view_607 = torch.ops.aten.view.default(addmm_104, [8, 196, 384]);  addmm_104 = None
        add_265 = torch.ops.aten.add.Tensor(slice_65, view_607);  slice_65 = view_607 = None
        cat_16 = torch.ops.aten.cat.default([slice_63, add_265], 1);  slice_63 = add_265 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(cat_16, [2], correction = 0, keepdim = True)
        getitem_214 = var_mean_78[0]
        getitem_215 = var_mean_78[1];  var_mean_78 = None
        add_266 = torch.ops.aten.add.Tensor(getitem_214, 1e-05);  getitem_214 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_107 = torch.ops.aten.sub.Tensor(cat_16, getitem_215);  getitem_215 = None
        mul_272 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_78);  sub_107 = rsqrt_78 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, arg84_1);  mul_272 = arg84_1 = None
        add_267 = torch.ops.aten.add.Tensor(mul_273, arg85_1);  mul_273 = arg85_1 = None
        permute_285 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        view_608 = torch.ops.aten.view.default(add_267, [1576, 384])
        mm_58 = torch.ops.aten.mm.default(view_608, permute_285);  view_608 = permute_285 = None
        view_609 = torch.ops.aten.view.default(mm_58, [8, 197, 768]);  mm_58 = None
        view_610 = torch.ops.aten.view.default(view_609, [8, 197, 2, 6, 64]);  view_609 = None
        permute_286 = torch.ops.aten.permute.default(view_610, [2, 0, 3, 1, 4]);  view_610 = None
        unbind_29 = torch.ops.aten.unbind.int(permute_286);  permute_286 = None
        getitem_216 = unbind_29[0]
        getitem_217 = unbind_29[1];  unbind_29 = None
        permute_287 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        view_611 = torch.ops.aten.view.default(add_267, [1576, 384]);  add_267 = None
        mm_59 = torch.ops.aten.mm.default(view_611, permute_287);  view_611 = permute_287 = None
        view_612 = torch.ops.aten.view.default(mm_59, [8, 197, 384]);  mm_59 = None
        view_613 = torch.ops.aten.view.default(view_612, [8, 197, 6, -1]);  view_612 = None
        permute_288 = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
        permute_289 = torch.ops.aten.permute.default(getitem_217, [0, 1, 3, 2]);  getitem_217 = None
        expand_118 = torch.ops.aten.expand.default(getitem_216, [8, 6, 197, 64]);  getitem_216 = None
        clone_228 = torch.ops.aten.clone.default(expand_118, memory_format = torch.contiguous_format);  expand_118 = None
        view_614 = torch.ops.aten.view.default(clone_228, [48, 197, 64]);  clone_228 = None
        expand_119 = torch.ops.aten.expand.default(permute_289, [8, 6, 64, 197]);  permute_289 = None
        clone_229 = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_615 = torch.ops.aten.view.default(clone_229, [48, 64, 197]);  clone_229 = None
        bmm_58 = torch.ops.aten.bmm.default(view_614, view_615);  view_614 = view_615 = None
        view_616 = torch.ops.aten.view.default(bmm_58, [8, 6, 197, 197]);  bmm_58 = None
        mul_tensor_36 = torch.ops.aten.mul.Tensor(view_616, 1);  view_616 = None
        amax_default_18 = torch.ops.aten.amax.default(mul_tensor_36, [-1], True)
        sub_tensor_18 = torch.ops.aten.sub.Tensor(mul_tensor_36, amax_default_18);  mul_tensor_36 = amax_default_18 = None
        mul_tensor_37 = torch.ops.aten.mul.Tensor(sub_tensor_18, 0.125);  sub_tensor_18 = None
        exp_29 = torch.ops.aten.exp.default(mul_tensor_37);  mul_tensor_37 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_29 = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
        expand_120 = torch.ops.aten.expand.default(div_29, [8, 6, 197, 197]);  div_29 = None
        view_617 = torch.ops.aten.view.default(expand_120, [48, 197, 197]);  expand_120 = None
        expand_121 = torch.ops.aten.expand.default(permute_288, [8, 6, 197, 64]);  permute_288 = None
        clone_230 = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_618 = torch.ops.aten.view.default(clone_230, [48, 197, 64]);  clone_230 = None
        bmm_59 = torch.ops.aten.bmm.default(view_617, view_618);  view_617 = view_618 = None
        view_619 = torch.ops.aten.view.default(bmm_59, [8, 6, 197, 64]);  bmm_59 = None
        permute_290 = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
        clone_231 = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        view_620 = torch.ops.aten.view.default(clone_231, [8, 197, 384]);  clone_231 = None
        view_621 = torch.ops.aten.view.default(view_620, [1576, 384]);  view_620 = None
        permute_291 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg89_1, view_621, permute_291);  arg89_1 = view_621 = permute_291 = None
        view_622 = torch.ops.aten.view.default(addmm_105, [8, 197, 384]);  addmm_105 = None
        add_268 = torch.ops.aten.add.Tensor(cat_16, view_622);  cat_16 = view_622 = None
        var_mean_79 = torch.ops.aten.var_mean.correction(add_268, [2], correction = 0, keepdim = True)
        getitem_218 = var_mean_79[0]
        getitem_219 = var_mean_79[1];  var_mean_79 = None
        add_269 = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        sub_109 = torch.ops.aten.sub.Tensor(add_268, getitem_219);  getitem_219 = None
        mul_275 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_79);  sub_109 = rsqrt_79 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, arg90_1);  mul_275 = arg90_1 = None
        add_270 = torch.ops.aten.add.Tensor(mul_276, arg91_1);  mul_276 = arg91_1 = None
        view_623 = torch.ops.aten.view.default(add_270, [1576, 384]);  add_270 = None
        permute_292 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg93_1, view_623, permute_292);  arg93_1 = view_623 = permute_292 = None
        view_624 = torch.ops.aten.view.default(addmm_106, [8, 197, 1536]);  addmm_106 = None
        mul_277 = torch.ops.aten.mul.Tensor(view_624, 0.5)
        mul_278 = torch.ops.aten.mul.Tensor(view_624, 0.7071067811865476);  view_624 = None
        erf_29 = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_271 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_277, add_271);  mul_277 = add_271 = None
        view_625 = torch.ops.aten.view.default(mul_279, [1576, 1536]);  mul_279 = None
        permute_293 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg95_1, view_625, permute_293);  arg95_1 = view_625 = permute_293 = None
        view_626 = torch.ops.aten.view.default(addmm_107, [8, 197, 384]);  addmm_107 = None
        add_272 = torch.ops.aten.add.Tensor(add_268, view_626);  add_268 = view_626 = None
        clone_234 = torch.ops.aten.clone.default(add_262, memory_format = torch.contiguous_format)
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_234, [2], correction = 0, keepdim = True)
        getitem_220 = var_mean_80[0]
        getitem_221 = var_mean_80[1];  var_mean_80 = None
        add_273 = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_273);  add_273 = None
        sub_110 = torch.ops.aten.sub.Tensor(clone_234, getitem_221);  clone_234 = getitem_221 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_110, rsqrt_80);  sub_110 = rsqrt_80 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, arg96_1);  mul_280 = arg96_1 = None
        add_274 = torch.ops.aten.add.Tensor(mul_281, arg97_1);  mul_281 = arg97_1 = None
        permute_294 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        view_627 = torch.ops.aten.view.default(add_274, [25088, 24])
        mm_60 = torch.ops.aten.mm.default(view_627, permute_294);  view_627 = permute_294 = None
        view_628 = torch.ops.aten.view.default(mm_60, [1568, 16, 48]);  mm_60 = None
        view_629 = torch.ops.aten.view.default(view_628, [1568, 16, 2, 4, 6]);  view_628 = None
        permute_295 = torch.ops.aten.permute.default(view_629, [2, 0, 3, 1, 4]);  view_629 = None
        unbind_30 = torch.ops.aten.unbind.int(permute_295);  permute_295 = None
        getitem_222 = unbind_30[0]
        getitem_223 = unbind_30[1];  unbind_30 = None
        permute_296 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        view_630 = torch.ops.aten.view.default(add_274, [25088, 24]);  add_274 = None
        mm_61 = torch.ops.aten.mm.default(view_630, permute_296);  view_630 = permute_296 = None
        view_631 = torch.ops.aten.view.default(mm_61, [1568, 16, 24]);  mm_61 = None
        view_632 = torch.ops.aten.view.default(view_631, [1568, 16, 4, -1]);  view_631 = None
        permute_297 = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
        permute_298 = torch.ops.aten.permute.default(getitem_223, [0, 1, 3, 2]);  getitem_223 = None
        expand_122 = torch.ops.aten.expand.default(getitem_222, [1568, 4, 16, 6]);  getitem_222 = None
        clone_235 = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_633 = torch.ops.aten.view.default(clone_235, [6272, 16, 6]);  clone_235 = None
        expand_123 = torch.ops.aten.expand.default(permute_298, [1568, 4, 6, 16]);  permute_298 = None
        clone_236 = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_634 = torch.ops.aten.view.default(clone_236, [6272, 6, 16]);  clone_236 = None
        bmm_60 = torch.ops.aten.bmm.default(view_633, view_634);  view_633 = view_634 = None
        view_635 = torch.ops.aten.view.default(bmm_60, [1568, 4, 16, 16]);  bmm_60 = None
        mul_tensor_34 = torch.ops.aten.mul.Tensor(view_635, 1);  view_635 = None
        amax_default_17 = torch.ops.aten.amax.default(mul_tensor_34, [-1], True)
        sub_tensor_17 = torch.ops.aten.sub.Tensor(mul_tensor_34, amax_default_17);  mul_tensor_34 = amax_default_17 = None
        mul_tensor_35 = torch.ops.aten.mul.Tensor(sub_tensor_17, 0.408248290463863);  sub_tensor_17 = None
        exp_30 = torch.ops.aten.exp.default(mul_tensor_35);  mul_tensor_35 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_30 = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        expand_124 = torch.ops.aten.expand.default(div_30, [1568, 4, 16, 16]);  div_30 = None
        view_636 = torch.ops.aten.view.default(expand_124, [6272, 16, 16]);  expand_124 = None
        expand_125 = torch.ops.aten.expand.default(permute_297, [1568, 4, 16, 6]);  permute_297 = None
        clone_237 = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_637 = torch.ops.aten.view.default(clone_237, [6272, 16, 6]);  clone_237 = None
        bmm_61 = torch.ops.aten.bmm.default(view_636, view_637);  view_636 = view_637 = None
        view_638 = torch.ops.aten.view.default(bmm_61, [1568, 4, 16, 6]);  bmm_61 = None
        permute_299 = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
        clone_238 = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
        view_639 = torch.ops.aten.view.default(clone_238, [1568, 16, 24]);  clone_238 = None
        view_640 = torch.ops.aten.view.default(view_639, [25088, 24]);  view_639 = None
        permute_300 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg101_1, view_640, permute_300);  arg101_1 = view_640 = permute_300 = None
        view_641 = torch.ops.aten.view.default(addmm_108, [1568, 16, 24]);  addmm_108 = None
        add_275 = torch.ops.aten.add.Tensor(add_262, view_641);  add_262 = view_641 = None
        clone_239 = torch.ops.aten.clone.default(add_275, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
        getitem_224 = var_mean_81[0]
        getitem_225 = var_mean_81[1];  var_mean_81 = None
        add_276 = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_112 = torch.ops.aten.sub.Tensor(clone_239, getitem_225);  clone_239 = getitem_225 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_81);  sub_112 = rsqrt_81 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, arg102_1);  mul_283 = arg102_1 = None
        add_277 = torch.ops.aten.add.Tensor(mul_284, arg103_1);  mul_284 = arg103_1 = None
        view_642 = torch.ops.aten.view.default(add_277, [25088, 24]);  add_277 = None
        permute_301 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg105_1, view_642, permute_301);  arg105_1 = view_642 = permute_301 = None
        view_643 = torch.ops.aten.view.default(addmm_109, [1568, 16, 96]);  addmm_109 = None
        mul_285 = torch.ops.aten.mul.Tensor(view_643, 0.5)
        mul_286 = torch.ops.aten.mul.Tensor(view_643, 0.7071067811865476);  view_643 = None
        erf_30 = torch.ops.aten.erf.default(mul_286);  mul_286 = None
        add_278 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_285, add_278);  mul_285 = add_278 = None
        view_644 = torch.ops.aten.view.default(mul_287, [25088, 96]);  mul_287 = None
        permute_302 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg107_1, view_644, permute_302);  arg107_1 = view_644 = permute_302 = None
        view_645 = torch.ops.aten.view.default(addmm_110, [1568, 16, 24]);  addmm_110 = None
        add_279 = torch.ops.aten.add.Tensor(add_275, view_645);  add_275 = view_645 = None
        slice_67 = torch.ops.aten.slice.Tensor(add_272, 1, 0, 1)
        slice_69 = torch.ops.aten.slice.Tensor(add_272, 1, 1, 9223372036854775807);  add_272 = None
        clone_242 = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format)
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_242, [2], correction = 0, keepdim = True)
        getitem_226 = var_mean_82[0]
        getitem_227 = var_mean_82[1];  var_mean_82 = None
        add_280 = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        sub_113 = torch.ops.aten.sub.Tensor(clone_242, getitem_227);  clone_242 = getitem_227 = None
        mul_288 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_82);  sub_113 = rsqrt_82 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_288, arg108_1);  mul_288 = arg108_1 = None
        add_281 = torch.ops.aten.add.Tensor(mul_289, arg109_1);  mul_289 = arg109_1 = None
        view_646 = torch.ops.aten.view.default(add_281, [8, 196, -1]);  add_281 = None
        view_647 = torch.ops.aten.view.default(view_646, [1568, 384]);  view_646 = None
        permute_303 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg111_1, view_647, permute_303);  arg111_1 = view_647 = permute_303 = None
        view_648 = torch.ops.aten.view.default(addmm_111, [8, 196, 384]);  addmm_111 = None
        add_282 = torch.ops.aten.add.Tensor(slice_69, view_648);  slice_69 = view_648 = None
        cat_17 = torch.ops.aten.cat.default([slice_67, add_282], 1);  slice_67 = add_282 = None
        var_mean_83 = torch.ops.aten.var_mean.correction(cat_17, [2], correction = 0, keepdim = True)
        getitem_228 = var_mean_83[0]
        getitem_229 = var_mean_83[1];  var_mean_83 = None
        add_283 = torch.ops.aten.add.Tensor(getitem_228, 1e-05);  getitem_228 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        sub_114 = torch.ops.aten.sub.Tensor(cat_17, getitem_229);  getitem_229 = None
        mul_290 = torch.ops.aten.mul.Tensor(sub_114, rsqrt_83);  sub_114 = rsqrt_83 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, arg112_1);  mul_290 = arg112_1 = None
        add_284 = torch.ops.aten.add.Tensor(mul_291, arg113_1);  mul_291 = arg113_1 = None
        permute_304 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        view_649 = torch.ops.aten.view.default(add_284, [1576, 384])
        mm_62 = torch.ops.aten.mm.default(view_649, permute_304);  view_649 = permute_304 = None
        view_650 = torch.ops.aten.view.default(mm_62, [8, 197, 768]);  mm_62 = None
        view_651 = torch.ops.aten.view.default(view_650, [8, 197, 2, 6, 64]);  view_650 = None
        permute_305 = torch.ops.aten.permute.default(view_651, [2, 0, 3, 1, 4]);  view_651 = None
        unbind_31 = torch.ops.aten.unbind.int(permute_305);  permute_305 = None
        getitem_230 = unbind_31[0]
        getitem_231 = unbind_31[1];  unbind_31 = None
        permute_306 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        view_652 = torch.ops.aten.view.default(add_284, [1576, 384]);  add_284 = None
        mm_63 = torch.ops.aten.mm.default(view_652, permute_306);  view_652 = permute_306 = None
        view_653 = torch.ops.aten.view.default(mm_63, [8, 197, 384]);  mm_63 = None
        view_654 = torch.ops.aten.view.default(view_653, [8, 197, 6, -1]);  view_653 = None
        permute_307 = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
        permute_308 = torch.ops.aten.permute.default(getitem_231, [0, 1, 3, 2]);  getitem_231 = None
        expand_126 = torch.ops.aten.expand.default(getitem_230, [8, 6, 197, 64]);  getitem_230 = None
        clone_243 = torch.ops.aten.clone.default(expand_126, memory_format = torch.contiguous_format);  expand_126 = None
        view_655 = torch.ops.aten.view.default(clone_243, [48, 197, 64]);  clone_243 = None
        expand_127 = torch.ops.aten.expand.default(permute_308, [8, 6, 64, 197]);  permute_308 = None
        clone_244 = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_656 = torch.ops.aten.view.default(clone_244, [48, 64, 197]);  clone_244 = None
        bmm_62 = torch.ops.aten.bmm.default(view_655, view_656);  view_655 = view_656 = None
        view_657 = torch.ops.aten.view.default(bmm_62, [8, 6, 197, 197]);  bmm_62 = None
        mul_tensor_32 = torch.ops.aten.mul.Tensor(view_657, 1);  view_657 = None
        amax_default_16 = torch.ops.aten.amax.default(mul_tensor_32, [-1], True)
        sub_tensor_16 = torch.ops.aten.sub.Tensor(mul_tensor_32, amax_default_16);  mul_tensor_32 = amax_default_16 = None
        mul_tensor_33 = torch.ops.aten.mul.Tensor(sub_tensor_16, 0.125);  sub_tensor_16 = None
        exp_31 = torch.ops.aten.exp.default(mul_tensor_33);  mul_tensor_33 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_31 = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
        expand_128 = torch.ops.aten.expand.default(div_31, [8, 6, 197, 197]);  div_31 = None
        view_658 = torch.ops.aten.view.default(expand_128, [48, 197, 197]);  expand_128 = None
        expand_129 = torch.ops.aten.expand.default(permute_307, [8, 6, 197, 64]);  permute_307 = None
        clone_245 = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_659 = torch.ops.aten.view.default(clone_245, [48, 197, 64]);  clone_245 = None
        bmm_63 = torch.ops.aten.bmm.default(view_658, view_659);  view_658 = view_659 = None
        view_660 = torch.ops.aten.view.default(bmm_63, [8, 6, 197, 64]);  bmm_63 = None
        permute_309 = torch.ops.aten.permute.default(view_660, [0, 2, 1, 3]);  view_660 = None
        clone_246 = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
        view_661 = torch.ops.aten.view.default(clone_246, [8, 197, 384]);  clone_246 = None
        view_662 = torch.ops.aten.view.default(view_661, [1576, 384]);  view_661 = None
        permute_310 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg117_1, view_662, permute_310);  arg117_1 = view_662 = permute_310 = None
        view_663 = torch.ops.aten.view.default(addmm_112, [8, 197, 384]);  addmm_112 = None
        add_285 = torch.ops.aten.add.Tensor(cat_17, view_663);  cat_17 = view_663 = None
        var_mean_84 = torch.ops.aten.var_mean.correction(add_285, [2], correction = 0, keepdim = True)
        getitem_232 = var_mean_84[0]
        getitem_233 = var_mean_84[1];  var_mean_84 = None
        add_286 = torch.ops.aten.add.Tensor(getitem_232, 1e-05);  getitem_232 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_116 = torch.ops.aten.sub.Tensor(add_285, getitem_233);  getitem_233 = None
        mul_293 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_84);  sub_116 = rsqrt_84 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_293, arg118_1);  mul_293 = arg118_1 = None
        add_287 = torch.ops.aten.add.Tensor(mul_294, arg119_1);  mul_294 = arg119_1 = None
        view_664 = torch.ops.aten.view.default(add_287, [1576, 384]);  add_287 = None
        permute_311 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg121_1, view_664, permute_311);  arg121_1 = view_664 = permute_311 = None
        view_665 = torch.ops.aten.view.default(addmm_113, [8, 197, 1536]);  addmm_113 = None
        mul_295 = torch.ops.aten.mul.Tensor(view_665, 0.5)
        mul_296 = torch.ops.aten.mul.Tensor(view_665, 0.7071067811865476);  view_665 = None
        erf_31 = torch.ops.aten.erf.default(mul_296);  mul_296 = None
        add_288 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_295, add_288);  mul_295 = add_288 = None
        view_666 = torch.ops.aten.view.default(mul_297, [1576, 1536]);  mul_297 = None
        permute_312 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg123_1, view_666, permute_312);  arg123_1 = view_666 = permute_312 = None
        view_667 = torch.ops.aten.view.default(addmm_114, [8, 197, 384]);  addmm_114 = None
        add_289 = torch.ops.aten.add.Tensor(add_285, view_667);  add_285 = view_667 = None
        clone_249 = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_249, [2], correction = 0, keepdim = True)
        getitem_234 = var_mean_85[0]
        getitem_235 = var_mean_85[1];  var_mean_85 = None
        add_290 = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        sub_117 = torch.ops.aten.sub.Tensor(clone_249, getitem_235);  clone_249 = getitem_235 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_117, rsqrt_85);  sub_117 = rsqrt_85 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, arg124_1);  mul_298 = arg124_1 = None
        add_291 = torch.ops.aten.add.Tensor(mul_299, arg125_1);  mul_299 = arg125_1 = None
        permute_313 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        view_668 = torch.ops.aten.view.default(add_291, [25088, 24])
        mm_64 = torch.ops.aten.mm.default(view_668, permute_313);  view_668 = permute_313 = None
        view_669 = torch.ops.aten.view.default(mm_64, [1568, 16, 48]);  mm_64 = None
        view_670 = torch.ops.aten.view.default(view_669, [1568, 16, 2, 4, 6]);  view_669 = None
        permute_314 = torch.ops.aten.permute.default(view_670, [2, 0, 3, 1, 4]);  view_670 = None
        unbind_32 = torch.ops.aten.unbind.int(permute_314);  permute_314 = None
        getitem_236 = unbind_32[0]
        getitem_237 = unbind_32[1];  unbind_32 = None
        permute_315 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        view_671 = torch.ops.aten.view.default(add_291, [25088, 24]);  add_291 = None
        mm_65 = torch.ops.aten.mm.default(view_671, permute_315);  view_671 = permute_315 = None
        view_672 = torch.ops.aten.view.default(mm_65, [1568, 16, 24]);  mm_65 = None
        view_673 = torch.ops.aten.view.default(view_672, [1568, 16, 4, -1]);  view_672 = None
        permute_316 = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
        permute_317 = torch.ops.aten.permute.default(getitem_237, [0, 1, 3, 2]);  getitem_237 = None
        expand_130 = torch.ops.aten.expand.default(getitem_236, [1568, 4, 16, 6]);  getitem_236 = None
        clone_250 = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_674 = torch.ops.aten.view.default(clone_250, [6272, 16, 6]);  clone_250 = None
        expand_131 = torch.ops.aten.expand.default(permute_317, [1568, 4, 6, 16]);  permute_317 = None
        clone_251 = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_675 = torch.ops.aten.view.default(clone_251, [6272, 6, 16]);  clone_251 = None
        bmm_64 = torch.ops.aten.bmm.default(view_674, view_675);  view_674 = view_675 = None
        view_676 = torch.ops.aten.view.default(bmm_64, [1568, 4, 16, 16]);  bmm_64 = None
        mul_tensor_30 = torch.ops.aten.mul.Tensor(view_676, 1);  view_676 = None
        amax_default_15 = torch.ops.aten.amax.default(mul_tensor_30, [-1], True)
        sub_tensor_15 = torch.ops.aten.sub.Tensor(mul_tensor_30, amax_default_15);  mul_tensor_30 = amax_default_15 = None
        mul_tensor_31 = torch.ops.aten.mul.Tensor(sub_tensor_15, 0.408248290463863);  sub_tensor_15 = None
        exp_32 = torch.ops.aten.exp.default(mul_tensor_31);  mul_tensor_31 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_32 = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        expand_132 = torch.ops.aten.expand.default(div_32, [1568, 4, 16, 16]);  div_32 = None
        view_677 = torch.ops.aten.view.default(expand_132, [6272, 16, 16]);  expand_132 = None
        expand_133 = torch.ops.aten.expand.default(permute_316, [1568, 4, 16, 6]);  permute_316 = None
        clone_252 = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_678 = torch.ops.aten.view.default(clone_252, [6272, 16, 6]);  clone_252 = None
        bmm_65 = torch.ops.aten.bmm.default(view_677, view_678);  view_677 = view_678 = None
        view_679 = torch.ops.aten.view.default(bmm_65, [1568, 4, 16, 6]);  bmm_65 = None
        permute_318 = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
        clone_253 = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        view_680 = torch.ops.aten.view.default(clone_253, [1568, 16, 24]);  clone_253 = None
        view_681 = torch.ops.aten.view.default(view_680, [25088, 24]);  view_680 = None
        permute_319 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg129_1, view_681, permute_319);  arg129_1 = view_681 = permute_319 = None
        view_682 = torch.ops.aten.view.default(addmm_115, [1568, 16, 24]);  addmm_115 = None
        add_292 = torch.ops.aten.add.Tensor(add_279, view_682);  add_279 = view_682 = None
        clone_254 = torch.ops.aten.clone.default(add_292, memory_format = torch.contiguous_format)
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
        getitem_238 = var_mean_86[0]
        getitem_239 = var_mean_86[1];  var_mean_86 = None
        add_293 = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        sub_119 = torch.ops.aten.sub.Tensor(clone_254, getitem_239);  clone_254 = getitem_239 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_86);  sub_119 = rsqrt_86 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, arg130_1);  mul_301 = arg130_1 = None
        add_294 = torch.ops.aten.add.Tensor(mul_302, arg131_1);  mul_302 = arg131_1 = None
        view_683 = torch.ops.aten.view.default(add_294, [25088, 24]);  add_294 = None
        permute_320 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg133_1, view_683, permute_320);  arg133_1 = view_683 = permute_320 = None
        view_684 = torch.ops.aten.view.default(addmm_116, [1568, 16, 96]);  addmm_116 = None
        mul_303 = torch.ops.aten.mul.Tensor(view_684, 0.5)
        mul_304 = torch.ops.aten.mul.Tensor(view_684, 0.7071067811865476);  view_684 = None
        erf_32 = torch.ops.aten.erf.default(mul_304);  mul_304 = None
        add_295 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_303, add_295);  mul_303 = add_295 = None
        view_685 = torch.ops.aten.view.default(mul_305, [25088, 96]);  mul_305 = None
        permute_321 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg135_1, view_685, permute_321);  arg135_1 = view_685 = permute_321 = None
        view_686 = torch.ops.aten.view.default(addmm_117, [1568, 16, 24]);  addmm_117 = None
        add_296 = torch.ops.aten.add.Tensor(add_292, view_686);  add_292 = view_686 = None
        slice_71 = torch.ops.aten.slice.Tensor(add_289, 1, 0, 1)
        slice_73 = torch.ops.aten.slice.Tensor(add_289, 1, 1, 9223372036854775807);  add_289 = None
        clone_257 = torch.ops.aten.clone.default(add_296, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_257, [2], correction = 0, keepdim = True)
        getitem_240 = var_mean_87[0]
        getitem_241 = var_mean_87[1];  var_mean_87 = None
        add_297 = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_297);  add_297 = None
        sub_120 = torch.ops.aten.sub.Tensor(clone_257, getitem_241);  clone_257 = getitem_241 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_120, rsqrt_87);  sub_120 = rsqrt_87 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_306, arg136_1);  mul_306 = arg136_1 = None
        add_298 = torch.ops.aten.add.Tensor(mul_307, arg137_1);  mul_307 = arg137_1 = None
        view_687 = torch.ops.aten.view.default(add_298, [8, 196, -1]);  add_298 = None
        view_688 = torch.ops.aten.view.default(view_687, [1568, 384]);  view_687 = None
        permute_322 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg139_1, view_688, permute_322);  arg139_1 = view_688 = permute_322 = None
        view_689 = torch.ops.aten.view.default(addmm_118, [8, 196, 384]);  addmm_118 = None
        add_299 = torch.ops.aten.add.Tensor(slice_73, view_689);  slice_73 = view_689 = None
        cat_18 = torch.ops.aten.cat.default([slice_71, add_299], 1);  slice_71 = add_299 = None
        var_mean_88 = torch.ops.aten.var_mean.correction(cat_18, [2], correction = 0, keepdim = True)
        getitem_242 = var_mean_88[0]
        getitem_243 = var_mean_88[1];  var_mean_88 = None
        add_300 = torch.ops.aten.add.Tensor(getitem_242, 1e-05);  getitem_242 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
        sub_121 = torch.ops.aten.sub.Tensor(cat_18, getitem_243);  getitem_243 = None
        mul_308 = torch.ops.aten.mul.Tensor(sub_121, rsqrt_88);  sub_121 = rsqrt_88 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_308, arg140_1);  mul_308 = arg140_1 = None
        add_301 = torch.ops.aten.add.Tensor(mul_309, arg141_1);  mul_309 = arg141_1 = None
        permute_323 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        view_690 = torch.ops.aten.view.default(add_301, [1576, 384])
        mm_66 = torch.ops.aten.mm.default(view_690, permute_323);  view_690 = permute_323 = None
        view_691 = torch.ops.aten.view.default(mm_66, [8, 197, 768]);  mm_66 = None
        view_692 = torch.ops.aten.view.default(view_691, [8, 197, 2, 6, 64]);  view_691 = None
        permute_324 = torch.ops.aten.permute.default(view_692, [2, 0, 3, 1, 4]);  view_692 = None
        unbind_33 = torch.ops.aten.unbind.int(permute_324);  permute_324 = None
        getitem_244 = unbind_33[0]
        getitem_245 = unbind_33[1];  unbind_33 = None
        permute_325 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        view_693 = torch.ops.aten.view.default(add_301, [1576, 384]);  add_301 = None
        mm_67 = torch.ops.aten.mm.default(view_693, permute_325);  view_693 = permute_325 = None
        view_694 = torch.ops.aten.view.default(mm_67, [8, 197, 384]);  mm_67 = None
        view_695 = torch.ops.aten.view.default(view_694, [8, 197, 6, -1]);  view_694 = None
        permute_326 = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
        permute_327 = torch.ops.aten.permute.default(getitem_245, [0, 1, 3, 2]);  getitem_245 = None
        expand_134 = torch.ops.aten.expand.default(getitem_244, [8, 6, 197, 64]);  getitem_244 = None
        clone_258 = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
        view_696 = torch.ops.aten.view.default(clone_258, [48, 197, 64]);  clone_258 = None
        expand_135 = torch.ops.aten.expand.default(permute_327, [8, 6, 64, 197]);  permute_327 = None
        clone_259 = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_697 = torch.ops.aten.view.default(clone_259, [48, 64, 197]);  clone_259 = None
        bmm_66 = torch.ops.aten.bmm.default(view_696, view_697);  view_696 = view_697 = None
        view_698 = torch.ops.aten.view.default(bmm_66, [8, 6, 197, 197]);  bmm_66 = None
        mul_tensor_28 = torch.ops.aten.mul.Tensor(view_698, 1);  view_698 = None
        amax_default_14 = torch.ops.aten.amax.default(mul_tensor_28, [-1], True)
        sub_tensor_14 = torch.ops.aten.sub.Tensor(mul_tensor_28, amax_default_14);  mul_tensor_28 = amax_default_14 = None
        mul_tensor_29 = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        exp_33 = torch.ops.aten.exp.default(mul_tensor_29);  mul_tensor_29 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
        expand_136 = torch.ops.aten.expand.default(div_33, [8, 6, 197, 197]);  div_33 = None
        view_699 = torch.ops.aten.view.default(expand_136, [48, 197, 197]);  expand_136 = None
        expand_137 = torch.ops.aten.expand.default(permute_326, [8, 6, 197, 64]);  permute_326 = None
        clone_260 = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_700 = torch.ops.aten.view.default(clone_260, [48, 197, 64]);  clone_260 = None
        bmm_67 = torch.ops.aten.bmm.default(view_699, view_700);  view_699 = view_700 = None
        view_701 = torch.ops.aten.view.default(bmm_67, [8, 6, 197, 64]);  bmm_67 = None
        permute_328 = torch.ops.aten.permute.default(view_701, [0, 2, 1, 3]);  view_701 = None
        clone_261 = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        view_702 = torch.ops.aten.view.default(clone_261, [8, 197, 384]);  clone_261 = None
        view_703 = torch.ops.aten.view.default(view_702, [1576, 384]);  view_702 = None
        permute_329 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg145_1, view_703, permute_329);  arg145_1 = view_703 = permute_329 = None
        view_704 = torch.ops.aten.view.default(addmm_119, [8, 197, 384]);  addmm_119 = None
        add_302 = torch.ops.aten.add.Tensor(cat_18, view_704);  cat_18 = view_704 = None
        var_mean_89 = torch.ops.aten.var_mean.correction(add_302, [2], correction = 0, keepdim = True)
        getitem_246 = var_mean_89[0]
        getitem_247 = var_mean_89[1];  var_mean_89 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_123 = torch.ops.aten.sub.Tensor(add_302, getitem_247);  getitem_247 = None
        mul_311 = torch.ops.aten.mul.Tensor(sub_123, rsqrt_89);  sub_123 = rsqrt_89 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_311, arg146_1);  mul_311 = arg146_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_312, arg147_1);  mul_312 = arg147_1 = None
        view_705 = torch.ops.aten.view.default(add_304, [1576, 384]);  add_304 = None
        permute_330 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg149_1, view_705, permute_330);  arg149_1 = view_705 = permute_330 = None
        view_706 = torch.ops.aten.view.default(addmm_120, [8, 197, 1536]);  addmm_120 = None
        mul_313 = torch.ops.aten.mul.Tensor(view_706, 0.5)
        mul_314 = torch.ops.aten.mul.Tensor(view_706, 0.7071067811865476);  view_706 = None
        erf_33 = torch.ops.aten.erf.default(mul_314);  mul_314 = None
        add_305 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_313, add_305);  mul_313 = add_305 = None
        view_707 = torch.ops.aten.view.default(mul_315, [1576, 1536]);  mul_315 = None
        permute_331 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg151_1, view_707, permute_331);  arg151_1 = view_707 = permute_331 = None
        view_708 = torch.ops.aten.view.default(addmm_121, [8, 197, 384]);  addmm_121 = None
        add_306 = torch.ops.aten.add.Tensor(add_302, view_708);  add_302 = view_708 = None
        clone_264 = torch.ops.aten.clone.default(add_296, memory_format = torch.contiguous_format)
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_264, [2], correction = 0, keepdim = True)
        getitem_248 = var_mean_90[0]
        getitem_249 = var_mean_90[1];  var_mean_90 = None
        add_307 = torch.ops.aten.add.Tensor(getitem_248, 1e-05);  getitem_248 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
        sub_124 = torch.ops.aten.sub.Tensor(clone_264, getitem_249);  clone_264 = getitem_249 = None
        mul_316 = torch.ops.aten.mul.Tensor(sub_124, rsqrt_90);  sub_124 = rsqrt_90 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, arg152_1);  mul_316 = arg152_1 = None
        add_308 = torch.ops.aten.add.Tensor(mul_317, arg153_1);  mul_317 = arg153_1 = None
        permute_332 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        view_709 = torch.ops.aten.view.default(add_308, [25088, 24])
        mm_68 = torch.ops.aten.mm.default(view_709, permute_332);  view_709 = permute_332 = None
        view_710 = torch.ops.aten.view.default(mm_68, [1568, 16, 48]);  mm_68 = None
        view_711 = torch.ops.aten.view.default(view_710, [1568, 16, 2, 4, 6]);  view_710 = None
        permute_333 = torch.ops.aten.permute.default(view_711, [2, 0, 3, 1, 4]);  view_711 = None
        unbind_34 = torch.ops.aten.unbind.int(permute_333);  permute_333 = None
        getitem_250 = unbind_34[0]
        getitem_251 = unbind_34[1];  unbind_34 = None
        permute_334 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        view_712 = torch.ops.aten.view.default(add_308, [25088, 24]);  add_308 = None
        mm_69 = torch.ops.aten.mm.default(view_712, permute_334);  view_712 = permute_334 = None
        view_713 = torch.ops.aten.view.default(mm_69, [1568, 16, 24]);  mm_69 = None
        view_714 = torch.ops.aten.view.default(view_713, [1568, 16, 4, -1]);  view_713 = None
        permute_335 = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None
        permute_336 = torch.ops.aten.permute.default(getitem_251, [0, 1, 3, 2]);  getitem_251 = None
        expand_138 = torch.ops.aten.expand.default(getitem_250, [1568, 4, 16, 6]);  getitem_250 = None
        clone_265 = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_715 = torch.ops.aten.view.default(clone_265, [6272, 16, 6]);  clone_265 = None
        expand_139 = torch.ops.aten.expand.default(permute_336, [1568, 4, 6, 16]);  permute_336 = None
        clone_266 = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_716 = torch.ops.aten.view.default(clone_266, [6272, 6, 16]);  clone_266 = None
        bmm_68 = torch.ops.aten.bmm.default(view_715, view_716);  view_715 = view_716 = None
        view_717 = torch.ops.aten.view.default(bmm_68, [1568, 4, 16, 16]);  bmm_68 = None
        mul_tensor_26 = torch.ops.aten.mul.Tensor(view_717, 1);  view_717 = None
        amax_default_13 = torch.ops.aten.amax.default(mul_tensor_26, [-1], True)
        sub_tensor_13 = torch.ops.aten.sub.Tensor(mul_tensor_26, amax_default_13);  mul_tensor_26 = amax_default_13 = None
        mul_tensor_27 = torch.ops.aten.mul.Tensor(sub_tensor_13, 0.408248290463863);  sub_tensor_13 = None
        exp_34 = torch.ops.aten.exp.default(mul_tensor_27);  mul_tensor_27 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_34 = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        expand_140 = torch.ops.aten.expand.default(div_34, [1568, 4, 16, 16]);  div_34 = None
        view_718 = torch.ops.aten.view.default(expand_140, [6272, 16, 16]);  expand_140 = None
        expand_141 = torch.ops.aten.expand.default(permute_335, [1568, 4, 16, 6]);  permute_335 = None
        clone_267 = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_719 = torch.ops.aten.view.default(clone_267, [6272, 16, 6]);  clone_267 = None
        bmm_69 = torch.ops.aten.bmm.default(view_718, view_719);  view_718 = view_719 = None
        view_720 = torch.ops.aten.view.default(bmm_69, [1568, 4, 16, 6]);  bmm_69 = None
        permute_337 = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
        clone_268 = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_721 = torch.ops.aten.view.default(clone_268, [1568, 16, 24]);  clone_268 = None
        view_722 = torch.ops.aten.view.default(view_721, [25088, 24]);  view_721 = None
        permute_338 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg157_1, view_722, permute_338);  arg157_1 = view_722 = permute_338 = None
        view_723 = torch.ops.aten.view.default(addmm_122, [1568, 16, 24]);  addmm_122 = None
        add_309 = torch.ops.aten.add.Tensor(add_296, view_723);  add_296 = view_723 = None
        clone_269 = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_269, [2], correction = 0, keepdim = True)
        getitem_252 = var_mean_91[0]
        getitem_253 = var_mean_91[1];  var_mean_91 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_126 = torch.ops.aten.sub.Tensor(clone_269, getitem_253);  clone_269 = getitem_253 = None
        mul_319 = torch.ops.aten.mul.Tensor(sub_126, rsqrt_91);  sub_126 = rsqrt_91 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, arg158_1);  mul_319 = arg158_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_320, arg159_1);  mul_320 = arg159_1 = None
        view_724 = torch.ops.aten.view.default(add_311, [25088, 24]);  add_311 = None
        permute_339 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg161_1, view_724, permute_339);  arg161_1 = view_724 = permute_339 = None
        view_725 = torch.ops.aten.view.default(addmm_123, [1568, 16, 96]);  addmm_123 = None
        mul_321 = torch.ops.aten.mul.Tensor(view_725, 0.5)
        mul_322 = torch.ops.aten.mul.Tensor(view_725, 0.7071067811865476);  view_725 = None
        erf_34 = torch.ops.aten.erf.default(mul_322);  mul_322 = None
        add_312 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_321, add_312);  mul_321 = add_312 = None
        view_726 = torch.ops.aten.view.default(mul_323, [25088, 96]);  mul_323 = None
        permute_340 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg163_1, view_726, permute_340);  arg163_1 = view_726 = permute_340 = None
        view_727 = torch.ops.aten.view.default(addmm_124, [1568, 16, 24]);  addmm_124 = None
        add_313 = torch.ops.aten.add.Tensor(add_309, view_727);  add_309 = view_727 = None
        slice_75 = torch.ops.aten.slice.Tensor(add_306, 1, 0, 1)
        slice_77 = torch.ops.aten.slice.Tensor(add_306, 1, 1, 9223372036854775807);  add_306 = None
        clone_272 = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_272, [2], correction = 0, keepdim = True)
        getitem_254 = var_mean_92[0]
        getitem_255 = var_mean_92[1];  var_mean_92 = None
        add_314 = torch.ops.aten.add.Tensor(getitem_254, 1e-05);  getitem_254 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        sub_127 = torch.ops.aten.sub.Tensor(clone_272, getitem_255);  clone_272 = getitem_255 = None
        mul_324 = torch.ops.aten.mul.Tensor(sub_127, rsqrt_92);  sub_127 = rsqrt_92 = None
        mul_325 = torch.ops.aten.mul.Tensor(mul_324, arg164_1);  mul_324 = arg164_1 = None
        add_315 = torch.ops.aten.add.Tensor(mul_325, arg165_1);  mul_325 = arg165_1 = None
        view_728 = torch.ops.aten.view.default(add_315, [8, 196, -1]);  add_315 = None
        view_729 = torch.ops.aten.view.default(view_728, [1568, 384]);  view_728 = None
        permute_341 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg167_1, view_729, permute_341);  arg167_1 = view_729 = permute_341 = None
        view_730 = torch.ops.aten.view.default(addmm_125, [8, 196, 384]);  addmm_125 = None
        add_316 = torch.ops.aten.add.Tensor(slice_77, view_730);  slice_77 = view_730 = None
        cat_19 = torch.ops.aten.cat.default([slice_75, add_316], 1);  slice_75 = add_316 = None
        var_mean_93 = torch.ops.aten.var_mean.correction(cat_19, [2], correction = 0, keepdim = True)
        getitem_256 = var_mean_93[0]
        getitem_257 = var_mean_93[1];  var_mean_93 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_256, 1e-05);  getitem_256 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_128 = torch.ops.aten.sub.Tensor(cat_19, getitem_257);  getitem_257 = None
        mul_326 = torch.ops.aten.mul.Tensor(sub_128, rsqrt_93);  sub_128 = rsqrt_93 = None
        mul_327 = torch.ops.aten.mul.Tensor(mul_326, arg168_1);  mul_326 = arg168_1 = None
        add_318 = torch.ops.aten.add.Tensor(mul_327, arg169_1);  mul_327 = arg169_1 = None
        permute_342 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        view_731 = torch.ops.aten.view.default(add_318, [1576, 384])
        mm_70 = torch.ops.aten.mm.default(view_731, permute_342);  view_731 = permute_342 = None
        view_732 = torch.ops.aten.view.default(mm_70, [8, 197, 768]);  mm_70 = None
        view_733 = torch.ops.aten.view.default(view_732, [8, 197, 2, 6, 64]);  view_732 = None
        permute_343 = torch.ops.aten.permute.default(view_733, [2, 0, 3, 1, 4]);  view_733 = None
        unbind_35 = torch.ops.aten.unbind.int(permute_343);  permute_343 = None
        getitem_258 = unbind_35[0]
        getitem_259 = unbind_35[1];  unbind_35 = None
        permute_344 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        view_734 = torch.ops.aten.view.default(add_318, [1576, 384]);  add_318 = None
        mm_71 = torch.ops.aten.mm.default(view_734, permute_344);  view_734 = permute_344 = None
        view_735 = torch.ops.aten.view.default(mm_71, [8, 197, 384]);  mm_71 = None
        view_736 = torch.ops.aten.view.default(view_735, [8, 197, 6, -1]);  view_735 = None
        permute_345 = torch.ops.aten.permute.default(view_736, [0, 2, 1, 3]);  view_736 = None
        permute_346 = torch.ops.aten.permute.default(getitem_259, [0, 1, 3, 2]);  getitem_259 = None
        expand_142 = torch.ops.aten.expand.default(getitem_258, [8, 6, 197, 64]);  getitem_258 = None
        clone_273 = torch.ops.aten.clone.default(expand_142, memory_format = torch.contiguous_format);  expand_142 = None
        view_737 = torch.ops.aten.view.default(clone_273, [48, 197, 64]);  clone_273 = None
        expand_143 = torch.ops.aten.expand.default(permute_346, [8, 6, 64, 197]);  permute_346 = None
        clone_274 = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
        view_738 = torch.ops.aten.view.default(clone_274, [48, 64, 197]);  clone_274 = None
        bmm_70 = torch.ops.aten.bmm.default(view_737, view_738);  view_737 = view_738 = None
        view_739 = torch.ops.aten.view.default(bmm_70, [8, 6, 197, 197]);  bmm_70 = None
        mul_tensor_24 = torch.ops.aten.mul.Tensor(view_739, 1);  view_739 = None
        amax_default_12 = torch.ops.aten.amax.default(mul_tensor_24, [-1], True)
        sub_tensor_12 = torch.ops.aten.sub.Tensor(mul_tensor_24, amax_default_12);  mul_tensor_24 = amax_default_12 = None
        mul_tensor_25 = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        exp_35 = torch.ops.aten.exp.default(mul_tensor_25);  mul_tensor_25 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_35 = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
        expand_144 = torch.ops.aten.expand.default(div_35, [8, 6, 197, 197]);  div_35 = None
        view_740 = torch.ops.aten.view.default(expand_144, [48, 197, 197]);  expand_144 = None
        expand_145 = torch.ops.aten.expand.default(permute_345, [8, 6, 197, 64]);  permute_345 = None
        clone_275 = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_741 = torch.ops.aten.view.default(clone_275, [48, 197, 64]);  clone_275 = None
        bmm_71 = torch.ops.aten.bmm.default(view_740, view_741);  view_740 = view_741 = None
        view_742 = torch.ops.aten.view.default(bmm_71, [8, 6, 197, 64]);  bmm_71 = None
        permute_347 = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
        clone_276 = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_743 = torch.ops.aten.view.default(clone_276, [8, 197, 384]);  clone_276 = None
        view_744 = torch.ops.aten.view.default(view_743, [1576, 384]);  view_743 = None
        permute_348 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg173_1, view_744, permute_348);  arg173_1 = view_744 = permute_348 = None
        view_745 = torch.ops.aten.view.default(addmm_126, [8, 197, 384]);  addmm_126 = None
        add_319 = torch.ops.aten.add.Tensor(cat_19, view_745);  cat_19 = view_745 = None
        var_mean_94 = torch.ops.aten.var_mean.correction(add_319, [2], correction = 0, keepdim = True)
        getitem_260 = var_mean_94[0]
        getitem_261 = var_mean_94[1];  var_mean_94 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_260, 1e-05);  getitem_260 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_130 = torch.ops.aten.sub.Tensor(add_319, getitem_261);  getitem_261 = None
        mul_329 = torch.ops.aten.mul.Tensor(sub_130, rsqrt_94);  sub_130 = rsqrt_94 = None
        mul_330 = torch.ops.aten.mul.Tensor(mul_329, arg174_1);  mul_329 = arg174_1 = None
        add_321 = torch.ops.aten.add.Tensor(mul_330, arg175_1);  mul_330 = arg175_1 = None
        view_746 = torch.ops.aten.view.default(add_321, [1576, 384]);  add_321 = None
        permute_349 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg177_1, view_746, permute_349);  arg177_1 = view_746 = permute_349 = None
        view_747 = torch.ops.aten.view.default(addmm_127, [8, 197, 1536]);  addmm_127 = None
        mul_331 = torch.ops.aten.mul.Tensor(view_747, 0.5)
        mul_332 = torch.ops.aten.mul.Tensor(view_747, 0.7071067811865476);  view_747 = None
        erf_35 = torch.ops.aten.erf.default(mul_332);  mul_332 = None
        add_322 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_331, add_322);  mul_331 = add_322 = None
        view_748 = torch.ops.aten.view.default(mul_333, [1576, 1536]);  mul_333 = None
        permute_350 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg179_1, view_748, permute_350);  arg179_1 = view_748 = permute_350 = None
        view_749 = torch.ops.aten.view.default(addmm_128, [8, 197, 384]);  addmm_128 = None
        add_323 = torch.ops.aten.add.Tensor(add_319, view_749);  add_319 = view_749 = None
        clone_279 = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_279, [2], correction = 0, keepdim = True)
        getitem_262 = var_mean_95[0]
        getitem_263 = var_mean_95[1];  var_mean_95 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_262, 1e-05);  getitem_262 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_131 = torch.ops.aten.sub.Tensor(clone_279, getitem_263);  clone_279 = getitem_263 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_131, rsqrt_95);  sub_131 = rsqrt_95 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, arg180_1);  mul_334 = arg180_1 = None
        add_325 = torch.ops.aten.add.Tensor(mul_335, arg181_1);  mul_335 = arg181_1 = None
        permute_351 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        view_750 = torch.ops.aten.view.default(add_325, [25088, 24])
        mm_72 = torch.ops.aten.mm.default(view_750, permute_351);  view_750 = permute_351 = None
        view_751 = torch.ops.aten.view.default(mm_72, [1568, 16, 48]);  mm_72 = None
        view_752 = torch.ops.aten.view.default(view_751, [1568, 16, 2, 4, 6]);  view_751 = None
        permute_352 = torch.ops.aten.permute.default(view_752, [2, 0, 3, 1, 4]);  view_752 = None
        unbind_36 = torch.ops.aten.unbind.int(permute_352);  permute_352 = None
        getitem_264 = unbind_36[0]
        getitem_265 = unbind_36[1];  unbind_36 = None
        permute_353 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        view_753 = torch.ops.aten.view.default(add_325, [25088, 24]);  add_325 = None
        mm_73 = torch.ops.aten.mm.default(view_753, permute_353);  view_753 = permute_353 = None
        view_754 = torch.ops.aten.view.default(mm_73, [1568, 16, 24]);  mm_73 = None
        view_755 = torch.ops.aten.view.default(view_754, [1568, 16, 4, -1]);  view_754 = None
        permute_354 = torch.ops.aten.permute.default(view_755, [0, 2, 1, 3]);  view_755 = None
        permute_355 = torch.ops.aten.permute.default(getitem_265, [0, 1, 3, 2]);  getitem_265 = None
        expand_146 = torch.ops.aten.expand.default(getitem_264, [1568, 4, 16, 6]);  getitem_264 = None
        clone_280 = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_756 = torch.ops.aten.view.default(clone_280, [6272, 16, 6]);  clone_280 = None
        expand_147 = torch.ops.aten.expand.default(permute_355, [1568, 4, 6, 16]);  permute_355 = None
        clone_281 = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_757 = torch.ops.aten.view.default(clone_281, [6272, 6, 16]);  clone_281 = None
        bmm_72 = torch.ops.aten.bmm.default(view_756, view_757);  view_756 = view_757 = None
        view_758 = torch.ops.aten.view.default(bmm_72, [1568, 4, 16, 16]);  bmm_72 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(view_758, 1);  view_758 = None
        amax_default_11 = torch.ops.aten.amax.default(mul_tensor_22, [-1], True)
        sub_tensor_11 = torch.ops.aten.sub.Tensor(mul_tensor_22, amax_default_11);  mul_tensor_22 = amax_default_11 = None
        mul_tensor_23 = torch.ops.aten.mul.Tensor(sub_tensor_11, 0.408248290463863);  sub_tensor_11 = None
        exp_36 = torch.ops.aten.exp.default(mul_tensor_23);  mul_tensor_23 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36 = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        expand_148 = torch.ops.aten.expand.default(div_36, [1568, 4, 16, 16]);  div_36 = None
        view_759 = torch.ops.aten.view.default(expand_148, [6272, 16, 16]);  expand_148 = None
        expand_149 = torch.ops.aten.expand.default(permute_354, [1568, 4, 16, 6]);  permute_354 = None
        clone_282 = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_760 = torch.ops.aten.view.default(clone_282, [6272, 16, 6]);  clone_282 = None
        bmm_73 = torch.ops.aten.bmm.default(view_759, view_760);  view_759 = view_760 = None
        view_761 = torch.ops.aten.view.default(bmm_73, [1568, 4, 16, 6]);  bmm_73 = None
        permute_356 = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
        clone_283 = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
        view_762 = torch.ops.aten.view.default(clone_283, [1568, 16, 24]);  clone_283 = None
        view_763 = torch.ops.aten.view.default(view_762, [25088, 24]);  view_762 = None
        permute_357 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg185_1, view_763, permute_357);  arg185_1 = view_763 = permute_357 = None
        view_764 = torch.ops.aten.view.default(addmm_129, [1568, 16, 24]);  addmm_129 = None
        add_326 = torch.ops.aten.add.Tensor(add_313, view_764);  add_313 = view_764 = None
        clone_284 = torch.ops.aten.clone.default(add_326, memory_format = torch.contiguous_format)
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_284, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_96[0]
        getitem_267 = var_mean_96[1];  var_mean_96 = None
        add_327 = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        sub_133 = torch.ops.aten.sub.Tensor(clone_284, getitem_267);  clone_284 = getitem_267 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_96);  sub_133 = rsqrt_96 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_337, arg186_1);  mul_337 = arg186_1 = None
        add_328 = torch.ops.aten.add.Tensor(mul_338, arg187_1);  mul_338 = arg187_1 = None
        view_765 = torch.ops.aten.view.default(add_328, [25088, 24]);  add_328 = None
        permute_358 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg189_1, view_765, permute_358);  arg189_1 = view_765 = permute_358 = None
        view_766 = torch.ops.aten.view.default(addmm_130, [1568, 16, 96]);  addmm_130 = None
        mul_339 = torch.ops.aten.mul.Tensor(view_766, 0.5)
        mul_340 = torch.ops.aten.mul.Tensor(view_766, 0.7071067811865476);  view_766 = None
        erf_36 = torch.ops.aten.erf.default(mul_340);  mul_340 = None
        add_329 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_339, add_329);  mul_339 = add_329 = None
        view_767 = torch.ops.aten.view.default(mul_341, [25088, 96]);  mul_341 = None
        permute_359 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg191_1, view_767, permute_359);  arg191_1 = view_767 = permute_359 = None
        view_768 = torch.ops.aten.view.default(addmm_131, [1568, 16, 24]);  addmm_131 = None
        add_330 = torch.ops.aten.add.Tensor(add_326, view_768);  add_326 = view_768 = None
        slice_79 = torch.ops.aten.slice.Tensor(add_323, 1, 0, 1)
        slice_81 = torch.ops.aten.slice.Tensor(add_323, 1, 1, 9223372036854775807);  add_323 = None
        clone_287 = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_287, [2], correction = 0, keepdim = True)
        getitem_268 = var_mean_97[0]
        getitem_269 = var_mean_97[1];  var_mean_97 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_134 = torch.ops.aten.sub.Tensor(clone_287, getitem_269);  clone_287 = getitem_269 = None
        mul_342 = torch.ops.aten.mul.Tensor(sub_134, rsqrt_97);  sub_134 = rsqrt_97 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_342, arg192_1);  mul_342 = arg192_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_343, arg193_1);  mul_343 = arg193_1 = None
        view_769 = torch.ops.aten.view.default(add_332, [8, 196, -1]);  add_332 = None
        view_770 = torch.ops.aten.view.default(view_769, [1568, 384]);  view_769 = None
        permute_360 = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg195_1, view_770, permute_360);  arg195_1 = view_770 = permute_360 = None
        view_771 = torch.ops.aten.view.default(addmm_132, [8, 196, 384]);  addmm_132 = None
        add_333 = torch.ops.aten.add.Tensor(slice_81, view_771);  slice_81 = view_771 = None
        cat_20 = torch.ops.aten.cat.default([slice_79, add_333], 1);  slice_79 = add_333 = None
        var_mean_98 = torch.ops.aten.var_mean.correction(cat_20, [2], correction = 0, keepdim = True)
        getitem_270 = var_mean_98[0]
        getitem_271 = var_mean_98[1];  var_mean_98 = None
        add_334 = torch.ops.aten.add.Tensor(getitem_270, 1e-05);  getitem_270 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_135 = torch.ops.aten.sub.Tensor(cat_20, getitem_271);  getitem_271 = None
        mul_344 = torch.ops.aten.mul.Tensor(sub_135, rsqrt_98);  sub_135 = rsqrt_98 = None
        mul_345 = torch.ops.aten.mul.Tensor(mul_344, arg196_1);  mul_344 = arg196_1 = None
        add_335 = torch.ops.aten.add.Tensor(mul_345, arg197_1);  mul_345 = arg197_1 = None
        permute_361 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        view_772 = torch.ops.aten.view.default(add_335, [1576, 384])
        mm_74 = torch.ops.aten.mm.default(view_772, permute_361);  view_772 = permute_361 = None
        view_773 = torch.ops.aten.view.default(mm_74, [8, 197, 768]);  mm_74 = None
        view_774 = torch.ops.aten.view.default(view_773, [8, 197, 2, 6, 64]);  view_773 = None
        permute_362 = torch.ops.aten.permute.default(view_774, [2, 0, 3, 1, 4]);  view_774 = None
        unbind_37 = torch.ops.aten.unbind.int(permute_362);  permute_362 = None
        getitem_272 = unbind_37[0]
        getitem_273 = unbind_37[1];  unbind_37 = None
        permute_363 = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        view_775 = torch.ops.aten.view.default(add_335, [1576, 384]);  add_335 = None
        mm_75 = torch.ops.aten.mm.default(view_775, permute_363);  view_775 = permute_363 = None
        view_776 = torch.ops.aten.view.default(mm_75, [8, 197, 384]);  mm_75 = None
        view_777 = torch.ops.aten.view.default(view_776, [8, 197, 6, -1]);  view_776 = None
        permute_364 = torch.ops.aten.permute.default(view_777, [0, 2, 1, 3]);  view_777 = None
        permute_365 = torch.ops.aten.permute.default(getitem_273, [0, 1, 3, 2]);  getitem_273 = None
        expand_150 = torch.ops.aten.expand.default(getitem_272, [8, 6, 197, 64]);  getitem_272 = None
        clone_288 = torch.ops.aten.clone.default(expand_150, memory_format = torch.contiguous_format);  expand_150 = None
        view_778 = torch.ops.aten.view.default(clone_288, [48, 197, 64]);  clone_288 = None
        expand_151 = torch.ops.aten.expand.default(permute_365, [8, 6, 64, 197]);  permute_365 = None
        clone_289 = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_779 = torch.ops.aten.view.default(clone_289, [48, 64, 197]);  clone_289 = None
        bmm_74 = torch.ops.aten.bmm.default(view_778, view_779);  view_778 = view_779 = None
        view_780 = torch.ops.aten.view.default(bmm_74, [8, 6, 197, 197]);  bmm_74 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(view_780, 1);  view_780 = None
        amax_default_10 = torch.ops.aten.amax.default(mul_tensor_20, [-1], True)
        sub_tensor_10 = torch.ops.aten.sub.Tensor(mul_tensor_20, amax_default_10);  mul_tensor_20 = amax_default_10 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        exp_37 = torch.ops.aten.exp.default(mul_tensor_21);  mul_tensor_21 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37 = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        expand_152 = torch.ops.aten.expand.default(div_37, [8, 6, 197, 197]);  div_37 = None
        view_781 = torch.ops.aten.view.default(expand_152, [48, 197, 197]);  expand_152 = None
        expand_153 = torch.ops.aten.expand.default(permute_364, [8, 6, 197, 64]);  permute_364 = None
        clone_290 = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_782 = torch.ops.aten.view.default(clone_290, [48, 197, 64]);  clone_290 = None
        bmm_75 = torch.ops.aten.bmm.default(view_781, view_782);  view_781 = view_782 = None
        view_783 = torch.ops.aten.view.default(bmm_75, [8, 6, 197, 64]);  bmm_75 = None
        permute_366 = torch.ops.aten.permute.default(view_783, [0, 2, 1, 3]);  view_783 = None
        clone_291 = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
        view_784 = torch.ops.aten.view.default(clone_291, [8, 197, 384]);  clone_291 = None
        view_785 = torch.ops.aten.view.default(view_784, [1576, 384]);  view_784 = None
        permute_367 = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg201_1, view_785, permute_367);  arg201_1 = view_785 = permute_367 = None
        view_786 = torch.ops.aten.view.default(addmm_133, [8, 197, 384]);  addmm_133 = None
        add_336 = torch.ops.aten.add.Tensor(cat_20, view_786);  cat_20 = view_786 = None
        var_mean_99 = torch.ops.aten.var_mean.correction(add_336, [2], correction = 0, keepdim = True)
        getitem_274 = var_mean_99[0]
        getitem_275 = var_mean_99[1];  var_mean_99 = None
        add_337 = torch.ops.aten.add.Tensor(getitem_274, 1e-05);  getitem_274 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_337);  add_337 = None
        sub_137 = torch.ops.aten.sub.Tensor(add_336, getitem_275);  getitem_275 = None
        mul_347 = torch.ops.aten.mul.Tensor(sub_137, rsqrt_99);  sub_137 = rsqrt_99 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_347, arg202_1);  mul_347 = arg202_1 = None
        add_338 = torch.ops.aten.add.Tensor(mul_348, arg203_1);  mul_348 = arg203_1 = None
        view_787 = torch.ops.aten.view.default(add_338, [1576, 384]);  add_338 = None
        permute_368 = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg205_1, view_787, permute_368);  arg205_1 = view_787 = permute_368 = None
        view_788 = torch.ops.aten.view.default(addmm_134, [8, 197, 1536]);  addmm_134 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_788, 0.5)
        mul_350 = torch.ops.aten.mul.Tensor(view_788, 0.7071067811865476);  view_788 = None
        erf_37 = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_339 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_349, add_339);  mul_349 = add_339 = None
        view_789 = torch.ops.aten.view.default(mul_351, [1576, 1536]);  mul_351 = None
        permute_369 = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg207_1, view_789, permute_369);  arg207_1 = view_789 = permute_369 = None
        view_790 = torch.ops.aten.view.default(addmm_135, [8, 197, 384]);  addmm_135 = None
        add_340 = torch.ops.aten.add.Tensor(add_336, view_790);  add_336 = view_790 = None
        clone_294 = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_100 = torch.ops.aten.var_mean.correction(clone_294, [2], correction = 0, keepdim = True)
        getitem_276 = var_mean_100[0]
        getitem_277 = var_mean_100[1];  var_mean_100 = None
        add_341 = torch.ops.aten.add.Tensor(getitem_276, 1e-05);  getitem_276 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_138 = torch.ops.aten.sub.Tensor(clone_294, getitem_277);  clone_294 = getitem_277 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_138, rsqrt_100);  sub_138 = rsqrt_100 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, arg208_1);  mul_352 = arg208_1 = None
        add_342 = torch.ops.aten.add.Tensor(mul_353, arg209_1);  mul_353 = arg209_1 = None
        permute_370 = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        view_791 = torch.ops.aten.view.default(add_342, [25088, 24])
        mm_76 = torch.ops.aten.mm.default(view_791, permute_370);  view_791 = permute_370 = None
        view_792 = torch.ops.aten.view.default(mm_76, [1568, 16, 48]);  mm_76 = None
        view_793 = torch.ops.aten.view.default(view_792, [1568, 16, 2, 4, 6]);  view_792 = None
        permute_371 = torch.ops.aten.permute.default(view_793, [2, 0, 3, 1, 4]);  view_793 = None
        unbind_38 = torch.ops.aten.unbind.int(permute_371);  permute_371 = None
        getitem_278 = unbind_38[0]
        getitem_279 = unbind_38[1];  unbind_38 = None
        permute_372 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        view_794 = torch.ops.aten.view.default(add_342, [25088, 24]);  add_342 = None
        mm_77 = torch.ops.aten.mm.default(view_794, permute_372);  view_794 = permute_372 = None
        view_795 = torch.ops.aten.view.default(mm_77, [1568, 16, 24]);  mm_77 = None
        view_796 = torch.ops.aten.view.default(view_795, [1568, 16, 4, -1]);  view_795 = None
        permute_373 = torch.ops.aten.permute.default(view_796, [0, 2, 1, 3]);  view_796 = None
        permute_374 = torch.ops.aten.permute.default(getitem_279, [0, 1, 3, 2]);  getitem_279 = None
        expand_154 = torch.ops.aten.expand.default(getitem_278, [1568, 4, 16, 6]);  getitem_278 = None
        clone_295 = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_797 = torch.ops.aten.view.default(clone_295, [6272, 16, 6]);  clone_295 = None
        expand_155 = torch.ops.aten.expand.default(permute_374, [1568, 4, 6, 16]);  permute_374 = None
        clone_296 = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_798 = torch.ops.aten.view.default(clone_296, [6272, 6, 16]);  clone_296 = None
        bmm_76 = torch.ops.aten.bmm.default(view_797, view_798);  view_797 = view_798 = None
        view_799 = torch.ops.aten.view.default(bmm_76, [1568, 4, 16, 16]);  bmm_76 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(view_799, 1);  view_799 = None
        amax_default_9 = torch.ops.aten.amax.default(mul_tensor_18, [-1], True)
        sub_tensor_9 = torch.ops.aten.sub.Tensor(mul_tensor_18, amax_default_9);  mul_tensor_18 = amax_default_9 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(sub_tensor_9, 0.408248290463863);  sub_tensor_9 = None
        exp_38 = torch.ops.aten.exp.default(mul_tensor_19);  mul_tensor_19 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38 = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        expand_156 = torch.ops.aten.expand.default(div_38, [1568, 4, 16, 16]);  div_38 = None
        view_800 = torch.ops.aten.view.default(expand_156, [6272, 16, 16]);  expand_156 = None
        expand_157 = torch.ops.aten.expand.default(permute_373, [1568, 4, 16, 6]);  permute_373 = None
        clone_297 = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_801 = torch.ops.aten.view.default(clone_297, [6272, 16, 6]);  clone_297 = None
        bmm_77 = torch.ops.aten.bmm.default(view_800, view_801);  view_800 = view_801 = None
        view_802 = torch.ops.aten.view.default(bmm_77, [1568, 4, 16, 6]);  bmm_77 = None
        permute_375 = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None
        clone_298 = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
        view_803 = torch.ops.aten.view.default(clone_298, [1568, 16, 24]);  clone_298 = None
        view_804 = torch.ops.aten.view.default(view_803, [25088, 24]);  view_803 = None
        permute_376 = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg213_1, view_804, permute_376);  arg213_1 = view_804 = permute_376 = None
        view_805 = torch.ops.aten.view.default(addmm_136, [1568, 16, 24]);  addmm_136 = None
        add_343 = torch.ops.aten.add.Tensor(add_330, view_805);  add_330 = view_805 = None
        clone_299 = torch.ops.aten.clone.default(add_343, memory_format = torch.contiguous_format)
        var_mean_101 = torch.ops.aten.var_mean.correction(clone_299, [2], correction = 0, keepdim = True)
        getitem_280 = var_mean_101[0]
        getitem_281 = var_mean_101[1];  var_mean_101 = None
        add_344 = torch.ops.aten.add.Tensor(getitem_280, 1e-05);  getitem_280 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_344);  add_344 = None
        sub_140 = torch.ops.aten.sub.Tensor(clone_299, getitem_281);  clone_299 = getitem_281 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_140, rsqrt_101);  sub_140 = rsqrt_101 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, arg214_1);  mul_355 = arg214_1 = None
        add_345 = torch.ops.aten.add.Tensor(mul_356, arg215_1);  mul_356 = arg215_1 = None
        view_806 = torch.ops.aten.view.default(add_345, [25088, 24]);  add_345 = None
        permute_377 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg217_1, view_806, permute_377);  arg217_1 = view_806 = permute_377 = None
        view_807 = torch.ops.aten.view.default(addmm_137, [1568, 16, 96]);  addmm_137 = None
        mul_357 = torch.ops.aten.mul.Tensor(view_807, 0.5)
        mul_358 = torch.ops.aten.mul.Tensor(view_807, 0.7071067811865476);  view_807 = None
        erf_38 = torch.ops.aten.erf.default(mul_358);  mul_358 = None
        add_346 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_357, add_346);  mul_357 = add_346 = None
        view_808 = torch.ops.aten.view.default(mul_359, [25088, 96]);  mul_359 = None
        permute_378 = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg219_1, view_808, permute_378);  arg219_1 = view_808 = permute_378 = None
        view_809 = torch.ops.aten.view.default(addmm_138, [1568, 16, 24]);  addmm_138 = None
        add_347 = torch.ops.aten.add.Tensor(add_343, view_809);  add_343 = view_809 = None
        slice_83 = torch.ops.aten.slice.Tensor(add_340, 1, 0, 1)
        slice_85 = torch.ops.aten.slice.Tensor(add_340, 1, 1, 9223372036854775807);  add_340 = None
        clone_302 = torch.ops.aten.clone.default(add_347, memory_format = torch.contiguous_format)
        var_mean_102 = torch.ops.aten.var_mean.correction(clone_302, [2], correction = 0, keepdim = True)
        getitem_282 = var_mean_102[0]
        getitem_283 = var_mean_102[1];  var_mean_102 = None
        add_348 = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        sub_141 = torch.ops.aten.sub.Tensor(clone_302, getitem_283);  clone_302 = getitem_283 = None
        mul_360 = torch.ops.aten.mul.Tensor(sub_141, rsqrt_102);  sub_141 = rsqrt_102 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_360, arg220_1);  mul_360 = arg220_1 = None
        add_349 = torch.ops.aten.add.Tensor(mul_361, arg221_1);  mul_361 = arg221_1 = None
        view_810 = torch.ops.aten.view.default(add_349, [8, 196, -1]);  add_349 = None
        view_811 = torch.ops.aten.view.default(view_810, [1568, 384]);  view_810 = None
        permute_379 = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg223_1, view_811, permute_379);  arg223_1 = view_811 = permute_379 = None
        view_812 = torch.ops.aten.view.default(addmm_139, [8, 196, 384]);  addmm_139 = None
        add_350 = torch.ops.aten.add.Tensor(slice_85, view_812);  slice_85 = view_812 = None
        cat_21 = torch.ops.aten.cat.default([slice_83, add_350], 1);  slice_83 = add_350 = None
        var_mean_103 = torch.ops.aten.var_mean.correction(cat_21, [2], correction = 0, keepdim = True)
        getitem_284 = var_mean_103[0]
        getitem_285 = var_mean_103[1];  var_mean_103 = None
        add_351 = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_142 = torch.ops.aten.sub.Tensor(cat_21, getitem_285);  getitem_285 = None
        mul_362 = torch.ops.aten.mul.Tensor(sub_142, rsqrt_103);  sub_142 = rsqrt_103 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_362, arg224_1);  mul_362 = arg224_1 = None
        add_352 = torch.ops.aten.add.Tensor(mul_363, arg225_1);  mul_363 = arg225_1 = None
        permute_380 = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        view_813 = torch.ops.aten.view.default(add_352, [1576, 384])
        mm_78 = torch.ops.aten.mm.default(view_813, permute_380);  view_813 = permute_380 = None
        view_814 = torch.ops.aten.view.default(mm_78, [8, 197, 768]);  mm_78 = None
        view_815 = torch.ops.aten.view.default(view_814, [8, 197, 2, 6, 64]);  view_814 = None
        permute_381 = torch.ops.aten.permute.default(view_815, [2, 0, 3, 1, 4]);  view_815 = None
        unbind_39 = torch.ops.aten.unbind.int(permute_381);  permute_381 = None
        getitem_286 = unbind_39[0]
        getitem_287 = unbind_39[1];  unbind_39 = None
        permute_382 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        view_816 = torch.ops.aten.view.default(add_352, [1576, 384]);  add_352 = None
        mm_79 = torch.ops.aten.mm.default(view_816, permute_382);  view_816 = permute_382 = None
        view_817 = torch.ops.aten.view.default(mm_79, [8, 197, 384]);  mm_79 = None
        view_818 = torch.ops.aten.view.default(view_817, [8, 197, 6, -1]);  view_817 = None
        permute_383 = torch.ops.aten.permute.default(view_818, [0, 2, 1, 3]);  view_818 = None
        permute_384 = torch.ops.aten.permute.default(getitem_287, [0, 1, 3, 2]);  getitem_287 = None
        expand_158 = torch.ops.aten.expand.default(getitem_286, [8, 6, 197, 64]);  getitem_286 = None
        clone_303 = torch.ops.aten.clone.default(expand_158, memory_format = torch.contiguous_format);  expand_158 = None
        view_819 = torch.ops.aten.view.default(clone_303, [48, 197, 64]);  clone_303 = None
        expand_159 = torch.ops.aten.expand.default(permute_384, [8, 6, 64, 197]);  permute_384 = None
        clone_304 = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_820 = torch.ops.aten.view.default(clone_304, [48, 64, 197]);  clone_304 = None
        bmm_78 = torch.ops.aten.bmm.default(view_819, view_820);  view_819 = view_820 = None
        view_821 = torch.ops.aten.view.default(bmm_78, [8, 6, 197, 197]);  bmm_78 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(view_821, 1);  view_821 = None
        amax_default_8 = torch.ops.aten.amax.default(mul_tensor_16, [-1], True)
        sub_tensor_8 = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = amax_default_8 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        exp_39 = torch.ops.aten.exp.default(mul_tensor_17);  mul_tensor_17 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39 = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        expand_160 = torch.ops.aten.expand.default(div_39, [8, 6, 197, 197]);  div_39 = None
        view_822 = torch.ops.aten.view.default(expand_160, [48, 197, 197]);  expand_160 = None
        expand_161 = torch.ops.aten.expand.default(permute_383, [8, 6, 197, 64]);  permute_383 = None
        clone_305 = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_823 = torch.ops.aten.view.default(clone_305, [48, 197, 64]);  clone_305 = None
        bmm_79 = torch.ops.aten.bmm.default(view_822, view_823);  view_822 = view_823 = None
        view_824 = torch.ops.aten.view.default(bmm_79, [8, 6, 197, 64]);  bmm_79 = None
        permute_385 = torch.ops.aten.permute.default(view_824, [0, 2, 1, 3]);  view_824 = None
        clone_306 = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_825 = torch.ops.aten.view.default(clone_306, [8, 197, 384]);  clone_306 = None
        view_826 = torch.ops.aten.view.default(view_825, [1576, 384]);  view_825 = None
        permute_386 = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg229_1, view_826, permute_386);  arg229_1 = view_826 = permute_386 = None
        view_827 = torch.ops.aten.view.default(addmm_140, [8, 197, 384]);  addmm_140 = None
        add_353 = torch.ops.aten.add.Tensor(cat_21, view_827);  cat_21 = view_827 = None
        var_mean_104 = torch.ops.aten.var_mean.correction(add_353, [2], correction = 0, keepdim = True)
        getitem_288 = var_mean_104[0]
        getitem_289 = var_mean_104[1];  var_mean_104 = None
        add_354 = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_354);  add_354 = None
        sub_144 = torch.ops.aten.sub.Tensor(add_353, getitem_289);  getitem_289 = None
        mul_365 = torch.ops.aten.mul.Tensor(sub_144, rsqrt_104);  sub_144 = rsqrt_104 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_365, arg230_1);  mul_365 = arg230_1 = None
        add_355 = torch.ops.aten.add.Tensor(mul_366, arg231_1);  mul_366 = arg231_1 = None
        view_828 = torch.ops.aten.view.default(add_355, [1576, 384]);  add_355 = None
        permute_387 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg233_1, view_828, permute_387);  arg233_1 = view_828 = permute_387 = None
        view_829 = torch.ops.aten.view.default(addmm_141, [8, 197, 1536]);  addmm_141 = None
        mul_367 = torch.ops.aten.mul.Tensor(view_829, 0.5)
        mul_368 = torch.ops.aten.mul.Tensor(view_829, 0.7071067811865476);  view_829 = None
        erf_39 = torch.ops.aten.erf.default(mul_368);  mul_368 = None
        add_356 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_367, add_356);  mul_367 = add_356 = None
        view_830 = torch.ops.aten.view.default(mul_369, [1576, 1536]);  mul_369 = None
        permute_388 = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg235_1, view_830, permute_388);  arg235_1 = view_830 = permute_388 = None
        view_831 = torch.ops.aten.view.default(addmm_142, [8, 197, 384]);  addmm_142 = None
        add_357 = torch.ops.aten.add.Tensor(add_353, view_831);  add_353 = view_831 = None
        clone_309 = torch.ops.aten.clone.default(add_347, memory_format = torch.contiguous_format)
        var_mean_105 = torch.ops.aten.var_mean.correction(clone_309, [2], correction = 0, keepdim = True)
        getitem_290 = var_mean_105[0]
        getitem_291 = var_mean_105[1];  var_mean_105 = None
        add_358 = torch.ops.aten.add.Tensor(getitem_290, 1e-05);  getitem_290 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_358);  add_358 = None
        sub_145 = torch.ops.aten.sub.Tensor(clone_309, getitem_291);  clone_309 = getitem_291 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_145, rsqrt_105);  sub_145 = rsqrt_105 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, arg236_1);  mul_370 = arg236_1 = None
        add_359 = torch.ops.aten.add.Tensor(mul_371, arg237_1);  mul_371 = arg237_1 = None
        permute_389 = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        view_832 = torch.ops.aten.view.default(add_359, [25088, 24])
        mm_80 = torch.ops.aten.mm.default(view_832, permute_389);  view_832 = permute_389 = None
        view_833 = torch.ops.aten.view.default(mm_80, [1568, 16, 48]);  mm_80 = None
        view_834 = torch.ops.aten.view.default(view_833, [1568, 16, 2, 4, 6]);  view_833 = None
        permute_390 = torch.ops.aten.permute.default(view_834, [2, 0, 3, 1, 4]);  view_834 = None
        unbind_40 = torch.ops.aten.unbind.int(permute_390);  permute_390 = None
        getitem_292 = unbind_40[0]
        getitem_293 = unbind_40[1];  unbind_40 = None
        permute_391 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        view_835 = torch.ops.aten.view.default(add_359, [25088, 24]);  add_359 = None
        mm_81 = torch.ops.aten.mm.default(view_835, permute_391);  view_835 = permute_391 = None
        view_836 = torch.ops.aten.view.default(mm_81, [1568, 16, 24]);  mm_81 = None
        view_837 = torch.ops.aten.view.default(view_836, [1568, 16, 4, -1]);  view_836 = None
        permute_392 = torch.ops.aten.permute.default(view_837, [0, 2, 1, 3]);  view_837 = None
        permute_393 = torch.ops.aten.permute.default(getitem_293, [0, 1, 3, 2]);  getitem_293 = None
        expand_162 = torch.ops.aten.expand.default(getitem_292, [1568, 4, 16, 6]);  getitem_292 = None
        clone_310 = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_838 = torch.ops.aten.view.default(clone_310, [6272, 16, 6]);  clone_310 = None
        expand_163 = torch.ops.aten.expand.default(permute_393, [1568, 4, 6, 16]);  permute_393 = None
        clone_311 = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_839 = torch.ops.aten.view.default(clone_311, [6272, 6, 16]);  clone_311 = None
        bmm_80 = torch.ops.aten.bmm.default(view_838, view_839);  view_838 = view_839 = None
        view_840 = torch.ops.aten.view.default(bmm_80, [1568, 4, 16, 16]);  bmm_80 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(view_840, 1);  view_840 = None
        amax_default_7 = torch.ops.aten.amax.default(mul_tensor_14, [-1], True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, 0.408248290463863);  sub_tensor_7 = None
        exp_40 = torch.ops.aten.exp.default(mul_tensor_15);  mul_tensor_15 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40 = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        expand_164 = torch.ops.aten.expand.default(div_40, [1568, 4, 16, 16]);  div_40 = None
        view_841 = torch.ops.aten.view.default(expand_164, [6272, 16, 16]);  expand_164 = None
        expand_165 = torch.ops.aten.expand.default(permute_392, [1568, 4, 16, 6]);  permute_392 = None
        clone_312 = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_842 = torch.ops.aten.view.default(clone_312, [6272, 16, 6]);  clone_312 = None
        bmm_81 = torch.ops.aten.bmm.default(view_841, view_842);  view_841 = view_842 = None
        view_843 = torch.ops.aten.view.default(bmm_81, [1568, 4, 16, 6]);  bmm_81 = None
        permute_394 = torch.ops.aten.permute.default(view_843, [0, 2, 1, 3]);  view_843 = None
        clone_313 = torch.ops.aten.clone.default(permute_394, memory_format = torch.contiguous_format);  permute_394 = None
        view_844 = torch.ops.aten.view.default(clone_313, [1568, 16, 24]);  clone_313 = None
        view_845 = torch.ops.aten.view.default(view_844, [25088, 24]);  view_844 = None
        permute_395 = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg241_1, view_845, permute_395);  arg241_1 = view_845 = permute_395 = None
        view_846 = torch.ops.aten.view.default(addmm_143, [1568, 16, 24]);  addmm_143 = None
        add_360 = torch.ops.aten.add.Tensor(add_347, view_846);  add_347 = view_846 = None
        clone_314 = torch.ops.aten.clone.default(add_360, memory_format = torch.contiguous_format)
        var_mean_106 = torch.ops.aten.var_mean.correction(clone_314, [2], correction = 0, keepdim = True)
        getitem_294 = var_mean_106[0]
        getitem_295 = var_mean_106[1];  var_mean_106 = None
        add_361 = torch.ops.aten.add.Tensor(getitem_294, 1e-05);  getitem_294 = None
        rsqrt_106 = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
        sub_147 = torch.ops.aten.sub.Tensor(clone_314, getitem_295);  clone_314 = getitem_295 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_147, rsqrt_106);  sub_147 = rsqrt_106 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, arg242_1);  mul_373 = arg242_1 = None
        add_362 = torch.ops.aten.add.Tensor(mul_374, arg243_1);  mul_374 = arg243_1 = None
        view_847 = torch.ops.aten.view.default(add_362, [25088, 24]);  add_362 = None
        permute_396 = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg245_1, view_847, permute_396);  arg245_1 = view_847 = permute_396 = None
        view_848 = torch.ops.aten.view.default(addmm_144, [1568, 16, 96]);  addmm_144 = None
        mul_375 = torch.ops.aten.mul.Tensor(view_848, 0.5)
        mul_376 = torch.ops.aten.mul.Tensor(view_848, 0.7071067811865476);  view_848 = None
        erf_40 = torch.ops.aten.erf.default(mul_376);  mul_376 = None
        add_363 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_375, add_363);  mul_375 = add_363 = None
        view_849 = torch.ops.aten.view.default(mul_377, [25088, 96]);  mul_377 = None
        permute_397 = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg247_1, view_849, permute_397);  arg247_1 = view_849 = permute_397 = None
        view_850 = torch.ops.aten.view.default(addmm_145, [1568, 16, 24]);  addmm_145 = None
        add_364 = torch.ops.aten.add.Tensor(add_360, view_850);  add_360 = view_850 = None
        slice_87 = torch.ops.aten.slice.Tensor(add_357, 1, 0, 1)
        slice_89 = torch.ops.aten.slice.Tensor(add_357, 1, 1, 9223372036854775807);  add_357 = None
        clone_317 = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
        var_mean_107 = torch.ops.aten.var_mean.correction(clone_317, [2], correction = 0, keepdim = True)
        getitem_296 = var_mean_107[0]
        getitem_297 = var_mean_107[1];  var_mean_107 = None
        add_365 = torch.ops.aten.add.Tensor(getitem_296, 1e-05);  getitem_296 = None
        rsqrt_107 = torch.ops.aten.rsqrt.default(add_365);  add_365 = None
        sub_148 = torch.ops.aten.sub.Tensor(clone_317, getitem_297);  clone_317 = getitem_297 = None
        mul_378 = torch.ops.aten.mul.Tensor(sub_148, rsqrt_107);  sub_148 = rsqrt_107 = None
        mul_379 = torch.ops.aten.mul.Tensor(mul_378, arg248_1);  mul_378 = arg248_1 = None
        add_366 = torch.ops.aten.add.Tensor(mul_379, arg249_1);  mul_379 = arg249_1 = None
        view_851 = torch.ops.aten.view.default(add_366, [8, 196, -1]);  add_366 = None
        view_852 = torch.ops.aten.view.default(view_851, [1568, 384]);  view_851 = None
        permute_398 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg251_1, view_852, permute_398);  arg251_1 = view_852 = permute_398 = None
        view_853 = torch.ops.aten.view.default(addmm_146, [8, 196, 384]);  addmm_146 = None
        add_367 = torch.ops.aten.add.Tensor(slice_89, view_853);  slice_89 = view_853 = None
        cat_22 = torch.ops.aten.cat.default([slice_87, add_367], 1);  slice_87 = add_367 = None
        var_mean_108 = torch.ops.aten.var_mean.correction(cat_22, [2], correction = 0, keepdim = True)
        getitem_298 = var_mean_108[0]
        getitem_299 = var_mean_108[1];  var_mean_108 = None
        add_368 = torch.ops.aten.add.Tensor(getitem_298, 1e-05);  getitem_298 = None
        rsqrt_108 = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
        sub_149 = torch.ops.aten.sub.Tensor(cat_22, getitem_299);  getitem_299 = None
        mul_380 = torch.ops.aten.mul.Tensor(sub_149, rsqrt_108);  sub_149 = rsqrt_108 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_380, arg252_1);  mul_380 = arg252_1 = None
        add_369 = torch.ops.aten.add.Tensor(mul_381, arg253_1);  mul_381 = arg253_1 = None
        permute_399 = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        view_854 = torch.ops.aten.view.default(add_369, [1576, 384])
        mm_82 = torch.ops.aten.mm.default(view_854, permute_399);  view_854 = permute_399 = None
        view_855 = torch.ops.aten.view.default(mm_82, [8, 197, 768]);  mm_82 = None
        view_856 = torch.ops.aten.view.default(view_855, [8, 197, 2, 6, 64]);  view_855 = None
        permute_400 = torch.ops.aten.permute.default(view_856, [2, 0, 3, 1, 4]);  view_856 = None
        unbind_41 = torch.ops.aten.unbind.int(permute_400);  permute_400 = None
        getitem_300 = unbind_41[0]
        getitem_301 = unbind_41[1];  unbind_41 = None
        permute_401 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        view_857 = torch.ops.aten.view.default(add_369, [1576, 384]);  add_369 = None
        mm_83 = torch.ops.aten.mm.default(view_857, permute_401);  view_857 = permute_401 = None
        view_858 = torch.ops.aten.view.default(mm_83, [8, 197, 384]);  mm_83 = None
        view_859 = torch.ops.aten.view.default(view_858, [8, 197, 6, -1]);  view_858 = None
        permute_402 = torch.ops.aten.permute.default(view_859, [0, 2, 1, 3]);  view_859 = None
        permute_403 = torch.ops.aten.permute.default(getitem_301, [0, 1, 3, 2]);  getitem_301 = None
        expand_166 = torch.ops.aten.expand.default(getitem_300, [8, 6, 197, 64]);  getitem_300 = None
        clone_318 = torch.ops.aten.clone.default(expand_166, memory_format = torch.contiguous_format);  expand_166 = None
        view_860 = torch.ops.aten.view.default(clone_318, [48, 197, 64]);  clone_318 = None
        expand_167 = torch.ops.aten.expand.default(permute_403, [8, 6, 64, 197]);  permute_403 = None
        clone_319 = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_861 = torch.ops.aten.view.default(clone_319, [48, 64, 197]);  clone_319 = None
        bmm_82 = torch.ops.aten.bmm.default(view_860, view_861);  view_860 = view_861 = None
        view_862 = torch.ops.aten.view.default(bmm_82, [8, 6, 197, 197]);  bmm_82 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_862, 1);  view_862 = None
        amax_default_6 = torch.ops.aten.amax.default(mul_tensor_12, [-1], True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        exp_41 = torch.ops.aten.exp.default(mul_tensor_13);  mul_tensor_13 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41 = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        expand_168 = torch.ops.aten.expand.default(div_41, [8, 6, 197, 197]);  div_41 = None
        view_863 = torch.ops.aten.view.default(expand_168, [48, 197, 197]);  expand_168 = None
        expand_169 = torch.ops.aten.expand.default(permute_402, [8, 6, 197, 64]);  permute_402 = None
        clone_320 = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_864 = torch.ops.aten.view.default(clone_320, [48, 197, 64]);  clone_320 = None
        bmm_83 = torch.ops.aten.bmm.default(view_863, view_864);  view_863 = view_864 = None
        view_865 = torch.ops.aten.view.default(bmm_83, [8, 6, 197, 64]);  bmm_83 = None
        permute_404 = torch.ops.aten.permute.default(view_865, [0, 2, 1, 3]);  view_865 = None
        clone_321 = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
        view_866 = torch.ops.aten.view.default(clone_321, [8, 197, 384]);  clone_321 = None
        view_867 = torch.ops.aten.view.default(view_866, [1576, 384]);  view_866 = None
        permute_405 = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg257_1, view_867, permute_405);  arg257_1 = view_867 = permute_405 = None
        view_868 = torch.ops.aten.view.default(addmm_147, [8, 197, 384]);  addmm_147 = None
        add_370 = torch.ops.aten.add.Tensor(cat_22, view_868);  cat_22 = view_868 = None
        var_mean_109 = torch.ops.aten.var_mean.correction(add_370, [2], correction = 0, keepdim = True)
        getitem_302 = var_mean_109[0]
        getitem_303 = var_mean_109[1];  var_mean_109 = None
        add_371 = torch.ops.aten.add.Tensor(getitem_302, 1e-05);  getitem_302 = None
        rsqrt_109 = torch.ops.aten.rsqrt.default(add_371);  add_371 = None
        sub_151 = torch.ops.aten.sub.Tensor(add_370, getitem_303);  getitem_303 = None
        mul_383 = torch.ops.aten.mul.Tensor(sub_151, rsqrt_109);  sub_151 = rsqrt_109 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_383, arg258_1);  mul_383 = arg258_1 = None
        add_372 = torch.ops.aten.add.Tensor(mul_384, arg259_1);  mul_384 = arg259_1 = None
        view_869 = torch.ops.aten.view.default(add_372, [1576, 384]);  add_372 = None
        permute_406 = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg261_1, view_869, permute_406);  arg261_1 = view_869 = permute_406 = None
        view_870 = torch.ops.aten.view.default(addmm_148, [8, 197, 1536]);  addmm_148 = None
        mul_385 = torch.ops.aten.mul.Tensor(view_870, 0.5)
        mul_386 = torch.ops.aten.mul.Tensor(view_870, 0.7071067811865476);  view_870 = None
        erf_41 = torch.ops.aten.erf.default(mul_386);  mul_386 = None
        add_373 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_385, add_373);  mul_385 = add_373 = None
        view_871 = torch.ops.aten.view.default(mul_387, [1576, 1536]);  mul_387 = None
        permute_407 = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg263_1, view_871, permute_407);  arg263_1 = view_871 = permute_407 = None
        view_872 = torch.ops.aten.view.default(addmm_149, [8, 197, 384]);  addmm_149 = None
        add_374 = torch.ops.aten.add.Tensor(add_370, view_872);  add_370 = view_872 = None
        clone_324 = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
        var_mean_110 = torch.ops.aten.var_mean.correction(clone_324, [2], correction = 0, keepdim = True)
        getitem_304 = var_mean_110[0]
        getitem_305 = var_mean_110[1];  var_mean_110 = None
        add_375 = torch.ops.aten.add.Tensor(getitem_304, 1e-05);  getitem_304 = None
        rsqrt_110 = torch.ops.aten.rsqrt.default(add_375);  add_375 = None
        sub_152 = torch.ops.aten.sub.Tensor(clone_324, getitem_305);  clone_324 = getitem_305 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_152, rsqrt_110);  sub_152 = rsqrt_110 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, arg264_1);  mul_388 = arg264_1 = None
        add_376 = torch.ops.aten.add.Tensor(mul_389, arg265_1);  mul_389 = arg265_1 = None
        permute_408 = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        view_873 = torch.ops.aten.view.default(add_376, [25088, 24])
        mm_84 = torch.ops.aten.mm.default(view_873, permute_408);  view_873 = permute_408 = None
        view_874 = torch.ops.aten.view.default(mm_84, [1568, 16, 48]);  mm_84 = None
        view_875 = torch.ops.aten.view.default(view_874, [1568, 16, 2, 4, 6]);  view_874 = None
        permute_409 = torch.ops.aten.permute.default(view_875, [2, 0, 3, 1, 4]);  view_875 = None
        unbind_42 = torch.ops.aten.unbind.int(permute_409);  permute_409 = None
        getitem_306 = unbind_42[0]
        getitem_307 = unbind_42[1];  unbind_42 = None
        permute_410 = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        view_876 = torch.ops.aten.view.default(add_376, [25088, 24]);  add_376 = None
        mm_85 = torch.ops.aten.mm.default(view_876, permute_410);  view_876 = permute_410 = None
        view_877 = torch.ops.aten.view.default(mm_85, [1568, 16, 24]);  mm_85 = None
        view_878 = torch.ops.aten.view.default(view_877, [1568, 16, 4, -1]);  view_877 = None
        permute_411 = torch.ops.aten.permute.default(view_878, [0, 2, 1, 3]);  view_878 = None
        permute_412 = torch.ops.aten.permute.default(getitem_307, [0, 1, 3, 2]);  getitem_307 = None
        expand_170 = torch.ops.aten.expand.default(getitem_306, [1568, 4, 16, 6]);  getitem_306 = None
        clone_325 = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_879 = torch.ops.aten.view.default(clone_325, [6272, 16, 6]);  clone_325 = None
        expand_171 = torch.ops.aten.expand.default(permute_412, [1568, 4, 6, 16]);  permute_412 = None
        clone_326 = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_880 = torch.ops.aten.view.default(clone_326, [6272, 6, 16]);  clone_326 = None
        bmm_84 = torch.ops.aten.bmm.default(view_879, view_880);  view_879 = view_880 = None
        view_881 = torch.ops.aten.view.default(bmm_84, [1568, 4, 16, 16]);  bmm_84 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_881, 1);  view_881 = None
        amax_default_5 = torch.ops.aten.amax.default(mul_tensor_10, [-1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, 0.408248290463863);  sub_tensor_5 = None
        exp_42 = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42 = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        expand_172 = torch.ops.aten.expand.default(div_42, [1568, 4, 16, 16]);  div_42 = None
        view_882 = torch.ops.aten.view.default(expand_172, [6272, 16, 16]);  expand_172 = None
        expand_173 = torch.ops.aten.expand.default(permute_411, [1568, 4, 16, 6]);  permute_411 = None
        clone_327 = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_883 = torch.ops.aten.view.default(clone_327, [6272, 16, 6]);  clone_327 = None
        bmm_85 = torch.ops.aten.bmm.default(view_882, view_883);  view_882 = view_883 = None
        view_884 = torch.ops.aten.view.default(bmm_85, [1568, 4, 16, 6]);  bmm_85 = None
        permute_413 = torch.ops.aten.permute.default(view_884, [0, 2, 1, 3]);  view_884 = None
        clone_328 = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_885 = torch.ops.aten.view.default(clone_328, [1568, 16, 24]);  clone_328 = None
        view_886 = torch.ops.aten.view.default(view_885, [25088, 24]);  view_885 = None
        permute_414 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg269_1, view_886, permute_414);  arg269_1 = view_886 = permute_414 = None
        view_887 = torch.ops.aten.view.default(addmm_150, [1568, 16, 24]);  addmm_150 = None
        add_377 = torch.ops.aten.add.Tensor(add_364, view_887);  add_364 = view_887 = None
        clone_329 = torch.ops.aten.clone.default(add_377, memory_format = torch.contiguous_format)
        var_mean_111 = torch.ops.aten.var_mean.correction(clone_329, [2], correction = 0, keepdim = True)
        getitem_308 = var_mean_111[0]
        getitem_309 = var_mean_111[1];  var_mean_111 = None
        add_378 = torch.ops.aten.add.Tensor(getitem_308, 1e-05);  getitem_308 = None
        rsqrt_111 = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
        sub_154 = torch.ops.aten.sub.Tensor(clone_329, getitem_309);  clone_329 = getitem_309 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_154, rsqrt_111);  sub_154 = rsqrt_111 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, arg270_1);  mul_391 = arg270_1 = None
        add_379 = torch.ops.aten.add.Tensor(mul_392, arg271_1);  mul_392 = arg271_1 = None
        view_888 = torch.ops.aten.view.default(add_379, [25088, 24]);  add_379 = None
        permute_415 = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg273_1, view_888, permute_415);  arg273_1 = view_888 = permute_415 = None
        view_889 = torch.ops.aten.view.default(addmm_151, [1568, 16, 96]);  addmm_151 = None
        mul_393 = torch.ops.aten.mul.Tensor(view_889, 0.5)
        mul_394 = torch.ops.aten.mul.Tensor(view_889, 0.7071067811865476);  view_889 = None
        erf_42 = torch.ops.aten.erf.default(mul_394);  mul_394 = None
        add_380 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_393, add_380);  mul_393 = add_380 = None
        view_890 = torch.ops.aten.view.default(mul_395, [25088, 96]);  mul_395 = None
        permute_416 = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg275_1, view_890, permute_416);  arg275_1 = view_890 = permute_416 = None
        view_891 = torch.ops.aten.view.default(addmm_152, [1568, 16, 24]);  addmm_152 = None
        add_381 = torch.ops.aten.add.Tensor(add_377, view_891);  add_377 = view_891 = None
        slice_91 = torch.ops.aten.slice.Tensor(add_374, 1, 0, 1)
        slice_93 = torch.ops.aten.slice.Tensor(add_374, 1, 1, 9223372036854775807);  add_374 = None
        clone_332 = torch.ops.aten.clone.default(add_381, memory_format = torch.contiguous_format)
        var_mean_112 = torch.ops.aten.var_mean.correction(clone_332, [2], correction = 0, keepdim = True)
        getitem_310 = var_mean_112[0]
        getitem_311 = var_mean_112[1];  var_mean_112 = None
        add_382 = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_112 = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
        sub_155 = torch.ops.aten.sub.Tensor(clone_332, getitem_311);  clone_332 = getitem_311 = None
        mul_396 = torch.ops.aten.mul.Tensor(sub_155, rsqrt_112);  sub_155 = rsqrt_112 = None
        mul_397 = torch.ops.aten.mul.Tensor(mul_396, arg276_1);  mul_396 = arg276_1 = None
        add_383 = torch.ops.aten.add.Tensor(mul_397, arg277_1);  mul_397 = arg277_1 = None
        view_892 = torch.ops.aten.view.default(add_383, [8, 196, -1]);  add_383 = None
        view_893 = torch.ops.aten.view.default(view_892, [1568, 384]);  view_892 = None
        permute_417 = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg279_1, view_893, permute_417);  arg279_1 = view_893 = permute_417 = None
        view_894 = torch.ops.aten.view.default(addmm_153, [8, 196, 384]);  addmm_153 = None
        add_384 = torch.ops.aten.add.Tensor(slice_93, view_894);  slice_93 = view_894 = None
        cat_23 = torch.ops.aten.cat.default([slice_91, add_384], 1);  slice_91 = add_384 = None
        var_mean_113 = torch.ops.aten.var_mean.correction(cat_23, [2], correction = 0, keepdim = True)
        getitem_312 = var_mean_113[0]
        getitem_313 = var_mean_113[1];  var_mean_113 = None
        add_385 = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_113 = torch.ops.aten.rsqrt.default(add_385);  add_385 = None
        sub_156 = torch.ops.aten.sub.Tensor(cat_23, getitem_313);  getitem_313 = None
        mul_398 = torch.ops.aten.mul.Tensor(sub_156, rsqrt_113);  sub_156 = rsqrt_113 = None
        mul_399 = torch.ops.aten.mul.Tensor(mul_398, arg280_1);  mul_398 = arg280_1 = None
        add_386 = torch.ops.aten.add.Tensor(mul_399, arg281_1);  mul_399 = arg281_1 = None
        permute_418 = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        view_895 = torch.ops.aten.view.default(add_386, [1576, 384])
        mm_86 = torch.ops.aten.mm.default(view_895, permute_418);  view_895 = permute_418 = None
        view_896 = torch.ops.aten.view.default(mm_86, [8, 197, 768]);  mm_86 = None
        view_897 = torch.ops.aten.view.default(view_896, [8, 197, 2, 6, 64]);  view_896 = None
        permute_419 = torch.ops.aten.permute.default(view_897, [2, 0, 3, 1, 4]);  view_897 = None
        unbind_43 = torch.ops.aten.unbind.int(permute_419);  permute_419 = None
        getitem_314 = unbind_43[0]
        getitem_315 = unbind_43[1];  unbind_43 = None
        permute_420 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        view_898 = torch.ops.aten.view.default(add_386, [1576, 384]);  add_386 = None
        mm_87 = torch.ops.aten.mm.default(view_898, permute_420);  view_898 = permute_420 = None
        view_899 = torch.ops.aten.view.default(mm_87, [8, 197, 384]);  mm_87 = None
        view_900 = torch.ops.aten.view.default(view_899, [8, 197, 6, -1]);  view_899 = None
        permute_421 = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None
        permute_422 = torch.ops.aten.permute.default(getitem_315, [0, 1, 3, 2]);  getitem_315 = None
        expand_174 = torch.ops.aten.expand.default(getitem_314, [8, 6, 197, 64]);  getitem_314 = None
        clone_333 = torch.ops.aten.clone.default(expand_174, memory_format = torch.contiguous_format);  expand_174 = None
        view_901 = torch.ops.aten.view.default(clone_333, [48, 197, 64]);  clone_333 = None
        expand_175 = torch.ops.aten.expand.default(permute_422, [8, 6, 64, 197]);  permute_422 = None
        clone_334 = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_902 = torch.ops.aten.view.default(clone_334, [48, 64, 197]);  clone_334 = None
        bmm_86 = torch.ops.aten.bmm.default(view_901, view_902);  view_901 = view_902 = None
        view_903 = torch.ops.aten.view.default(bmm_86, [8, 6, 197, 197]);  bmm_86 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_903, 1);  view_903 = None
        amax_default_4 = torch.ops.aten.amax.default(mul_tensor_8, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        exp_43 = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43 = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        expand_176 = torch.ops.aten.expand.default(div_43, [8, 6, 197, 197]);  div_43 = None
        view_904 = torch.ops.aten.view.default(expand_176, [48, 197, 197]);  expand_176 = None
        expand_177 = torch.ops.aten.expand.default(permute_421, [8, 6, 197, 64]);  permute_421 = None
        clone_335 = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_905 = torch.ops.aten.view.default(clone_335, [48, 197, 64]);  clone_335 = None
        bmm_87 = torch.ops.aten.bmm.default(view_904, view_905);  view_904 = view_905 = None
        view_906 = torch.ops.aten.view.default(bmm_87, [8, 6, 197, 64]);  bmm_87 = None
        permute_423 = torch.ops.aten.permute.default(view_906, [0, 2, 1, 3]);  view_906 = None
        clone_336 = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
        view_907 = torch.ops.aten.view.default(clone_336, [8, 197, 384]);  clone_336 = None
        view_908 = torch.ops.aten.view.default(view_907, [1576, 384]);  view_907 = None
        permute_424 = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg285_1, view_908, permute_424);  arg285_1 = view_908 = permute_424 = None
        view_909 = torch.ops.aten.view.default(addmm_154, [8, 197, 384]);  addmm_154 = None
        add_387 = torch.ops.aten.add.Tensor(cat_23, view_909);  cat_23 = view_909 = None
        var_mean_114 = torch.ops.aten.var_mean.correction(add_387, [2], correction = 0, keepdim = True)
        getitem_316 = var_mean_114[0]
        getitem_317 = var_mean_114[1];  var_mean_114 = None
        add_388 = torch.ops.aten.add.Tensor(getitem_316, 1e-05);  getitem_316 = None
        rsqrt_114 = torch.ops.aten.rsqrt.default(add_388);  add_388 = None
        sub_158 = torch.ops.aten.sub.Tensor(add_387, getitem_317);  getitem_317 = None
        mul_401 = torch.ops.aten.mul.Tensor(sub_158, rsqrt_114);  sub_158 = rsqrt_114 = None
        mul_402 = torch.ops.aten.mul.Tensor(mul_401, arg286_1);  mul_401 = arg286_1 = None
        add_389 = torch.ops.aten.add.Tensor(mul_402, arg287_1);  mul_402 = arg287_1 = None
        view_910 = torch.ops.aten.view.default(add_389, [1576, 384]);  add_389 = None
        permute_425 = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg289_1, view_910, permute_425);  arg289_1 = view_910 = permute_425 = None
        view_911 = torch.ops.aten.view.default(addmm_155, [8, 197, 1536]);  addmm_155 = None
        mul_403 = torch.ops.aten.mul.Tensor(view_911, 0.5)
        mul_404 = torch.ops.aten.mul.Tensor(view_911, 0.7071067811865476);  view_911 = None
        erf_43 = torch.ops.aten.erf.default(mul_404);  mul_404 = None
        add_390 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_403, add_390);  mul_403 = add_390 = None
        view_912 = torch.ops.aten.view.default(mul_405, [1576, 1536]);  mul_405 = None
        permute_426 = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg291_1, view_912, permute_426);  arg291_1 = view_912 = permute_426 = None
        view_913 = torch.ops.aten.view.default(addmm_156, [8, 197, 384]);  addmm_156 = None
        add_391 = torch.ops.aten.add.Tensor(add_387, view_913);  add_387 = view_913 = None
        clone_339 = torch.ops.aten.clone.default(add_381, memory_format = torch.contiguous_format)
        var_mean_115 = torch.ops.aten.var_mean.correction(clone_339, [2], correction = 0, keepdim = True)
        getitem_318 = var_mean_115[0]
        getitem_319 = var_mean_115[1];  var_mean_115 = None
        add_392 = torch.ops.aten.add.Tensor(getitem_318, 1e-05);  getitem_318 = None
        rsqrt_115 = torch.ops.aten.rsqrt.default(add_392);  add_392 = None
        sub_159 = torch.ops.aten.sub.Tensor(clone_339, getitem_319);  clone_339 = getitem_319 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_159, rsqrt_115);  sub_159 = rsqrt_115 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, arg292_1);  mul_406 = arg292_1 = None
        add_393 = torch.ops.aten.add.Tensor(mul_407, arg293_1);  mul_407 = arg293_1 = None
        permute_427 = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        view_914 = torch.ops.aten.view.default(add_393, [25088, 24])
        mm_88 = torch.ops.aten.mm.default(view_914, permute_427);  view_914 = permute_427 = None
        view_915 = torch.ops.aten.view.default(mm_88, [1568, 16, 48]);  mm_88 = None
        view_916 = torch.ops.aten.view.default(view_915, [1568, 16, 2, 4, 6]);  view_915 = None
        permute_428 = torch.ops.aten.permute.default(view_916, [2, 0, 3, 1, 4]);  view_916 = None
        unbind_44 = torch.ops.aten.unbind.int(permute_428);  permute_428 = None
        getitem_320 = unbind_44[0]
        getitem_321 = unbind_44[1];  unbind_44 = None
        permute_429 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        view_917 = torch.ops.aten.view.default(add_393, [25088, 24]);  add_393 = None
        mm_89 = torch.ops.aten.mm.default(view_917, permute_429);  view_917 = permute_429 = None
        view_918 = torch.ops.aten.view.default(mm_89, [1568, 16, 24]);  mm_89 = None
        view_919 = torch.ops.aten.view.default(view_918, [1568, 16, 4, -1]);  view_918 = None
        permute_430 = torch.ops.aten.permute.default(view_919, [0, 2, 1, 3]);  view_919 = None
        permute_431 = torch.ops.aten.permute.default(getitem_321, [0, 1, 3, 2]);  getitem_321 = None
        expand_178 = torch.ops.aten.expand.default(getitem_320, [1568, 4, 16, 6]);  getitem_320 = None
        clone_340 = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_920 = torch.ops.aten.view.default(clone_340, [6272, 16, 6]);  clone_340 = None
        expand_179 = torch.ops.aten.expand.default(permute_431, [1568, 4, 6, 16]);  permute_431 = None
        clone_341 = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_921 = torch.ops.aten.view.default(clone_341, [6272, 6, 16]);  clone_341 = None
        bmm_88 = torch.ops.aten.bmm.default(view_920, view_921);  view_920 = view_921 = None
        view_922 = torch.ops.aten.view.default(bmm_88, [1568, 4, 16, 16]);  bmm_88 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_922, 1);  view_922 = None
        amax_default_3 = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, 0.408248290463863);  sub_tensor_3 = None
        exp_44 = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44 = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        expand_180 = torch.ops.aten.expand.default(div_44, [1568, 4, 16, 16]);  div_44 = None
        view_923 = torch.ops.aten.view.default(expand_180, [6272, 16, 16]);  expand_180 = None
        expand_181 = torch.ops.aten.expand.default(permute_430, [1568, 4, 16, 6]);  permute_430 = None
        clone_342 = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_924 = torch.ops.aten.view.default(clone_342, [6272, 16, 6]);  clone_342 = None
        bmm_89 = torch.ops.aten.bmm.default(view_923, view_924);  view_923 = view_924 = None
        view_925 = torch.ops.aten.view.default(bmm_89, [1568, 4, 16, 6]);  bmm_89 = None
        permute_432 = torch.ops.aten.permute.default(view_925, [0, 2, 1, 3]);  view_925 = None
        clone_343 = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
        view_926 = torch.ops.aten.view.default(clone_343, [1568, 16, 24]);  clone_343 = None
        view_927 = torch.ops.aten.view.default(view_926, [25088, 24]);  view_926 = None
        permute_433 = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg297_1, view_927, permute_433);  arg297_1 = view_927 = permute_433 = None
        view_928 = torch.ops.aten.view.default(addmm_157, [1568, 16, 24]);  addmm_157 = None
        add_394 = torch.ops.aten.add.Tensor(add_381, view_928);  add_381 = view_928 = None
        clone_344 = torch.ops.aten.clone.default(add_394, memory_format = torch.contiguous_format)
        var_mean_116 = torch.ops.aten.var_mean.correction(clone_344, [2], correction = 0, keepdim = True)
        getitem_322 = var_mean_116[0]
        getitem_323 = var_mean_116[1];  var_mean_116 = None
        add_395 = torch.ops.aten.add.Tensor(getitem_322, 1e-05);  getitem_322 = None
        rsqrt_116 = torch.ops.aten.rsqrt.default(add_395);  add_395 = None
        sub_161 = torch.ops.aten.sub.Tensor(clone_344, getitem_323);  clone_344 = getitem_323 = None
        mul_409 = torch.ops.aten.mul.Tensor(sub_161, rsqrt_116);  sub_161 = rsqrt_116 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, arg298_1);  mul_409 = arg298_1 = None
        add_396 = torch.ops.aten.add.Tensor(mul_410, arg299_1);  mul_410 = arg299_1 = None
        view_929 = torch.ops.aten.view.default(add_396, [25088, 24]);  add_396 = None
        permute_434 = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg301_1, view_929, permute_434);  arg301_1 = view_929 = permute_434 = None
        view_930 = torch.ops.aten.view.default(addmm_158, [1568, 16, 96]);  addmm_158 = None
        mul_411 = torch.ops.aten.mul.Tensor(view_930, 0.5)
        mul_412 = torch.ops.aten.mul.Tensor(view_930, 0.7071067811865476);  view_930 = None
        erf_44 = torch.ops.aten.erf.default(mul_412);  mul_412 = None
        add_397 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_411, add_397);  mul_411 = add_397 = None
        view_931 = torch.ops.aten.view.default(mul_413, [25088, 96]);  mul_413 = None
        permute_435 = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg303_1, view_931, permute_435);  arg303_1 = view_931 = permute_435 = None
        view_932 = torch.ops.aten.view.default(addmm_159, [1568, 16, 24]);  addmm_159 = None
        add_398 = torch.ops.aten.add.Tensor(add_394, view_932);  add_394 = view_932 = None
        slice_95 = torch.ops.aten.slice.Tensor(add_391, 1, 0, 1)
        slice_97 = torch.ops.aten.slice.Tensor(add_391, 1, 1, 9223372036854775807);  add_391 = None
        clone_347 = torch.ops.aten.clone.default(add_398, memory_format = torch.contiguous_format)
        var_mean_117 = torch.ops.aten.var_mean.correction(clone_347, [2], correction = 0, keepdim = True)
        getitem_324 = var_mean_117[0]
        getitem_325 = var_mean_117[1];  var_mean_117 = None
        add_399 = torch.ops.aten.add.Tensor(getitem_324, 1e-05);  getitem_324 = None
        rsqrt_117 = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
        sub_162 = torch.ops.aten.sub.Tensor(clone_347, getitem_325);  clone_347 = getitem_325 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_162, rsqrt_117);  sub_162 = rsqrt_117 = None
        mul_415 = torch.ops.aten.mul.Tensor(mul_414, arg304_1);  mul_414 = arg304_1 = None
        add_400 = torch.ops.aten.add.Tensor(mul_415, arg305_1);  mul_415 = arg305_1 = None
        view_933 = torch.ops.aten.view.default(add_400, [8, 196, -1]);  add_400 = None
        view_934 = torch.ops.aten.view.default(view_933, [1568, 384]);  view_933 = None
        permute_436 = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg307_1, view_934, permute_436);  arg307_1 = view_934 = permute_436 = None
        view_935 = torch.ops.aten.view.default(addmm_160, [8, 196, 384]);  addmm_160 = None
        add_401 = torch.ops.aten.add.Tensor(slice_97, view_935);  slice_97 = view_935 = None
        cat_24 = torch.ops.aten.cat.default([slice_95, add_401], 1);  slice_95 = add_401 = None
        var_mean_118 = torch.ops.aten.var_mean.correction(cat_24, [2], correction = 0, keepdim = True)
        getitem_326 = var_mean_118[0]
        getitem_327 = var_mean_118[1];  var_mean_118 = None
        add_402 = torch.ops.aten.add.Tensor(getitem_326, 1e-05);  getitem_326 = None
        rsqrt_118 = torch.ops.aten.rsqrt.default(add_402);  add_402 = None
        sub_163 = torch.ops.aten.sub.Tensor(cat_24, getitem_327);  getitem_327 = None
        mul_416 = torch.ops.aten.mul.Tensor(sub_163, rsqrt_118);  sub_163 = rsqrt_118 = None
        mul_417 = torch.ops.aten.mul.Tensor(mul_416, arg308_1);  mul_416 = arg308_1 = None
        add_403 = torch.ops.aten.add.Tensor(mul_417, arg309_1);  mul_417 = arg309_1 = None
        permute_437 = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        view_936 = torch.ops.aten.view.default(add_403, [1576, 384])
        mm_90 = torch.ops.aten.mm.default(view_936, permute_437);  view_936 = permute_437 = None
        view_937 = torch.ops.aten.view.default(mm_90, [8, 197, 768]);  mm_90 = None
        view_938 = torch.ops.aten.view.default(view_937, [8, 197, 2, 6, 64]);  view_937 = None
        permute_438 = torch.ops.aten.permute.default(view_938, [2, 0, 3, 1, 4]);  view_938 = None
        unbind_45 = torch.ops.aten.unbind.int(permute_438);  permute_438 = None
        getitem_328 = unbind_45[0]
        getitem_329 = unbind_45[1];  unbind_45 = None
        permute_439 = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        view_939 = torch.ops.aten.view.default(add_403, [1576, 384]);  add_403 = None
        mm_91 = torch.ops.aten.mm.default(view_939, permute_439);  view_939 = permute_439 = None
        view_940 = torch.ops.aten.view.default(mm_91, [8, 197, 384]);  mm_91 = None
        view_941 = torch.ops.aten.view.default(view_940, [8, 197, 6, -1]);  view_940 = None
        permute_440 = torch.ops.aten.permute.default(view_941, [0, 2, 1, 3]);  view_941 = None
        permute_441 = torch.ops.aten.permute.default(getitem_329, [0, 1, 3, 2]);  getitem_329 = None
        expand_182 = torch.ops.aten.expand.default(getitem_328, [8, 6, 197, 64]);  getitem_328 = None
        clone_348 = torch.ops.aten.clone.default(expand_182, memory_format = torch.contiguous_format);  expand_182 = None
        view_942 = torch.ops.aten.view.default(clone_348, [48, 197, 64]);  clone_348 = None
        expand_183 = torch.ops.aten.expand.default(permute_441, [8, 6, 64, 197]);  permute_441 = None
        clone_349 = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_943 = torch.ops.aten.view.default(clone_349, [48, 64, 197]);  clone_349 = None
        bmm_90 = torch.ops.aten.bmm.default(view_942, view_943);  view_942 = view_943 = None
        view_944 = torch.ops.aten.view.default(bmm_90, [8, 6, 197, 197]);  bmm_90 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_944, 1);  view_944 = None
        amax_default_2 = torch.ops.aten.amax.default(mul_tensor_4, [-1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.125);  sub_tensor_2 = None
        exp_45 = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45 = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        expand_184 = torch.ops.aten.expand.default(div_45, [8, 6, 197, 197]);  div_45 = None
        view_945 = torch.ops.aten.view.default(expand_184, [48, 197, 197]);  expand_184 = None
        expand_185 = torch.ops.aten.expand.default(permute_440, [8, 6, 197, 64]);  permute_440 = None
        clone_350 = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_946 = torch.ops.aten.view.default(clone_350, [48, 197, 64]);  clone_350 = None
        bmm_91 = torch.ops.aten.bmm.default(view_945, view_946);  view_945 = view_946 = None
        view_947 = torch.ops.aten.view.default(bmm_91, [8, 6, 197, 64]);  bmm_91 = None
        permute_442 = torch.ops.aten.permute.default(view_947, [0, 2, 1, 3]);  view_947 = None
        clone_351 = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
        view_948 = torch.ops.aten.view.default(clone_351, [8, 197, 384]);  clone_351 = None
        view_949 = torch.ops.aten.view.default(view_948, [1576, 384]);  view_948 = None
        permute_443 = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg313_1, view_949, permute_443);  arg313_1 = view_949 = permute_443 = None
        view_950 = torch.ops.aten.view.default(addmm_161, [8, 197, 384]);  addmm_161 = None
        add_404 = torch.ops.aten.add.Tensor(cat_24, view_950);  cat_24 = view_950 = None
        var_mean_119 = torch.ops.aten.var_mean.correction(add_404, [2], correction = 0, keepdim = True)
        getitem_330 = var_mean_119[0]
        getitem_331 = var_mean_119[1];  var_mean_119 = None
        add_405 = torch.ops.aten.add.Tensor(getitem_330, 1e-05);  getitem_330 = None
        rsqrt_119 = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
        sub_165 = torch.ops.aten.sub.Tensor(add_404, getitem_331);  getitem_331 = None
        mul_419 = torch.ops.aten.mul.Tensor(sub_165, rsqrt_119);  sub_165 = rsqrt_119 = None
        mul_420 = torch.ops.aten.mul.Tensor(mul_419, arg314_1);  mul_419 = arg314_1 = None
        add_406 = torch.ops.aten.add.Tensor(mul_420, arg315_1);  mul_420 = arg315_1 = None
        view_951 = torch.ops.aten.view.default(add_406, [1576, 384]);  add_406 = None
        permute_444 = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg317_1, view_951, permute_444);  arg317_1 = view_951 = permute_444 = None
        view_952 = torch.ops.aten.view.default(addmm_162, [8, 197, 1536]);  addmm_162 = None
        mul_421 = torch.ops.aten.mul.Tensor(view_952, 0.5)
        mul_422 = torch.ops.aten.mul.Tensor(view_952, 0.7071067811865476);  view_952 = None
        erf_45 = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_407 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_421, add_407);  mul_421 = add_407 = None
        view_953 = torch.ops.aten.view.default(mul_423, [1576, 1536]);  mul_423 = None
        permute_445 = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg319_1, view_953, permute_445);  arg319_1 = view_953 = permute_445 = None
        view_954 = torch.ops.aten.view.default(addmm_163, [8, 197, 384]);  addmm_163 = None
        add_408 = torch.ops.aten.add.Tensor(add_404, view_954);  add_404 = view_954 = None
        clone_354 = torch.ops.aten.clone.default(add_398, memory_format = torch.contiguous_format)
        var_mean_120 = torch.ops.aten.var_mean.correction(clone_354, [2], correction = 0, keepdim = True)
        getitem_332 = var_mean_120[0]
        getitem_333 = var_mean_120[1];  var_mean_120 = None
        add_409 = torch.ops.aten.add.Tensor(getitem_332, 1e-05);  getitem_332 = None
        rsqrt_120 = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
        sub_166 = torch.ops.aten.sub.Tensor(clone_354, getitem_333);  clone_354 = getitem_333 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_166, rsqrt_120);  sub_166 = rsqrt_120 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, arg320_1);  mul_424 = arg320_1 = None
        add_410 = torch.ops.aten.add.Tensor(mul_425, arg321_1);  mul_425 = arg321_1 = None
        permute_446 = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        view_955 = torch.ops.aten.view.default(add_410, [25088, 24])
        mm_92 = torch.ops.aten.mm.default(view_955, permute_446);  view_955 = permute_446 = None
        view_956 = torch.ops.aten.view.default(mm_92, [1568, 16, 48]);  mm_92 = None
        view_957 = torch.ops.aten.view.default(view_956, [1568, 16, 2, 4, 6]);  view_956 = None
        permute_447 = torch.ops.aten.permute.default(view_957, [2, 0, 3, 1, 4]);  view_957 = None
        unbind_46 = torch.ops.aten.unbind.int(permute_447);  permute_447 = None
        getitem_334 = unbind_46[0]
        getitem_335 = unbind_46[1];  unbind_46 = None
        permute_448 = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        view_958 = torch.ops.aten.view.default(add_410, [25088, 24]);  add_410 = None
        mm_93 = torch.ops.aten.mm.default(view_958, permute_448);  view_958 = permute_448 = None
        view_959 = torch.ops.aten.view.default(mm_93, [1568, 16, 24]);  mm_93 = None
        view_960 = torch.ops.aten.view.default(view_959, [1568, 16, 4, -1]);  view_959 = None
        permute_449 = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None
        permute_450 = torch.ops.aten.permute.default(getitem_335, [0, 1, 3, 2]);  getitem_335 = None
        expand_186 = torch.ops.aten.expand.default(getitem_334, [1568, 4, 16, 6]);  getitem_334 = None
        clone_355 = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_961 = torch.ops.aten.view.default(clone_355, [6272, 16, 6]);  clone_355 = None
        expand_187 = torch.ops.aten.expand.default(permute_450, [1568, 4, 6, 16]);  permute_450 = None
        clone_356 = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_962 = torch.ops.aten.view.default(clone_356, [6272, 6, 16]);  clone_356 = None
        bmm_92 = torch.ops.aten.bmm.default(view_961, view_962);  view_961 = view_962 = None
        view_963 = torch.ops.aten.view.default(bmm_92, [1568, 4, 16, 16]);  bmm_92 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_963, 1);  view_963 = None
        amax_default_1 = torch.ops.aten.amax.default(mul_tensor_2, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, 0.408248290463863);  sub_tensor_1 = None
        exp_46 = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46 = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        expand_188 = torch.ops.aten.expand.default(div_46, [1568, 4, 16, 16]);  div_46 = None
        view_964 = torch.ops.aten.view.default(expand_188, [6272, 16, 16]);  expand_188 = None
        expand_189 = torch.ops.aten.expand.default(permute_449, [1568, 4, 16, 6]);  permute_449 = None
        clone_357 = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_965 = torch.ops.aten.view.default(clone_357, [6272, 16, 6]);  clone_357 = None
        bmm_93 = torch.ops.aten.bmm.default(view_964, view_965);  view_964 = view_965 = None
        view_966 = torch.ops.aten.view.default(bmm_93, [1568, 4, 16, 6]);  bmm_93 = None
        permute_451 = torch.ops.aten.permute.default(view_966, [0, 2, 1, 3]);  view_966 = None
        clone_358 = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_967 = torch.ops.aten.view.default(clone_358, [1568, 16, 24]);  clone_358 = None
        view_968 = torch.ops.aten.view.default(view_967, [25088, 24]);  view_967 = None
        permute_452 = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg325_1, view_968, permute_452);  arg325_1 = view_968 = permute_452 = None
        view_969 = torch.ops.aten.view.default(addmm_164, [1568, 16, 24]);  addmm_164 = None
        add_411 = torch.ops.aten.add.Tensor(add_398, view_969);  add_398 = view_969 = None
        clone_359 = torch.ops.aten.clone.default(add_411, memory_format = torch.contiguous_format)
        var_mean_121 = torch.ops.aten.var_mean.correction(clone_359, [2], correction = 0, keepdim = True)
        getitem_336 = var_mean_121[0]
        getitem_337 = var_mean_121[1];  var_mean_121 = None
        add_412 = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_121 = torch.ops.aten.rsqrt.default(add_412);  add_412 = None
        sub_168 = torch.ops.aten.sub.Tensor(clone_359, getitem_337);  clone_359 = getitem_337 = None
        mul_427 = torch.ops.aten.mul.Tensor(sub_168, rsqrt_121);  sub_168 = rsqrt_121 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_427, arg326_1);  mul_427 = arg326_1 = None
        add_413 = torch.ops.aten.add.Tensor(mul_428, arg327_1);  mul_428 = arg327_1 = None
        view_970 = torch.ops.aten.view.default(add_413, [25088, 24]);  add_413 = None
        permute_453 = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg329_1, view_970, permute_453);  arg329_1 = view_970 = permute_453 = None
        view_971 = torch.ops.aten.view.default(addmm_165, [1568, 16, 96]);  addmm_165 = None
        mul_429 = torch.ops.aten.mul.Tensor(view_971, 0.5)
        mul_430 = torch.ops.aten.mul.Tensor(view_971, 0.7071067811865476);  view_971 = None
        erf_46 = torch.ops.aten.erf.default(mul_430);  mul_430 = None
        add_414 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_431 = torch.ops.aten.mul.Tensor(mul_429, add_414);  mul_429 = add_414 = None
        view_972 = torch.ops.aten.view.default(mul_431, [25088, 96]);  mul_431 = None
        permute_454 = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg331_1, view_972, permute_454);  arg331_1 = view_972 = permute_454 = None
        view_973 = torch.ops.aten.view.default(addmm_166, [1568, 16, 24]);  addmm_166 = None
        add_415 = torch.ops.aten.add.Tensor(add_411, view_973);  add_411 = view_973 = None
        slice_99 = torch.ops.aten.slice.Tensor(add_408, 1, 0, 1)
        slice_101 = torch.ops.aten.slice.Tensor(add_408, 1, 1, 9223372036854775807);  add_408 = None
        clone_362 = torch.ops.aten.clone.default(add_415, memory_format = torch.contiguous_format);  add_415 = None
        var_mean_122 = torch.ops.aten.var_mean.correction(clone_362, [2], correction = 0, keepdim = True)
        getitem_338 = var_mean_122[0]
        getitem_339 = var_mean_122[1];  var_mean_122 = None
        add_416 = torch.ops.aten.add.Tensor(getitem_338, 1e-05);  getitem_338 = None
        rsqrt_122 = torch.ops.aten.rsqrt.default(add_416);  add_416 = None
        sub_169 = torch.ops.aten.sub.Tensor(clone_362, getitem_339);  clone_362 = getitem_339 = None
        mul_432 = torch.ops.aten.mul.Tensor(sub_169, rsqrt_122);  sub_169 = rsqrt_122 = None
        mul_433 = torch.ops.aten.mul.Tensor(mul_432, arg332_1);  mul_432 = arg332_1 = None
        add_417 = torch.ops.aten.add.Tensor(mul_433, arg333_1);  mul_433 = arg333_1 = None
        view_974 = torch.ops.aten.view.default(add_417, [8, 196, -1]);  add_417 = None
        view_975 = torch.ops.aten.view.default(view_974, [1568, 384]);  view_974 = None
        permute_455 = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg335_1, view_975, permute_455);  arg335_1 = view_975 = permute_455 = None
        view_976 = torch.ops.aten.view.default(addmm_167, [8, 196, 384]);  addmm_167 = None
        add_418 = torch.ops.aten.add.Tensor(slice_101, view_976);  slice_101 = view_976 = None
        cat_25 = torch.ops.aten.cat.default([slice_99, add_418], 1);  slice_99 = add_418 = None
        var_mean_123 = torch.ops.aten.var_mean.correction(cat_25, [2], correction = 0, keepdim = True)
        getitem_340 = var_mean_123[0]
        getitem_341 = var_mean_123[1];  var_mean_123 = None
        add_419 = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_123 = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
        sub_170 = torch.ops.aten.sub.Tensor(cat_25, getitem_341);  getitem_341 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_170, rsqrt_123);  sub_170 = rsqrt_123 = None
        mul_435 = torch.ops.aten.mul.Tensor(mul_434, arg336_1);  mul_434 = arg336_1 = None
        add_420 = torch.ops.aten.add.Tensor(mul_435, arg337_1);  mul_435 = arg337_1 = None
        permute_456 = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        view_977 = torch.ops.aten.view.default(add_420, [1576, 384])
        mm_94 = torch.ops.aten.mm.default(view_977, permute_456);  view_977 = permute_456 = None
        view_978 = torch.ops.aten.view.default(mm_94, [8, 197, 768]);  mm_94 = None
        view_979 = torch.ops.aten.view.default(view_978, [8, 197, 2, 6, 64]);  view_978 = None
        permute_457 = torch.ops.aten.permute.default(view_979, [2, 0, 3, 1, 4]);  view_979 = None
        unbind_47 = torch.ops.aten.unbind.int(permute_457);  permute_457 = None
        getitem_342 = unbind_47[0]
        getitem_343 = unbind_47[1];  unbind_47 = None
        permute_458 = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        view_980 = torch.ops.aten.view.default(add_420, [1576, 384]);  add_420 = None
        mm_95 = torch.ops.aten.mm.default(view_980, permute_458);  view_980 = permute_458 = None
        view_981 = torch.ops.aten.view.default(mm_95, [8, 197, 384]);  mm_95 = None
        view_982 = torch.ops.aten.view.default(view_981, [8, 197, 6, -1]);  view_981 = None
        permute_459 = torch.ops.aten.permute.default(view_982, [0, 2, 1, 3]);  view_982 = None
        permute_460 = torch.ops.aten.permute.default(getitem_343, [0, 1, 3, 2]);  getitem_343 = None
        expand_190 = torch.ops.aten.expand.default(getitem_342, [8, 6, 197, 64]);  getitem_342 = None
        clone_363 = torch.ops.aten.clone.default(expand_190, memory_format = torch.contiguous_format);  expand_190 = None
        view_983 = torch.ops.aten.view.default(clone_363, [48, 197, 64]);  clone_363 = None
        expand_191 = torch.ops.aten.expand.default(permute_460, [8, 6, 64, 197]);  permute_460 = None
        clone_364 = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_984 = torch.ops.aten.view.default(clone_364, [48, 64, 197]);  clone_364 = None
        bmm_94 = torch.ops.aten.bmm.default(view_983, view_984);  view_983 = view_984 = None
        view_985 = torch.ops.aten.view.default(bmm_94, [8, 6, 197, 197]);  bmm_94 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_985, 1);  view_985 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_47 = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        expand_192 = torch.ops.aten.expand.default(div_47, [8, 6, 197, 197]);  div_47 = None
        view_986 = torch.ops.aten.view.default(expand_192, [48, 197, 197]);  expand_192 = None
        expand_193 = torch.ops.aten.expand.default(permute_459, [8, 6, 197, 64]);  permute_459 = None
        clone_365 = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_987 = torch.ops.aten.view.default(clone_365, [48, 197, 64]);  clone_365 = None
        bmm_95 = torch.ops.aten.bmm.default(view_986, view_987);  view_986 = view_987 = None
        view_988 = torch.ops.aten.view.default(bmm_95, [8, 6, 197, 64]);  bmm_95 = None
        permute_461 = torch.ops.aten.permute.default(view_988, [0, 2, 1, 3]);  view_988 = None
        clone_366 = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
        view_989 = torch.ops.aten.view.default(clone_366, [8, 197, 384]);  clone_366 = None
        view_990 = torch.ops.aten.view.default(view_989, [1576, 384]);  view_989 = None
        permute_462 = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg341_1, view_990, permute_462);  arg341_1 = view_990 = permute_462 = None
        view_991 = torch.ops.aten.view.default(addmm_168, [8, 197, 384]);  addmm_168 = None
        add_421 = torch.ops.aten.add.Tensor(cat_25, view_991);  cat_25 = view_991 = None
        var_mean_124 = torch.ops.aten.var_mean.correction(add_421, [2], correction = 0, keepdim = True)
        getitem_344 = var_mean_124[0]
        getitem_345 = var_mean_124[1];  var_mean_124 = None
        add_422 = torch.ops.aten.add.Tensor(getitem_344, 1e-05);  getitem_344 = None
        rsqrt_124 = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        sub_172 = torch.ops.aten.sub.Tensor(add_421, getitem_345);  getitem_345 = None
        mul_437 = torch.ops.aten.mul.Tensor(sub_172, rsqrt_124);  sub_172 = rsqrt_124 = None
        mul_438 = torch.ops.aten.mul.Tensor(mul_437, arg342_1);  mul_437 = arg342_1 = None
        add_423 = torch.ops.aten.add.Tensor(mul_438, arg343_1);  mul_438 = arg343_1 = None
        view_992 = torch.ops.aten.view.default(add_423, [1576, 384]);  add_423 = None
        permute_463 = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg345_1, view_992, permute_463);  arg345_1 = view_992 = permute_463 = None
        view_993 = torch.ops.aten.view.default(addmm_169, [8, 197, 1536]);  addmm_169 = None
        mul_439 = torch.ops.aten.mul.Tensor(view_993, 0.5)
        mul_440 = torch.ops.aten.mul.Tensor(view_993, 0.7071067811865476);  view_993 = None
        erf_47 = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_424 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_439, add_424);  mul_439 = add_424 = None
        view_994 = torch.ops.aten.view.default(mul_441, [1576, 1536]);  mul_441 = None
        permute_464 = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg347_1, view_994, permute_464);  arg347_1 = view_994 = permute_464 = None
        view_995 = torch.ops.aten.view.default(addmm_170, [8, 197, 384]);  addmm_170 = None
        add_425 = torch.ops.aten.add.Tensor(add_421, view_995);  add_421 = view_995 = None
        var_mean_125 = torch.ops.aten.var_mean.correction(add_425, [2], correction = 0, keepdim = True)
        getitem_346 = var_mean_125[0]
        getitem_347 = var_mean_125[1];  var_mean_125 = None
        add_426 = torch.ops.aten.add.Tensor(getitem_346, 1e-05);  getitem_346 = None
        rsqrt_125 = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
        sub_173 = torch.ops.aten.sub.Tensor(add_425, getitem_347);  add_425 = getitem_347 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_173, rsqrt_125);  sub_173 = rsqrt_125 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, arg348_1);  mul_442 = arg348_1 = None
        add_427 = torch.ops.aten.add.Tensor(mul_443, arg349_1);  mul_443 = arg349_1 = None
        select_1 = torch.ops.aten.select.int(add_427, 1, 0);  add_427 = None
        clone_369 = torch.ops.aten.clone.default(select_1);  select_1 = None
        permute_465 = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg351_1, clone_369, permute_465);  arg351_1 = clone_369 = permute_465 = None
        return (addmm_171,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 24, 4, 4), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 14112, device=device(type='cuda', index=0))
    reader.tensor(buf2, (24, 3, 7, 7), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf3, (24,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf4, (384,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf5, (384,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384, 384), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf7, (384,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (384,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf9, (384,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1, 1, 384), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 302592, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1, 197, 384), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf12, (24,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf13, (24,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf14, (48, 24), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf15, (24, 24), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf16, (24, 24), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf17, (24,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf18, (24,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf19, (24,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf20, (96, 24), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf21, (96,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf22, (24, 96), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf23, (24,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf24, (24,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf25, (24,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf26, (384, 384), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf27, (384,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf28, (384,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf29, (384,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768, 384), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf31, (384, 384), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf32, (384, 384), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf33, (384,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf34, (384,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf35, (384,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1536, 384), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1536,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf38, (384, 1536), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf39, (384,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf40, (24,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf41, (24,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf42, (48, 24), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf43, (24, 24), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf44, (24, 24), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf45, (24,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf46, (24,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf47, (24,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf48, (96, 24), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf49, (96,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf50, (24, 96), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf51, (24,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf52, (24,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf53, (24,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf54, (384, 384), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf55, (384,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf56, (384,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf57, (384,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768, 384), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf59, (384, 384), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf60, (384, 384), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf61, (384,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf62, (384,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf63, (384,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf64, (1536, 384), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf65, (1536,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf66, (384, 1536), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf67, (384,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf68, (24,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf69, (24,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf70, (48, 24), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf71, (24, 24), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf72, (24, 24), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf73, (24,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf74, (24,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf75, (24,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf76, (96, 24), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf77, (96,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf78, (24, 96), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf79, (24,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf80, (24,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf81, (24,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf82, (384, 384), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf83, (384,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf84, (384,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf85, (384,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 384), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384, 384), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf88, (384, 384), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf89, (384,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf90, (384,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf91, (384,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1536, 384), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1536,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf94, (384, 1536), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf95, (384,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf96, (24,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf97, (24,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf98, (48, 24), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf99, (24, 24), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf100, (24, 24), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf101, (24,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf102, (24,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf103, (24,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf104, (96, 24), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf105, (96,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf106, (24, 96), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf107, (24,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf108, (24,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf109, (24,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384, 384), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf112, (384,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (384,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768, 384), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf115, (384, 384), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf116, (384, 384), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf117, (384,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf118, (384,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf119, (384,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1536, 384), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1536,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384, 1536), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf124, (24,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf125, (24,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf126, (48, 24), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf127, (24, 24), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf128, (24, 24), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf129, (24,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf130, (24,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf131, (24,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf132, (96, 24), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf133, (96,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf134, (24, 96), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf135, (24,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf136, (24,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf137, (24,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf138, (384, 384), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf139, (384,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf140, (384,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf141, (384,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768, 384), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf143, (384, 384), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf144, (384, 384), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf146, (384,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf147, (384,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1536, 384), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1536,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf150, (384, 1536), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf151, (384,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf152, (24,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf153, (24,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf154, (48, 24), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf155, (24, 24), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf156, (24, 24), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf157, (24,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf158, (24,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf159, (24,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf160, (96, 24), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf161, (96,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf162, (24, 96), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf163, (24,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf164, (24,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf165, (24,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf166, (384, 384), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf167, (384,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf168, (384,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf169, (384,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768, 384), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf171, (384, 384), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf172, (384, 384), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf173, (384,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf174, (384,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf175, (384,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf176, (1536, 384), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1536,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf178, (384, 1536), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf179, (384,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf180, (24,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf181, (24,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf182, (48, 24), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf183, (24, 24), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf184, (24, 24), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf185, (24,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf186, (24,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf187, (24,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf188, (96, 24), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf189, (96,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf190, (24, 96), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf191, (24,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf192, (24,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf193, (24,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf194, (384, 384), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf195, (384,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf196, (384,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf197, (384,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768, 384), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf199, (384, 384), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf200, (384, 384), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf201, (384,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf202, (384,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf203, (384,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1536, 384), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1536,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf206, (384, 1536), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf207, (384,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf208, (24,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf209, (24,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf210, (48, 24), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf211, (24, 24), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf212, (24, 24), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf213, (24,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf214, (24,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf215, (24,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf216, (96, 24), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf217, (96,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf218, (24, 96), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf219, (24,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf220, (24,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf221, (24,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf222, (384, 384), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf223, (384,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf224, (384,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf225, (384,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf226, (768, 384), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf227, (384, 384), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf228, (384, 384), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf229, (384,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf230, (384,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf231, (384,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1536, 384), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1536,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf234, (384, 1536), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf235, (384,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf236, (24,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf237, (24,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf238, (48, 24), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf239, (24, 24), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf240, (24, 24), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf241, (24,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf242, (24,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf243, (24,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf244, (96, 24), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf245, (96,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf246, (24, 96), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf247, (24,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf248, (24,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf249, (24,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf250, (384, 384), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf251, (384,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf252, (384,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf253, (384,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf254, (768, 384), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf255, (384, 384), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf256, (384, 384), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf257, (384,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf258, (384,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf259, (384,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1536, 384), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1536,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf262, (384, 1536), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf263, (384,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf264, (24,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf265, (24,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf266, (48, 24), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf267, (24, 24), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf268, (24, 24), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf269, (24,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf270, (24,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf271, (24,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf272, (96, 24), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf273, (96,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf274, (24, 96), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf275, (24,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf276, (24,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf277, (24,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf278, (384, 384), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf279, (384,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf280, (384,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf281, (384,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf282, (768, 384), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf283, (384, 384), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf284, (384, 384), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf285, (384,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf286, (384,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf287, (384,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf288, (1536, 384), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf289, (1536,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf290, (384, 1536), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf291, (384,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf292, (24,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf293, (24,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf294, (48, 24), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf295, (24, 24), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf296, (24, 24), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf297, (24,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf298, (24,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf299, (24,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf300, (96, 24), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf301, (96,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf302, (24, 96), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf303, (24,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf304, (24,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf305, (24,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf306, (384, 384), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf307, (384,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf308, (384,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf309, (384,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf310, (768, 384), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf311, (384, 384), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf312, (384, 384), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf313, (384,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf314, (384,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf315, (384,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1536, 384), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1536,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf318, (384, 1536), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf319, (384,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf320, (24,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf321, (24,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf322, (48, 24), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf323, (24, 24), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf324, (24, 24), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf325, (24,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf326, (24,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf327, (24,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf328, (96, 24), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf329, (96,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf330, (24, 96), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf331, (24,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf332, (24,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf333, (24,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf334, (384, 384), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf335, (384,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf336, (384,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf337, (384,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf338, (768, 384), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf339, (384, 384), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf340, (384, 384), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf341, (384,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf342, (384,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf343, (384,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1536, 384), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1536,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf346, (384, 1536), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf347, (384,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf348, (384,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf349, (384,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1000, 384), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1000,), is_leaf=True)  # arg351_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)