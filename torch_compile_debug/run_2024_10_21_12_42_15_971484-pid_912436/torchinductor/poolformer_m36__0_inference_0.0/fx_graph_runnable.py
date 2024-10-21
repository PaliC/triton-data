
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1):
        convolution_76 = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        view_217 = torch.ops.aten.view.default(convolution_76, [8, 1, 96, 3136])
        var_mean_73 = torch.ops.aten.var_mean.correction(view_217, [2, 3], correction = 0, keepdim = True)
        getitem_146 = var_mean_73[0]
        getitem_147 = var_mean_73[1];  var_mean_73 = None
        add_254 = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_109 = torch.ops.aten.sub.Tensor(view_217, getitem_147);  view_217 = getitem_147 = None
        mul_326 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_73);  sub_109 = rsqrt_73 = None
        view_218 = torch.ops.aten.view.default(mul_326, [8, 96, 56, 56]);  mul_326 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(arg3_1, 0);  arg3_1 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
        mul_327 = torch.ops.aten.mul.Tensor(view_218, unsqueeze_437);  view_218 = unsqueeze_437 = None
        add_255 = torch.ops.aten.add.Tensor(mul_327, unsqueeze_434);  mul_327 = unsqueeze_434 = None
        avg_pool2d_36 = torch.ops.aten.avg_pool2d.default(add_255, [3, 3], [1, 1], [1, 1], False, False)
        sub_110 = torch.ops.aten.sub.Tensor(avg_pool2d_36, add_255);  avg_pool2d_36 = add_255 = None
        view_219 = torch.ops.aten.view.default(arg5_1, [96, 1, 1]);  arg5_1 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_110, view_219);  sub_110 = view_219 = None
        add_256 = torch.ops.aten.add.Tensor(convolution_76, mul_328);  convolution_76 = mul_328 = None
        view_220 = torch.ops.aten.view.default(add_256, [8, 1, 96, 3136])
        var_mean_74 = torch.ops.aten.var_mean.correction(view_220, [2, 3], correction = 0, keepdim = True)
        getitem_148 = var_mean_74[0]
        getitem_149 = var_mean_74[1];  var_mean_74 = None
        add_257 = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_111 = torch.ops.aten.sub.Tensor(view_220, getitem_149);  view_220 = getitem_149 = None
        mul_329 = torch.ops.aten.mul.Tensor(sub_111, rsqrt_74);  sub_111 = rsqrt_74 = None
        view_221 = torch.ops.aten.view.default(mul_329, [8, 96, 56, 56]);  mul_329 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg7_1, 0);  arg7_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(arg6_1, 0);  arg6_1 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_330 = torch.ops.aten.mul.Tensor(view_221, unsqueeze_443);  view_221 = unsqueeze_443 = None
        add_258 = torch.ops.aten.add.Tensor(mul_330, unsqueeze_440);  mul_330 = unsqueeze_440 = None
        convolution_77 = torch.ops.aten.convolution.default(add_258, arg8_1, arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_258 = arg8_1 = arg9_1 = None
        mul_331 = torch.ops.aten.mul.Tensor(convolution_77, 0.5)
        mul_332 = torch.ops.aten.mul.Tensor(convolution_77, 0.7071067811865476);  convolution_77 = None
        erf_36 = torch.ops.aten.erf.default(mul_332);  mul_332 = None
        add_259 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_331, add_259);  mul_331 = add_259 = None
        convolution_78 = torch.ops.aten.convolution.default(mul_333, arg10_1, arg11_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_333 = arg10_1 = arg11_1 = None
        view_222 = torch.ops.aten.view.default(arg12_1, [96, 1, 1]);  arg12_1 = None
        mul_334 = torch.ops.aten.mul.Tensor(convolution_78, view_222);  convolution_78 = view_222 = None
        add_260 = torch.ops.aten.add.Tensor(add_256, mul_334);  add_256 = mul_334 = None
        view_223 = torch.ops.aten.view.default(add_260, [8, 1, 96, 3136])
        var_mean_75 = torch.ops.aten.var_mean.correction(view_223, [2, 3], correction = 0, keepdim = True)
        getitem_150 = var_mean_75[0]
        getitem_151 = var_mean_75[1];  var_mean_75 = None
        add_261 = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_112 = torch.ops.aten.sub.Tensor(view_223, getitem_151);  view_223 = getitem_151 = None
        mul_335 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_75);  sub_112 = rsqrt_75 = None
        view_224 = torch.ops.aten.view.default(mul_335, [8, 96, 56, 56]);  mul_335 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(arg13_1, 0);  arg13_1 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
        mul_336 = torch.ops.aten.mul.Tensor(view_224, unsqueeze_449);  view_224 = unsqueeze_449 = None
        add_262 = torch.ops.aten.add.Tensor(mul_336, unsqueeze_446);  mul_336 = unsqueeze_446 = None
        avg_pool2d_37 = torch.ops.aten.avg_pool2d.default(add_262, [3, 3], [1, 1], [1, 1], False, False)
        sub_113 = torch.ops.aten.sub.Tensor(avg_pool2d_37, add_262);  avg_pool2d_37 = add_262 = None
        view_225 = torch.ops.aten.view.default(arg15_1, [96, 1, 1]);  arg15_1 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_113, view_225);  sub_113 = view_225 = None
        add_263 = torch.ops.aten.add.Tensor(add_260, mul_337);  add_260 = mul_337 = None
        view_226 = torch.ops.aten.view.default(add_263, [8, 1, 96, 3136])
        var_mean_76 = torch.ops.aten.var_mean.correction(view_226, [2, 3], correction = 0, keepdim = True)
        getitem_152 = var_mean_76[0]
        getitem_153 = var_mean_76[1];  var_mean_76 = None
        add_264 = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_114 = torch.ops.aten.sub.Tensor(view_226, getitem_153);  view_226 = getitem_153 = None
        mul_338 = torch.ops.aten.mul.Tensor(sub_114, rsqrt_76);  sub_114 = rsqrt_76 = None
        view_227 = torch.ops.aten.view.default(mul_338, [8, 96, 56, 56]);  mul_338 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(arg17_1, 0);  arg17_1 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(arg16_1, 0);  arg16_1 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
        mul_339 = torch.ops.aten.mul.Tensor(view_227, unsqueeze_455);  view_227 = unsqueeze_455 = None
        add_265 = torch.ops.aten.add.Tensor(mul_339, unsqueeze_452);  mul_339 = unsqueeze_452 = None
        convolution_79 = torch.ops.aten.convolution.default(add_265, arg18_1, arg19_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_265 = arg18_1 = arg19_1 = None
        mul_340 = torch.ops.aten.mul.Tensor(convolution_79, 0.5)
        mul_341 = torch.ops.aten.mul.Tensor(convolution_79, 0.7071067811865476);  convolution_79 = None
        erf_37 = torch.ops.aten.erf.default(mul_341);  mul_341 = None
        add_266 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_340, add_266);  mul_340 = add_266 = None
        convolution_80 = torch.ops.aten.convolution.default(mul_342, arg20_1, arg21_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_342 = arg20_1 = arg21_1 = None
        view_228 = torch.ops.aten.view.default(arg22_1, [96, 1, 1]);  arg22_1 = None
        mul_343 = torch.ops.aten.mul.Tensor(convolution_80, view_228);  convolution_80 = view_228 = None
        add_267 = torch.ops.aten.add.Tensor(add_263, mul_343);  add_263 = mul_343 = None
        view_229 = torch.ops.aten.view.default(add_267, [8, 1, 96, 3136])
        var_mean_77 = torch.ops.aten.var_mean.correction(view_229, [2, 3], correction = 0, keepdim = True)
        getitem_154 = var_mean_77[0]
        getitem_155 = var_mean_77[1];  var_mean_77 = None
        add_268 = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_115 = torch.ops.aten.sub.Tensor(view_229, getitem_155);  view_229 = getitem_155 = None
        mul_344 = torch.ops.aten.mul.Tensor(sub_115, rsqrt_77);  sub_115 = rsqrt_77 = None
        view_230 = torch.ops.aten.view.default(mul_344, [8, 96, 56, 56]);  mul_344 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(arg24_1, 0);  arg24_1 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(arg23_1, 0);  arg23_1 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
        mul_345 = torch.ops.aten.mul.Tensor(view_230, unsqueeze_461);  view_230 = unsqueeze_461 = None
        add_269 = torch.ops.aten.add.Tensor(mul_345, unsqueeze_458);  mul_345 = unsqueeze_458 = None
        avg_pool2d_38 = torch.ops.aten.avg_pool2d.default(add_269, [3, 3], [1, 1], [1, 1], False, False)
        sub_116 = torch.ops.aten.sub.Tensor(avg_pool2d_38, add_269);  avg_pool2d_38 = add_269 = None
        view_231 = torch.ops.aten.view.default(arg25_1, [96, 1, 1]);  arg25_1 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_116, view_231);  sub_116 = view_231 = None
        add_270 = torch.ops.aten.add.Tensor(add_267, mul_346);  add_267 = mul_346 = None
        view_232 = torch.ops.aten.view.default(add_270, [8, 1, 96, 3136])
        var_mean_78 = torch.ops.aten.var_mean.correction(view_232, [2, 3], correction = 0, keepdim = True)
        getitem_156 = var_mean_78[0]
        getitem_157 = var_mean_78[1];  var_mean_78 = None
        add_271 = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_117 = torch.ops.aten.sub.Tensor(view_232, getitem_157);  view_232 = getitem_157 = None
        mul_347 = torch.ops.aten.mul.Tensor(sub_117, rsqrt_78);  sub_117 = rsqrt_78 = None
        view_233 = torch.ops.aten.view.default(mul_347, [8, 96, 56, 56]);  mul_347 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(arg27_1, 0);  arg27_1 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(arg26_1, 0);  arg26_1 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
        mul_348 = torch.ops.aten.mul.Tensor(view_233, unsqueeze_467);  view_233 = unsqueeze_467 = None
        add_272 = torch.ops.aten.add.Tensor(mul_348, unsqueeze_464);  mul_348 = unsqueeze_464 = None
        convolution_81 = torch.ops.aten.convolution.default(add_272, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_272 = arg28_1 = arg29_1 = None
        mul_349 = torch.ops.aten.mul.Tensor(convolution_81, 0.5)
        mul_350 = torch.ops.aten.mul.Tensor(convolution_81, 0.7071067811865476);  convolution_81 = None
        erf_38 = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_273 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_349, add_273);  mul_349 = add_273 = None
        convolution_82 = torch.ops.aten.convolution.default(mul_351, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_351 = arg30_1 = arg31_1 = None
        view_234 = torch.ops.aten.view.default(arg32_1, [96, 1, 1]);  arg32_1 = None
        mul_352 = torch.ops.aten.mul.Tensor(convolution_82, view_234);  convolution_82 = view_234 = None
        add_274 = torch.ops.aten.add.Tensor(add_270, mul_352);  add_270 = mul_352 = None
        view_235 = torch.ops.aten.view.default(add_274, [8, 1, 96, 3136])
        var_mean_79 = torch.ops.aten.var_mean.correction(view_235, [2, 3], correction = 0, keepdim = True)
        getitem_158 = var_mean_79[0]
        getitem_159 = var_mean_79[1];  var_mean_79 = None
        add_275 = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        sub_118 = torch.ops.aten.sub.Tensor(view_235, getitem_159);  view_235 = getitem_159 = None
        mul_353 = torch.ops.aten.mul.Tensor(sub_118, rsqrt_79);  sub_118 = rsqrt_79 = None
        view_236 = torch.ops.aten.view.default(mul_353, [8, 96, 56, 56]);  mul_353 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg34_1, 0);  arg34_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(arg33_1, 0);  arg33_1 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
        mul_354 = torch.ops.aten.mul.Tensor(view_236, unsqueeze_473);  view_236 = unsqueeze_473 = None
        add_276 = torch.ops.aten.add.Tensor(mul_354, unsqueeze_470);  mul_354 = unsqueeze_470 = None
        avg_pool2d_39 = torch.ops.aten.avg_pool2d.default(add_276, [3, 3], [1, 1], [1, 1], False, False)
        sub_119 = torch.ops.aten.sub.Tensor(avg_pool2d_39, add_276);  avg_pool2d_39 = add_276 = None
        view_237 = torch.ops.aten.view.default(arg35_1, [96, 1, 1]);  arg35_1 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_119, view_237);  sub_119 = view_237 = None
        add_277 = torch.ops.aten.add.Tensor(add_274, mul_355);  add_274 = mul_355 = None
        view_238 = torch.ops.aten.view.default(add_277, [8, 1, 96, 3136])
        var_mean_80 = torch.ops.aten.var_mean.correction(view_238, [2, 3], correction = 0, keepdim = True)
        getitem_160 = var_mean_80[0]
        getitem_161 = var_mean_80[1];  var_mean_80 = None
        add_278 = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        sub_120 = torch.ops.aten.sub.Tensor(view_238, getitem_161);  view_238 = getitem_161 = None
        mul_356 = torch.ops.aten.mul.Tensor(sub_120, rsqrt_80);  sub_120 = rsqrt_80 = None
        view_239 = torch.ops.aten.view.default(mul_356, [8, 96, 56, 56]);  mul_356 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(arg37_1, 0);  arg37_1 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(arg36_1, 0);  arg36_1 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
        mul_357 = torch.ops.aten.mul.Tensor(view_239, unsqueeze_479);  view_239 = unsqueeze_479 = None
        add_279 = torch.ops.aten.add.Tensor(mul_357, unsqueeze_476);  mul_357 = unsqueeze_476 = None
        convolution_83 = torch.ops.aten.convolution.default(add_279, arg38_1, arg39_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_279 = arg38_1 = arg39_1 = None
        mul_358 = torch.ops.aten.mul.Tensor(convolution_83, 0.5)
        mul_359 = torch.ops.aten.mul.Tensor(convolution_83, 0.7071067811865476);  convolution_83 = None
        erf_39 = torch.ops.aten.erf.default(mul_359);  mul_359 = None
        add_280 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_360 = torch.ops.aten.mul.Tensor(mul_358, add_280);  mul_358 = add_280 = None
        convolution_84 = torch.ops.aten.convolution.default(mul_360, arg40_1, arg41_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_360 = arg40_1 = arg41_1 = None
        view_240 = torch.ops.aten.view.default(arg42_1, [96, 1, 1]);  arg42_1 = None
        mul_361 = torch.ops.aten.mul.Tensor(convolution_84, view_240);  convolution_84 = view_240 = None
        add_281 = torch.ops.aten.add.Tensor(add_277, mul_361);  add_277 = mul_361 = None
        view_241 = torch.ops.aten.view.default(add_281, [8, 1, 96, 3136])
        var_mean_81 = torch.ops.aten.var_mean.correction(view_241, [2, 3], correction = 0, keepdim = True)
        getitem_162 = var_mean_81[0]
        getitem_163 = var_mean_81[1];  var_mean_81 = None
        add_282 = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_121 = torch.ops.aten.sub.Tensor(view_241, getitem_163);  view_241 = getitem_163 = None
        mul_362 = torch.ops.aten.mul.Tensor(sub_121, rsqrt_81);  sub_121 = rsqrt_81 = None
        view_242 = torch.ops.aten.view.default(mul_362, [8, 96, 56, 56]);  mul_362 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg44_1, 0);  arg44_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(arg43_1, 0);  arg43_1 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
        mul_363 = torch.ops.aten.mul.Tensor(view_242, unsqueeze_485);  view_242 = unsqueeze_485 = None
        add_283 = torch.ops.aten.add.Tensor(mul_363, unsqueeze_482);  mul_363 = unsqueeze_482 = None
        avg_pool2d_40 = torch.ops.aten.avg_pool2d.default(add_283, [3, 3], [1, 1], [1, 1], False, False)
        sub_122 = torch.ops.aten.sub.Tensor(avg_pool2d_40, add_283);  avg_pool2d_40 = add_283 = None
        view_243 = torch.ops.aten.view.default(arg45_1, [96, 1, 1]);  arg45_1 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_122, view_243);  sub_122 = view_243 = None
        add_284 = torch.ops.aten.add.Tensor(add_281, mul_364);  add_281 = mul_364 = None
        view_244 = torch.ops.aten.view.default(add_284, [8, 1, 96, 3136])
        var_mean_82 = torch.ops.aten.var_mean.correction(view_244, [2, 3], correction = 0, keepdim = True)
        getitem_164 = var_mean_82[0]
        getitem_165 = var_mean_82[1];  var_mean_82 = None
        add_285 = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        sub_123 = torch.ops.aten.sub.Tensor(view_244, getitem_165);  view_244 = getitem_165 = None
        mul_365 = torch.ops.aten.mul.Tensor(sub_123, rsqrt_82);  sub_123 = rsqrt_82 = None
        view_245 = torch.ops.aten.view.default(mul_365, [8, 96, 56, 56]);  mul_365 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg47_1, 0);  arg47_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(arg46_1, 0);  arg46_1 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
        mul_366 = torch.ops.aten.mul.Tensor(view_245, unsqueeze_491);  view_245 = unsqueeze_491 = None
        add_286 = torch.ops.aten.add.Tensor(mul_366, unsqueeze_488);  mul_366 = unsqueeze_488 = None
        convolution_85 = torch.ops.aten.convolution.default(add_286, arg48_1, arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_286 = arg48_1 = arg49_1 = None
        mul_367 = torch.ops.aten.mul.Tensor(convolution_85, 0.5)
        mul_368 = torch.ops.aten.mul.Tensor(convolution_85, 0.7071067811865476);  convolution_85 = None
        erf_40 = torch.ops.aten.erf.default(mul_368);  mul_368 = None
        add_287 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_367, add_287);  mul_367 = add_287 = None
        convolution_86 = torch.ops.aten.convolution.default(mul_369, arg50_1, arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_369 = arg50_1 = arg51_1 = None
        view_246 = torch.ops.aten.view.default(arg52_1, [96, 1, 1]);  arg52_1 = None
        mul_370 = torch.ops.aten.mul.Tensor(convolution_86, view_246);  convolution_86 = view_246 = None
        add_288 = torch.ops.aten.add.Tensor(add_284, mul_370);  add_284 = mul_370 = None
        view_247 = torch.ops.aten.view.default(add_288, [8, 1, 96, 3136])
        var_mean_83 = torch.ops.aten.var_mean.correction(view_247, [2, 3], correction = 0, keepdim = True)
        getitem_166 = var_mean_83[0]
        getitem_167 = var_mean_83[1];  var_mean_83 = None
        add_289 = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        sub_124 = torch.ops.aten.sub.Tensor(view_247, getitem_167);  view_247 = getitem_167 = None
        mul_371 = torch.ops.aten.mul.Tensor(sub_124, rsqrt_83);  sub_124 = rsqrt_83 = None
        view_248 = torch.ops.aten.view.default(mul_371, [8, 96, 56, 56]);  mul_371 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg54_1, 0);  arg54_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(arg53_1, 0);  arg53_1 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
        mul_372 = torch.ops.aten.mul.Tensor(view_248, unsqueeze_497);  view_248 = unsqueeze_497 = None
        add_290 = torch.ops.aten.add.Tensor(mul_372, unsqueeze_494);  mul_372 = unsqueeze_494 = None
        avg_pool2d_41 = torch.ops.aten.avg_pool2d.default(add_290, [3, 3], [1, 1], [1, 1], False, False)
        sub_125 = torch.ops.aten.sub.Tensor(avg_pool2d_41, add_290);  avg_pool2d_41 = add_290 = None
        view_249 = torch.ops.aten.view.default(arg55_1, [96, 1, 1]);  arg55_1 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_125, view_249);  sub_125 = view_249 = None
        add_291 = torch.ops.aten.add.Tensor(add_288, mul_373);  add_288 = mul_373 = None
        view_250 = torch.ops.aten.view.default(add_291, [8, 1, 96, 3136])
        var_mean_84 = torch.ops.aten.var_mean.correction(view_250, [2, 3], correction = 0, keepdim = True)
        getitem_168 = var_mean_84[0]
        getitem_169 = var_mean_84[1];  var_mean_84 = None
        add_292 = torch.ops.aten.add.Tensor(getitem_168, 1e-05);  getitem_168 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_126 = torch.ops.aten.sub.Tensor(view_250, getitem_169);  view_250 = getitem_169 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_126, rsqrt_84);  sub_126 = rsqrt_84 = None
        view_251 = torch.ops.aten.view.default(mul_374, [8, 96, 56, 56]);  mul_374 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(arg57_1, 0);  arg57_1 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(arg56_1, 0);  arg56_1 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
        mul_375 = torch.ops.aten.mul.Tensor(view_251, unsqueeze_503);  view_251 = unsqueeze_503 = None
        add_293 = torch.ops.aten.add.Tensor(mul_375, unsqueeze_500);  mul_375 = unsqueeze_500 = None
        convolution_87 = torch.ops.aten.convolution.default(add_293, arg58_1, arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_293 = arg58_1 = arg59_1 = None
        mul_376 = torch.ops.aten.mul.Tensor(convolution_87, 0.5)
        mul_377 = torch.ops.aten.mul.Tensor(convolution_87, 0.7071067811865476);  convolution_87 = None
        erf_41 = torch.ops.aten.erf.default(mul_377);  mul_377 = None
        add_294 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_378 = torch.ops.aten.mul.Tensor(mul_376, add_294);  mul_376 = add_294 = None
        convolution_88 = torch.ops.aten.convolution.default(mul_378, arg60_1, arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_378 = arg60_1 = arg61_1 = None
        view_252 = torch.ops.aten.view.default(arg62_1, [96, 1, 1]);  arg62_1 = None
        mul_379 = torch.ops.aten.mul.Tensor(convolution_88, view_252);  convolution_88 = view_252 = None
        add_295 = torch.ops.aten.add.Tensor(add_291, mul_379);  add_291 = mul_379 = None
        convolution_89 = torch.ops.aten.convolution.default(add_295, arg63_1, arg64_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  add_295 = arg63_1 = arg64_1 = None
        view_253 = torch.ops.aten.view.default(convolution_89, [8, 1, 192, 784])
        var_mean_85 = torch.ops.aten.var_mean.correction(view_253, [2, 3], correction = 0, keepdim = True)
        getitem_170 = var_mean_85[0]
        getitem_171 = var_mean_85[1];  var_mean_85 = None
        add_296 = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_127 = torch.ops.aten.sub.Tensor(view_253, getitem_171);  view_253 = getitem_171 = None
        mul_380 = torch.ops.aten.mul.Tensor(sub_127, rsqrt_85);  sub_127 = rsqrt_85 = None
        view_254 = torch.ops.aten.view.default(mul_380, [8, 192, 28, 28]);  mul_380 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg66_1, 0);  arg66_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(arg65_1, 0);  arg65_1 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
        mul_381 = torch.ops.aten.mul.Tensor(view_254, unsqueeze_509);  view_254 = unsqueeze_509 = None
        add_297 = torch.ops.aten.add.Tensor(mul_381, unsqueeze_506);  mul_381 = unsqueeze_506 = None
        avg_pool2d_42 = torch.ops.aten.avg_pool2d.default(add_297, [3, 3], [1, 1], [1, 1], False, False)
        sub_128 = torch.ops.aten.sub.Tensor(avg_pool2d_42, add_297);  avg_pool2d_42 = add_297 = None
        view_255 = torch.ops.aten.view.default(arg67_1, [192, 1, 1]);  arg67_1 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_128, view_255);  sub_128 = view_255 = None
        add_298 = torch.ops.aten.add.Tensor(convolution_89, mul_382);  convolution_89 = mul_382 = None
        view_256 = torch.ops.aten.view.default(add_298, [8, 1, 192, 784])
        var_mean_86 = torch.ops.aten.var_mean.correction(view_256, [2, 3], correction = 0, keepdim = True)
        getitem_172 = var_mean_86[0]
        getitem_173 = var_mean_86[1];  var_mean_86 = None
        add_299 = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_129 = torch.ops.aten.sub.Tensor(view_256, getitem_173);  view_256 = getitem_173 = None
        mul_383 = torch.ops.aten.mul.Tensor(sub_129, rsqrt_86);  sub_129 = rsqrt_86 = None
        view_257 = torch.ops.aten.view.default(mul_383, [8, 192, 28, 28]);  mul_383 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg69_1, 0);  arg69_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(arg68_1, 0);  arg68_1 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
        mul_384 = torch.ops.aten.mul.Tensor(view_257, unsqueeze_515);  view_257 = unsqueeze_515 = None
        add_300 = torch.ops.aten.add.Tensor(mul_384, unsqueeze_512);  mul_384 = unsqueeze_512 = None
        convolution_90 = torch.ops.aten.convolution.default(add_300, arg70_1, arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_300 = arg70_1 = arg71_1 = None
        mul_385 = torch.ops.aten.mul.Tensor(convolution_90, 0.5)
        mul_386 = torch.ops.aten.mul.Tensor(convolution_90, 0.7071067811865476);  convolution_90 = None
        erf_42 = torch.ops.aten.erf.default(mul_386);  mul_386 = None
        add_301 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_385, add_301);  mul_385 = add_301 = None
        convolution_91 = torch.ops.aten.convolution.default(mul_387, arg72_1, arg73_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_387 = arg72_1 = arg73_1 = None
        view_258 = torch.ops.aten.view.default(arg74_1, [192, 1, 1]);  arg74_1 = None
        mul_388 = torch.ops.aten.mul.Tensor(convolution_91, view_258);  convolution_91 = view_258 = None
        add_302 = torch.ops.aten.add.Tensor(add_298, mul_388);  add_298 = mul_388 = None
        view_259 = torch.ops.aten.view.default(add_302, [8, 1, 192, 784])
        var_mean_87 = torch.ops.aten.var_mean.correction(view_259, [2, 3], correction = 0, keepdim = True)
        getitem_174 = var_mean_87[0]
        getitem_175 = var_mean_87[1];  var_mean_87 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_130 = torch.ops.aten.sub.Tensor(view_259, getitem_175);  view_259 = getitem_175 = None
        mul_389 = torch.ops.aten.mul.Tensor(sub_130, rsqrt_87);  sub_130 = rsqrt_87 = None
        view_260 = torch.ops.aten.view.default(mul_389, [8, 192, 28, 28]);  mul_389 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg76_1, 0);  arg76_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(arg75_1, 0);  arg75_1 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
        mul_390 = torch.ops.aten.mul.Tensor(view_260, unsqueeze_521);  view_260 = unsqueeze_521 = None
        add_304 = torch.ops.aten.add.Tensor(mul_390, unsqueeze_518);  mul_390 = unsqueeze_518 = None
        avg_pool2d_43 = torch.ops.aten.avg_pool2d.default(add_304, [3, 3], [1, 1], [1, 1], False, False)
        sub_131 = torch.ops.aten.sub.Tensor(avg_pool2d_43, add_304);  avg_pool2d_43 = add_304 = None
        view_261 = torch.ops.aten.view.default(arg77_1, [192, 1, 1]);  arg77_1 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_131, view_261);  sub_131 = view_261 = None
        add_305 = torch.ops.aten.add.Tensor(add_302, mul_391);  add_302 = mul_391 = None
        view_262 = torch.ops.aten.view.default(add_305, [8, 1, 192, 784])
        var_mean_88 = torch.ops.aten.var_mean.correction(view_262, [2, 3], correction = 0, keepdim = True)
        getitem_176 = var_mean_88[0]
        getitem_177 = var_mean_88[1];  var_mean_88 = None
        add_306 = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_132 = torch.ops.aten.sub.Tensor(view_262, getitem_177);  view_262 = getitem_177 = None
        mul_392 = torch.ops.aten.mul.Tensor(sub_132, rsqrt_88);  sub_132 = rsqrt_88 = None
        view_263 = torch.ops.aten.view.default(mul_392, [8, 192, 28, 28]);  mul_392 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(arg79_1, 0);  arg79_1 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(arg78_1, 0);  arg78_1 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
        mul_393 = torch.ops.aten.mul.Tensor(view_263, unsqueeze_527);  view_263 = unsqueeze_527 = None
        add_307 = torch.ops.aten.add.Tensor(mul_393, unsqueeze_524);  mul_393 = unsqueeze_524 = None
        convolution_92 = torch.ops.aten.convolution.default(add_307, arg80_1, arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_307 = arg80_1 = arg81_1 = None
        mul_394 = torch.ops.aten.mul.Tensor(convolution_92, 0.5)
        mul_395 = torch.ops.aten.mul.Tensor(convolution_92, 0.7071067811865476);  convolution_92 = None
        erf_43 = torch.ops.aten.erf.default(mul_395);  mul_395 = None
        add_308 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_394, add_308);  mul_394 = add_308 = None
        convolution_93 = torch.ops.aten.convolution.default(mul_396, arg82_1, arg83_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_396 = arg82_1 = arg83_1 = None
        view_264 = torch.ops.aten.view.default(arg84_1, [192, 1, 1]);  arg84_1 = None
        mul_397 = torch.ops.aten.mul.Tensor(convolution_93, view_264);  convolution_93 = view_264 = None
        add_309 = torch.ops.aten.add.Tensor(add_305, mul_397);  add_305 = mul_397 = None
        view_265 = torch.ops.aten.view.default(add_309, [8, 1, 192, 784])
        var_mean_89 = torch.ops.aten.var_mean.correction(view_265, [2, 3], correction = 0, keepdim = True)
        getitem_178 = var_mean_89[0]
        getitem_179 = var_mean_89[1];  var_mean_89 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_133 = torch.ops.aten.sub.Tensor(view_265, getitem_179);  view_265 = getitem_179 = None
        mul_398 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_89);  sub_133 = rsqrt_89 = None
        view_266 = torch.ops.aten.view.default(mul_398, [8, 192, 28, 28]);  mul_398 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg86_1, 0);  arg86_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(arg85_1, 0);  arg85_1 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
        mul_399 = torch.ops.aten.mul.Tensor(view_266, unsqueeze_533);  view_266 = unsqueeze_533 = None
        add_311 = torch.ops.aten.add.Tensor(mul_399, unsqueeze_530);  mul_399 = unsqueeze_530 = None
        avg_pool2d_44 = torch.ops.aten.avg_pool2d.default(add_311, [3, 3], [1, 1], [1, 1], False, False)
        sub_134 = torch.ops.aten.sub.Tensor(avg_pool2d_44, add_311);  avg_pool2d_44 = add_311 = None
        view_267 = torch.ops.aten.view.default(arg87_1, [192, 1, 1]);  arg87_1 = None
        mul_400 = torch.ops.aten.mul.Tensor(sub_134, view_267);  sub_134 = view_267 = None
        add_312 = torch.ops.aten.add.Tensor(add_309, mul_400);  add_309 = mul_400 = None
        view_268 = torch.ops.aten.view.default(add_312, [8, 1, 192, 784])
        var_mean_90 = torch.ops.aten.var_mean.correction(view_268, [2, 3], correction = 0, keepdim = True)
        getitem_180 = var_mean_90[0]
        getitem_181 = var_mean_90[1];  var_mean_90 = None
        add_313 = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_135 = torch.ops.aten.sub.Tensor(view_268, getitem_181);  view_268 = getitem_181 = None
        mul_401 = torch.ops.aten.mul.Tensor(sub_135, rsqrt_90);  sub_135 = rsqrt_90 = None
        view_269 = torch.ops.aten.view.default(mul_401, [8, 192, 28, 28]);  mul_401 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg89_1, 0);  arg89_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(arg88_1, 0);  arg88_1 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
        mul_402 = torch.ops.aten.mul.Tensor(view_269, unsqueeze_539);  view_269 = unsqueeze_539 = None
        add_314 = torch.ops.aten.add.Tensor(mul_402, unsqueeze_536);  mul_402 = unsqueeze_536 = None
        convolution_94 = torch.ops.aten.convolution.default(add_314, arg90_1, arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_314 = arg90_1 = arg91_1 = None
        mul_403 = torch.ops.aten.mul.Tensor(convolution_94, 0.5)
        mul_404 = torch.ops.aten.mul.Tensor(convolution_94, 0.7071067811865476);  convolution_94 = None
        erf_44 = torch.ops.aten.erf.default(mul_404);  mul_404 = None
        add_315 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_403, add_315);  mul_403 = add_315 = None
        convolution_95 = torch.ops.aten.convolution.default(mul_405, arg92_1, arg93_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_405 = arg92_1 = arg93_1 = None
        view_270 = torch.ops.aten.view.default(arg94_1, [192, 1, 1]);  arg94_1 = None
        mul_406 = torch.ops.aten.mul.Tensor(convolution_95, view_270);  convolution_95 = view_270 = None
        add_316 = torch.ops.aten.add.Tensor(add_312, mul_406);  add_312 = mul_406 = None
        view_271 = torch.ops.aten.view.default(add_316, [8, 1, 192, 784])
        var_mean_91 = torch.ops.aten.var_mean.correction(view_271, [2, 3], correction = 0, keepdim = True)
        getitem_182 = var_mean_91[0]
        getitem_183 = var_mean_91[1];  var_mean_91 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_136 = torch.ops.aten.sub.Tensor(view_271, getitem_183);  view_271 = getitem_183 = None
        mul_407 = torch.ops.aten.mul.Tensor(sub_136, rsqrt_91);  sub_136 = rsqrt_91 = None
        view_272 = torch.ops.aten.view.default(mul_407, [8, 192, 28, 28]);  mul_407 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg96_1, 0);  arg96_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(arg95_1, 0);  arg95_1 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
        mul_408 = torch.ops.aten.mul.Tensor(view_272, unsqueeze_545);  view_272 = unsqueeze_545 = None
        add_318 = torch.ops.aten.add.Tensor(mul_408, unsqueeze_542);  mul_408 = unsqueeze_542 = None
        avg_pool2d_45 = torch.ops.aten.avg_pool2d.default(add_318, [3, 3], [1, 1], [1, 1], False, False)
        sub_137 = torch.ops.aten.sub.Tensor(avg_pool2d_45, add_318);  avg_pool2d_45 = add_318 = None
        view_273 = torch.ops.aten.view.default(arg97_1, [192, 1, 1]);  arg97_1 = None
        mul_409 = torch.ops.aten.mul.Tensor(sub_137, view_273);  sub_137 = view_273 = None
        add_319 = torch.ops.aten.add.Tensor(add_316, mul_409);  add_316 = mul_409 = None
        view_274 = torch.ops.aten.view.default(add_319, [8, 1, 192, 784])
        var_mean_92 = torch.ops.aten.var_mean.correction(view_274, [2, 3], correction = 0, keepdim = True)
        getitem_184 = var_mean_92[0]
        getitem_185 = var_mean_92[1];  var_mean_92 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_138 = torch.ops.aten.sub.Tensor(view_274, getitem_185);  view_274 = getitem_185 = None
        mul_410 = torch.ops.aten.mul.Tensor(sub_138, rsqrt_92);  sub_138 = rsqrt_92 = None
        view_275 = torch.ops.aten.view.default(mul_410, [8, 192, 28, 28]);  mul_410 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(arg99_1, 0);  arg99_1 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(arg98_1, 0);  arg98_1 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
        mul_411 = torch.ops.aten.mul.Tensor(view_275, unsqueeze_551);  view_275 = unsqueeze_551 = None
        add_321 = torch.ops.aten.add.Tensor(mul_411, unsqueeze_548);  mul_411 = unsqueeze_548 = None
        convolution_96 = torch.ops.aten.convolution.default(add_321, arg100_1, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_321 = arg100_1 = arg101_1 = None
        mul_412 = torch.ops.aten.mul.Tensor(convolution_96, 0.5)
        mul_413 = torch.ops.aten.mul.Tensor(convolution_96, 0.7071067811865476);  convolution_96 = None
        erf_45 = torch.ops.aten.erf.default(mul_413);  mul_413 = None
        add_322 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_414 = torch.ops.aten.mul.Tensor(mul_412, add_322);  mul_412 = add_322 = None
        convolution_97 = torch.ops.aten.convolution.default(mul_414, arg102_1, arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_414 = arg102_1 = arg103_1 = None
        view_276 = torch.ops.aten.view.default(arg104_1, [192, 1, 1]);  arg104_1 = None
        mul_415 = torch.ops.aten.mul.Tensor(convolution_97, view_276);  convolution_97 = view_276 = None
        add_323 = torch.ops.aten.add.Tensor(add_319, mul_415);  add_319 = mul_415 = None
        view_277 = torch.ops.aten.view.default(add_323, [8, 1, 192, 784])
        var_mean_93 = torch.ops.aten.var_mean.correction(view_277, [2, 3], correction = 0, keepdim = True)
        getitem_186 = var_mean_93[0]
        getitem_187 = var_mean_93[1];  var_mean_93 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_139 = torch.ops.aten.sub.Tensor(view_277, getitem_187);  view_277 = getitem_187 = None
        mul_416 = torch.ops.aten.mul.Tensor(sub_139, rsqrt_93);  sub_139 = rsqrt_93 = None
        view_278 = torch.ops.aten.view.default(mul_416, [8, 192, 28, 28]);  mul_416 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg106_1, 0);  arg106_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(arg105_1, 0);  arg105_1 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
        mul_417 = torch.ops.aten.mul.Tensor(view_278, unsqueeze_557);  view_278 = unsqueeze_557 = None
        add_325 = torch.ops.aten.add.Tensor(mul_417, unsqueeze_554);  mul_417 = unsqueeze_554 = None
        avg_pool2d_46 = torch.ops.aten.avg_pool2d.default(add_325, [3, 3], [1, 1], [1, 1], False, False)
        sub_140 = torch.ops.aten.sub.Tensor(avg_pool2d_46, add_325);  avg_pool2d_46 = add_325 = None
        view_279 = torch.ops.aten.view.default(arg107_1, [192, 1, 1]);  arg107_1 = None
        mul_418 = torch.ops.aten.mul.Tensor(sub_140, view_279);  sub_140 = view_279 = None
        add_326 = torch.ops.aten.add.Tensor(add_323, mul_418);  add_323 = mul_418 = None
        view_280 = torch.ops.aten.view.default(add_326, [8, 1, 192, 784])
        var_mean_94 = torch.ops.aten.var_mean.correction(view_280, [2, 3], correction = 0, keepdim = True)
        getitem_188 = var_mean_94[0]
        getitem_189 = var_mean_94[1];  var_mean_94 = None
        add_327 = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        sub_141 = torch.ops.aten.sub.Tensor(view_280, getitem_189);  view_280 = getitem_189 = None
        mul_419 = torch.ops.aten.mul.Tensor(sub_141, rsqrt_94);  sub_141 = rsqrt_94 = None
        view_281 = torch.ops.aten.view.default(mul_419, [8, 192, 28, 28]);  mul_419 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg109_1, 0);  arg109_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(arg108_1, 0);  arg108_1 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
        mul_420 = torch.ops.aten.mul.Tensor(view_281, unsqueeze_563);  view_281 = unsqueeze_563 = None
        add_328 = torch.ops.aten.add.Tensor(mul_420, unsqueeze_560);  mul_420 = unsqueeze_560 = None
        convolution_98 = torch.ops.aten.convolution.default(add_328, arg110_1, arg111_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_328 = arg110_1 = arg111_1 = None
        mul_421 = torch.ops.aten.mul.Tensor(convolution_98, 0.5)
        mul_422 = torch.ops.aten.mul.Tensor(convolution_98, 0.7071067811865476);  convolution_98 = None
        erf_46 = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_329 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_421, add_329);  mul_421 = add_329 = None
        convolution_99 = torch.ops.aten.convolution.default(mul_423, arg112_1, arg113_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_423 = arg112_1 = arg113_1 = None
        view_282 = torch.ops.aten.view.default(arg114_1, [192, 1, 1]);  arg114_1 = None
        mul_424 = torch.ops.aten.mul.Tensor(convolution_99, view_282);  convolution_99 = view_282 = None
        add_330 = torch.ops.aten.add.Tensor(add_326, mul_424);  add_326 = mul_424 = None
        view_283 = torch.ops.aten.view.default(add_330, [8, 1, 192, 784])
        var_mean_95 = torch.ops.aten.var_mean.correction(view_283, [2, 3], correction = 0, keepdim = True)
        getitem_190 = var_mean_95[0]
        getitem_191 = var_mean_95[1];  var_mean_95 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_142 = torch.ops.aten.sub.Tensor(view_283, getitem_191);  view_283 = getitem_191 = None
        mul_425 = torch.ops.aten.mul.Tensor(sub_142, rsqrt_95);  sub_142 = rsqrt_95 = None
        view_284 = torch.ops.aten.view.default(mul_425, [8, 192, 28, 28]);  mul_425 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg116_1, 0);  arg116_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(arg115_1, 0);  arg115_1 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
        mul_426 = torch.ops.aten.mul.Tensor(view_284, unsqueeze_569);  view_284 = unsqueeze_569 = None
        add_332 = torch.ops.aten.add.Tensor(mul_426, unsqueeze_566);  mul_426 = unsqueeze_566 = None
        avg_pool2d_47 = torch.ops.aten.avg_pool2d.default(add_332, [3, 3], [1, 1], [1, 1], False, False)
        sub_143 = torch.ops.aten.sub.Tensor(avg_pool2d_47, add_332);  avg_pool2d_47 = add_332 = None
        view_285 = torch.ops.aten.view.default(arg117_1, [192, 1, 1]);  arg117_1 = None
        mul_427 = torch.ops.aten.mul.Tensor(sub_143, view_285);  sub_143 = view_285 = None
        add_333 = torch.ops.aten.add.Tensor(add_330, mul_427);  add_330 = mul_427 = None
        view_286 = torch.ops.aten.view.default(add_333, [8, 1, 192, 784])
        var_mean_96 = torch.ops.aten.var_mean.correction(view_286, [2, 3], correction = 0, keepdim = True)
        getitem_192 = var_mean_96[0]
        getitem_193 = var_mean_96[1];  var_mean_96 = None
        add_334 = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_144 = torch.ops.aten.sub.Tensor(view_286, getitem_193);  view_286 = getitem_193 = None
        mul_428 = torch.ops.aten.mul.Tensor(sub_144, rsqrt_96);  sub_144 = rsqrt_96 = None
        view_287 = torch.ops.aten.view.default(mul_428, [8, 192, 28, 28]);  mul_428 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(arg119_1, 0);  arg119_1 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(arg118_1, 0);  arg118_1 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
        mul_429 = torch.ops.aten.mul.Tensor(view_287, unsqueeze_575);  view_287 = unsqueeze_575 = None
        add_335 = torch.ops.aten.add.Tensor(mul_429, unsqueeze_572);  mul_429 = unsqueeze_572 = None
        convolution_100 = torch.ops.aten.convolution.default(add_335, arg120_1, arg121_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_335 = arg120_1 = arg121_1 = None
        mul_430 = torch.ops.aten.mul.Tensor(convolution_100, 0.5)
        mul_431 = torch.ops.aten.mul.Tensor(convolution_100, 0.7071067811865476);  convolution_100 = None
        erf_47 = torch.ops.aten.erf.default(mul_431);  mul_431 = None
        add_336 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_430, add_336);  mul_430 = add_336 = None
        convolution_101 = torch.ops.aten.convolution.default(mul_432, arg122_1, arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_432 = arg122_1 = arg123_1 = None
        view_288 = torch.ops.aten.view.default(arg124_1, [192, 1, 1]);  arg124_1 = None
        mul_433 = torch.ops.aten.mul.Tensor(convolution_101, view_288);  convolution_101 = view_288 = None
        add_337 = torch.ops.aten.add.Tensor(add_333, mul_433);  add_333 = mul_433 = None
        convolution_102 = torch.ops.aten.convolution.default(add_337, arg125_1, arg126_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  add_337 = arg125_1 = arg126_1 = None
        view_289 = torch.ops.aten.view.default(convolution_102, [8, 1, 384, 196])
        var_mean_97 = torch.ops.aten.var_mean.correction(view_289, [2, 3], correction = 0, keepdim = True)
        getitem_194 = var_mean_97[0]
        getitem_195 = var_mean_97[1];  var_mean_97 = None
        add_338 = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        sub_145 = torch.ops.aten.sub.Tensor(view_289, getitem_195);  view_289 = getitem_195 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_145, rsqrt_97);  sub_145 = rsqrt_97 = None
        view_290 = torch.ops.aten.view.default(mul_434, [8, 384, 14, 14]);  mul_434 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg128_1, 0);  arg128_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(arg127_1, 0);  arg127_1 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
        mul_435 = torch.ops.aten.mul.Tensor(view_290, unsqueeze_581);  view_290 = unsqueeze_581 = None
        add_339 = torch.ops.aten.add.Tensor(mul_435, unsqueeze_578);  mul_435 = unsqueeze_578 = None
        avg_pool2d_48 = torch.ops.aten.avg_pool2d.default(add_339, [3, 3], [1, 1], [1, 1], False, False)
        sub_146 = torch.ops.aten.sub.Tensor(avg_pool2d_48, add_339);  avg_pool2d_48 = add_339 = None
        view_291 = torch.ops.aten.view.default(arg129_1, [384, 1, 1]);  arg129_1 = None
        mul_436 = torch.ops.aten.mul.Tensor(sub_146, view_291);  sub_146 = view_291 = None
        add_340 = torch.ops.aten.add.Tensor(convolution_102, mul_436);  convolution_102 = mul_436 = None
        view_292 = torch.ops.aten.view.default(add_340, [8, 1, 384, 196])
        var_mean_98 = torch.ops.aten.var_mean.correction(view_292, [2, 3], correction = 0, keepdim = True)
        getitem_196 = var_mean_98[0]
        getitem_197 = var_mean_98[1];  var_mean_98 = None
        add_341 = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_147 = torch.ops.aten.sub.Tensor(view_292, getitem_197);  view_292 = getitem_197 = None
        mul_437 = torch.ops.aten.mul.Tensor(sub_147, rsqrt_98);  sub_147 = rsqrt_98 = None
        view_293 = torch.ops.aten.view.default(mul_437, [8, 384, 14, 14]);  mul_437 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg131_1, 0);  arg131_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(arg130_1, 0);  arg130_1 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
        mul_438 = torch.ops.aten.mul.Tensor(view_293, unsqueeze_587);  view_293 = unsqueeze_587 = None
        add_342 = torch.ops.aten.add.Tensor(mul_438, unsqueeze_584);  mul_438 = unsqueeze_584 = None
        convolution_103 = torch.ops.aten.convolution.default(add_342, arg132_1, arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_342 = arg132_1 = arg133_1 = None
        mul_439 = torch.ops.aten.mul.Tensor(convolution_103, 0.5)
        mul_440 = torch.ops.aten.mul.Tensor(convolution_103, 0.7071067811865476);  convolution_103 = None
        erf_48 = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_343 = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_439, add_343);  mul_439 = add_343 = None
        convolution_104 = torch.ops.aten.convolution.default(mul_441, arg134_1, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_441 = arg134_1 = arg135_1 = None
        view_294 = torch.ops.aten.view.default(arg136_1, [384, 1, 1]);  arg136_1 = None
        mul_442 = torch.ops.aten.mul.Tensor(convolution_104, view_294);  convolution_104 = view_294 = None
        add_344 = torch.ops.aten.add.Tensor(add_340, mul_442);  add_340 = mul_442 = None
        view_295 = torch.ops.aten.view.default(add_344, [8, 1, 384, 196])
        var_mean_99 = torch.ops.aten.var_mean.correction(view_295, [2, 3], correction = 0, keepdim = True)
        getitem_198 = var_mean_99[0]
        getitem_199 = var_mean_99[1];  var_mean_99 = None
        add_345 = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
        sub_148 = torch.ops.aten.sub.Tensor(view_295, getitem_199);  view_295 = getitem_199 = None
        mul_443 = torch.ops.aten.mul.Tensor(sub_148, rsqrt_99);  sub_148 = rsqrt_99 = None
        view_296 = torch.ops.aten.view.default(mul_443, [8, 384, 14, 14]);  mul_443 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg138_1, 0);  arg138_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(arg137_1, 0);  arg137_1 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
        mul_444 = torch.ops.aten.mul.Tensor(view_296, unsqueeze_593);  view_296 = unsqueeze_593 = None
        add_346 = torch.ops.aten.add.Tensor(mul_444, unsqueeze_590);  mul_444 = unsqueeze_590 = None
        avg_pool2d_49 = torch.ops.aten.avg_pool2d.default(add_346, [3, 3], [1, 1], [1, 1], False, False)
        sub_149 = torch.ops.aten.sub.Tensor(avg_pool2d_49, add_346);  avg_pool2d_49 = add_346 = None
        view_297 = torch.ops.aten.view.default(arg139_1, [384, 1, 1]);  arg139_1 = None
        mul_445 = torch.ops.aten.mul.Tensor(sub_149, view_297);  sub_149 = view_297 = None
        add_347 = torch.ops.aten.add.Tensor(add_344, mul_445);  add_344 = mul_445 = None
        view_298 = torch.ops.aten.view.default(add_347, [8, 1, 384, 196])
        var_mean_100 = torch.ops.aten.var_mean.correction(view_298, [2, 3], correction = 0, keepdim = True)
        getitem_200 = var_mean_100[0]
        getitem_201 = var_mean_100[1];  var_mean_100 = None
        add_348 = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        sub_150 = torch.ops.aten.sub.Tensor(view_298, getitem_201);  view_298 = getitem_201 = None
        mul_446 = torch.ops.aten.mul.Tensor(sub_150, rsqrt_100);  sub_150 = rsqrt_100 = None
        view_299 = torch.ops.aten.view.default(mul_446, [8, 384, 14, 14]);  mul_446 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(arg141_1, 0);  arg141_1 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(arg140_1, 0);  arg140_1 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
        mul_447 = torch.ops.aten.mul.Tensor(view_299, unsqueeze_599);  view_299 = unsqueeze_599 = None
        add_349 = torch.ops.aten.add.Tensor(mul_447, unsqueeze_596);  mul_447 = unsqueeze_596 = None
        convolution_105 = torch.ops.aten.convolution.default(add_349, arg142_1, arg143_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_349 = arg142_1 = arg143_1 = None
        mul_448 = torch.ops.aten.mul.Tensor(convolution_105, 0.5)
        mul_449 = torch.ops.aten.mul.Tensor(convolution_105, 0.7071067811865476);  convolution_105 = None
        erf_49 = torch.ops.aten.erf.default(mul_449);  mul_449 = None
        add_350 = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_450 = torch.ops.aten.mul.Tensor(mul_448, add_350);  mul_448 = add_350 = None
        convolution_106 = torch.ops.aten.convolution.default(mul_450, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_450 = arg144_1 = arg145_1 = None
        view_300 = torch.ops.aten.view.default(arg146_1, [384, 1, 1]);  arg146_1 = None
        mul_451 = torch.ops.aten.mul.Tensor(convolution_106, view_300);  convolution_106 = view_300 = None
        add_351 = torch.ops.aten.add.Tensor(add_347, mul_451);  add_347 = mul_451 = None
        view_301 = torch.ops.aten.view.default(add_351, [8, 1, 384, 196])
        var_mean_101 = torch.ops.aten.var_mean.correction(view_301, [2, 3], correction = 0, keepdim = True)
        getitem_202 = var_mean_101[0]
        getitem_203 = var_mean_101[1];  var_mean_101 = None
        add_352 = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        sub_151 = torch.ops.aten.sub.Tensor(view_301, getitem_203);  view_301 = getitem_203 = None
        mul_452 = torch.ops.aten.mul.Tensor(sub_151, rsqrt_101);  sub_151 = rsqrt_101 = None
        view_302 = torch.ops.aten.view.default(mul_452, [8, 384, 14, 14]);  mul_452 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg148_1, 0);  arg148_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(arg147_1, 0);  arg147_1 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
        mul_453 = torch.ops.aten.mul.Tensor(view_302, unsqueeze_605);  view_302 = unsqueeze_605 = None
        add_353 = torch.ops.aten.add.Tensor(mul_453, unsqueeze_602);  mul_453 = unsqueeze_602 = None
        avg_pool2d_50 = torch.ops.aten.avg_pool2d.default(add_353, [3, 3], [1, 1], [1, 1], False, False)
        sub_152 = torch.ops.aten.sub.Tensor(avg_pool2d_50, add_353);  avg_pool2d_50 = add_353 = None
        view_303 = torch.ops.aten.view.default(arg149_1, [384, 1, 1]);  arg149_1 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_152, view_303);  sub_152 = view_303 = None
        add_354 = torch.ops.aten.add.Tensor(add_351, mul_454);  add_351 = mul_454 = None
        view_304 = torch.ops.aten.view.default(add_354, [8, 1, 384, 196])
        var_mean_102 = torch.ops.aten.var_mean.correction(view_304, [2, 3], correction = 0, keepdim = True)
        getitem_204 = var_mean_102[0]
        getitem_205 = var_mean_102[1];  var_mean_102 = None
        add_355 = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        sub_153 = torch.ops.aten.sub.Tensor(view_304, getitem_205);  view_304 = getitem_205 = None
        mul_455 = torch.ops.aten.mul.Tensor(sub_153, rsqrt_102);  sub_153 = rsqrt_102 = None
        view_305 = torch.ops.aten.view.default(mul_455, [8, 384, 14, 14]);  mul_455 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg151_1, 0);  arg151_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(arg150_1, 0);  arg150_1 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
        mul_456 = torch.ops.aten.mul.Tensor(view_305, unsqueeze_611);  view_305 = unsqueeze_611 = None
        add_356 = torch.ops.aten.add.Tensor(mul_456, unsqueeze_608);  mul_456 = unsqueeze_608 = None
        convolution_107 = torch.ops.aten.convolution.default(add_356, arg152_1, arg153_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_356 = arg152_1 = arg153_1 = None
        mul_457 = torch.ops.aten.mul.Tensor(convolution_107, 0.5)
        mul_458 = torch.ops.aten.mul.Tensor(convolution_107, 0.7071067811865476);  convolution_107 = None
        erf_50 = torch.ops.aten.erf.default(mul_458);  mul_458 = None
        add_357 = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_459 = torch.ops.aten.mul.Tensor(mul_457, add_357);  mul_457 = add_357 = None
        convolution_108 = torch.ops.aten.convolution.default(mul_459, arg154_1, arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_459 = arg154_1 = arg155_1 = None
        view_306 = torch.ops.aten.view.default(arg156_1, [384, 1, 1]);  arg156_1 = None
        mul_460 = torch.ops.aten.mul.Tensor(convolution_108, view_306);  convolution_108 = view_306 = None
        add_358 = torch.ops.aten.add.Tensor(add_354, mul_460);  add_354 = mul_460 = None
        view_307 = torch.ops.aten.view.default(add_358, [8, 1, 384, 196])
        var_mean_103 = torch.ops.aten.var_mean.correction(view_307, [2, 3], correction = 0, keepdim = True)
        getitem_206 = var_mean_103[0]
        getitem_207 = var_mean_103[1];  var_mean_103 = None
        add_359 = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        sub_154 = torch.ops.aten.sub.Tensor(view_307, getitem_207);  view_307 = getitem_207 = None
        mul_461 = torch.ops.aten.mul.Tensor(sub_154, rsqrt_103);  sub_154 = rsqrt_103 = None
        view_308 = torch.ops.aten.view.default(mul_461, [8, 384, 14, 14]);  mul_461 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(arg158_1, 0);  arg158_1 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(arg157_1, 0);  arg157_1 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
        mul_462 = torch.ops.aten.mul.Tensor(view_308, unsqueeze_617);  view_308 = unsqueeze_617 = None
        add_360 = torch.ops.aten.add.Tensor(mul_462, unsqueeze_614);  mul_462 = unsqueeze_614 = None
        avg_pool2d_51 = torch.ops.aten.avg_pool2d.default(add_360, [3, 3], [1, 1], [1, 1], False, False)
        sub_155 = torch.ops.aten.sub.Tensor(avg_pool2d_51, add_360);  avg_pool2d_51 = add_360 = None
        view_309 = torch.ops.aten.view.default(arg159_1, [384, 1, 1]);  arg159_1 = None
        mul_463 = torch.ops.aten.mul.Tensor(sub_155, view_309);  sub_155 = view_309 = None
        add_361 = torch.ops.aten.add.Tensor(add_358, mul_463);  add_358 = mul_463 = None
        view_310 = torch.ops.aten.view.default(add_361, [8, 1, 384, 196])
        var_mean_104 = torch.ops.aten.var_mean.correction(view_310, [2, 3], correction = 0, keepdim = True)
        getitem_208 = var_mean_104[0]
        getitem_209 = var_mean_104[1];  var_mean_104 = None
        add_362 = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
        sub_156 = torch.ops.aten.sub.Tensor(view_310, getitem_209);  view_310 = getitem_209 = None
        mul_464 = torch.ops.aten.mul.Tensor(sub_156, rsqrt_104);  sub_156 = rsqrt_104 = None
        view_311 = torch.ops.aten.view.default(mul_464, [8, 384, 14, 14]);  mul_464 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(arg161_1, 0);  arg161_1 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(arg160_1, 0);  arg160_1 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
        mul_465 = torch.ops.aten.mul.Tensor(view_311, unsqueeze_623);  view_311 = unsqueeze_623 = None
        add_363 = torch.ops.aten.add.Tensor(mul_465, unsqueeze_620);  mul_465 = unsqueeze_620 = None
        convolution_109 = torch.ops.aten.convolution.default(add_363, arg162_1, arg163_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_363 = arg162_1 = arg163_1 = None
        mul_466 = torch.ops.aten.mul.Tensor(convolution_109, 0.5)
        mul_467 = torch.ops.aten.mul.Tensor(convolution_109, 0.7071067811865476);  convolution_109 = None
        erf_51 = torch.ops.aten.erf.default(mul_467);  mul_467 = None
        add_364 = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_468 = torch.ops.aten.mul.Tensor(mul_466, add_364);  mul_466 = add_364 = None
        convolution_110 = torch.ops.aten.convolution.default(mul_468, arg164_1, arg165_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_468 = arg164_1 = arg165_1 = None
        view_312 = torch.ops.aten.view.default(arg166_1, [384, 1, 1]);  arg166_1 = None
        mul_469 = torch.ops.aten.mul.Tensor(convolution_110, view_312);  convolution_110 = view_312 = None
        add_365 = torch.ops.aten.add.Tensor(add_361, mul_469);  add_361 = mul_469 = None
        view_313 = torch.ops.aten.view.default(add_365, [8, 1, 384, 196])
        var_mean_105 = torch.ops.aten.var_mean.correction(view_313, [2, 3], correction = 0, keepdim = True)
        getitem_210 = var_mean_105[0]
        getitem_211 = var_mean_105[1];  var_mean_105 = None
        add_366 = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
        sub_157 = torch.ops.aten.sub.Tensor(view_313, getitem_211);  view_313 = getitem_211 = None
        mul_470 = torch.ops.aten.mul.Tensor(sub_157, rsqrt_105);  sub_157 = rsqrt_105 = None
        view_314 = torch.ops.aten.view.default(mul_470, [8, 384, 14, 14]);  mul_470 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(arg168_1, 0);  arg168_1 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(arg167_1, 0);  arg167_1 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
        mul_471 = torch.ops.aten.mul.Tensor(view_314, unsqueeze_629);  view_314 = unsqueeze_629 = None
        add_367 = torch.ops.aten.add.Tensor(mul_471, unsqueeze_626);  mul_471 = unsqueeze_626 = None
        avg_pool2d_52 = torch.ops.aten.avg_pool2d.default(add_367, [3, 3], [1, 1], [1, 1], False, False)
        sub_158 = torch.ops.aten.sub.Tensor(avg_pool2d_52, add_367);  avg_pool2d_52 = add_367 = None
        view_315 = torch.ops.aten.view.default(arg169_1, [384, 1, 1]);  arg169_1 = None
        mul_472 = torch.ops.aten.mul.Tensor(sub_158, view_315);  sub_158 = view_315 = None
        add_368 = torch.ops.aten.add.Tensor(add_365, mul_472);  add_365 = mul_472 = None
        view_316 = torch.ops.aten.view.default(add_368, [8, 1, 384, 196])
        var_mean_106 = torch.ops.aten.var_mean.correction(view_316, [2, 3], correction = 0, keepdim = True)
        getitem_212 = var_mean_106[0]
        getitem_213 = var_mean_106[1];  var_mean_106 = None
        add_369 = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_106 = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
        sub_159 = torch.ops.aten.sub.Tensor(view_316, getitem_213);  view_316 = getitem_213 = None
        mul_473 = torch.ops.aten.mul.Tensor(sub_159, rsqrt_106);  sub_159 = rsqrt_106 = None
        view_317 = torch.ops.aten.view.default(mul_473, [8, 384, 14, 14]);  mul_473 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(arg171_1, 0);  arg171_1 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(arg170_1, 0);  arg170_1 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
        mul_474 = torch.ops.aten.mul.Tensor(view_317, unsqueeze_635);  view_317 = unsqueeze_635 = None
        add_370 = torch.ops.aten.add.Tensor(mul_474, unsqueeze_632);  mul_474 = unsqueeze_632 = None
        convolution_111 = torch.ops.aten.convolution.default(add_370, arg172_1, arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_370 = arg172_1 = arg173_1 = None
        mul_475 = torch.ops.aten.mul.Tensor(convolution_111, 0.5)
        mul_476 = torch.ops.aten.mul.Tensor(convolution_111, 0.7071067811865476);  convolution_111 = None
        erf_52 = torch.ops.aten.erf.default(mul_476);  mul_476 = None
        add_371 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_477 = torch.ops.aten.mul.Tensor(mul_475, add_371);  mul_475 = add_371 = None
        convolution_112 = torch.ops.aten.convolution.default(mul_477, arg174_1, arg175_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_477 = arg174_1 = arg175_1 = None
        view_318 = torch.ops.aten.view.default(arg176_1, [384, 1, 1]);  arg176_1 = None
        mul_478 = torch.ops.aten.mul.Tensor(convolution_112, view_318);  convolution_112 = view_318 = None
        add_372 = torch.ops.aten.add.Tensor(add_368, mul_478);  add_368 = mul_478 = None
        view_319 = torch.ops.aten.view.default(add_372, [8, 1, 384, 196])
        var_mean_107 = torch.ops.aten.var_mean.correction(view_319, [2, 3], correction = 0, keepdim = True)
        getitem_214 = var_mean_107[0]
        getitem_215 = var_mean_107[1];  var_mean_107 = None
        add_373 = torch.ops.aten.add.Tensor(getitem_214, 1e-05);  getitem_214 = None
        rsqrt_107 = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
        sub_160 = torch.ops.aten.sub.Tensor(view_319, getitem_215);  view_319 = getitem_215 = None
        mul_479 = torch.ops.aten.mul.Tensor(sub_160, rsqrt_107);  sub_160 = rsqrt_107 = None
        view_320 = torch.ops.aten.view.default(mul_479, [8, 384, 14, 14]);  mul_479 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(arg178_1, 0);  arg178_1 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(arg177_1, 0);  arg177_1 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
        mul_480 = torch.ops.aten.mul.Tensor(view_320, unsqueeze_641);  view_320 = unsqueeze_641 = None
        add_374 = torch.ops.aten.add.Tensor(mul_480, unsqueeze_638);  mul_480 = unsqueeze_638 = None
        avg_pool2d_53 = torch.ops.aten.avg_pool2d.default(add_374, [3, 3], [1, 1], [1, 1], False, False)
        sub_161 = torch.ops.aten.sub.Tensor(avg_pool2d_53, add_374);  avg_pool2d_53 = add_374 = None
        view_321 = torch.ops.aten.view.default(arg179_1, [384, 1, 1]);  arg179_1 = None
        mul_481 = torch.ops.aten.mul.Tensor(sub_161, view_321);  sub_161 = view_321 = None
        add_375 = torch.ops.aten.add.Tensor(add_372, mul_481);  add_372 = mul_481 = None
        view_322 = torch.ops.aten.view.default(add_375, [8, 1, 384, 196])
        var_mean_108 = torch.ops.aten.var_mean.correction(view_322, [2, 3], correction = 0, keepdim = True)
        getitem_216 = var_mean_108[0]
        getitem_217 = var_mean_108[1];  var_mean_108 = None
        add_376 = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_108 = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_162 = torch.ops.aten.sub.Tensor(view_322, getitem_217);  view_322 = getitem_217 = None
        mul_482 = torch.ops.aten.mul.Tensor(sub_162, rsqrt_108);  sub_162 = rsqrt_108 = None
        view_323 = torch.ops.aten.view.default(mul_482, [8, 384, 14, 14]);  mul_482 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(arg181_1, 0);  arg181_1 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(arg180_1, 0);  arg180_1 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
        mul_483 = torch.ops.aten.mul.Tensor(view_323, unsqueeze_647);  view_323 = unsqueeze_647 = None
        add_377 = torch.ops.aten.add.Tensor(mul_483, unsqueeze_644);  mul_483 = unsqueeze_644 = None
        convolution_113 = torch.ops.aten.convolution.default(add_377, arg182_1, arg183_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_377 = arg182_1 = arg183_1 = None
        mul_484 = torch.ops.aten.mul.Tensor(convolution_113, 0.5)
        mul_485 = torch.ops.aten.mul.Tensor(convolution_113, 0.7071067811865476);  convolution_113 = None
        erf_53 = torch.ops.aten.erf.default(mul_485);  mul_485 = None
        add_378 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_486 = torch.ops.aten.mul.Tensor(mul_484, add_378);  mul_484 = add_378 = None
        convolution_114 = torch.ops.aten.convolution.default(mul_486, arg184_1, arg185_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_486 = arg184_1 = arg185_1 = None
        view_324 = torch.ops.aten.view.default(arg186_1, [384, 1, 1]);  arg186_1 = None
        mul_487 = torch.ops.aten.mul.Tensor(convolution_114, view_324);  convolution_114 = view_324 = None
        add_379 = torch.ops.aten.add.Tensor(add_375, mul_487);  add_375 = mul_487 = None
        view_325 = torch.ops.aten.view.default(add_379, [8, 1, 384, 196])
        var_mean_109 = torch.ops.aten.var_mean.correction(view_325, [2, 3], correction = 0, keepdim = True)
        getitem_218 = var_mean_109[0]
        getitem_219 = var_mean_109[1];  var_mean_109 = None
        add_380 = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_109 = torch.ops.aten.rsqrt.default(add_380);  add_380 = None
        sub_163 = torch.ops.aten.sub.Tensor(view_325, getitem_219);  view_325 = getitem_219 = None
        mul_488 = torch.ops.aten.mul.Tensor(sub_163, rsqrt_109);  sub_163 = rsqrt_109 = None
        view_326 = torch.ops.aten.view.default(mul_488, [8, 384, 14, 14]);  mul_488 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg188_1, 0);  arg188_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(arg187_1, 0);  arg187_1 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
        mul_489 = torch.ops.aten.mul.Tensor(view_326, unsqueeze_653);  view_326 = unsqueeze_653 = None
        add_381 = torch.ops.aten.add.Tensor(mul_489, unsqueeze_650);  mul_489 = unsqueeze_650 = None
        avg_pool2d_54 = torch.ops.aten.avg_pool2d.default(add_381, [3, 3], [1, 1], [1, 1], False, False)
        sub_164 = torch.ops.aten.sub.Tensor(avg_pool2d_54, add_381);  avg_pool2d_54 = add_381 = None
        view_327 = torch.ops.aten.view.default(arg189_1, [384, 1, 1]);  arg189_1 = None
        mul_490 = torch.ops.aten.mul.Tensor(sub_164, view_327);  sub_164 = view_327 = None
        add_382 = torch.ops.aten.add.Tensor(add_379, mul_490);  add_379 = mul_490 = None
        view_328 = torch.ops.aten.view.default(add_382, [8, 1, 384, 196])
        var_mean_110 = torch.ops.aten.var_mean.correction(view_328, [2, 3], correction = 0, keepdim = True)
        getitem_220 = var_mean_110[0]
        getitem_221 = var_mean_110[1];  var_mean_110 = None
        add_383 = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_110 = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
        sub_165 = torch.ops.aten.sub.Tensor(view_328, getitem_221);  view_328 = getitem_221 = None
        mul_491 = torch.ops.aten.mul.Tensor(sub_165, rsqrt_110);  sub_165 = rsqrt_110 = None
        view_329 = torch.ops.aten.view.default(mul_491, [8, 384, 14, 14]);  mul_491 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg191_1, 0);  arg191_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(arg190_1, 0);  arg190_1 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
        mul_492 = torch.ops.aten.mul.Tensor(view_329, unsqueeze_659);  view_329 = unsqueeze_659 = None
        add_384 = torch.ops.aten.add.Tensor(mul_492, unsqueeze_656);  mul_492 = unsqueeze_656 = None
        convolution_115 = torch.ops.aten.convolution.default(add_384, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_384 = arg192_1 = arg193_1 = None
        mul_493 = torch.ops.aten.mul.Tensor(convolution_115, 0.5)
        mul_494 = torch.ops.aten.mul.Tensor(convolution_115, 0.7071067811865476);  convolution_115 = None
        erf_54 = torch.ops.aten.erf.default(mul_494);  mul_494 = None
        add_385 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_495 = torch.ops.aten.mul.Tensor(mul_493, add_385);  mul_493 = add_385 = None
        convolution_116 = torch.ops.aten.convolution.default(mul_495, arg194_1, arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_495 = arg194_1 = arg195_1 = None
        view_330 = torch.ops.aten.view.default(arg196_1, [384, 1, 1]);  arg196_1 = None
        mul_496 = torch.ops.aten.mul.Tensor(convolution_116, view_330);  convolution_116 = view_330 = None
        add_386 = torch.ops.aten.add.Tensor(add_382, mul_496);  add_382 = mul_496 = None
        view_331 = torch.ops.aten.view.default(add_386, [8, 1, 384, 196])
        var_mean_111 = torch.ops.aten.var_mean.correction(view_331, [2, 3], correction = 0, keepdim = True)
        getitem_222 = var_mean_111[0]
        getitem_223 = var_mean_111[1];  var_mean_111 = None
        add_387 = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_111 = torch.ops.aten.rsqrt.default(add_387);  add_387 = None
        sub_166 = torch.ops.aten.sub.Tensor(view_331, getitem_223);  view_331 = getitem_223 = None
        mul_497 = torch.ops.aten.mul.Tensor(sub_166, rsqrt_111);  sub_166 = rsqrt_111 = None
        view_332 = torch.ops.aten.view.default(mul_497, [8, 384, 14, 14]);  mul_497 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(arg198_1, 0);  arg198_1 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(arg197_1, 0);  arg197_1 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
        mul_498 = torch.ops.aten.mul.Tensor(view_332, unsqueeze_665);  view_332 = unsqueeze_665 = None
        add_388 = torch.ops.aten.add.Tensor(mul_498, unsqueeze_662);  mul_498 = unsqueeze_662 = None
        avg_pool2d_55 = torch.ops.aten.avg_pool2d.default(add_388, [3, 3], [1, 1], [1, 1], False, False)
        sub_167 = torch.ops.aten.sub.Tensor(avg_pool2d_55, add_388);  avg_pool2d_55 = add_388 = None
        view_333 = torch.ops.aten.view.default(arg199_1, [384, 1, 1]);  arg199_1 = None
        mul_499 = torch.ops.aten.mul.Tensor(sub_167, view_333);  sub_167 = view_333 = None
        add_389 = torch.ops.aten.add.Tensor(add_386, mul_499);  add_386 = mul_499 = None
        view_334 = torch.ops.aten.view.default(add_389, [8, 1, 384, 196])
        var_mean_112 = torch.ops.aten.var_mean.correction(view_334, [2, 3], correction = 0, keepdim = True)
        getitem_224 = var_mean_112[0]
        getitem_225 = var_mean_112[1];  var_mean_112 = None
        add_390 = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_112 = torch.ops.aten.rsqrt.default(add_390);  add_390 = None
        sub_168 = torch.ops.aten.sub.Tensor(view_334, getitem_225);  view_334 = getitem_225 = None
        mul_500 = torch.ops.aten.mul.Tensor(sub_168, rsqrt_112);  sub_168 = rsqrt_112 = None
        view_335 = torch.ops.aten.view.default(mul_500, [8, 384, 14, 14]);  mul_500 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(arg201_1, 0);  arg201_1 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(arg200_1, 0);  arg200_1 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
        mul_501 = torch.ops.aten.mul.Tensor(view_335, unsqueeze_671);  view_335 = unsqueeze_671 = None
        add_391 = torch.ops.aten.add.Tensor(mul_501, unsqueeze_668);  mul_501 = unsqueeze_668 = None
        convolution_117 = torch.ops.aten.convolution.default(add_391, arg202_1, arg203_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_391 = arg202_1 = arg203_1 = None
        mul_502 = torch.ops.aten.mul.Tensor(convolution_117, 0.5)
        mul_503 = torch.ops.aten.mul.Tensor(convolution_117, 0.7071067811865476);  convolution_117 = None
        erf_55 = torch.ops.aten.erf.default(mul_503);  mul_503 = None
        add_392 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_504 = torch.ops.aten.mul.Tensor(mul_502, add_392);  mul_502 = add_392 = None
        convolution_118 = torch.ops.aten.convolution.default(mul_504, arg204_1, arg205_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_504 = arg204_1 = arg205_1 = None
        view_336 = torch.ops.aten.view.default(arg206_1, [384, 1, 1]);  arg206_1 = None
        mul_505 = torch.ops.aten.mul.Tensor(convolution_118, view_336);  convolution_118 = view_336 = None
        add_393 = torch.ops.aten.add.Tensor(add_389, mul_505);  add_389 = mul_505 = None
        view_337 = torch.ops.aten.view.default(add_393, [8, 1, 384, 196])
        var_mean_113 = torch.ops.aten.var_mean.correction(view_337, [2, 3], correction = 0, keepdim = True)
        getitem_226 = var_mean_113[0]
        getitem_227 = var_mean_113[1];  var_mean_113 = None
        add_394 = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_113 = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
        sub_169 = torch.ops.aten.sub.Tensor(view_337, getitem_227);  view_337 = getitem_227 = None
        mul_506 = torch.ops.aten.mul.Tensor(sub_169, rsqrt_113);  sub_169 = rsqrt_113 = None
        view_338 = torch.ops.aten.view.default(mul_506, [8, 384, 14, 14]);  mul_506 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(arg208_1, 0);  arg208_1 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(arg207_1, 0);  arg207_1 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
        mul_507 = torch.ops.aten.mul.Tensor(view_338, unsqueeze_677);  view_338 = unsqueeze_677 = None
        add_395 = torch.ops.aten.add.Tensor(mul_507, unsqueeze_674);  mul_507 = unsqueeze_674 = None
        avg_pool2d_56 = torch.ops.aten.avg_pool2d.default(add_395, [3, 3], [1, 1], [1, 1], False, False)
        sub_170 = torch.ops.aten.sub.Tensor(avg_pool2d_56, add_395);  avg_pool2d_56 = add_395 = None
        view_339 = torch.ops.aten.view.default(arg209_1, [384, 1, 1]);  arg209_1 = None
        mul_508 = torch.ops.aten.mul.Tensor(sub_170, view_339);  sub_170 = view_339 = None
        add_396 = torch.ops.aten.add.Tensor(add_393, mul_508);  add_393 = mul_508 = None
        view_340 = torch.ops.aten.view.default(add_396, [8, 1, 384, 196])
        var_mean_114 = torch.ops.aten.var_mean.correction(view_340, [2, 3], correction = 0, keepdim = True)
        getitem_228 = var_mean_114[0]
        getitem_229 = var_mean_114[1];  var_mean_114 = None
        add_397 = torch.ops.aten.add.Tensor(getitem_228, 1e-05);  getitem_228 = None
        rsqrt_114 = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
        sub_171 = torch.ops.aten.sub.Tensor(view_340, getitem_229);  view_340 = getitem_229 = None
        mul_509 = torch.ops.aten.mul.Tensor(sub_171, rsqrt_114);  sub_171 = rsqrt_114 = None
        view_341 = torch.ops.aten.view.default(mul_509, [8, 384, 14, 14]);  mul_509 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(arg211_1, 0);  arg211_1 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(arg210_1, 0);  arg210_1 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
        mul_510 = torch.ops.aten.mul.Tensor(view_341, unsqueeze_683);  view_341 = unsqueeze_683 = None
        add_398 = torch.ops.aten.add.Tensor(mul_510, unsqueeze_680);  mul_510 = unsqueeze_680 = None
        convolution_119 = torch.ops.aten.convolution.default(add_398, arg212_1, arg213_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_398 = arg212_1 = arg213_1 = None
        mul_511 = torch.ops.aten.mul.Tensor(convolution_119, 0.5)
        mul_512 = torch.ops.aten.mul.Tensor(convolution_119, 0.7071067811865476);  convolution_119 = None
        erf_56 = torch.ops.aten.erf.default(mul_512);  mul_512 = None
        add_399 = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_513 = torch.ops.aten.mul.Tensor(mul_511, add_399);  mul_511 = add_399 = None
        convolution_120 = torch.ops.aten.convolution.default(mul_513, arg214_1, arg215_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_513 = arg214_1 = arg215_1 = None
        view_342 = torch.ops.aten.view.default(arg216_1, [384, 1, 1]);  arg216_1 = None
        mul_514 = torch.ops.aten.mul.Tensor(convolution_120, view_342);  convolution_120 = view_342 = None
        add_400 = torch.ops.aten.add.Tensor(add_396, mul_514);  add_396 = mul_514 = None
        view_343 = torch.ops.aten.view.default(add_400, [8, 1, 384, 196])
        var_mean_115 = torch.ops.aten.var_mean.correction(view_343, [2, 3], correction = 0, keepdim = True)
        getitem_230 = var_mean_115[0]
        getitem_231 = var_mean_115[1];  var_mean_115 = None
        add_401 = torch.ops.aten.add.Tensor(getitem_230, 1e-05);  getitem_230 = None
        rsqrt_115 = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
        sub_172 = torch.ops.aten.sub.Tensor(view_343, getitem_231);  view_343 = getitem_231 = None
        mul_515 = torch.ops.aten.mul.Tensor(sub_172, rsqrt_115);  sub_172 = rsqrt_115 = None
        view_344 = torch.ops.aten.view.default(mul_515, [8, 384, 14, 14]);  mul_515 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg218_1, 0);  arg218_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(arg217_1, 0);  arg217_1 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
        mul_516 = torch.ops.aten.mul.Tensor(view_344, unsqueeze_689);  view_344 = unsqueeze_689 = None
        add_402 = torch.ops.aten.add.Tensor(mul_516, unsqueeze_686);  mul_516 = unsqueeze_686 = None
        avg_pool2d_57 = torch.ops.aten.avg_pool2d.default(add_402, [3, 3], [1, 1], [1, 1], False, False)
        sub_173 = torch.ops.aten.sub.Tensor(avg_pool2d_57, add_402);  avg_pool2d_57 = add_402 = None
        view_345 = torch.ops.aten.view.default(arg219_1, [384, 1, 1]);  arg219_1 = None
        mul_517 = torch.ops.aten.mul.Tensor(sub_173, view_345);  sub_173 = view_345 = None
        add_403 = torch.ops.aten.add.Tensor(add_400, mul_517);  add_400 = mul_517 = None
        view_346 = torch.ops.aten.view.default(add_403, [8, 1, 384, 196])
        var_mean_116 = torch.ops.aten.var_mean.correction(view_346, [2, 3], correction = 0, keepdim = True)
        getitem_232 = var_mean_116[0]
        getitem_233 = var_mean_116[1];  var_mean_116 = None
        add_404 = torch.ops.aten.add.Tensor(getitem_232, 1e-05);  getitem_232 = None
        rsqrt_116 = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        sub_174 = torch.ops.aten.sub.Tensor(view_346, getitem_233);  view_346 = getitem_233 = None
        mul_518 = torch.ops.aten.mul.Tensor(sub_174, rsqrt_116);  sub_174 = rsqrt_116 = None
        view_347 = torch.ops.aten.view.default(mul_518, [8, 384, 14, 14]);  mul_518 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(arg221_1, 0);  arg221_1 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(arg220_1, 0);  arg220_1 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
        mul_519 = torch.ops.aten.mul.Tensor(view_347, unsqueeze_695);  view_347 = unsqueeze_695 = None
        add_405 = torch.ops.aten.add.Tensor(mul_519, unsqueeze_692);  mul_519 = unsqueeze_692 = None
        convolution_121 = torch.ops.aten.convolution.default(add_405, arg222_1, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_405 = arg222_1 = arg223_1 = None
        mul_520 = torch.ops.aten.mul.Tensor(convolution_121, 0.5)
        mul_521 = torch.ops.aten.mul.Tensor(convolution_121, 0.7071067811865476);  convolution_121 = None
        erf_57 = torch.ops.aten.erf.default(mul_521);  mul_521 = None
        add_406 = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_522 = torch.ops.aten.mul.Tensor(mul_520, add_406);  mul_520 = add_406 = None
        convolution_122 = torch.ops.aten.convolution.default(mul_522, arg224_1, arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_522 = arg224_1 = arg225_1 = None
        view_348 = torch.ops.aten.view.default(arg226_1, [384, 1, 1]);  arg226_1 = None
        mul_523 = torch.ops.aten.mul.Tensor(convolution_122, view_348);  convolution_122 = view_348 = None
        add_407 = torch.ops.aten.add.Tensor(add_403, mul_523);  add_403 = mul_523 = None
        view_349 = torch.ops.aten.view.default(add_407, [8, 1, 384, 196])
        var_mean_117 = torch.ops.aten.var_mean.correction(view_349, [2, 3], correction = 0, keepdim = True)
        getitem_234 = var_mean_117[0]
        getitem_235 = var_mean_117[1];  var_mean_117 = None
        add_408 = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
        rsqrt_117 = torch.ops.aten.rsqrt.default(add_408);  add_408 = None
        sub_175 = torch.ops.aten.sub.Tensor(view_349, getitem_235);  view_349 = getitem_235 = None
        mul_524 = torch.ops.aten.mul.Tensor(sub_175, rsqrt_117);  sub_175 = rsqrt_117 = None
        view_350 = torch.ops.aten.view.default(mul_524, [8, 384, 14, 14]);  mul_524 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg228_1, 0);  arg228_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(arg227_1, 0);  arg227_1 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
        mul_525 = torch.ops.aten.mul.Tensor(view_350, unsqueeze_701);  view_350 = unsqueeze_701 = None
        add_409 = torch.ops.aten.add.Tensor(mul_525, unsqueeze_698);  mul_525 = unsqueeze_698 = None
        avg_pool2d_58 = torch.ops.aten.avg_pool2d.default(add_409, [3, 3], [1, 1], [1, 1], False, False)
        sub_176 = torch.ops.aten.sub.Tensor(avg_pool2d_58, add_409);  avg_pool2d_58 = add_409 = None
        view_351 = torch.ops.aten.view.default(arg229_1, [384, 1, 1]);  arg229_1 = None
        mul_526 = torch.ops.aten.mul.Tensor(sub_176, view_351);  sub_176 = view_351 = None
        add_410 = torch.ops.aten.add.Tensor(add_407, mul_526);  add_407 = mul_526 = None
        view_352 = torch.ops.aten.view.default(add_410, [8, 1, 384, 196])
        var_mean_118 = torch.ops.aten.var_mean.correction(view_352, [2, 3], correction = 0, keepdim = True)
        getitem_236 = var_mean_118[0]
        getitem_237 = var_mean_118[1];  var_mean_118 = None
        add_411 = torch.ops.aten.add.Tensor(getitem_236, 1e-05);  getitem_236 = None
        rsqrt_118 = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
        sub_177 = torch.ops.aten.sub.Tensor(view_352, getitem_237);  view_352 = getitem_237 = None
        mul_527 = torch.ops.aten.mul.Tensor(sub_177, rsqrt_118);  sub_177 = rsqrt_118 = None
        view_353 = torch.ops.aten.view.default(mul_527, [8, 384, 14, 14]);  mul_527 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg231_1, 0);  arg231_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(arg230_1, 0);  arg230_1 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
        mul_528 = torch.ops.aten.mul.Tensor(view_353, unsqueeze_707);  view_353 = unsqueeze_707 = None
        add_412 = torch.ops.aten.add.Tensor(mul_528, unsqueeze_704);  mul_528 = unsqueeze_704 = None
        convolution_123 = torch.ops.aten.convolution.default(add_412, arg232_1, arg233_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_412 = arg232_1 = arg233_1 = None
        mul_529 = torch.ops.aten.mul.Tensor(convolution_123, 0.5)
        mul_530 = torch.ops.aten.mul.Tensor(convolution_123, 0.7071067811865476);  convolution_123 = None
        erf_58 = torch.ops.aten.erf.default(mul_530);  mul_530 = None
        add_413 = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_531 = torch.ops.aten.mul.Tensor(mul_529, add_413);  mul_529 = add_413 = None
        convolution_124 = torch.ops.aten.convolution.default(mul_531, arg234_1, arg235_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_531 = arg234_1 = arg235_1 = None
        view_354 = torch.ops.aten.view.default(arg236_1, [384, 1, 1]);  arg236_1 = None
        mul_532 = torch.ops.aten.mul.Tensor(convolution_124, view_354);  convolution_124 = view_354 = None
        add_414 = torch.ops.aten.add.Tensor(add_410, mul_532);  add_410 = mul_532 = None
        view_355 = torch.ops.aten.view.default(add_414, [8, 1, 384, 196])
        var_mean_119 = torch.ops.aten.var_mean.correction(view_355, [2, 3], correction = 0, keepdim = True)
        getitem_238 = var_mean_119[0]
        getitem_239 = var_mean_119[1];  var_mean_119 = None
        add_415 = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_119 = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
        sub_178 = torch.ops.aten.sub.Tensor(view_355, getitem_239);  view_355 = getitem_239 = None
        mul_533 = torch.ops.aten.mul.Tensor(sub_178, rsqrt_119);  sub_178 = rsqrt_119 = None
        view_356 = torch.ops.aten.view.default(mul_533, [8, 384, 14, 14]);  mul_533 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg238_1, 0);  arg238_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(arg237_1, 0);  arg237_1 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
        mul_534 = torch.ops.aten.mul.Tensor(view_356, unsqueeze_713);  view_356 = unsqueeze_713 = None
        add_416 = torch.ops.aten.add.Tensor(mul_534, unsqueeze_710);  mul_534 = unsqueeze_710 = None
        avg_pool2d_59 = torch.ops.aten.avg_pool2d.default(add_416, [3, 3], [1, 1], [1, 1], False, False)
        sub_179 = torch.ops.aten.sub.Tensor(avg_pool2d_59, add_416);  avg_pool2d_59 = add_416 = None
        view_357 = torch.ops.aten.view.default(arg239_1, [384, 1, 1]);  arg239_1 = None
        mul_535 = torch.ops.aten.mul.Tensor(sub_179, view_357);  sub_179 = view_357 = None
        add_417 = torch.ops.aten.add.Tensor(add_414, mul_535);  add_414 = mul_535 = None
        view_358 = torch.ops.aten.view.default(add_417, [8, 1, 384, 196])
        var_mean_120 = torch.ops.aten.var_mean.correction(view_358, [2, 3], correction = 0, keepdim = True)
        getitem_240 = var_mean_120[0]
        getitem_241 = var_mean_120[1];  var_mean_120 = None
        add_418 = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_120 = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        sub_180 = torch.ops.aten.sub.Tensor(view_358, getitem_241);  view_358 = getitem_241 = None
        mul_536 = torch.ops.aten.mul.Tensor(sub_180, rsqrt_120);  sub_180 = rsqrt_120 = None
        view_359 = torch.ops.aten.view.default(mul_536, [8, 384, 14, 14]);  mul_536 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(arg241_1, 0);  arg241_1 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(arg240_1, 0);  arg240_1 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
        mul_537 = torch.ops.aten.mul.Tensor(view_359, unsqueeze_719);  view_359 = unsqueeze_719 = None
        add_419 = torch.ops.aten.add.Tensor(mul_537, unsqueeze_716);  mul_537 = unsqueeze_716 = None
        convolution_125 = torch.ops.aten.convolution.default(add_419, arg242_1, arg243_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_419 = arg242_1 = arg243_1 = None
        mul_538 = torch.ops.aten.mul.Tensor(convolution_125, 0.5)
        mul_539 = torch.ops.aten.mul.Tensor(convolution_125, 0.7071067811865476);  convolution_125 = None
        erf_59 = torch.ops.aten.erf.default(mul_539);  mul_539 = None
        add_420 = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_540 = torch.ops.aten.mul.Tensor(mul_538, add_420);  mul_538 = add_420 = None
        convolution_126 = torch.ops.aten.convolution.default(mul_540, arg244_1, arg245_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_540 = arg244_1 = arg245_1 = None
        view_360 = torch.ops.aten.view.default(arg246_1, [384, 1, 1]);  arg246_1 = None
        mul_541 = torch.ops.aten.mul.Tensor(convolution_126, view_360);  convolution_126 = view_360 = None
        add_421 = torch.ops.aten.add.Tensor(add_417, mul_541);  add_417 = mul_541 = None
        view_361 = torch.ops.aten.view.default(add_421, [8, 1, 384, 196])
        var_mean_121 = torch.ops.aten.var_mean.correction(view_361, [2, 3], correction = 0, keepdim = True)
        getitem_242 = var_mean_121[0]
        getitem_243 = var_mean_121[1];  var_mean_121 = None
        add_422 = torch.ops.aten.add.Tensor(getitem_242, 1e-05);  getitem_242 = None
        rsqrt_121 = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        sub_181 = torch.ops.aten.sub.Tensor(view_361, getitem_243);  view_361 = getitem_243 = None
        mul_542 = torch.ops.aten.mul.Tensor(sub_181, rsqrt_121);  sub_181 = rsqrt_121 = None
        view_362 = torch.ops.aten.view.default(mul_542, [8, 384, 14, 14]);  mul_542 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg248_1, 0);  arg248_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(arg247_1, 0);  arg247_1 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
        mul_543 = torch.ops.aten.mul.Tensor(view_362, unsqueeze_725);  view_362 = unsqueeze_725 = None
        add_423 = torch.ops.aten.add.Tensor(mul_543, unsqueeze_722);  mul_543 = unsqueeze_722 = None
        avg_pool2d_60 = torch.ops.aten.avg_pool2d.default(add_423, [3, 3], [1, 1], [1, 1], False, False)
        sub_182 = torch.ops.aten.sub.Tensor(avg_pool2d_60, add_423);  avg_pool2d_60 = add_423 = None
        view_363 = torch.ops.aten.view.default(arg249_1, [384, 1, 1]);  arg249_1 = None
        mul_544 = torch.ops.aten.mul.Tensor(sub_182, view_363);  sub_182 = view_363 = None
        add_424 = torch.ops.aten.add.Tensor(add_421, mul_544);  add_421 = mul_544 = None
        view_364 = torch.ops.aten.view.default(add_424, [8, 1, 384, 196])
        var_mean_122 = torch.ops.aten.var_mean.correction(view_364, [2, 3], correction = 0, keepdim = True)
        getitem_244 = var_mean_122[0]
        getitem_245 = var_mean_122[1];  var_mean_122 = None
        add_425 = torch.ops.aten.add.Tensor(getitem_244, 1e-05);  getitem_244 = None
        rsqrt_122 = torch.ops.aten.rsqrt.default(add_425);  add_425 = None
        sub_183 = torch.ops.aten.sub.Tensor(view_364, getitem_245);  view_364 = getitem_245 = None
        mul_545 = torch.ops.aten.mul.Tensor(sub_183, rsqrt_122);  sub_183 = rsqrt_122 = None
        view_365 = torch.ops.aten.view.default(mul_545, [8, 384, 14, 14]);  mul_545 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg251_1, 0);  arg251_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(arg250_1, 0);  arg250_1 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
        mul_546 = torch.ops.aten.mul.Tensor(view_365, unsqueeze_731);  view_365 = unsqueeze_731 = None
        add_426 = torch.ops.aten.add.Tensor(mul_546, unsqueeze_728);  mul_546 = unsqueeze_728 = None
        convolution_127 = torch.ops.aten.convolution.default(add_426, arg252_1, arg253_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_426 = arg252_1 = arg253_1 = None
        mul_547 = torch.ops.aten.mul.Tensor(convolution_127, 0.5)
        mul_548 = torch.ops.aten.mul.Tensor(convolution_127, 0.7071067811865476);  convolution_127 = None
        erf_60 = torch.ops.aten.erf.default(mul_548);  mul_548 = None
        add_427 = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_549 = torch.ops.aten.mul.Tensor(mul_547, add_427);  mul_547 = add_427 = None
        convolution_128 = torch.ops.aten.convolution.default(mul_549, arg254_1, arg255_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_549 = arg254_1 = arg255_1 = None
        view_366 = torch.ops.aten.view.default(arg256_1, [384, 1, 1]);  arg256_1 = None
        mul_550 = torch.ops.aten.mul.Tensor(convolution_128, view_366);  convolution_128 = view_366 = None
        add_428 = torch.ops.aten.add.Tensor(add_424, mul_550);  add_424 = mul_550 = None
        view_367 = torch.ops.aten.view.default(add_428, [8, 1, 384, 196])
        var_mean_123 = torch.ops.aten.var_mean.correction(view_367, [2, 3], correction = 0, keepdim = True)
        getitem_246 = var_mean_123[0]
        getitem_247 = var_mean_123[1];  var_mean_123 = None
        add_429 = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_123 = torch.ops.aten.rsqrt.default(add_429);  add_429 = None
        sub_184 = torch.ops.aten.sub.Tensor(view_367, getitem_247);  view_367 = getitem_247 = None
        mul_551 = torch.ops.aten.mul.Tensor(sub_184, rsqrt_123);  sub_184 = rsqrt_123 = None
        view_368 = torch.ops.aten.view.default(mul_551, [8, 384, 14, 14]);  mul_551 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg258_1, 0);  arg258_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(arg257_1, 0);  arg257_1 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
        mul_552 = torch.ops.aten.mul.Tensor(view_368, unsqueeze_737);  view_368 = unsqueeze_737 = None
        add_430 = torch.ops.aten.add.Tensor(mul_552, unsqueeze_734);  mul_552 = unsqueeze_734 = None
        avg_pool2d_61 = torch.ops.aten.avg_pool2d.default(add_430, [3, 3], [1, 1], [1, 1], False, False)
        sub_185 = torch.ops.aten.sub.Tensor(avg_pool2d_61, add_430);  avg_pool2d_61 = add_430 = None
        view_369 = torch.ops.aten.view.default(arg259_1, [384, 1, 1]);  arg259_1 = None
        mul_553 = torch.ops.aten.mul.Tensor(sub_185, view_369);  sub_185 = view_369 = None
        add_431 = torch.ops.aten.add.Tensor(add_428, mul_553);  add_428 = mul_553 = None
        view_370 = torch.ops.aten.view.default(add_431, [8, 1, 384, 196])
        var_mean_124 = torch.ops.aten.var_mean.correction(view_370, [2, 3], correction = 0, keepdim = True)
        getitem_248 = var_mean_124[0]
        getitem_249 = var_mean_124[1];  var_mean_124 = None
        add_432 = torch.ops.aten.add.Tensor(getitem_248, 1e-05);  getitem_248 = None
        rsqrt_124 = torch.ops.aten.rsqrt.default(add_432);  add_432 = None
        sub_186 = torch.ops.aten.sub.Tensor(view_370, getitem_249);  view_370 = getitem_249 = None
        mul_554 = torch.ops.aten.mul.Tensor(sub_186, rsqrt_124);  sub_186 = rsqrt_124 = None
        view_371 = torch.ops.aten.view.default(mul_554, [8, 384, 14, 14]);  mul_554 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(arg261_1, 0);  arg261_1 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(arg260_1, 0);  arg260_1 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
        mul_555 = torch.ops.aten.mul.Tensor(view_371, unsqueeze_743);  view_371 = unsqueeze_743 = None
        add_433 = torch.ops.aten.add.Tensor(mul_555, unsqueeze_740);  mul_555 = unsqueeze_740 = None
        convolution_129 = torch.ops.aten.convolution.default(add_433, arg262_1, arg263_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_433 = arg262_1 = arg263_1 = None
        mul_556 = torch.ops.aten.mul.Tensor(convolution_129, 0.5)
        mul_557 = torch.ops.aten.mul.Tensor(convolution_129, 0.7071067811865476);  convolution_129 = None
        erf_61 = torch.ops.aten.erf.default(mul_557);  mul_557 = None
        add_434 = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_558 = torch.ops.aten.mul.Tensor(mul_556, add_434);  mul_556 = add_434 = None
        convolution_130 = torch.ops.aten.convolution.default(mul_558, arg264_1, arg265_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_558 = arg264_1 = arg265_1 = None
        view_372 = torch.ops.aten.view.default(arg266_1, [384, 1, 1]);  arg266_1 = None
        mul_559 = torch.ops.aten.mul.Tensor(convolution_130, view_372);  convolution_130 = view_372 = None
        add_435 = torch.ops.aten.add.Tensor(add_431, mul_559);  add_431 = mul_559 = None
        view_373 = torch.ops.aten.view.default(add_435, [8, 1, 384, 196])
        var_mean_125 = torch.ops.aten.var_mean.correction(view_373, [2, 3], correction = 0, keepdim = True)
        getitem_250 = var_mean_125[0]
        getitem_251 = var_mean_125[1];  var_mean_125 = None
        add_436 = torch.ops.aten.add.Tensor(getitem_250, 1e-05);  getitem_250 = None
        rsqrt_125 = torch.ops.aten.rsqrt.default(add_436);  add_436 = None
        sub_187 = torch.ops.aten.sub.Tensor(view_373, getitem_251);  view_373 = getitem_251 = None
        mul_560 = torch.ops.aten.mul.Tensor(sub_187, rsqrt_125);  sub_187 = rsqrt_125 = None
        view_374 = torch.ops.aten.view.default(mul_560, [8, 384, 14, 14]);  mul_560 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg268_1, 0);  arg268_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(arg267_1, 0);  arg267_1 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
        mul_561 = torch.ops.aten.mul.Tensor(view_374, unsqueeze_749);  view_374 = unsqueeze_749 = None
        add_437 = torch.ops.aten.add.Tensor(mul_561, unsqueeze_746);  mul_561 = unsqueeze_746 = None
        avg_pool2d_62 = torch.ops.aten.avg_pool2d.default(add_437, [3, 3], [1, 1], [1, 1], False, False)
        sub_188 = torch.ops.aten.sub.Tensor(avg_pool2d_62, add_437);  avg_pool2d_62 = add_437 = None
        view_375 = torch.ops.aten.view.default(arg269_1, [384, 1, 1]);  arg269_1 = None
        mul_562 = torch.ops.aten.mul.Tensor(sub_188, view_375);  sub_188 = view_375 = None
        add_438 = torch.ops.aten.add.Tensor(add_435, mul_562);  add_435 = mul_562 = None
        view_376 = torch.ops.aten.view.default(add_438, [8, 1, 384, 196])
        var_mean_126 = torch.ops.aten.var_mean.correction(view_376, [2, 3], correction = 0, keepdim = True)
        getitem_252 = var_mean_126[0]
        getitem_253 = var_mean_126[1];  var_mean_126 = None
        add_439 = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_126 = torch.ops.aten.rsqrt.default(add_439);  add_439 = None
        sub_189 = torch.ops.aten.sub.Tensor(view_376, getitem_253);  view_376 = getitem_253 = None
        mul_563 = torch.ops.aten.mul.Tensor(sub_189, rsqrt_126);  sub_189 = rsqrt_126 = None
        view_377 = torch.ops.aten.view.default(mul_563, [8, 384, 14, 14]);  mul_563 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg271_1, 0);  arg271_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(arg270_1, 0);  arg270_1 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
        mul_564 = torch.ops.aten.mul.Tensor(view_377, unsqueeze_755);  view_377 = unsqueeze_755 = None
        add_440 = torch.ops.aten.add.Tensor(mul_564, unsqueeze_752);  mul_564 = unsqueeze_752 = None
        convolution_131 = torch.ops.aten.convolution.default(add_440, arg272_1, arg273_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_440 = arg272_1 = arg273_1 = None
        mul_565 = torch.ops.aten.mul.Tensor(convolution_131, 0.5)
        mul_566 = torch.ops.aten.mul.Tensor(convolution_131, 0.7071067811865476);  convolution_131 = None
        erf_62 = torch.ops.aten.erf.default(mul_566);  mul_566 = None
        add_441 = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_567 = torch.ops.aten.mul.Tensor(mul_565, add_441);  mul_565 = add_441 = None
        convolution_132 = torch.ops.aten.convolution.default(mul_567, arg274_1, arg275_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_567 = arg274_1 = arg275_1 = None
        view_378 = torch.ops.aten.view.default(arg276_1, [384, 1, 1]);  arg276_1 = None
        mul_568 = torch.ops.aten.mul.Tensor(convolution_132, view_378);  convolution_132 = view_378 = None
        add_442 = torch.ops.aten.add.Tensor(add_438, mul_568);  add_438 = mul_568 = None
        view_379 = torch.ops.aten.view.default(add_442, [8, 1, 384, 196])
        var_mean_127 = torch.ops.aten.var_mean.correction(view_379, [2, 3], correction = 0, keepdim = True)
        getitem_254 = var_mean_127[0]
        getitem_255 = var_mean_127[1];  var_mean_127 = None
        add_443 = torch.ops.aten.add.Tensor(getitem_254, 1e-05);  getitem_254 = None
        rsqrt_127 = torch.ops.aten.rsqrt.default(add_443);  add_443 = None
        sub_190 = torch.ops.aten.sub.Tensor(view_379, getitem_255);  view_379 = getitem_255 = None
        mul_569 = torch.ops.aten.mul.Tensor(sub_190, rsqrt_127);  sub_190 = rsqrt_127 = None
        view_380 = torch.ops.aten.view.default(mul_569, [8, 384, 14, 14]);  mul_569 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg278_1, 0);  arg278_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(arg277_1, 0);  arg277_1 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
        mul_570 = torch.ops.aten.mul.Tensor(view_380, unsqueeze_761);  view_380 = unsqueeze_761 = None
        add_444 = torch.ops.aten.add.Tensor(mul_570, unsqueeze_758);  mul_570 = unsqueeze_758 = None
        avg_pool2d_63 = torch.ops.aten.avg_pool2d.default(add_444, [3, 3], [1, 1], [1, 1], False, False)
        sub_191 = torch.ops.aten.sub.Tensor(avg_pool2d_63, add_444);  avg_pool2d_63 = add_444 = None
        view_381 = torch.ops.aten.view.default(arg279_1, [384, 1, 1]);  arg279_1 = None
        mul_571 = torch.ops.aten.mul.Tensor(sub_191, view_381);  sub_191 = view_381 = None
        add_445 = torch.ops.aten.add.Tensor(add_442, mul_571);  add_442 = mul_571 = None
        view_382 = torch.ops.aten.view.default(add_445, [8, 1, 384, 196])
        var_mean_128 = torch.ops.aten.var_mean.correction(view_382, [2, 3], correction = 0, keepdim = True)
        getitem_256 = var_mean_128[0]
        getitem_257 = var_mean_128[1];  var_mean_128 = None
        add_446 = torch.ops.aten.add.Tensor(getitem_256, 1e-05);  getitem_256 = None
        rsqrt_128 = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
        sub_192 = torch.ops.aten.sub.Tensor(view_382, getitem_257);  view_382 = getitem_257 = None
        mul_572 = torch.ops.aten.mul.Tensor(sub_192, rsqrt_128);  sub_192 = rsqrt_128 = None
        view_383 = torch.ops.aten.view.default(mul_572, [8, 384, 14, 14]);  mul_572 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(arg281_1, 0);  arg281_1 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(arg280_1, 0);  arg280_1 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
        mul_573 = torch.ops.aten.mul.Tensor(view_383, unsqueeze_767);  view_383 = unsqueeze_767 = None
        add_447 = torch.ops.aten.add.Tensor(mul_573, unsqueeze_764);  mul_573 = unsqueeze_764 = None
        convolution_133 = torch.ops.aten.convolution.default(add_447, arg282_1, arg283_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_447 = arg282_1 = arg283_1 = None
        mul_574 = torch.ops.aten.mul.Tensor(convolution_133, 0.5)
        mul_575 = torch.ops.aten.mul.Tensor(convolution_133, 0.7071067811865476);  convolution_133 = None
        erf_63 = torch.ops.aten.erf.default(mul_575);  mul_575 = None
        add_448 = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_576 = torch.ops.aten.mul.Tensor(mul_574, add_448);  mul_574 = add_448 = None
        convolution_134 = torch.ops.aten.convolution.default(mul_576, arg284_1, arg285_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_576 = arg284_1 = arg285_1 = None
        view_384 = torch.ops.aten.view.default(arg286_1, [384, 1, 1]);  arg286_1 = None
        mul_577 = torch.ops.aten.mul.Tensor(convolution_134, view_384);  convolution_134 = view_384 = None
        add_449 = torch.ops.aten.add.Tensor(add_445, mul_577);  add_445 = mul_577 = None
        view_385 = torch.ops.aten.view.default(add_449, [8, 1, 384, 196])
        var_mean_129 = torch.ops.aten.var_mean.correction(view_385, [2, 3], correction = 0, keepdim = True)
        getitem_258 = var_mean_129[0]
        getitem_259 = var_mean_129[1];  var_mean_129 = None
        add_450 = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
        rsqrt_129 = torch.ops.aten.rsqrt.default(add_450);  add_450 = None
        sub_193 = torch.ops.aten.sub.Tensor(view_385, getitem_259);  view_385 = getitem_259 = None
        mul_578 = torch.ops.aten.mul.Tensor(sub_193, rsqrt_129);  sub_193 = rsqrt_129 = None
        view_386 = torch.ops.aten.view.default(mul_578, [8, 384, 14, 14]);  mul_578 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg288_1, 0);  arg288_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(arg287_1, 0);  arg287_1 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
        mul_579 = torch.ops.aten.mul.Tensor(view_386, unsqueeze_773);  view_386 = unsqueeze_773 = None
        add_451 = torch.ops.aten.add.Tensor(mul_579, unsqueeze_770);  mul_579 = unsqueeze_770 = None
        avg_pool2d_64 = torch.ops.aten.avg_pool2d.default(add_451, [3, 3], [1, 1], [1, 1], False, False)
        sub_194 = torch.ops.aten.sub.Tensor(avg_pool2d_64, add_451);  avg_pool2d_64 = add_451 = None
        view_387 = torch.ops.aten.view.default(arg289_1, [384, 1, 1]);  arg289_1 = None
        mul_580 = torch.ops.aten.mul.Tensor(sub_194, view_387);  sub_194 = view_387 = None
        add_452 = torch.ops.aten.add.Tensor(add_449, mul_580);  add_449 = mul_580 = None
        view_388 = torch.ops.aten.view.default(add_452, [8, 1, 384, 196])
        var_mean_130 = torch.ops.aten.var_mean.correction(view_388, [2, 3], correction = 0, keepdim = True)
        getitem_260 = var_mean_130[0]
        getitem_261 = var_mean_130[1];  var_mean_130 = None
        add_453 = torch.ops.aten.add.Tensor(getitem_260, 1e-05);  getitem_260 = None
        rsqrt_130 = torch.ops.aten.rsqrt.default(add_453);  add_453 = None
        sub_195 = torch.ops.aten.sub.Tensor(view_388, getitem_261);  view_388 = getitem_261 = None
        mul_581 = torch.ops.aten.mul.Tensor(sub_195, rsqrt_130);  sub_195 = rsqrt_130 = None
        view_389 = torch.ops.aten.view.default(mul_581, [8, 384, 14, 14]);  mul_581 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg291_1, 0);  arg291_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(arg290_1, 0);  arg290_1 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
        mul_582 = torch.ops.aten.mul.Tensor(view_389, unsqueeze_779);  view_389 = unsqueeze_779 = None
        add_454 = torch.ops.aten.add.Tensor(mul_582, unsqueeze_776);  mul_582 = unsqueeze_776 = None
        convolution_135 = torch.ops.aten.convolution.default(add_454, arg292_1, arg293_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_454 = arg292_1 = arg293_1 = None
        mul_583 = torch.ops.aten.mul.Tensor(convolution_135, 0.5)
        mul_584 = torch.ops.aten.mul.Tensor(convolution_135, 0.7071067811865476);  convolution_135 = None
        erf_64 = torch.ops.aten.erf.default(mul_584);  mul_584 = None
        add_455 = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_585 = torch.ops.aten.mul.Tensor(mul_583, add_455);  mul_583 = add_455 = None
        convolution_136 = torch.ops.aten.convolution.default(mul_585, arg294_1, arg295_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_585 = arg294_1 = arg295_1 = None
        view_390 = torch.ops.aten.view.default(arg296_1, [384, 1, 1]);  arg296_1 = None
        mul_586 = torch.ops.aten.mul.Tensor(convolution_136, view_390);  convolution_136 = view_390 = None
        add_456 = torch.ops.aten.add.Tensor(add_452, mul_586);  add_452 = mul_586 = None
        view_391 = torch.ops.aten.view.default(add_456, [8, 1, 384, 196])
        var_mean_131 = torch.ops.aten.var_mean.correction(view_391, [2, 3], correction = 0, keepdim = True)
        getitem_262 = var_mean_131[0]
        getitem_263 = var_mean_131[1];  var_mean_131 = None
        add_457 = torch.ops.aten.add.Tensor(getitem_262, 1e-05);  getitem_262 = None
        rsqrt_131 = torch.ops.aten.rsqrt.default(add_457);  add_457 = None
        sub_196 = torch.ops.aten.sub.Tensor(view_391, getitem_263);  view_391 = getitem_263 = None
        mul_587 = torch.ops.aten.mul.Tensor(sub_196, rsqrt_131);  sub_196 = rsqrt_131 = None
        view_392 = torch.ops.aten.view.default(mul_587, [8, 384, 14, 14]);  mul_587 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg298_1, 0);  arg298_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(arg297_1, 0);  arg297_1 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
        mul_588 = torch.ops.aten.mul.Tensor(view_392, unsqueeze_785);  view_392 = unsqueeze_785 = None
        add_458 = torch.ops.aten.add.Tensor(mul_588, unsqueeze_782);  mul_588 = unsqueeze_782 = None
        avg_pool2d_65 = torch.ops.aten.avg_pool2d.default(add_458, [3, 3], [1, 1], [1, 1], False, False)
        sub_197 = torch.ops.aten.sub.Tensor(avg_pool2d_65, add_458);  avg_pool2d_65 = add_458 = None
        view_393 = torch.ops.aten.view.default(arg299_1, [384, 1, 1]);  arg299_1 = None
        mul_589 = torch.ops.aten.mul.Tensor(sub_197, view_393);  sub_197 = view_393 = None
        add_459 = torch.ops.aten.add.Tensor(add_456, mul_589);  add_456 = mul_589 = None
        view_394 = torch.ops.aten.view.default(add_459, [8, 1, 384, 196])
        var_mean_132 = torch.ops.aten.var_mean.correction(view_394, [2, 3], correction = 0, keepdim = True)
        getitem_264 = var_mean_132[0]
        getitem_265 = var_mean_132[1];  var_mean_132 = None
        add_460 = torch.ops.aten.add.Tensor(getitem_264, 1e-05);  getitem_264 = None
        rsqrt_132 = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
        sub_198 = torch.ops.aten.sub.Tensor(view_394, getitem_265);  view_394 = getitem_265 = None
        mul_590 = torch.ops.aten.mul.Tensor(sub_198, rsqrt_132);  sub_198 = rsqrt_132 = None
        view_395 = torch.ops.aten.view.default(mul_590, [8, 384, 14, 14]);  mul_590 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(arg301_1, 0);  arg301_1 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(arg300_1, 0);  arg300_1 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
        mul_591 = torch.ops.aten.mul.Tensor(view_395, unsqueeze_791);  view_395 = unsqueeze_791 = None
        add_461 = torch.ops.aten.add.Tensor(mul_591, unsqueeze_788);  mul_591 = unsqueeze_788 = None
        convolution_137 = torch.ops.aten.convolution.default(add_461, arg302_1, arg303_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_461 = arg302_1 = arg303_1 = None
        mul_592 = torch.ops.aten.mul.Tensor(convolution_137, 0.5)
        mul_593 = torch.ops.aten.mul.Tensor(convolution_137, 0.7071067811865476);  convolution_137 = None
        erf_65 = torch.ops.aten.erf.default(mul_593);  mul_593 = None
        add_462 = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_594 = torch.ops.aten.mul.Tensor(mul_592, add_462);  mul_592 = add_462 = None
        convolution_138 = torch.ops.aten.convolution.default(mul_594, arg304_1, arg305_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_594 = arg304_1 = arg305_1 = None
        view_396 = torch.ops.aten.view.default(arg306_1, [384, 1, 1]);  arg306_1 = None
        mul_595 = torch.ops.aten.mul.Tensor(convolution_138, view_396);  convolution_138 = view_396 = None
        add_463 = torch.ops.aten.add.Tensor(add_459, mul_595);  add_459 = mul_595 = None
        convolution_139 = torch.ops.aten.convolution.default(add_463, arg307_1, arg308_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  add_463 = arg307_1 = arg308_1 = None
        view_397 = torch.ops.aten.view.default(convolution_139, [8, 1, 768, 49])
        var_mean_133 = torch.ops.aten.var_mean.correction(view_397, [2, 3], correction = 0, keepdim = True)
        getitem_266 = var_mean_133[0]
        getitem_267 = var_mean_133[1];  var_mean_133 = None
        add_464 = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_133 = torch.ops.aten.rsqrt.default(add_464);  add_464 = None
        sub_199 = torch.ops.aten.sub.Tensor(view_397, getitem_267);  view_397 = getitem_267 = None
        mul_596 = torch.ops.aten.mul.Tensor(sub_199, rsqrt_133);  sub_199 = rsqrt_133 = None
        view_398 = torch.ops.aten.view.default(mul_596, [8, 768, 7, 7]);  mul_596 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg310_1, 0);  arg310_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(arg309_1, 0);  arg309_1 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
        mul_597 = torch.ops.aten.mul.Tensor(view_398, unsqueeze_797);  view_398 = unsqueeze_797 = None
        add_465 = torch.ops.aten.add.Tensor(mul_597, unsqueeze_794);  mul_597 = unsqueeze_794 = None
        avg_pool2d_66 = torch.ops.aten.avg_pool2d.default(add_465, [3, 3], [1, 1], [1, 1], False, False)
        sub_200 = torch.ops.aten.sub.Tensor(avg_pool2d_66, add_465);  avg_pool2d_66 = add_465 = None
        view_399 = torch.ops.aten.view.default(arg311_1, [768, 1, 1]);  arg311_1 = None
        mul_598 = torch.ops.aten.mul.Tensor(sub_200, view_399);  sub_200 = view_399 = None
        add_466 = torch.ops.aten.add.Tensor(convolution_139, mul_598);  convolution_139 = mul_598 = None
        view_400 = torch.ops.aten.view.default(add_466, [8, 1, 768, 49])
        var_mean_134 = torch.ops.aten.var_mean.correction(view_400, [2, 3], correction = 0, keepdim = True)
        getitem_268 = var_mean_134[0]
        getitem_269 = var_mean_134[1];  var_mean_134 = None
        add_467 = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_134 = torch.ops.aten.rsqrt.default(add_467);  add_467 = None
        sub_201 = torch.ops.aten.sub.Tensor(view_400, getitem_269);  view_400 = getitem_269 = None
        mul_599 = torch.ops.aten.mul.Tensor(sub_201, rsqrt_134);  sub_201 = rsqrt_134 = None
        view_401 = torch.ops.aten.view.default(mul_599, [8, 768, 7, 7]);  mul_599 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg313_1, 0);  arg313_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(arg312_1, 0);  arg312_1 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
        mul_600 = torch.ops.aten.mul.Tensor(view_401, unsqueeze_803);  view_401 = unsqueeze_803 = None
        add_468 = torch.ops.aten.add.Tensor(mul_600, unsqueeze_800);  mul_600 = unsqueeze_800 = None
        convolution_140 = torch.ops.aten.convolution.default(add_468, arg314_1, arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_468 = arg314_1 = arg315_1 = None
        mul_601 = torch.ops.aten.mul.Tensor(convolution_140, 0.5)
        mul_602 = torch.ops.aten.mul.Tensor(convolution_140, 0.7071067811865476);  convolution_140 = None
        erf_66 = torch.ops.aten.erf.default(mul_602);  mul_602 = None
        add_469 = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_603 = torch.ops.aten.mul.Tensor(mul_601, add_469);  mul_601 = add_469 = None
        convolution_141 = torch.ops.aten.convolution.default(mul_603, arg316_1, arg317_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_603 = arg316_1 = arg317_1 = None
        view_402 = torch.ops.aten.view.default(arg318_1, [768, 1, 1]);  arg318_1 = None
        mul_604 = torch.ops.aten.mul.Tensor(convolution_141, view_402);  convolution_141 = view_402 = None
        add_470 = torch.ops.aten.add.Tensor(add_466, mul_604);  add_466 = mul_604 = None
        view_403 = torch.ops.aten.view.default(add_470, [8, 1, 768, 49])
        var_mean_135 = torch.ops.aten.var_mean.correction(view_403, [2, 3], correction = 0, keepdim = True)
        getitem_270 = var_mean_135[0]
        getitem_271 = var_mean_135[1];  var_mean_135 = None
        add_471 = torch.ops.aten.add.Tensor(getitem_270, 1e-05);  getitem_270 = None
        rsqrt_135 = torch.ops.aten.rsqrt.default(add_471);  add_471 = None
        sub_202 = torch.ops.aten.sub.Tensor(view_403, getitem_271);  view_403 = getitem_271 = None
        mul_605 = torch.ops.aten.mul.Tensor(sub_202, rsqrt_135);  sub_202 = rsqrt_135 = None
        view_404 = torch.ops.aten.view.default(mul_605, [8, 768, 7, 7]);  mul_605 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg320_1, 0);  arg320_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(arg319_1, 0);  arg319_1 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
        mul_606 = torch.ops.aten.mul.Tensor(view_404, unsqueeze_809);  view_404 = unsqueeze_809 = None
        add_472 = torch.ops.aten.add.Tensor(mul_606, unsqueeze_806);  mul_606 = unsqueeze_806 = None
        avg_pool2d_67 = torch.ops.aten.avg_pool2d.default(add_472, [3, 3], [1, 1], [1, 1], False, False)
        sub_203 = torch.ops.aten.sub.Tensor(avg_pool2d_67, add_472);  avg_pool2d_67 = add_472 = None
        view_405 = torch.ops.aten.view.default(arg321_1, [768, 1, 1]);  arg321_1 = None
        mul_607 = torch.ops.aten.mul.Tensor(sub_203, view_405);  sub_203 = view_405 = None
        add_473 = torch.ops.aten.add.Tensor(add_470, mul_607);  add_470 = mul_607 = None
        view_406 = torch.ops.aten.view.default(add_473, [8, 1, 768, 49])
        var_mean_136 = torch.ops.aten.var_mean.correction(view_406, [2, 3], correction = 0, keepdim = True)
        getitem_272 = var_mean_136[0]
        getitem_273 = var_mean_136[1];  var_mean_136 = None
        add_474 = torch.ops.aten.add.Tensor(getitem_272, 1e-05);  getitem_272 = None
        rsqrt_136 = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
        sub_204 = torch.ops.aten.sub.Tensor(view_406, getitem_273);  view_406 = getitem_273 = None
        mul_608 = torch.ops.aten.mul.Tensor(sub_204, rsqrt_136);  sub_204 = rsqrt_136 = None
        view_407 = torch.ops.aten.view.default(mul_608, [8, 768, 7, 7]);  mul_608 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(arg323_1, 0);  arg323_1 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(arg322_1, 0);  arg322_1 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
        mul_609 = torch.ops.aten.mul.Tensor(view_407, unsqueeze_815);  view_407 = unsqueeze_815 = None
        add_475 = torch.ops.aten.add.Tensor(mul_609, unsqueeze_812);  mul_609 = unsqueeze_812 = None
        convolution_142 = torch.ops.aten.convolution.default(add_475, arg324_1, arg325_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_475 = arg324_1 = arg325_1 = None
        mul_610 = torch.ops.aten.mul.Tensor(convolution_142, 0.5)
        mul_611 = torch.ops.aten.mul.Tensor(convolution_142, 0.7071067811865476);  convolution_142 = None
        erf_67 = torch.ops.aten.erf.default(mul_611);  mul_611 = None
        add_476 = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_612 = torch.ops.aten.mul.Tensor(mul_610, add_476);  mul_610 = add_476 = None
        convolution_143 = torch.ops.aten.convolution.default(mul_612, arg326_1, arg327_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_612 = arg326_1 = arg327_1 = None
        view_408 = torch.ops.aten.view.default(arg328_1, [768, 1, 1]);  arg328_1 = None
        mul_613 = torch.ops.aten.mul.Tensor(convolution_143, view_408);  convolution_143 = view_408 = None
        add_477 = torch.ops.aten.add.Tensor(add_473, mul_613);  add_473 = mul_613 = None
        view_409 = torch.ops.aten.view.default(add_477, [8, 1, 768, 49])
        var_mean_137 = torch.ops.aten.var_mean.correction(view_409, [2, 3], correction = 0, keepdim = True)
        getitem_274 = var_mean_137[0]
        getitem_275 = var_mean_137[1];  var_mean_137 = None
        add_478 = torch.ops.aten.add.Tensor(getitem_274, 1e-05);  getitem_274 = None
        rsqrt_137 = torch.ops.aten.rsqrt.default(add_478);  add_478 = None
        sub_205 = torch.ops.aten.sub.Tensor(view_409, getitem_275);  view_409 = getitem_275 = None
        mul_614 = torch.ops.aten.mul.Tensor(sub_205, rsqrt_137);  sub_205 = rsqrt_137 = None
        view_410 = torch.ops.aten.view.default(mul_614, [8, 768, 7, 7]);  mul_614 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg330_1, 0);  arg330_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(arg329_1, 0);  arg329_1 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
        mul_615 = torch.ops.aten.mul.Tensor(view_410, unsqueeze_821);  view_410 = unsqueeze_821 = None
        add_479 = torch.ops.aten.add.Tensor(mul_615, unsqueeze_818);  mul_615 = unsqueeze_818 = None
        avg_pool2d_68 = torch.ops.aten.avg_pool2d.default(add_479, [3, 3], [1, 1], [1, 1], False, False)
        sub_206 = torch.ops.aten.sub.Tensor(avg_pool2d_68, add_479);  avg_pool2d_68 = add_479 = None
        view_411 = torch.ops.aten.view.default(arg331_1, [768, 1, 1]);  arg331_1 = None
        mul_616 = torch.ops.aten.mul.Tensor(sub_206, view_411);  sub_206 = view_411 = None
        add_480 = torch.ops.aten.add.Tensor(add_477, mul_616);  add_477 = mul_616 = None
        view_412 = torch.ops.aten.view.default(add_480, [8, 1, 768, 49])
        var_mean_138 = torch.ops.aten.var_mean.correction(view_412, [2, 3], correction = 0, keepdim = True)
        getitem_276 = var_mean_138[0]
        getitem_277 = var_mean_138[1];  var_mean_138 = None
        add_481 = torch.ops.aten.add.Tensor(getitem_276, 1e-05);  getitem_276 = None
        rsqrt_138 = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        sub_207 = torch.ops.aten.sub.Tensor(view_412, getitem_277);  view_412 = getitem_277 = None
        mul_617 = torch.ops.aten.mul.Tensor(sub_207, rsqrt_138);  sub_207 = rsqrt_138 = None
        view_413 = torch.ops.aten.view.default(mul_617, [8, 768, 7, 7]);  mul_617 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg333_1, 0);  arg333_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(arg332_1, 0);  arg332_1 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
        mul_618 = torch.ops.aten.mul.Tensor(view_413, unsqueeze_827);  view_413 = unsqueeze_827 = None
        add_482 = torch.ops.aten.add.Tensor(mul_618, unsqueeze_824);  mul_618 = unsqueeze_824 = None
        convolution_144 = torch.ops.aten.convolution.default(add_482, arg334_1, arg335_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_482 = arg334_1 = arg335_1 = None
        mul_619 = torch.ops.aten.mul.Tensor(convolution_144, 0.5)
        mul_620 = torch.ops.aten.mul.Tensor(convolution_144, 0.7071067811865476);  convolution_144 = None
        erf_68 = torch.ops.aten.erf.default(mul_620);  mul_620 = None
        add_483 = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_621 = torch.ops.aten.mul.Tensor(mul_619, add_483);  mul_619 = add_483 = None
        convolution_145 = torch.ops.aten.convolution.default(mul_621, arg336_1, arg337_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_621 = arg336_1 = arg337_1 = None
        view_414 = torch.ops.aten.view.default(arg338_1, [768, 1, 1]);  arg338_1 = None
        mul_622 = torch.ops.aten.mul.Tensor(convolution_145, view_414);  convolution_145 = view_414 = None
        add_484 = torch.ops.aten.add.Tensor(add_480, mul_622);  add_480 = mul_622 = None
        view_415 = torch.ops.aten.view.default(add_484, [8, 1, 768, 49])
        var_mean_139 = torch.ops.aten.var_mean.correction(view_415, [2, 3], correction = 0, keepdim = True)
        getitem_278 = var_mean_139[0]
        getitem_279 = var_mean_139[1];  var_mean_139 = None
        add_485 = torch.ops.aten.add.Tensor(getitem_278, 1e-05);  getitem_278 = None
        rsqrt_139 = torch.ops.aten.rsqrt.default(add_485);  add_485 = None
        sub_208 = torch.ops.aten.sub.Tensor(view_415, getitem_279);  view_415 = getitem_279 = None
        mul_623 = torch.ops.aten.mul.Tensor(sub_208, rsqrt_139);  sub_208 = rsqrt_139 = None
        view_416 = torch.ops.aten.view.default(mul_623, [8, 768, 7, 7]);  mul_623 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg340_1, 0);  arg340_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(arg339_1, 0);  arg339_1 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
        mul_624 = torch.ops.aten.mul.Tensor(view_416, unsqueeze_833);  view_416 = unsqueeze_833 = None
        add_486 = torch.ops.aten.add.Tensor(mul_624, unsqueeze_830);  mul_624 = unsqueeze_830 = None
        avg_pool2d_69 = torch.ops.aten.avg_pool2d.default(add_486, [3, 3], [1, 1], [1, 1], False, False)
        sub_209 = torch.ops.aten.sub.Tensor(avg_pool2d_69, add_486);  avg_pool2d_69 = add_486 = None
        view_417 = torch.ops.aten.view.default(arg341_1, [768, 1, 1]);  arg341_1 = None
        mul_625 = torch.ops.aten.mul.Tensor(sub_209, view_417);  sub_209 = view_417 = None
        add_487 = torch.ops.aten.add.Tensor(add_484, mul_625);  add_484 = mul_625 = None
        view_418 = torch.ops.aten.view.default(add_487, [8, 1, 768, 49])
        var_mean_140 = torch.ops.aten.var_mean.correction(view_418, [2, 3], correction = 0, keepdim = True)
        getitem_280 = var_mean_140[0]
        getitem_281 = var_mean_140[1];  var_mean_140 = None
        add_488 = torch.ops.aten.add.Tensor(getitem_280, 1e-05);  getitem_280 = None
        rsqrt_140 = torch.ops.aten.rsqrt.default(add_488);  add_488 = None
        sub_210 = torch.ops.aten.sub.Tensor(view_418, getitem_281);  view_418 = getitem_281 = None
        mul_626 = torch.ops.aten.mul.Tensor(sub_210, rsqrt_140);  sub_210 = rsqrt_140 = None
        view_419 = torch.ops.aten.view.default(mul_626, [8, 768, 7, 7]);  mul_626 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(arg343_1, 0);  arg343_1 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(arg342_1, 0);  arg342_1 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
        mul_627 = torch.ops.aten.mul.Tensor(view_419, unsqueeze_839);  view_419 = unsqueeze_839 = None
        add_489 = torch.ops.aten.add.Tensor(mul_627, unsqueeze_836);  mul_627 = unsqueeze_836 = None
        convolution_146 = torch.ops.aten.convolution.default(add_489, arg344_1, arg345_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_489 = arg344_1 = arg345_1 = None
        mul_628 = torch.ops.aten.mul.Tensor(convolution_146, 0.5)
        mul_629 = torch.ops.aten.mul.Tensor(convolution_146, 0.7071067811865476);  convolution_146 = None
        erf_69 = torch.ops.aten.erf.default(mul_629);  mul_629 = None
        add_490 = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_630 = torch.ops.aten.mul.Tensor(mul_628, add_490);  mul_628 = add_490 = None
        convolution_147 = torch.ops.aten.convolution.default(mul_630, arg346_1, arg347_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_630 = arg346_1 = arg347_1 = None
        view_420 = torch.ops.aten.view.default(arg348_1, [768, 1, 1]);  arg348_1 = None
        mul_631 = torch.ops.aten.mul.Tensor(convolution_147, view_420);  convolution_147 = view_420 = None
        add_491 = torch.ops.aten.add.Tensor(add_487, mul_631);  add_487 = mul_631 = None
        view_421 = torch.ops.aten.view.default(add_491, [8, 1, 768, 49])
        var_mean_141 = torch.ops.aten.var_mean.correction(view_421, [2, 3], correction = 0, keepdim = True)
        getitem_282 = var_mean_141[0]
        getitem_283 = var_mean_141[1];  var_mean_141 = None
        add_492 = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_141 = torch.ops.aten.rsqrt.default(add_492);  add_492 = None
        sub_211 = torch.ops.aten.sub.Tensor(view_421, getitem_283);  view_421 = getitem_283 = None
        mul_632 = torch.ops.aten.mul.Tensor(sub_211, rsqrt_141);  sub_211 = rsqrt_141 = None
        view_422 = torch.ops.aten.view.default(mul_632, [8, 768, 7, 7]);  mul_632 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg350_1, 0);  arg350_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(arg349_1, 0);  arg349_1 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
        mul_633 = torch.ops.aten.mul.Tensor(view_422, unsqueeze_845);  view_422 = unsqueeze_845 = None
        add_493 = torch.ops.aten.add.Tensor(mul_633, unsqueeze_842);  mul_633 = unsqueeze_842 = None
        avg_pool2d_70 = torch.ops.aten.avg_pool2d.default(add_493, [3, 3], [1, 1], [1, 1], False, False)
        sub_212 = torch.ops.aten.sub.Tensor(avg_pool2d_70, add_493);  avg_pool2d_70 = add_493 = None
        view_423 = torch.ops.aten.view.default(arg351_1, [768, 1, 1]);  arg351_1 = None
        mul_634 = torch.ops.aten.mul.Tensor(sub_212, view_423);  sub_212 = view_423 = None
        add_494 = torch.ops.aten.add.Tensor(add_491, mul_634);  add_491 = mul_634 = None
        view_424 = torch.ops.aten.view.default(add_494, [8, 1, 768, 49])
        var_mean_142 = torch.ops.aten.var_mean.correction(view_424, [2, 3], correction = 0, keepdim = True)
        getitem_284 = var_mean_142[0]
        getitem_285 = var_mean_142[1];  var_mean_142 = None
        add_495 = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
        rsqrt_142 = torch.ops.aten.rsqrt.default(add_495);  add_495 = None
        sub_213 = torch.ops.aten.sub.Tensor(view_424, getitem_285);  view_424 = getitem_285 = None
        mul_635 = torch.ops.aten.mul.Tensor(sub_213, rsqrt_142);  sub_213 = rsqrt_142 = None
        view_425 = torch.ops.aten.view.default(mul_635, [8, 768, 7, 7]);  mul_635 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg353_1, 0);  arg353_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(arg352_1, 0);  arg352_1 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
        mul_636 = torch.ops.aten.mul.Tensor(view_425, unsqueeze_851);  view_425 = unsqueeze_851 = None
        add_496 = torch.ops.aten.add.Tensor(mul_636, unsqueeze_848);  mul_636 = unsqueeze_848 = None
        convolution_148 = torch.ops.aten.convolution.default(add_496, arg354_1, arg355_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_496 = arg354_1 = arg355_1 = None
        mul_637 = torch.ops.aten.mul.Tensor(convolution_148, 0.5)
        mul_638 = torch.ops.aten.mul.Tensor(convolution_148, 0.7071067811865476);  convolution_148 = None
        erf_70 = torch.ops.aten.erf.default(mul_638);  mul_638 = None
        add_497 = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_639 = torch.ops.aten.mul.Tensor(mul_637, add_497);  mul_637 = add_497 = None
        convolution_149 = torch.ops.aten.convolution.default(mul_639, arg356_1, arg357_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_639 = arg356_1 = arg357_1 = None
        view_426 = torch.ops.aten.view.default(arg358_1, [768, 1, 1]);  arg358_1 = None
        mul_640 = torch.ops.aten.mul.Tensor(convolution_149, view_426);  convolution_149 = view_426 = None
        add_498 = torch.ops.aten.add.Tensor(add_494, mul_640);  add_494 = mul_640 = None
        view_427 = torch.ops.aten.view.default(add_498, [8, 1, 768, 49])
        var_mean_143 = torch.ops.aten.var_mean.correction(view_427, [2, 3], correction = 0, keepdim = True)
        getitem_286 = var_mean_143[0]
        getitem_287 = var_mean_143[1];  var_mean_143 = None
        add_499 = torch.ops.aten.add.Tensor(getitem_286, 1e-05);  getitem_286 = None
        rsqrt_143 = torch.ops.aten.rsqrt.default(add_499);  add_499 = None
        sub_214 = torch.ops.aten.sub.Tensor(view_427, getitem_287);  view_427 = getitem_287 = None
        mul_641 = torch.ops.aten.mul.Tensor(sub_214, rsqrt_143);  sub_214 = rsqrt_143 = None
        view_428 = torch.ops.aten.view.default(mul_641, [8, 768, 7, 7]);  mul_641 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg360_1, 0);  arg360_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(arg359_1, 0);  arg359_1 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
        mul_642 = torch.ops.aten.mul.Tensor(view_428, unsqueeze_857);  view_428 = unsqueeze_857 = None
        add_500 = torch.ops.aten.add.Tensor(mul_642, unsqueeze_854);  mul_642 = unsqueeze_854 = None
        avg_pool2d_71 = torch.ops.aten.avg_pool2d.default(add_500, [3, 3], [1, 1], [1, 1], False, False)
        sub_215 = torch.ops.aten.sub.Tensor(avg_pool2d_71, add_500);  avg_pool2d_71 = add_500 = None
        view_429 = torch.ops.aten.view.default(arg361_1, [768, 1, 1]);  arg361_1 = None
        mul_643 = torch.ops.aten.mul.Tensor(sub_215, view_429);  sub_215 = view_429 = None
        add_501 = torch.ops.aten.add.Tensor(add_498, mul_643);  add_498 = mul_643 = None
        view_430 = torch.ops.aten.view.default(add_501, [8, 1, 768, 49])
        var_mean_144 = torch.ops.aten.var_mean.correction(view_430, [2, 3], correction = 0, keepdim = True)
        getitem_288 = var_mean_144[0]
        getitem_289 = var_mean_144[1];  var_mean_144 = None
        add_502 = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_144 = torch.ops.aten.rsqrt.default(add_502);  add_502 = None
        sub_216 = torch.ops.aten.sub.Tensor(view_430, getitem_289);  view_430 = getitem_289 = None
        mul_644 = torch.ops.aten.mul.Tensor(sub_216, rsqrt_144);  sub_216 = rsqrt_144 = None
        view_431 = torch.ops.aten.view.default(mul_644, [8, 768, 7, 7]);  mul_644 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(arg363_1, 0);  arg363_1 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(arg362_1, 0);  arg362_1 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
        mul_645 = torch.ops.aten.mul.Tensor(view_431, unsqueeze_863);  view_431 = unsqueeze_863 = None
        add_503 = torch.ops.aten.add.Tensor(mul_645, unsqueeze_860);  mul_645 = unsqueeze_860 = None
        convolution_150 = torch.ops.aten.convolution.default(add_503, arg364_1, arg365_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_503 = arg364_1 = arg365_1 = None
        mul_646 = torch.ops.aten.mul.Tensor(convolution_150, 0.5)
        mul_647 = torch.ops.aten.mul.Tensor(convolution_150, 0.7071067811865476);  convolution_150 = None
        erf_71 = torch.ops.aten.erf.default(mul_647);  mul_647 = None
        add_504 = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_648 = torch.ops.aten.mul.Tensor(mul_646, add_504);  mul_646 = add_504 = None
        convolution_151 = torch.ops.aten.convolution.default(mul_648, arg366_1, arg367_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_648 = arg366_1 = arg367_1 = None
        view_432 = torch.ops.aten.view.default(arg368_1, [768, 1, 1]);  arg368_1 = None
        mul_649 = torch.ops.aten.mul.Tensor(convolution_151, view_432);  convolution_151 = view_432 = None
        add_505 = torch.ops.aten.add.Tensor(add_501, mul_649);  add_501 = mul_649 = None
        mean_1 = torch.ops.aten.mean.dim(add_505, [-1, -2], True);  add_505 = None
        permute_3 = torch.ops.aten.permute.default(mean_1, [0, 2, 3, 1]);  mean_1 = None
        var_mean_145 = torch.ops.aten.var_mean.correction(permute_3, [3], correction = 0, keepdim = True)
        getitem_290 = var_mean_145[0]
        getitem_291 = var_mean_145[1];  var_mean_145 = None
        add_506 = torch.ops.aten.add.Tensor(getitem_290, 1e-06);  getitem_290 = None
        rsqrt_145 = torch.ops.aten.rsqrt.default(add_506);  add_506 = None
        sub_217 = torch.ops.aten.sub.Tensor(permute_3, getitem_291);  permute_3 = getitem_291 = None
        mul_650 = torch.ops.aten.mul.Tensor(sub_217, rsqrt_145);  sub_217 = rsqrt_145 = None
        mul_651 = torch.ops.aten.mul.Tensor(mul_650, arg369_1);  mul_650 = arg369_1 = None
        add_507 = torch.ops.aten.add.Tensor(mul_651, arg370_1);  mul_651 = arg370_1 = None
        permute_4 = torch.ops.aten.permute.default(add_507, [0, 3, 1, 2]);  add_507 = None
        view_433 = torch.ops.aten.view.default(permute_4, [8, 768]);  permute_4 = None
        permute_5 = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg372_1, view_433, permute_5);  arg372_1 = view_433 = permute_5 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 56448, device=device(type='cuda', index=0))
    reader.tensor(buf0, (96, 3, 7, 7), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf1, (96,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf2, (8, 3, 224, 224), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf3, (96,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf4, (96,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf5, (96,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf6, (96,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf7, (96,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf8, (384, 96, 1, 1), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf9, (384,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf10, (96, 384, 1, 1), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf11, (96,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf12, (96,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf13, (96,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf14, (96,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf15, (96,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf16, (96,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf17, (96,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 96, 1, 1), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf19, (384,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf20, (96, 384, 1, 1), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf21, (96,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf22, (96,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf23, (96,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf24, (96,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf25, (96,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf26, (96,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf27, (96,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf28, (384, 96, 1, 1), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf29, (384,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf30, (96, 384, 1, 1), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf31, (96,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf32, (96,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf33, (96,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf34, (96,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf35, (96,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf36, (96,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf37, (96,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf38, (384, 96, 1, 1), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf39, (384,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf40, (96, 384, 1, 1), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf41, (96,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf42, (96,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf43, (96,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf44, (96,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf45, (96,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf46, (96,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf47, (96,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf48, (384, 96, 1, 1), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf49, (384,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf50, (96, 384, 1, 1), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf51, (96,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf52, (96,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf53, (96,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf54, (96,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf55, (96,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf56, (96,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf57, (96,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf58, (384, 96, 1, 1), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf59, (384,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf60, (96, 384, 1, 1), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf61, (96,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf62, (96,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf63, (192, 96, 3, 3), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf64, (192,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf65, (192,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf66, (192,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf67, (192,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf68, (192,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf69, (192,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768, 192, 1, 1), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf72, (192, 768, 1, 1), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf73, (192,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf74, (192,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf75, (192,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf76, (192,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf77, (192,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf78, (192,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf79, (192,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768, 192, 1, 1), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf82, (192, 768, 1, 1), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf83, (192,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf84, (192,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf85, (192,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf86, (192,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf87, (192,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf88, (192,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf89, (192,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768, 192, 1, 1), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf92, (192, 768, 1, 1), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf93, (192,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf94, (192,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf95, (192,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf96, (192,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf97, (192,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf98, (192,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf99, (192,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768, 192, 1, 1), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf102, (192, 768, 1, 1), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf103, (192,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf104, (192,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf105, (192,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf106, (192,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf107, (192,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf108, (192,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf109, (192,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768, 192, 1, 1), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf112, (192, 768, 1, 1), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf113, (192,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf114, (192,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf115, (192,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf116, (192,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf117, (192,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf118, (192,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf119, (192,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768, 192, 1, 1), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf122, (192, 768, 1, 1), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf123, (192,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf124, (192,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 2654208, device=device(type='cuda', index=0))
    reader.tensor(buf125, (384, 192, 3, 3), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf127, (384,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf128, (384,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf129, (384,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf130, (384,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf131, (384,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1536, 384, 1, 1), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1536,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf134, (384, 1536, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf135, (384,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf136, (384,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf137, (384,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf138, (384,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf139, (384,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf140, (384,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf141, (384,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1536, 384, 1, 1), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf143, (1536,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf144, (384, 1536, 1, 1), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf146, (384,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf147, (384,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf148, (384,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf149, (384,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf150, (384,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf151, (384,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1536, 384, 1, 1), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1536,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf154, (384, 1536, 1, 1), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf155, (384,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf156, (384,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf157, (384,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf158, (384,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf159, (384,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf160, (384,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf161, (384,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf162, (1536, 384, 1, 1), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1536,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf164, (384, 1536, 1, 1), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf165, (384,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf166, (384,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf167, (384,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf168, (384,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf169, (384,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf170, (384,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf171, (384,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1536, 384, 1, 1), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1536,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf174, (384, 1536, 1, 1), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf175, (384,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf176, (384,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf177, (384,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf178, (384,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf179, (384,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf180, (384,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf181, (384,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1536, 384, 1, 1), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1536,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf184, (384, 1536, 1, 1), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf185, (384,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf186, (384,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf187, (384,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf188, (384,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf189, (384,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf190, (384,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf191, (384,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf192, (1536, 384, 1, 1), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf193, (1536,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf194, (384, 1536, 1, 1), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf195, (384,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf196, (384,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf197, (384,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf198, (384,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf199, (384,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf200, (384,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf201, (384,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1536, 384, 1, 1), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1536,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf204, (384, 1536, 1, 1), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf205, (384,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf206, (384,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf207, (384,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf208, (384,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf209, (384,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf210, (384,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf211, (384,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1536, 384, 1, 1), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1536,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf214, (384, 1536, 1, 1), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf215, (384,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf216, (384,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf217, (384,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf218, (384,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf219, (384,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf220, (384,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf221, (384,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf222, (1536, 384, 1, 1), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1536,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf224, (384, 1536, 1, 1), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf225, (384,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf226, (384,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf227, (384,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf228, (384,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf229, (384,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf230, (384,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf231, (384,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1536, 384, 1, 1), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1536,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf234, (384, 1536, 1, 1), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf235, (384,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf236, (384,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf237, (384,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf238, (384,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf239, (384,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf240, (384,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf241, (384,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf242, (1536, 384, 1, 1), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf243, (1536,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf244, (384, 1536, 1, 1), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf245, (384,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf246, (384,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf247, (384,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf248, (384,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf249, (384,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf250, (384,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf251, (384,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf252, (1536, 384, 1, 1), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf253, (1536,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf254, (384, 1536, 1, 1), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf255, (384,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf256, (384,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf257, (384,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf258, (384,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf259, (384,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf260, (384,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf261, (384,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf262, (1536, 384, 1, 1), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1536,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf264, (384, 1536, 1, 1), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf265, (384,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf266, (384,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf267, (384,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf268, (384,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf269, (384,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf270, (384,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf271, (384,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf272, (1536, 384, 1, 1), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf273, (1536,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf274, (384, 1536, 1, 1), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf275, (384,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf276, (384,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf277, (384,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf278, (384,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf279, (384,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf280, (384,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf281, (384,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf282, (1536, 384, 1, 1), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf283, (1536,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf284, (384, 1536, 1, 1), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf285, (384,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf286, (384,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf287, (384,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf288, (384,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf289, (384,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf290, (384,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf291, (384,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1536, 384, 1, 1), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1536,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf294, (384, 1536, 1, 1), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf295, (384,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf296, (384,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf297, (384,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf298, (384,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf299, (384,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf300, (384,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf301, (384,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf302, (1536, 384, 1, 1), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1536,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf304, (384, 1536, 1, 1), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf305, (384,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf306, (384,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 10616832, device=device(type='cuda', index=0))
    reader.tensor(buf307, (768, 384, 3, 3), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf308, (768,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf309, (768,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf310, (768,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf311, (768,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf312, (768,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf313, (768,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf314, (3072, 768, 1, 1), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf315, (3072,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf316, (768, 3072, 1, 1), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf317, (768,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf318, (768,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf319, (768,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf320, (768,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf321, (768,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf322, (768,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf323, (768,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf324, (3072, 768, 1, 1), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf325, (3072,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf326, (768, 3072, 1, 1), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf327, (768,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf328, (768,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf329, (768,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf330, (768,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf331, (768,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf332, (768,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf333, (768,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf334, (3072, 768, 1, 1), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf335, (3072,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf336, (768, 3072, 1, 1), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf337, (768,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf338, (768,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf339, (768,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf340, (768,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf341, (768,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf342, (768,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf343, (768,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf344, (3072, 768, 1, 1), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf345, (3072,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf346, (768, 3072, 1, 1), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf347, (768,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf348, (768,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf349, (768,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf350, (768,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf351, (768,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf352, (768,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf353, (768,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf354, (3072, 768, 1, 1), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf355, (3072,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf356, (768, 3072, 1, 1), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf357, (768,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf358, (768,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf359, (768,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf360, (768,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf361, (768,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf362, (768,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf363, (768,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf364, (3072, 768, 1, 1), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf365, (3072,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf366, (768, 3072, 1, 1), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf367, (768,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf368, (768,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf369, (768,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf370, (768,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf371, (1000, 768), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf372, (1000,), is_leaf=True)  # arg372_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)