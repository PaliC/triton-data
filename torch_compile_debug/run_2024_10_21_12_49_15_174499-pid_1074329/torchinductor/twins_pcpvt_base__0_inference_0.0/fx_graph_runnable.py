
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1):
        convolution_33 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_433 = torch.ops.aten.view.default(convolution_33, [8, 64, 3136]);  convolution_33 = None
        permute_294 = torch.ops.aten.permute.default(view_433, [0, 2, 1]);  view_433 = None
        clone_96 = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
        getitem_340 = var_mean_86[0]
        getitem_341 = var_mean_86[1];  var_mean_86 = None
        add_260 = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_86 = torch.ops.aten.sub.Tensor(clone_96, getitem_341);  clone_96 = getitem_341 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, arg3_1);  mul_256 = arg3_1 = None
        add_261 = torch.ops.aten.add.Tensor(mul_257, arg4_1);  mul_257 = arg4_1 = None
        var_mean_87 = torch.ops.aten.var_mean.correction(add_261, [2], correction = 0, keepdim = True)
        getitem_342 = var_mean_87[0]
        getitem_343 = var_mean_87[1];  var_mean_87 = None
        add_262 = torch.ops.aten.add.Tensor(getitem_342, 1e-06);  getitem_342 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        sub_87 = torch.ops.aten.sub.Tensor(add_261, getitem_343);  getitem_343 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, arg5_1);  mul_258 = arg5_1 = None
        add_263 = torch.ops.aten.add.Tensor(mul_259, arg6_1);  mul_259 = arg6_1 = None
        view_434 = torch.ops.aten.view.default(add_263, [25088, 64])
        permute_295 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg8_1, view_434, permute_295);  arg8_1 = view_434 = permute_295 = None
        view_435 = torch.ops.aten.view.default(addmm_141, [8, 3136, 64]);  addmm_141 = None
        view_436 = torch.ops.aten.view.default(view_435, [8, 3136, 1, 64]);  view_435 = None
        permute_296 = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
        permute_297 = torch.ops.aten.permute.default(add_263, [0, 2, 1]);  add_263 = None
        view_437 = torch.ops.aten.view.default(permute_297, [8, 64, 56, 56]);  permute_297 = None
        convolution_34 = torch.ops.aten.convolution.default(view_437, arg9_1, arg10_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_437 = arg9_1 = arg10_1 = None
        view_438 = torch.ops.aten.view.default(convolution_34, [8, 64, 49]);  convolution_34 = None
        permute_298 = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
        var_mean_88 = torch.ops.aten.var_mean.correction(permute_298, [2], correction = 0, keepdim = True)
        getitem_344 = var_mean_88[0]
        getitem_345 = var_mean_88[1];  var_mean_88 = None
        add_264 = torch.ops.aten.add.Tensor(getitem_344, 1e-05);  getitem_344 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_88 = torch.ops.aten.sub.Tensor(permute_298, getitem_345);  permute_298 = getitem_345 = None
        mul_260 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, arg11_1);  mul_260 = arg11_1 = None
        add_265 = torch.ops.aten.add.Tensor(mul_261, arg12_1);  mul_261 = arg12_1 = None
        view_439 = torch.ops.aten.view.default(add_265, [392, 64]);  add_265 = None
        permute_299 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg14_1, view_439, permute_299);  arg14_1 = view_439 = permute_299 = None
        view_440 = torch.ops.aten.view.default(addmm_142, [8, 49, 128]);  addmm_142 = None
        view_441 = torch.ops.aten.view.default(view_440, [8, -1, 2, 1, 64]);  view_440 = None
        permute_300 = torch.ops.aten.permute.default(view_441, [2, 0, 3, 1, 4]);  view_441 = None
        unbind_28 = torch.ops.aten.unbind.int(permute_300);  permute_300 = None
        getitem_346 = unbind_28[0]
        getitem_347 = unbind_28[1];  unbind_28 = None
        _scaled_dot_product_efficient_attention_28 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_296, getitem_346, getitem_347, None, False);  permute_296 = getitem_346 = getitem_347 = None
        getitem_348 = _scaled_dot_product_efficient_attention_28[0];  _scaled_dot_product_efficient_attention_28 = None
        permute_301 = torch.ops.aten.permute.default(getitem_348, [0, 2, 1, 3]);  getitem_348 = None
        view_442 = torch.ops.aten.view.default(permute_301, [8, 3136, 64]);  permute_301 = None
        view_443 = torch.ops.aten.view.default(view_442, [25088, 64]);  view_442 = None
        permute_302 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg16_1, view_443, permute_302);  arg16_1 = view_443 = permute_302 = None
        view_444 = torch.ops.aten.view.default(addmm_143, [8, 3136, 64]);  addmm_143 = None
        add_266 = torch.ops.aten.add.Tensor(add_261, view_444);  add_261 = view_444 = None
        var_mean_89 = torch.ops.aten.var_mean.correction(add_266, [2], correction = 0, keepdim = True)
        getitem_352 = var_mean_89[0]
        getitem_353 = var_mean_89[1];  var_mean_89 = None
        add_267 = torch.ops.aten.add.Tensor(getitem_352, 1e-06);  getitem_352 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
        sub_89 = torch.ops.aten.sub.Tensor(add_266, getitem_353);  getitem_353 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, arg17_1);  mul_262 = arg17_1 = None
        add_268 = torch.ops.aten.add.Tensor(mul_263, arg18_1);  mul_263 = arg18_1 = None
        view_445 = torch.ops.aten.view.default(add_268, [25088, 64]);  add_268 = None
        permute_303 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg20_1, view_445, permute_303);  arg20_1 = view_445 = permute_303 = None
        view_446 = torch.ops.aten.view.default(addmm_144, [8, 3136, 512]);  addmm_144 = None
        mul_264 = torch.ops.aten.mul.Tensor(view_446, 0.5)
        mul_265 = torch.ops.aten.mul.Tensor(view_446, 0.7071067811865476);  view_446 = None
        erf_28 = torch.ops.aten.erf.default(mul_265);  mul_265 = None
        add_269 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_264, add_269);  mul_264 = add_269 = None
        view_447 = torch.ops.aten.view.default(mul_266, [25088, 512]);  mul_266 = None
        permute_304 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg22_1, view_447, permute_304);  arg22_1 = view_447 = permute_304 = None
        view_448 = torch.ops.aten.view.default(addmm_145, [8, 3136, 64]);  addmm_145 = None
        add_270 = torch.ops.aten.add.Tensor(add_266, view_448);  add_266 = view_448 = None
        permute_305 = torch.ops.aten.permute.default(add_270, [0, 2, 1]);  add_270 = None
        view_449 = torch.ops.aten.view.default(permute_305, [8, 64, 56, 56]);  permute_305 = None
        convolution_35 = torch.ops.aten.convolution.default(view_449, arg23_1, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  arg23_1 = arg24_1 = None
        add_271 = torch.ops.aten.add.Tensor(convolution_35, view_449);  convolution_35 = view_449 = None
        view_451 = torch.ops.aten.view.default(add_271, [8, 64, 3136]);  add_271 = None
        permute_307 = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        var_mean_90 = torch.ops.aten.var_mean.correction(permute_307, [2], correction = 0, keepdim = True)
        getitem_354 = var_mean_90[0]
        getitem_355 = var_mean_90[1];  var_mean_90 = None
        add_272 = torch.ops.aten.add.Tensor(getitem_354, 1e-06);  getitem_354 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        sub_90 = torch.ops.aten.sub.Tensor(permute_307, getitem_355);  getitem_355 = None
        mul_267 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, arg25_1);  mul_267 = arg25_1 = None
        add_273 = torch.ops.aten.add.Tensor(mul_268, arg26_1);  mul_268 = arg26_1 = None
        view_452 = torch.ops.aten.view.default(add_273, [25088, 64])
        permute_308 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg28_1, view_452, permute_308);  arg28_1 = view_452 = permute_308 = None
        view_453 = torch.ops.aten.view.default(addmm_146, [8, 3136, 64]);  addmm_146 = None
        view_454 = torch.ops.aten.view.default(view_453, [8, 3136, 1, 64]);  view_453 = None
        permute_309 = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        permute_310 = torch.ops.aten.permute.default(add_273, [0, 2, 1]);  add_273 = None
        view_455 = torch.ops.aten.view.default(permute_310, [8, 64, 56, 56]);  permute_310 = None
        convolution_36 = torch.ops.aten.convolution.default(view_455, arg29_1, arg30_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_455 = arg29_1 = arg30_1 = None
        view_456 = torch.ops.aten.view.default(convolution_36, [8, 64, 49]);  convolution_36 = None
        permute_311 = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
        var_mean_91 = torch.ops.aten.var_mean.correction(permute_311, [2], correction = 0, keepdim = True)
        getitem_356 = var_mean_91[0]
        getitem_357 = var_mean_91[1];  var_mean_91 = None
        add_274 = torch.ops.aten.add.Tensor(getitem_356, 1e-05);  getitem_356 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
        sub_91 = torch.ops.aten.sub.Tensor(permute_311, getitem_357);  permute_311 = getitem_357 = None
        mul_269 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, arg31_1);  mul_269 = arg31_1 = None
        add_275 = torch.ops.aten.add.Tensor(mul_270, arg32_1);  mul_270 = arg32_1 = None
        view_457 = torch.ops.aten.view.default(add_275, [392, 64]);  add_275 = None
        permute_312 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg34_1, view_457, permute_312);  arg34_1 = view_457 = permute_312 = None
        view_458 = torch.ops.aten.view.default(addmm_147, [8, 49, 128]);  addmm_147 = None
        view_459 = torch.ops.aten.view.default(view_458, [8, -1, 2, 1, 64]);  view_458 = None
        permute_313 = torch.ops.aten.permute.default(view_459, [2, 0, 3, 1, 4]);  view_459 = None
        unbind_29 = torch.ops.aten.unbind.int(permute_313);  permute_313 = None
        getitem_358 = unbind_29[0]
        getitem_359 = unbind_29[1];  unbind_29 = None
        _scaled_dot_product_efficient_attention_29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_309, getitem_358, getitem_359, None, False);  permute_309 = getitem_358 = getitem_359 = None
        getitem_360 = _scaled_dot_product_efficient_attention_29[0];  _scaled_dot_product_efficient_attention_29 = None
        permute_314 = torch.ops.aten.permute.default(getitem_360, [0, 2, 1, 3]);  getitem_360 = None
        view_460 = torch.ops.aten.view.default(permute_314, [8, 3136, 64]);  permute_314 = None
        view_461 = torch.ops.aten.view.default(view_460, [25088, 64]);  view_460 = None
        permute_315 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg36_1, view_461, permute_315);  arg36_1 = view_461 = permute_315 = None
        view_462 = torch.ops.aten.view.default(addmm_148, [8, 3136, 64]);  addmm_148 = None
        add_276 = torch.ops.aten.add.Tensor(permute_307, view_462);  permute_307 = view_462 = None
        var_mean_92 = torch.ops.aten.var_mean.correction(add_276, [2], correction = 0, keepdim = True)
        getitem_364 = var_mean_92[0]
        getitem_365 = var_mean_92[1];  var_mean_92 = None
        add_277 = torch.ops.aten.add.Tensor(getitem_364, 1e-06);  getitem_364 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        sub_92 = torch.ops.aten.sub.Tensor(add_276, getitem_365);  getitem_365 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, arg37_1);  mul_271 = arg37_1 = None
        add_278 = torch.ops.aten.add.Tensor(mul_272, arg38_1);  mul_272 = arg38_1 = None
        view_463 = torch.ops.aten.view.default(add_278, [25088, 64]);  add_278 = None
        permute_316 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg40_1, view_463, permute_316);  arg40_1 = view_463 = permute_316 = None
        view_464 = torch.ops.aten.view.default(addmm_149, [8, 3136, 512]);  addmm_149 = None
        mul_273 = torch.ops.aten.mul.Tensor(view_464, 0.5)
        mul_274 = torch.ops.aten.mul.Tensor(view_464, 0.7071067811865476);  view_464 = None
        erf_29 = torch.ops.aten.erf.default(mul_274);  mul_274 = None
        add_279 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_273, add_279);  mul_273 = add_279 = None
        view_465 = torch.ops.aten.view.default(mul_275, [25088, 512]);  mul_275 = None
        permute_317 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg42_1, view_465, permute_317);  arg42_1 = view_465 = permute_317 = None
        view_466 = torch.ops.aten.view.default(addmm_150, [8, 3136, 64]);  addmm_150 = None
        add_280 = torch.ops.aten.add.Tensor(add_276, view_466);  add_276 = view_466 = None
        var_mean_93 = torch.ops.aten.var_mean.correction(add_280, [2], correction = 0, keepdim = True)
        getitem_366 = var_mean_93[0]
        getitem_367 = var_mean_93[1];  var_mean_93 = None
        add_281 = torch.ops.aten.add.Tensor(getitem_366, 1e-06);  getitem_366 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
        sub_93 = torch.ops.aten.sub.Tensor(add_280, getitem_367);  getitem_367 = None
        mul_276 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, arg43_1);  mul_276 = arg43_1 = None
        add_282 = torch.ops.aten.add.Tensor(mul_277, arg44_1);  mul_277 = arg44_1 = None
        view_467 = torch.ops.aten.view.default(add_282, [25088, 64])
        permute_318 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg46_1, view_467, permute_318);  arg46_1 = view_467 = permute_318 = None
        view_468 = torch.ops.aten.view.default(addmm_151, [8, 3136, 64]);  addmm_151 = None
        view_469 = torch.ops.aten.view.default(view_468, [8, 3136, 1, 64]);  view_468 = None
        permute_319 = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
        permute_320 = torch.ops.aten.permute.default(add_282, [0, 2, 1]);  add_282 = None
        view_470 = torch.ops.aten.view.default(permute_320, [8, 64, 56, 56]);  permute_320 = None
        convolution_37 = torch.ops.aten.convolution.default(view_470, arg47_1, arg48_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_470 = arg47_1 = arg48_1 = None
        view_471 = torch.ops.aten.view.default(convolution_37, [8, 64, 49]);  convolution_37 = None
        permute_321 = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        var_mean_94 = torch.ops.aten.var_mean.correction(permute_321, [2], correction = 0, keepdim = True)
        getitem_368 = var_mean_94[0]
        getitem_369 = var_mean_94[1];  var_mean_94 = None
        add_283 = torch.ops.aten.add.Tensor(getitem_368, 1e-05);  getitem_368 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        sub_94 = torch.ops.aten.sub.Tensor(permute_321, getitem_369);  permute_321 = getitem_369 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, arg49_1);  mul_278 = arg49_1 = None
        add_284 = torch.ops.aten.add.Tensor(mul_279, arg50_1);  mul_279 = arg50_1 = None
        view_472 = torch.ops.aten.view.default(add_284, [392, 64]);  add_284 = None
        permute_322 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg52_1, view_472, permute_322);  arg52_1 = view_472 = permute_322 = None
        view_473 = torch.ops.aten.view.default(addmm_152, [8, 49, 128]);  addmm_152 = None
        view_474 = torch.ops.aten.view.default(view_473, [8, -1, 2, 1, 64]);  view_473 = None
        permute_323 = torch.ops.aten.permute.default(view_474, [2, 0, 3, 1, 4]);  view_474 = None
        unbind_30 = torch.ops.aten.unbind.int(permute_323);  permute_323 = None
        getitem_370 = unbind_30[0]
        getitem_371 = unbind_30[1];  unbind_30 = None
        _scaled_dot_product_efficient_attention_30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_319, getitem_370, getitem_371, None, False);  permute_319 = getitem_370 = getitem_371 = None
        getitem_372 = _scaled_dot_product_efficient_attention_30[0];  _scaled_dot_product_efficient_attention_30 = None
        permute_324 = torch.ops.aten.permute.default(getitem_372, [0, 2, 1, 3]);  getitem_372 = None
        view_475 = torch.ops.aten.view.default(permute_324, [8, 3136, 64]);  permute_324 = None
        view_476 = torch.ops.aten.view.default(view_475, [25088, 64]);  view_475 = None
        permute_325 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg54_1, view_476, permute_325);  arg54_1 = view_476 = permute_325 = None
        view_477 = torch.ops.aten.view.default(addmm_153, [8, 3136, 64]);  addmm_153 = None
        add_285 = torch.ops.aten.add.Tensor(add_280, view_477);  add_280 = view_477 = None
        var_mean_95 = torch.ops.aten.var_mean.correction(add_285, [2], correction = 0, keepdim = True)
        getitem_376 = var_mean_95[0]
        getitem_377 = var_mean_95[1];  var_mean_95 = None
        add_286 = torch.ops.aten.add.Tensor(getitem_376, 1e-06);  getitem_376 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_95 = torch.ops.aten.sub.Tensor(add_285, getitem_377);  getitem_377 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, arg55_1);  mul_280 = arg55_1 = None
        add_287 = torch.ops.aten.add.Tensor(mul_281, arg56_1);  mul_281 = arg56_1 = None
        view_478 = torch.ops.aten.view.default(add_287, [25088, 64]);  add_287 = None
        permute_326 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg58_1, view_478, permute_326);  arg58_1 = view_478 = permute_326 = None
        view_479 = torch.ops.aten.view.default(addmm_154, [8, 3136, 512]);  addmm_154 = None
        mul_282 = torch.ops.aten.mul.Tensor(view_479, 0.5)
        mul_283 = torch.ops.aten.mul.Tensor(view_479, 0.7071067811865476);  view_479 = None
        erf_30 = torch.ops.aten.erf.default(mul_283);  mul_283 = None
        add_288 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_282, add_288);  mul_282 = add_288 = None
        view_480 = torch.ops.aten.view.default(mul_284, [25088, 512]);  mul_284 = None
        permute_327 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg60_1, view_480, permute_327);  arg60_1 = view_480 = permute_327 = None
        view_481 = torch.ops.aten.view.default(addmm_155, [8, 3136, 64]);  addmm_155 = None
        add_289 = torch.ops.aten.add.Tensor(add_285, view_481);  add_285 = view_481 = None
        view_482 = torch.ops.aten.view.default(add_289, [8, 56, 56, -1]);  add_289 = None
        permute_328 = torch.ops.aten.permute.default(view_482, [0, 3, 1, 2]);  view_482 = None
        clone_107 = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        convolution_38 = torch.ops.aten.convolution.default(clone_107, arg61_1, arg62_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_107 = arg61_1 = arg62_1 = None
        view_483 = torch.ops.aten.view.default(convolution_38, [8, 128, 784]);  convolution_38 = None
        permute_329 = torch.ops.aten.permute.default(view_483, [0, 2, 1]);  view_483 = None
        clone_108 = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_108, [2], correction = 0, keepdim = True)
        getitem_378 = var_mean_96[0]
        getitem_379 = var_mean_96[1];  var_mean_96 = None
        add_290 = torch.ops.aten.add.Tensor(getitem_378, 1e-05);  getitem_378 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        sub_96 = torch.ops.aten.sub.Tensor(clone_108, getitem_379);  clone_108 = getitem_379 = None
        mul_285 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_285, arg63_1);  mul_285 = arg63_1 = None
        add_291 = torch.ops.aten.add.Tensor(mul_286, arg64_1);  mul_286 = arg64_1 = None
        var_mean_97 = torch.ops.aten.var_mean.correction(add_291, [2], correction = 0, keepdim = True)
        getitem_380 = var_mean_97[0]
        getitem_381 = var_mean_97[1];  var_mean_97 = None
        add_292 = torch.ops.aten.add.Tensor(getitem_380, 1e-06);  getitem_380 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_97 = torch.ops.aten.sub.Tensor(add_291, getitem_381);  getitem_381 = None
        mul_287 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_287, arg65_1);  mul_287 = arg65_1 = None
        add_293 = torch.ops.aten.add.Tensor(mul_288, arg66_1);  mul_288 = arg66_1 = None
        view_484 = torch.ops.aten.view.default(add_293, [6272, 128])
        permute_330 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg68_1, view_484, permute_330);  arg68_1 = view_484 = permute_330 = None
        view_485 = torch.ops.aten.view.default(addmm_156, [8, 784, 128]);  addmm_156 = None
        view_486 = torch.ops.aten.view.default(view_485, [8, 784, 2, 64]);  view_485 = None
        permute_331 = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        permute_332 = torch.ops.aten.permute.default(add_293, [0, 2, 1]);  add_293 = None
        view_487 = torch.ops.aten.view.default(permute_332, [8, 128, 28, 28]);  permute_332 = None
        convolution_39 = torch.ops.aten.convolution.default(view_487, arg69_1, arg70_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_487 = arg69_1 = arg70_1 = None
        view_488 = torch.ops.aten.view.default(convolution_39, [8, 128, 49]);  convolution_39 = None
        permute_333 = torch.ops.aten.permute.default(view_488, [0, 2, 1]);  view_488 = None
        var_mean_98 = torch.ops.aten.var_mean.correction(permute_333, [2], correction = 0, keepdim = True)
        getitem_382 = var_mean_98[0]
        getitem_383 = var_mean_98[1];  var_mean_98 = None
        add_294 = torch.ops.aten.add.Tensor(getitem_382, 1e-05);  getitem_382 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        sub_98 = torch.ops.aten.sub.Tensor(permute_333, getitem_383);  permute_333 = getitem_383 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = rsqrt_98 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, arg71_1);  mul_289 = arg71_1 = None
        add_295 = torch.ops.aten.add.Tensor(mul_290, arg72_1);  mul_290 = arg72_1 = None
        view_489 = torch.ops.aten.view.default(add_295, [392, 128]);  add_295 = None
        permute_334 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg74_1, view_489, permute_334);  arg74_1 = view_489 = permute_334 = None
        view_490 = torch.ops.aten.view.default(addmm_157, [8, 49, 256]);  addmm_157 = None
        view_491 = torch.ops.aten.view.default(view_490, [8, -1, 2, 2, 64]);  view_490 = None
        permute_335 = torch.ops.aten.permute.default(view_491, [2, 0, 3, 1, 4]);  view_491 = None
        unbind_31 = torch.ops.aten.unbind.int(permute_335);  permute_335 = None
        getitem_384 = unbind_31[0]
        getitem_385 = unbind_31[1];  unbind_31 = None
        _scaled_dot_product_efficient_attention_31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_331, getitem_384, getitem_385, None, False);  permute_331 = getitem_384 = getitem_385 = None
        getitem_386 = _scaled_dot_product_efficient_attention_31[0];  _scaled_dot_product_efficient_attention_31 = None
        permute_336 = torch.ops.aten.permute.default(getitem_386, [0, 2, 1, 3]);  getitem_386 = None
        view_492 = torch.ops.aten.view.default(permute_336, [8, 784, 128]);  permute_336 = None
        view_493 = torch.ops.aten.view.default(view_492, [6272, 128]);  view_492 = None
        permute_337 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg76_1, view_493, permute_337);  arg76_1 = view_493 = permute_337 = None
        view_494 = torch.ops.aten.view.default(addmm_158, [8, 784, 128]);  addmm_158 = None
        add_296 = torch.ops.aten.add.Tensor(add_291, view_494);  add_291 = view_494 = None
        var_mean_99 = torch.ops.aten.var_mean.correction(add_296, [2], correction = 0, keepdim = True)
        getitem_390 = var_mean_99[0]
        getitem_391 = var_mean_99[1];  var_mean_99 = None
        add_297 = torch.ops.aten.add.Tensor(getitem_390, 1e-06);  getitem_390 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_297);  add_297 = None
        sub_99 = torch.ops.aten.sub.Tensor(add_296, getitem_391);  getitem_391 = None
        mul_291 = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = rsqrt_99 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_291, arg77_1);  mul_291 = arg77_1 = None
        add_298 = torch.ops.aten.add.Tensor(mul_292, arg78_1);  mul_292 = arg78_1 = None
        view_495 = torch.ops.aten.view.default(add_298, [6272, 128]);  add_298 = None
        permute_338 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg80_1, view_495, permute_338);  arg80_1 = view_495 = permute_338 = None
        view_496 = torch.ops.aten.view.default(addmm_159, [8, 784, 1024]);  addmm_159 = None
        mul_293 = torch.ops.aten.mul.Tensor(view_496, 0.5)
        mul_294 = torch.ops.aten.mul.Tensor(view_496, 0.7071067811865476);  view_496 = None
        erf_31 = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_299 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_293, add_299);  mul_293 = add_299 = None
        view_497 = torch.ops.aten.view.default(mul_295, [6272, 1024]);  mul_295 = None
        permute_339 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg82_1, view_497, permute_339);  arg82_1 = view_497 = permute_339 = None
        view_498 = torch.ops.aten.view.default(addmm_160, [8, 784, 128]);  addmm_160 = None
        add_300 = torch.ops.aten.add.Tensor(add_296, view_498);  add_296 = view_498 = None
        permute_340 = torch.ops.aten.permute.default(add_300, [0, 2, 1]);  add_300 = None
        view_499 = torch.ops.aten.view.default(permute_340, [8, 128, 28, 28]);  permute_340 = None
        convolution_40 = torch.ops.aten.convolution.default(view_499, arg83_1, arg84_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg83_1 = arg84_1 = None
        add_301 = torch.ops.aten.add.Tensor(convolution_40, view_499);  convolution_40 = view_499 = None
        view_501 = torch.ops.aten.view.default(add_301, [8, 128, 784]);  add_301 = None
        permute_342 = torch.ops.aten.permute.default(view_501, [0, 2, 1]);  view_501 = None
        var_mean_100 = torch.ops.aten.var_mean.correction(permute_342, [2], correction = 0, keepdim = True)
        getitem_392 = var_mean_100[0]
        getitem_393 = var_mean_100[1];  var_mean_100 = None
        add_302 = torch.ops.aten.add.Tensor(getitem_392, 1e-06);  getitem_392 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
        sub_100 = torch.ops.aten.sub.Tensor(permute_342, getitem_393);  getitem_393 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = rsqrt_100 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, arg85_1);  mul_296 = arg85_1 = None
        add_303 = torch.ops.aten.add.Tensor(mul_297, arg86_1);  mul_297 = arg86_1 = None
        view_502 = torch.ops.aten.view.default(add_303, [6272, 128])
        permute_343 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg88_1, view_502, permute_343);  arg88_1 = view_502 = permute_343 = None
        view_503 = torch.ops.aten.view.default(addmm_161, [8, 784, 128]);  addmm_161 = None
        view_504 = torch.ops.aten.view.default(view_503, [8, 784, 2, 64]);  view_503 = None
        permute_344 = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        permute_345 = torch.ops.aten.permute.default(add_303, [0, 2, 1]);  add_303 = None
        view_505 = torch.ops.aten.view.default(permute_345, [8, 128, 28, 28]);  permute_345 = None
        convolution_41 = torch.ops.aten.convolution.default(view_505, arg89_1, arg90_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_505 = arg89_1 = arg90_1 = None
        view_506 = torch.ops.aten.view.default(convolution_41, [8, 128, 49]);  convolution_41 = None
        permute_346 = torch.ops.aten.permute.default(view_506, [0, 2, 1]);  view_506 = None
        var_mean_101 = torch.ops.aten.var_mean.correction(permute_346, [2], correction = 0, keepdim = True)
        getitem_394 = var_mean_101[0]
        getitem_395 = var_mean_101[1];  var_mean_101 = None
        add_304 = torch.ops.aten.add.Tensor(getitem_394, 1e-05);  getitem_394 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_304);  add_304 = None
        sub_101 = torch.ops.aten.sub.Tensor(permute_346, getitem_395);  permute_346 = getitem_395 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = rsqrt_101 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, arg91_1);  mul_298 = arg91_1 = None
        add_305 = torch.ops.aten.add.Tensor(mul_299, arg92_1);  mul_299 = arg92_1 = None
        view_507 = torch.ops.aten.view.default(add_305, [392, 128]);  add_305 = None
        permute_347 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg94_1, view_507, permute_347);  arg94_1 = view_507 = permute_347 = None
        view_508 = torch.ops.aten.view.default(addmm_162, [8, 49, 256]);  addmm_162 = None
        view_509 = torch.ops.aten.view.default(view_508, [8, -1, 2, 2, 64]);  view_508 = None
        permute_348 = torch.ops.aten.permute.default(view_509, [2, 0, 3, 1, 4]);  view_509 = None
        unbind_32 = torch.ops.aten.unbind.int(permute_348);  permute_348 = None
        getitem_396 = unbind_32[0]
        getitem_397 = unbind_32[1];  unbind_32 = None
        _scaled_dot_product_efficient_attention_32 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_344, getitem_396, getitem_397, None, False);  permute_344 = getitem_396 = getitem_397 = None
        getitem_398 = _scaled_dot_product_efficient_attention_32[0];  _scaled_dot_product_efficient_attention_32 = None
        permute_349 = torch.ops.aten.permute.default(getitem_398, [0, 2, 1, 3]);  getitem_398 = None
        view_510 = torch.ops.aten.view.default(permute_349, [8, 784, 128]);  permute_349 = None
        view_511 = torch.ops.aten.view.default(view_510, [6272, 128]);  view_510 = None
        permute_350 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg96_1, view_511, permute_350);  arg96_1 = view_511 = permute_350 = None
        view_512 = torch.ops.aten.view.default(addmm_163, [8, 784, 128]);  addmm_163 = None
        add_306 = torch.ops.aten.add.Tensor(permute_342, view_512);  permute_342 = view_512 = None
        var_mean_102 = torch.ops.aten.var_mean.correction(add_306, [2], correction = 0, keepdim = True)
        getitem_402 = var_mean_102[0]
        getitem_403 = var_mean_102[1];  var_mean_102 = None
        add_307 = torch.ops.aten.add.Tensor(getitem_402, 1e-06);  getitem_402 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
        sub_102 = torch.ops.aten.sub.Tensor(add_306, getitem_403);  getitem_403 = None
        mul_300 = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = rsqrt_102 = None
        mul_301 = torch.ops.aten.mul.Tensor(mul_300, arg97_1);  mul_300 = arg97_1 = None
        add_308 = torch.ops.aten.add.Tensor(mul_301, arg98_1);  mul_301 = arg98_1 = None
        view_513 = torch.ops.aten.view.default(add_308, [6272, 128]);  add_308 = None
        permute_351 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg100_1, view_513, permute_351);  arg100_1 = view_513 = permute_351 = None
        view_514 = torch.ops.aten.view.default(addmm_164, [8, 784, 1024]);  addmm_164 = None
        mul_302 = torch.ops.aten.mul.Tensor(view_514, 0.5)
        mul_303 = torch.ops.aten.mul.Tensor(view_514, 0.7071067811865476);  view_514 = None
        erf_32 = torch.ops.aten.erf.default(mul_303);  mul_303 = None
        add_309 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_302, add_309);  mul_302 = add_309 = None
        view_515 = torch.ops.aten.view.default(mul_304, [6272, 1024]);  mul_304 = None
        permute_352 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg102_1, view_515, permute_352);  arg102_1 = view_515 = permute_352 = None
        view_516 = torch.ops.aten.view.default(addmm_165, [8, 784, 128]);  addmm_165 = None
        add_310 = torch.ops.aten.add.Tensor(add_306, view_516);  add_306 = view_516 = None
        var_mean_103 = torch.ops.aten.var_mean.correction(add_310, [2], correction = 0, keepdim = True)
        getitem_404 = var_mean_103[0]
        getitem_405 = var_mean_103[1];  var_mean_103 = None
        add_311 = torch.ops.aten.add.Tensor(getitem_404, 1e-06);  getitem_404 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_311);  add_311 = None
        sub_103 = torch.ops.aten.sub.Tensor(add_310, getitem_405);  getitem_405 = None
        mul_305 = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = rsqrt_103 = None
        mul_306 = torch.ops.aten.mul.Tensor(mul_305, arg103_1);  mul_305 = arg103_1 = None
        add_312 = torch.ops.aten.add.Tensor(mul_306, arg104_1);  mul_306 = arg104_1 = None
        view_517 = torch.ops.aten.view.default(add_312, [6272, 128])
        permute_353 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg106_1, view_517, permute_353);  arg106_1 = view_517 = permute_353 = None
        view_518 = torch.ops.aten.view.default(addmm_166, [8, 784, 128]);  addmm_166 = None
        view_519 = torch.ops.aten.view.default(view_518, [8, 784, 2, 64]);  view_518 = None
        permute_354 = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
        permute_355 = torch.ops.aten.permute.default(add_312, [0, 2, 1]);  add_312 = None
        view_520 = torch.ops.aten.view.default(permute_355, [8, 128, 28, 28]);  permute_355 = None
        convolution_42 = torch.ops.aten.convolution.default(view_520, arg107_1, arg108_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_520 = arg107_1 = arg108_1 = None
        view_521 = torch.ops.aten.view.default(convolution_42, [8, 128, 49]);  convolution_42 = None
        permute_356 = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
        var_mean_104 = torch.ops.aten.var_mean.correction(permute_356, [2], correction = 0, keepdim = True)
        getitem_406 = var_mean_104[0]
        getitem_407 = var_mean_104[1];  var_mean_104 = None
        add_313 = torch.ops.aten.add.Tensor(getitem_406, 1e-05);  getitem_406 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_104 = torch.ops.aten.sub.Tensor(permute_356, getitem_407);  permute_356 = getitem_407 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = rsqrt_104 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, arg109_1);  mul_307 = arg109_1 = None
        add_314 = torch.ops.aten.add.Tensor(mul_308, arg110_1);  mul_308 = arg110_1 = None
        view_522 = torch.ops.aten.view.default(add_314, [392, 128]);  add_314 = None
        permute_357 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg112_1, view_522, permute_357);  arg112_1 = view_522 = permute_357 = None
        view_523 = torch.ops.aten.view.default(addmm_167, [8, 49, 256]);  addmm_167 = None
        view_524 = torch.ops.aten.view.default(view_523, [8, -1, 2, 2, 64]);  view_523 = None
        permute_358 = torch.ops.aten.permute.default(view_524, [2, 0, 3, 1, 4]);  view_524 = None
        unbind_33 = torch.ops.aten.unbind.int(permute_358);  permute_358 = None
        getitem_408 = unbind_33[0]
        getitem_409 = unbind_33[1];  unbind_33 = None
        _scaled_dot_product_efficient_attention_33 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_354, getitem_408, getitem_409, None, False);  permute_354 = getitem_408 = getitem_409 = None
        getitem_410 = _scaled_dot_product_efficient_attention_33[0];  _scaled_dot_product_efficient_attention_33 = None
        permute_359 = torch.ops.aten.permute.default(getitem_410, [0, 2, 1, 3]);  getitem_410 = None
        view_525 = torch.ops.aten.view.default(permute_359, [8, 784, 128]);  permute_359 = None
        view_526 = torch.ops.aten.view.default(view_525, [6272, 128]);  view_525 = None
        permute_360 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg114_1, view_526, permute_360);  arg114_1 = view_526 = permute_360 = None
        view_527 = torch.ops.aten.view.default(addmm_168, [8, 784, 128]);  addmm_168 = None
        add_315 = torch.ops.aten.add.Tensor(add_310, view_527);  add_310 = view_527 = None
        var_mean_105 = torch.ops.aten.var_mean.correction(add_315, [2], correction = 0, keepdim = True)
        getitem_414 = var_mean_105[0]
        getitem_415 = var_mean_105[1];  var_mean_105 = None
        add_316 = torch.ops.aten.add.Tensor(getitem_414, 1e-06);  getitem_414 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
        sub_105 = torch.ops.aten.sub.Tensor(add_315, getitem_415);  getitem_415 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = rsqrt_105 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_309, arg115_1);  mul_309 = arg115_1 = None
        add_317 = torch.ops.aten.add.Tensor(mul_310, arg116_1);  mul_310 = arg116_1 = None
        view_528 = torch.ops.aten.view.default(add_317, [6272, 128]);  add_317 = None
        permute_361 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg118_1, view_528, permute_361);  arg118_1 = view_528 = permute_361 = None
        view_529 = torch.ops.aten.view.default(addmm_169, [8, 784, 1024]);  addmm_169 = None
        mul_311 = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_312 = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
        erf_33 = torch.ops.aten.erf.default(mul_312);  mul_312 = None
        add_318 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_311, add_318);  mul_311 = add_318 = None
        view_530 = torch.ops.aten.view.default(mul_313, [6272, 1024]);  mul_313 = None
        permute_362 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg120_1, view_530, permute_362);  arg120_1 = view_530 = permute_362 = None
        view_531 = torch.ops.aten.view.default(addmm_170, [8, 784, 128]);  addmm_170 = None
        add_319 = torch.ops.aten.add.Tensor(add_315, view_531);  add_315 = view_531 = None
        var_mean_106 = torch.ops.aten.var_mean.correction(add_319, [2], correction = 0, keepdim = True)
        getitem_416 = var_mean_106[0]
        getitem_417 = var_mean_106[1];  var_mean_106 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_416, 1e-06);  getitem_416 = None
        rsqrt_106 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_106 = torch.ops.aten.sub.Tensor(add_319, getitem_417);  getitem_417 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = rsqrt_106 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, arg121_1);  mul_314 = arg121_1 = None
        add_321 = torch.ops.aten.add.Tensor(mul_315, arg122_1);  mul_315 = arg122_1 = None
        view_532 = torch.ops.aten.view.default(add_321, [6272, 128])
        permute_363 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg124_1, view_532, permute_363);  arg124_1 = view_532 = permute_363 = None
        view_533 = torch.ops.aten.view.default(addmm_171, [8, 784, 128]);  addmm_171 = None
        view_534 = torch.ops.aten.view.default(view_533, [8, 784, 2, 64]);  view_533 = None
        permute_364 = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        permute_365 = torch.ops.aten.permute.default(add_321, [0, 2, 1]);  add_321 = None
        view_535 = torch.ops.aten.view.default(permute_365, [8, 128, 28, 28]);  permute_365 = None
        convolution_43 = torch.ops.aten.convolution.default(view_535, arg125_1, arg126_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_535 = arg125_1 = arg126_1 = None
        view_536 = torch.ops.aten.view.default(convolution_43, [8, 128, 49]);  convolution_43 = None
        permute_366 = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None
        var_mean_107 = torch.ops.aten.var_mean.correction(permute_366, [2], correction = 0, keepdim = True)
        getitem_418 = var_mean_107[0]
        getitem_419 = var_mean_107[1];  var_mean_107 = None
        add_322 = torch.ops.aten.add.Tensor(getitem_418, 1e-05);  getitem_418 = None
        rsqrt_107 = torch.ops.aten.rsqrt.default(add_322);  add_322 = None
        sub_107 = torch.ops.aten.sub.Tensor(permute_366, getitem_419);  permute_366 = getitem_419 = None
        mul_316 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = rsqrt_107 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, arg127_1);  mul_316 = arg127_1 = None
        add_323 = torch.ops.aten.add.Tensor(mul_317, arg128_1);  mul_317 = arg128_1 = None
        view_537 = torch.ops.aten.view.default(add_323, [392, 128]);  add_323 = None
        permute_367 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_172 = torch.ops.aten.addmm.default(arg130_1, view_537, permute_367);  arg130_1 = view_537 = permute_367 = None
        view_538 = torch.ops.aten.view.default(addmm_172, [8, 49, 256]);  addmm_172 = None
        view_539 = torch.ops.aten.view.default(view_538, [8, -1, 2, 2, 64]);  view_538 = None
        permute_368 = torch.ops.aten.permute.default(view_539, [2, 0, 3, 1, 4]);  view_539 = None
        unbind_34 = torch.ops.aten.unbind.int(permute_368);  permute_368 = None
        getitem_420 = unbind_34[0]
        getitem_421 = unbind_34[1];  unbind_34 = None
        _scaled_dot_product_efficient_attention_34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_364, getitem_420, getitem_421, None, False);  permute_364 = getitem_420 = getitem_421 = None
        getitem_422 = _scaled_dot_product_efficient_attention_34[0];  _scaled_dot_product_efficient_attention_34 = None
        permute_369 = torch.ops.aten.permute.default(getitem_422, [0, 2, 1, 3]);  getitem_422 = None
        view_540 = torch.ops.aten.view.default(permute_369, [8, 784, 128]);  permute_369 = None
        view_541 = torch.ops.aten.view.default(view_540, [6272, 128]);  view_540 = None
        permute_370 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_173 = torch.ops.aten.addmm.default(arg132_1, view_541, permute_370);  arg132_1 = view_541 = permute_370 = None
        view_542 = torch.ops.aten.view.default(addmm_173, [8, 784, 128]);  addmm_173 = None
        add_324 = torch.ops.aten.add.Tensor(add_319, view_542);  add_319 = view_542 = None
        var_mean_108 = torch.ops.aten.var_mean.correction(add_324, [2], correction = 0, keepdim = True)
        getitem_426 = var_mean_108[0]
        getitem_427 = var_mean_108[1];  var_mean_108 = None
        add_325 = torch.ops.aten.add.Tensor(getitem_426, 1e-06);  getitem_426 = None
        rsqrt_108 = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
        sub_108 = torch.ops.aten.sub.Tensor(add_324, getitem_427);  getitem_427 = None
        mul_318 = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = rsqrt_108 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, arg133_1);  mul_318 = arg133_1 = None
        add_326 = torch.ops.aten.add.Tensor(mul_319, arg134_1);  mul_319 = arg134_1 = None
        view_543 = torch.ops.aten.view.default(add_326, [6272, 128]);  add_326 = None
        permute_371 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_174 = torch.ops.aten.addmm.default(arg136_1, view_543, permute_371);  arg136_1 = view_543 = permute_371 = None
        view_544 = torch.ops.aten.view.default(addmm_174, [8, 784, 1024]);  addmm_174 = None
        mul_320 = torch.ops.aten.mul.Tensor(view_544, 0.5)
        mul_321 = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476);  view_544 = None
        erf_34 = torch.ops.aten.erf.default(mul_321);  mul_321 = None
        add_327 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_320, add_327);  mul_320 = add_327 = None
        view_545 = torch.ops.aten.view.default(mul_322, [6272, 1024]);  mul_322 = None
        permute_372 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_175 = torch.ops.aten.addmm.default(arg138_1, view_545, permute_372);  arg138_1 = view_545 = permute_372 = None
        view_546 = torch.ops.aten.view.default(addmm_175, [8, 784, 128]);  addmm_175 = None
        add_328 = torch.ops.aten.add.Tensor(add_324, view_546);  add_324 = view_546 = None
        view_547 = torch.ops.aten.view.default(add_328, [8, 28, 28, -1]);  add_328 = None
        permute_373 = torch.ops.aten.permute.default(view_547, [0, 3, 1, 2]);  view_547 = None
        clone_122 = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
        convolution_44 = torch.ops.aten.convolution.default(clone_122, arg139_1, arg140_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_122 = arg139_1 = arg140_1 = None
        view_548 = torch.ops.aten.view.default(convolution_44, [8, 320, 196]);  convolution_44 = None
        permute_374 = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None
        clone_123 = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        var_mean_109 = torch.ops.aten.var_mean.correction(clone_123, [2], correction = 0, keepdim = True)
        getitem_428 = var_mean_109[0]
        getitem_429 = var_mean_109[1];  var_mean_109 = None
        add_329 = torch.ops.aten.add.Tensor(getitem_428, 1e-05);  getitem_428 = None
        rsqrt_109 = torch.ops.aten.rsqrt.default(add_329);  add_329 = None
        sub_109 = torch.ops.aten.sub.Tensor(clone_123, getitem_429);  clone_123 = getitem_429 = None
        mul_323 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = rsqrt_109 = None
        mul_324 = torch.ops.aten.mul.Tensor(mul_323, arg141_1);  mul_323 = arg141_1 = None
        add_330 = torch.ops.aten.add.Tensor(mul_324, arg142_1);  mul_324 = arg142_1 = None
        var_mean_110 = torch.ops.aten.var_mean.correction(add_330, [2], correction = 0, keepdim = True)
        getitem_430 = var_mean_110[0]
        getitem_431 = var_mean_110[1];  var_mean_110 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_430, 1e-06);  getitem_430 = None
        rsqrt_110 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_110 = torch.ops.aten.sub.Tensor(add_330, getitem_431);  getitem_431 = None
        mul_325 = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = rsqrt_110 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, arg143_1);  mul_325 = arg143_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_326, arg144_1);  mul_326 = arg144_1 = None
        view_549 = torch.ops.aten.view.default(add_332, [1568, 320])
        permute_375 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_176 = torch.ops.aten.addmm.default(arg146_1, view_549, permute_375);  arg146_1 = view_549 = permute_375 = None
        view_550 = torch.ops.aten.view.default(addmm_176, [8, 196, 320]);  addmm_176 = None
        view_551 = torch.ops.aten.view.default(view_550, [8, 196, 5, 64]);  view_550 = None
        permute_376 = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
        permute_377 = torch.ops.aten.permute.default(add_332, [0, 2, 1]);  add_332 = None
        view_552 = torch.ops.aten.view.default(permute_377, [8, 320, 14, 14]);  permute_377 = None
        convolution_45 = torch.ops.aten.convolution.default(view_552, arg147_1, arg148_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_552 = arg147_1 = arg148_1 = None
        view_553 = torch.ops.aten.view.default(convolution_45, [8, 320, 49]);  convolution_45 = None
        permute_378 = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None
        var_mean_111 = torch.ops.aten.var_mean.correction(permute_378, [2], correction = 0, keepdim = True)
        getitem_432 = var_mean_111[0]
        getitem_433 = var_mean_111[1];  var_mean_111 = None
        add_333 = torch.ops.aten.add.Tensor(getitem_432, 1e-05);  getitem_432 = None
        rsqrt_111 = torch.ops.aten.rsqrt.default(add_333);  add_333 = None
        sub_111 = torch.ops.aten.sub.Tensor(permute_378, getitem_433);  permute_378 = getitem_433 = None
        mul_327 = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = rsqrt_111 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_327, arg149_1);  mul_327 = arg149_1 = None
        add_334 = torch.ops.aten.add.Tensor(mul_328, arg150_1);  mul_328 = arg150_1 = None
        view_554 = torch.ops.aten.view.default(add_334, [392, 320]);  add_334 = None
        permute_379 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_177 = torch.ops.aten.addmm.default(arg152_1, view_554, permute_379);  arg152_1 = view_554 = permute_379 = None
        view_555 = torch.ops.aten.view.default(addmm_177, [8, 49, 640]);  addmm_177 = None
        view_556 = torch.ops.aten.view.default(view_555, [8, -1, 2, 5, 64]);  view_555 = None
        permute_380 = torch.ops.aten.permute.default(view_556, [2, 0, 3, 1, 4]);  view_556 = None
        unbind_35 = torch.ops.aten.unbind.int(permute_380);  permute_380 = None
        getitem_434 = unbind_35[0]
        getitem_435 = unbind_35[1];  unbind_35 = None
        _scaled_dot_product_efficient_attention_35 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_376, getitem_434, getitem_435, None, False);  permute_376 = getitem_434 = getitem_435 = None
        getitem_436 = _scaled_dot_product_efficient_attention_35[0];  _scaled_dot_product_efficient_attention_35 = None
        permute_381 = torch.ops.aten.permute.default(getitem_436, [0, 2, 1, 3]);  getitem_436 = None
        view_557 = torch.ops.aten.view.default(permute_381, [8, 196, 320]);  permute_381 = None
        view_558 = torch.ops.aten.view.default(view_557, [1568, 320]);  view_557 = None
        permute_382 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_178 = torch.ops.aten.addmm.default(arg154_1, view_558, permute_382);  arg154_1 = view_558 = permute_382 = None
        view_559 = torch.ops.aten.view.default(addmm_178, [8, 196, 320]);  addmm_178 = None
        add_335 = torch.ops.aten.add.Tensor(add_330, view_559);  add_330 = view_559 = None
        var_mean_112 = torch.ops.aten.var_mean.correction(add_335, [2], correction = 0, keepdim = True)
        getitem_440 = var_mean_112[0]
        getitem_441 = var_mean_112[1];  var_mean_112 = None
        add_336 = torch.ops.aten.add.Tensor(getitem_440, 1e-06);  getitem_440 = None
        rsqrt_112 = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
        sub_112 = torch.ops.aten.sub.Tensor(add_335, getitem_441);  getitem_441 = None
        mul_329 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = rsqrt_112 = None
        mul_330 = torch.ops.aten.mul.Tensor(mul_329, arg155_1);  mul_329 = arg155_1 = None
        add_337 = torch.ops.aten.add.Tensor(mul_330, arg156_1);  mul_330 = arg156_1 = None
        view_560 = torch.ops.aten.view.default(add_337, [1568, 320]);  add_337 = None
        permute_383 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_179 = torch.ops.aten.addmm.default(arg158_1, view_560, permute_383);  arg158_1 = view_560 = permute_383 = None
        view_561 = torch.ops.aten.view.default(addmm_179, [8, 196, 1280]);  addmm_179 = None
        mul_331 = torch.ops.aten.mul.Tensor(view_561, 0.5)
        mul_332 = torch.ops.aten.mul.Tensor(view_561, 0.7071067811865476);  view_561 = None
        erf_35 = torch.ops.aten.erf.default(mul_332);  mul_332 = None
        add_338 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_331, add_338);  mul_331 = add_338 = None
        view_562 = torch.ops.aten.view.default(mul_333, [1568, 1280]);  mul_333 = None
        permute_384 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_180 = torch.ops.aten.addmm.default(arg160_1, view_562, permute_384);  arg160_1 = view_562 = permute_384 = None
        view_563 = torch.ops.aten.view.default(addmm_180, [8, 196, 320]);  addmm_180 = None
        add_339 = torch.ops.aten.add.Tensor(add_335, view_563);  add_335 = view_563 = None
        permute_385 = torch.ops.aten.permute.default(add_339, [0, 2, 1]);  add_339 = None
        view_564 = torch.ops.aten.view.default(permute_385, [8, 320, 14, 14]);  permute_385 = None
        convolution_46 = torch.ops.aten.convolution.default(view_564, arg161_1, arg162_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  arg161_1 = arg162_1 = None
        add_340 = torch.ops.aten.add.Tensor(convolution_46, view_564);  convolution_46 = view_564 = None
        view_566 = torch.ops.aten.view.default(add_340, [8, 320, 196]);  add_340 = None
        permute_387 = torch.ops.aten.permute.default(view_566, [0, 2, 1]);  view_566 = None
        var_mean_113 = torch.ops.aten.var_mean.correction(permute_387, [2], correction = 0, keepdim = True)
        getitem_442 = var_mean_113[0]
        getitem_443 = var_mean_113[1];  var_mean_113 = None
        add_341 = torch.ops.aten.add.Tensor(getitem_442, 1e-06);  getitem_442 = None
        rsqrt_113 = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_113 = torch.ops.aten.sub.Tensor(permute_387, getitem_443);  getitem_443 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = rsqrt_113 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, arg163_1);  mul_334 = arg163_1 = None
        add_342 = torch.ops.aten.add.Tensor(mul_335, arg164_1);  mul_335 = arg164_1 = None
        view_567 = torch.ops.aten.view.default(add_342, [1568, 320])
        permute_388 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg166_1, view_567, permute_388);  arg166_1 = view_567 = permute_388 = None
        view_568 = torch.ops.aten.view.default(addmm_181, [8, 196, 320]);  addmm_181 = None
        view_569 = torch.ops.aten.view.default(view_568, [8, 196, 5, 64]);  view_568 = None
        permute_389 = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
        permute_390 = torch.ops.aten.permute.default(add_342, [0, 2, 1]);  add_342 = None
        view_570 = torch.ops.aten.view.default(permute_390, [8, 320, 14, 14]);  permute_390 = None
        convolution_47 = torch.ops.aten.convolution.default(view_570, arg167_1, arg168_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_570 = arg167_1 = arg168_1 = None
        view_571 = torch.ops.aten.view.default(convolution_47, [8, 320, 49]);  convolution_47 = None
        permute_391 = torch.ops.aten.permute.default(view_571, [0, 2, 1]);  view_571 = None
        var_mean_114 = torch.ops.aten.var_mean.correction(permute_391, [2], correction = 0, keepdim = True)
        getitem_444 = var_mean_114[0]
        getitem_445 = var_mean_114[1];  var_mean_114 = None
        add_343 = torch.ops.aten.add.Tensor(getitem_444, 1e-05);  getitem_444 = None
        rsqrt_114 = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
        sub_114 = torch.ops.aten.sub.Tensor(permute_391, getitem_445);  permute_391 = getitem_445 = None
        mul_336 = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = rsqrt_114 = None
        mul_337 = torch.ops.aten.mul.Tensor(mul_336, arg169_1);  mul_336 = arg169_1 = None
        add_344 = torch.ops.aten.add.Tensor(mul_337, arg170_1);  mul_337 = arg170_1 = None
        view_572 = torch.ops.aten.view.default(add_344, [392, 320]);  add_344 = None
        permute_392 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg172_1, view_572, permute_392);  arg172_1 = view_572 = permute_392 = None
        view_573 = torch.ops.aten.view.default(addmm_182, [8, 49, 640]);  addmm_182 = None
        view_574 = torch.ops.aten.view.default(view_573, [8, -1, 2, 5, 64]);  view_573 = None
        permute_393 = torch.ops.aten.permute.default(view_574, [2, 0, 3, 1, 4]);  view_574 = None
        unbind_36 = torch.ops.aten.unbind.int(permute_393);  permute_393 = None
        getitem_446 = unbind_36[0]
        getitem_447 = unbind_36[1];  unbind_36 = None
        _scaled_dot_product_efficient_attention_36 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_389, getitem_446, getitem_447, None, False);  permute_389 = getitem_446 = getitem_447 = None
        getitem_448 = _scaled_dot_product_efficient_attention_36[0];  _scaled_dot_product_efficient_attention_36 = None
        permute_394 = torch.ops.aten.permute.default(getitem_448, [0, 2, 1, 3]);  getitem_448 = None
        view_575 = torch.ops.aten.view.default(permute_394, [8, 196, 320]);  permute_394 = None
        view_576 = torch.ops.aten.view.default(view_575, [1568, 320]);  view_575 = None
        permute_395 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg174_1, view_576, permute_395);  arg174_1 = view_576 = permute_395 = None
        view_577 = torch.ops.aten.view.default(addmm_183, [8, 196, 320]);  addmm_183 = None
        add_345 = torch.ops.aten.add.Tensor(permute_387, view_577);  permute_387 = view_577 = None
        var_mean_115 = torch.ops.aten.var_mean.correction(add_345, [2], correction = 0, keepdim = True)
        getitem_452 = var_mean_115[0]
        getitem_453 = var_mean_115[1];  var_mean_115 = None
        add_346 = torch.ops.aten.add.Tensor(getitem_452, 1e-06);  getitem_452 = None
        rsqrt_115 = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
        sub_115 = torch.ops.aten.sub.Tensor(add_345, getitem_453);  getitem_453 = None
        mul_338 = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = rsqrt_115 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, arg175_1);  mul_338 = arg175_1 = None
        add_347 = torch.ops.aten.add.Tensor(mul_339, arg176_1);  mul_339 = arg176_1 = None
        view_578 = torch.ops.aten.view.default(add_347, [1568, 320]);  add_347 = None
        permute_396 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_184 = torch.ops.aten.addmm.default(arg178_1, view_578, permute_396);  arg178_1 = view_578 = permute_396 = None
        view_579 = torch.ops.aten.view.default(addmm_184, [8, 196, 1280]);  addmm_184 = None
        mul_340 = torch.ops.aten.mul.Tensor(view_579, 0.5)
        mul_341 = torch.ops.aten.mul.Tensor(view_579, 0.7071067811865476);  view_579 = None
        erf_36 = torch.ops.aten.erf.default(mul_341);  mul_341 = None
        add_348 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_340, add_348);  mul_340 = add_348 = None
        view_580 = torch.ops.aten.view.default(mul_342, [1568, 1280]);  mul_342 = None
        permute_397 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg180_1, view_580, permute_397);  arg180_1 = view_580 = permute_397 = None
        view_581 = torch.ops.aten.view.default(addmm_185, [8, 196, 320]);  addmm_185 = None
        add_349 = torch.ops.aten.add.Tensor(add_345, view_581);  add_345 = view_581 = None
        var_mean_116 = torch.ops.aten.var_mean.correction(add_349, [2], correction = 0, keepdim = True)
        getitem_454 = var_mean_116[0]
        getitem_455 = var_mean_116[1];  var_mean_116 = None
        add_350 = torch.ops.aten.add.Tensor(getitem_454, 1e-06);  getitem_454 = None
        rsqrt_116 = torch.ops.aten.rsqrt.default(add_350);  add_350 = None
        sub_116 = torch.ops.aten.sub.Tensor(add_349, getitem_455);  getitem_455 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = rsqrt_116 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, arg181_1);  mul_343 = arg181_1 = None
        add_351 = torch.ops.aten.add.Tensor(mul_344, arg182_1);  mul_344 = arg182_1 = None
        view_582 = torch.ops.aten.view.default(add_351, [1568, 320])
        permute_398 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_186 = torch.ops.aten.addmm.default(arg184_1, view_582, permute_398);  arg184_1 = view_582 = permute_398 = None
        view_583 = torch.ops.aten.view.default(addmm_186, [8, 196, 320]);  addmm_186 = None
        view_584 = torch.ops.aten.view.default(view_583, [8, 196, 5, 64]);  view_583 = None
        permute_399 = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
        permute_400 = torch.ops.aten.permute.default(add_351, [0, 2, 1]);  add_351 = None
        view_585 = torch.ops.aten.view.default(permute_400, [8, 320, 14, 14]);  permute_400 = None
        convolution_48 = torch.ops.aten.convolution.default(view_585, arg185_1, arg186_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_585 = arg185_1 = arg186_1 = None
        view_586 = torch.ops.aten.view.default(convolution_48, [8, 320, 49]);  convolution_48 = None
        permute_401 = torch.ops.aten.permute.default(view_586, [0, 2, 1]);  view_586 = None
        var_mean_117 = torch.ops.aten.var_mean.correction(permute_401, [2], correction = 0, keepdim = True)
        getitem_456 = var_mean_117[0]
        getitem_457 = var_mean_117[1];  var_mean_117 = None
        add_352 = torch.ops.aten.add.Tensor(getitem_456, 1e-05);  getitem_456 = None
        rsqrt_117 = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        sub_117 = torch.ops.aten.sub.Tensor(permute_401, getitem_457);  permute_401 = getitem_457 = None
        mul_345 = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = rsqrt_117 = None
        mul_346 = torch.ops.aten.mul.Tensor(mul_345, arg187_1);  mul_345 = arg187_1 = None
        add_353 = torch.ops.aten.add.Tensor(mul_346, arg188_1);  mul_346 = arg188_1 = None
        view_587 = torch.ops.aten.view.default(add_353, [392, 320]);  add_353 = None
        permute_402 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg190_1, view_587, permute_402);  arg190_1 = view_587 = permute_402 = None
        view_588 = torch.ops.aten.view.default(addmm_187, [8, 49, 640]);  addmm_187 = None
        view_589 = torch.ops.aten.view.default(view_588, [8, -1, 2, 5, 64]);  view_588 = None
        permute_403 = torch.ops.aten.permute.default(view_589, [2, 0, 3, 1, 4]);  view_589 = None
        unbind_37 = torch.ops.aten.unbind.int(permute_403);  permute_403 = None
        getitem_458 = unbind_37[0]
        getitem_459 = unbind_37[1];  unbind_37 = None
        _scaled_dot_product_efficient_attention_37 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_399, getitem_458, getitem_459, None, False);  permute_399 = getitem_458 = getitem_459 = None
        getitem_460 = _scaled_dot_product_efficient_attention_37[0];  _scaled_dot_product_efficient_attention_37 = None
        permute_404 = torch.ops.aten.permute.default(getitem_460, [0, 2, 1, 3]);  getitem_460 = None
        view_590 = torch.ops.aten.view.default(permute_404, [8, 196, 320]);  permute_404 = None
        view_591 = torch.ops.aten.view.default(view_590, [1568, 320]);  view_590 = None
        permute_405 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg192_1, view_591, permute_405);  arg192_1 = view_591 = permute_405 = None
        view_592 = torch.ops.aten.view.default(addmm_188, [8, 196, 320]);  addmm_188 = None
        add_354 = torch.ops.aten.add.Tensor(add_349, view_592);  add_349 = view_592 = None
        var_mean_118 = torch.ops.aten.var_mean.correction(add_354, [2], correction = 0, keepdim = True)
        getitem_464 = var_mean_118[0]
        getitem_465 = var_mean_118[1];  var_mean_118 = None
        add_355 = torch.ops.aten.add.Tensor(getitem_464, 1e-06);  getitem_464 = None
        rsqrt_118 = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        sub_118 = torch.ops.aten.sub.Tensor(add_354, getitem_465);  getitem_465 = None
        mul_347 = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = rsqrt_118 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_347, arg193_1);  mul_347 = arg193_1 = None
        add_356 = torch.ops.aten.add.Tensor(mul_348, arg194_1);  mul_348 = arg194_1 = None
        view_593 = torch.ops.aten.view.default(add_356, [1568, 320]);  add_356 = None
        permute_406 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg196_1, view_593, permute_406);  arg196_1 = view_593 = permute_406 = None
        view_594 = torch.ops.aten.view.default(addmm_189, [8, 196, 1280]);  addmm_189 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_594, 0.5)
        mul_350 = torch.ops.aten.mul.Tensor(view_594, 0.7071067811865476);  view_594 = None
        erf_37 = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_357 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_349, add_357);  mul_349 = add_357 = None
        view_595 = torch.ops.aten.view.default(mul_351, [1568, 1280]);  mul_351 = None
        permute_407 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg198_1, view_595, permute_407);  arg198_1 = view_595 = permute_407 = None
        view_596 = torch.ops.aten.view.default(addmm_190, [8, 196, 320]);  addmm_190 = None
        add_358 = torch.ops.aten.add.Tensor(add_354, view_596);  add_354 = view_596 = None
        var_mean_119 = torch.ops.aten.var_mean.correction(add_358, [2], correction = 0, keepdim = True)
        getitem_466 = var_mean_119[0]
        getitem_467 = var_mean_119[1];  var_mean_119 = None
        add_359 = torch.ops.aten.add.Tensor(getitem_466, 1e-06);  getitem_466 = None
        rsqrt_119 = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        sub_119 = torch.ops.aten.sub.Tensor(add_358, getitem_467);  getitem_467 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = rsqrt_119 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, arg199_1);  mul_352 = arg199_1 = None
        add_360 = torch.ops.aten.add.Tensor(mul_353, arg200_1);  mul_353 = arg200_1 = None
        view_597 = torch.ops.aten.view.default(add_360, [1568, 320])
        permute_408 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_191 = torch.ops.aten.addmm.default(arg202_1, view_597, permute_408);  arg202_1 = view_597 = permute_408 = None
        view_598 = torch.ops.aten.view.default(addmm_191, [8, 196, 320]);  addmm_191 = None
        view_599 = torch.ops.aten.view.default(view_598, [8, 196, 5, 64]);  view_598 = None
        permute_409 = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
        permute_410 = torch.ops.aten.permute.default(add_360, [0, 2, 1]);  add_360 = None
        view_600 = torch.ops.aten.view.default(permute_410, [8, 320, 14, 14]);  permute_410 = None
        convolution_49 = torch.ops.aten.convolution.default(view_600, arg203_1, arg204_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_600 = arg203_1 = arg204_1 = None
        view_601 = torch.ops.aten.view.default(convolution_49, [8, 320, 49]);  convolution_49 = None
        permute_411 = torch.ops.aten.permute.default(view_601, [0, 2, 1]);  view_601 = None
        var_mean_120 = torch.ops.aten.var_mean.correction(permute_411, [2], correction = 0, keepdim = True)
        getitem_468 = var_mean_120[0]
        getitem_469 = var_mean_120[1];  var_mean_120 = None
        add_361 = torch.ops.aten.add.Tensor(getitem_468, 1e-05);  getitem_468 = None
        rsqrt_120 = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
        sub_120 = torch.ops.aten.sub.Tensor(permute_411, getitem_469);  permute_411 = getitem_469 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = rsqrt_120 = None
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, arg205_1);  mul_354 = arg205_1 = None
        add_362 = torch.ops.aten.add.Tensor(mul_355, arg206_1);  mul_355 = arg206_1 = None
        view_602 = torch.ops.aten.view.default(add_362, [392, 320]);  add_362 = None
        permute_412 = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_192 = torch.ops.aten.addmm.default(arg208_1, view_602, permute_412);  arg208_1 = view_602 = permute_412 = None
        view_603 = torch.ops.aten.view.default(addmm_192, [8, 49, 640]);  addmm_192 = None
        view_604 = torch.ops.aten.view.default(view_603, [8, -1, 2, 5, 64]);  view_603 = None
        permute_413 = torch.ops.aten.permute.default(view_604, [2, 0, 3, 1, 4]);  view_604 = None
        unbind_38 = torch.ops.aten.unbind.int(permute_413);  permute_413 = None
        getitem_470 = unbind_38[0]
        getitem_471 = unbind_38[1];  unbind_38 = None
        _scaled_dot_product_efficient_attention_38 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_409, getitem_470, getitem_471, None, False);  permute_409 = getitem_470 = getitem_471 = None
        getitem_472 = _scaled_dot_product_efficient_attention_38[0];  _scaled_dot_product_efficient_attention_38 = None
        permute_414 = torch.ops.aten.permute.default(getitem_472, [0, 2, 1, 3]);  getitem_472 = None
        view_605 = torch.ops.aten.view.default(permute_414, [8, 196, 320]);  permute_414 = None
        view_606 = torch.ops.aten.view.default(view_605, [1568, 320]);  view_605 = None
        permute_415 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_193 = torch.ops.aten.addmm.default(arg210_1, view_606, permute_415);  arg210_1 = view_606 = permute_415 = None
        view_607 = torch.ops.aten.view.default(addmm_193, [8, 196, 320]);  addmm_193 = None
        add_363 = torch.ops.aten.add.Tensor(add_358, view_607);  add_358 = view_607 = None
        var_mean_121 = torch.ops.aten.var_mean.correction(add_363, [2], correction = 0, keepdim = True)
        getitem_476 = var_mean_121[0]
        getitem_477 = var_mean_121[1];  var_mean_121 = None
        add_364 = torch.ops.aten.add.Tensor(getitem_476, 1e-06);  getitem_476 = None
        rsqrt_121 = torch.ops.aten.rsqrt.default(add_364);  add_364 = None
        sub_121 = torch.ops.aten.sub.Tensor(add_363, getitem_477);  getitem_477 = None
        mul_356 = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = rsqrt_121 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_356, arg211_1);  mul_356 = arg211_1 = None
        add_365 = torch.ops.aten.add.Tensor(mul_357, arg212_1);  mul_357 = arg212_1 = None
        view_608 = torch.ops.aten.view.default(add_365, [1568, 320]);  add_365 = None
        permute_416 = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_194 = torch.ops.aten.addmm.default(arg214_1, view_608, permute_416);  arg214_1 = view_608 = permute_416 = None
        view_609 = torch.ops.aten.view.default(addmm_194, [8, 196, 1280]);  addmm_194 = None
        mul_358 = torch.ops.aten.mul.Tensor(view_609, 0.5)
        mul_359 = torch.ops.aten.mul.Tensor(view_609, 0.7071067811865476);  view_609 = None
        erf_38 = torch.ops.aten.erf.default(mul_359);  mul_359 = None
        add_366 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_360 = torch.ops.aten.mul.Tensor(mul_358, add_366);  mul_358 = add_366 = None
        view_610 = torch.ops.aten.view.default(mul_360, [1568, 1280]);  mul_360 = None
        permute_417 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_195 = torch.ops.aten.addmm.default(arg216_1, view_610, permute_417);  arg216_1 = view_610 = permute_417 = None
        view_611 = torch.ops.aten.view.default(addmm_195, [8, 196, 320]);  addmm_195 = None
        add_367 = torch.ops.aten.add.Tensor(add_363, view_611);  add_363 = view_611 = None
        var_mean_122 = torch.ops.aten.var_mean.correction(add_367, [2], correction = 0, keepdim = True)
        getitem_478 = var_mean_122[0]
        getitem_479 = var_mean_122[1];  var_mean_122 = None
        add_368 = torch.ops.aten.add.Tensor(getitem_478, 1e-06);  getitem_478 = None
        rsqrt_122 = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
        sub_122 = torch.ops.aten.sub.Tensor(add_367, getitem_479);  getitem_479 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_122, rsqrt_122);  sub_122 = rsqrt_122 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, arg217_1);  mul_361 = arg217_1 = None
        add_369 = torch.ops.aten.add.Tensor(mul_362, arg218_1);  mul_362 = arg218_1 = None
        view_612 = torch.ops.aten.view.default(add_369, [1568, 320])
        permute_418 = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_196 = torch.ops.aten.addmm.default(arg220_1, view_612, permute_418);  arg220_1 = view_612 = permute_418 = None
        view_613 = torch.ops.aten.view.default(addmm_196, [8, 196, 320]);  addmm_196 = None
        view_614 = torch.ops.aten.view.default(view_613, [8, 196, 5, 64]);  view_613 = None
        permute_419 = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        permute_420 = torch.ops.aten.permute.default(add_369, [0, 2, 1]);  add_369 = None
        view_615 = torch.ops.aten.view.default(permute_420, [8, 320, 14, 14]);  permute_420 = None
        convolution_50 = torch.ops.aten.convolution.default(view_615, arg221_1, arg222_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_615 = arg221_1 = arg222_1 = None
        view_616 = torch.ops.aten.view.default(convolution_50, [8, 320, 49]);  convolution_50 = None
        permute_421 = torch.ops.aten.permute.default(view_616, [0, 2, 1]);  view_616 = None
        var_mean_123 = torch.ops.aten.var_mean.correction(permute_421, [2], correction = 0, keepdim = True)
        getitem_480 = var_mean_123[0]
        getitem_481 = var_mean_123[1];  var_mean_123 = None
        add_370 = torch.ops.aten.add.Tensor(getitem_480, 1e-05);  getitem_480 = None
        rsqrt_123 = torch.ops.aten.rsqrt.default(add_370);  add_370 = None
        sub_123 = torch.ops.aten.sub.Tensor(permute_421, getitem_481);  permute_421 = getitem_481 = None
        mul_363 = torch.ops.aten.mul.Tensor(sub_123, rsqrt_123);  sub_123 = rsqrt_123 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, arg223_1);  mul_363 = arg223_1 = None
        add_371 = torch.ops.aten.add.Tensor(mul_364, arg224_1);  mul_364 = arg224_1 = None
        view_617 = torch.ops.aten.view.default(add_371, [392, 320]);  add_371 = None
        permute_422 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_197 = torch.ops.aten.addmm.default(arg226_1, view_617, permute_422);  arg226_1 = view_617 = permute_422 = None
        view_618 = torch.ops.aten.view.default(addmm_197, [8, 49, 640]);  addmm_197 = None
        view_619 = torch.ops.aten.view.default(view_618, [8, -1, 2, 5, 64]);  view_618 = None
        permute_423 = torch.ops.aten.permute.default(view_619, [2, 0, 3, 1, 4]);  view_619 = None
        unbind_39 = torch.ops.aten.unbind.int(permute_423);  permute_423 = None
        getitem_482 = unbind_39[0]
        getitem_483 = unbind_39[1];  unbind_39 = None
        _scaled_dot_product_efficient_attention_39 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_419, getitem_482, getitem_483, None, False);  permute_419 = getitem_482 = getitem_483 = None
        getitem_484 = _scaled_dot_product_efficient_attention_39[0];  _scaled_dot_product_efficient_attention_39 = None
        permute_424 = torch.ops.aten.permute.default(getitem_484, [0, 2, 1, 3]);  getitem_484 = None
        view_620 = torch.ops.aten.view.default(permute_424, [8, 196, 320]);  permute_424 = None
        view_621 = torch.ops.aten.view.default(view_620, [1568, 320]);  view_620 = None
        permute_425 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_198 = torch.ops.aten.addmm.default(arg228_1, view_621, permute_425);  arg228_1 = view_621 = permute_425 = None
        view_622 = torch.ops.aten.view.default(addmm_198, [8, 196, 320]);  addmm_198 = None
        add_372 = torch.ops.aten.add.Tensor(add_367, view_622);  add_367 = view_622 = None
        var_mean_124 = torch.ops.aten.var_mean.correction(add_372, [2], correction = 0, keepdim = True)
        getitem_488 = var_mean_124[0]
        getitem_489 = var_mean_124[1];  var_mean_124 = None
        add_373 = torch.ops.aten.add.Tensor(getitem_488, 1e-06);  getitem_488 = None
        rsqrt_124 = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
        sub_124 = torch.ops.aten.sub.Tensor(add_372, getitem_489);  getitem_489 = None
        mul_365 = torch.ops.aten.mul.Tensor(sub_124, rsqrt_124);  sub_124 = rsqrt_124 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_365, arg229_1);  mul_365 = arg229_1 = None
        add_374 = torch.ops.aten.add.Tensor(mul_366, arg230_1);  mul_366 = arg230_1 = None
        view_623 = torch.ops.aten.view.default(add_374, [1568, 320]);  add_374 = None
        permute_426 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_199 = torch.ops.aten.addmm.default(arg232_1, view_623, permute_426);  arg232_1 = view_623 = permute_426 = None
        view_624 = torch.ops.aten.view.default(addmm_199, [8, 196, 1280]);  addmm_199 = None
        mul_367 = torch.ops.aten.mul.Tensor(view_624, 0.5)
        mul_368 = torch.ops.aten.mul.Tensor(view_624, 0.7071067811865476);  view_624 = None
        erf_39 = torch.ops.aten.erf.default(mul_368);  mul_368 = None
        add_375 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_367, add_375);  mul_367 = add_375 = None
        view_625 = torch.ops.aten.view.default(mul_369, [1568, 1280]);  mul_369 = None
        permute_427 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_200 = torch.ops.aten.addmm.default(arg234_1, view_625, permute_427);  arg234_1 = view_625 = permute_427 = None
        view_626 = torch.ops.aten.view.default(addmm_200, [8, 196, 320]);  addmm_200 = None
        add_376 = torch.ops.aten.add.Tensor(add_372, view_626);  add_372 = view_626 = None
        var_mean_125 = torch.ops.aten.var_mean.correction(add_376, [2], correction = 0, keepdim = True)
        getitem_490 = var_mean_125[0]
        getitem_491 = var_mean_125[1];  var_mean_125 = None
        add_377 = torch.ops.aten.add.Tensor(getitem_490, 1e-06);  getitem_490 = None
        rsqrt_125 = torch.ops.aten.rsqrt.default(add_377);  add_377 = None
        sub_125 = torch.ops.aten.sub.Tensor(add_376, getitem_491);  getitem_491 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_125, rsqrt_125);  sub_125 = rsqrt_125 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, arg235_1);  mul_370 = arg235_1 = None
        add_378 = torch.ops.aten.add.Tensor(mul_371, arg236_1);  mul_371 = arg236_1 = None
        view_627 = torch.ops.aten.view.default(add_378, [1568, 320])
        permute_428 = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_201 = torch.ops.aten.addmm.default(arg238_1, view_627, permute_428);  arg238_1 = view_627 = permute_428 = None
        view_628 = torch.ops.aten.view.default(addmm_201, [8, 196, 320]);  addmm_201 = None
        view_629 = torch.ops.aten.view.default(view_628, [8, 196, 5, 64]);  view_628 = None
        permute_429 = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
        permute_430 = torch.ops.aten.permute.default(add_378, [0, 2, 1]);  add_378 = None
        view_630 = torch.ops.aten.view.default(permute_430, [8, 320, 14, 14]);  permute_430 = None
        convolution_51 = torch.ops.aten.convolution.default(view_630, arg239_1, arg240_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_630 = arg239_1 = arg240_1 = None
        view_631 = torch.ops.aten.view.default(convolution_51, [8, 320, 49]);  convolution_51 = None
        permute_431 = torch.ops.aten.permute.default(view_631, [0, 2, 1]);  view_631 = None
        var_mean_126 = torch.ops.aten.var_mean.correction(permute_431, [2], correction = 0, keepdim = True)
        getitem_492 = var_mean_126[0]
        getitem_493 = var_mean_126[1];  var_mean_126 = None
        add_379 = torch.ops.aten.add.Tensor(getitem_492, 1e-05);  getitem_492 = None
        rsqrt_126 = torch.ops.aten.rsqrt.default(add_379);  add_379 = None
        sub_126 = torch.ops.aten.sub.Tensor(permute_431, getitem_493);  permute_431 = getitem_493 = None
        mul_372 = torch.ops.aten.mul.Tensor(sub_126, rsqrt_126);  sub_126 = rsqrt_126 = None
        mul_373 = torch.ops.aten.mul.Tensor(mul_372, arg241_1);  mul_372 = arg241_1 = None
        add_380 = torch.ops.aten.add.Tensor(mul_373, arg242_1);  mul_373 = arg242_1 = None
        view_632 = torch.ops.aten.view.default(add_380, [392, 320]);  add_380 = None
        permute_432 = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_202 = torch.ops.aten.addmm.default(arg244_1, view_632, permute_432);  arg244_1 = view_632 = permute_432 = None
        view_633 = torch.ops.aten.view.default(addmm_202, [8, 49, 640]);  addmm_202 = None
        view_634 = torch.ops.aten.view.default(view_633, [8, -1, 2, 5, 64]);  view_633 = None
        permute_433 = torch.ops.aten.permute.default(view_634, [2, 0, 3, 1, 4]);  view_634 = None
        unbind_40 = torch.ops.aten.unbind.int(permute_433);  permute_433 = None
        getitem_494 = unbind_40[0]
        getitem_495 = unbind_40[1];  unbind_40 = None
        _scaled_dot_product_efficient_attention_40 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_429, getitem_494, getitem_495, None, False);  permute_429 = getitem_494 = getitem_495 = None
        getitem_496 = _scaled_dot_product_efficient_attention_40[0];  _scaled_dot_product_efficient_attention_40 = None
        permute_434 = torch.ops.aten.permute.default(getitem_496, [0, 2, 1, 3]);  getitem_496 = None
        view_635 = torch.ops.aten.view.default(permute_434, [8, 196, 320]);  permute_434 = None
        view_636 = torch.ops.aten.view.default(view_635, [1568, 320]);  view_635 = None
        permute_435 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_203 = torch.ops.aten.addmm.default(arg246_1, view_636, permute_435);  arg246_1 = view_636 = permute_435 = None
        view_637 = torch.ops.aten.view.default(addmm_203, [8, 196, 320]);  addmm_203 = None
        add_381 = torch.ops.aten.add.Tensor(add_376, view_637);  add_376 = view_637 = None
        var_mean_127 = torch.ops.aten.var_mean.correction(add_381, [2], correction = 0, keepdim = True)
        getitem_500 = var_mean_127[0]
        getitem_501 = var_mean_127[1];  var_mean_127 = None
        add_382 = torch.ops.aten.add.Tensor(getitem_500, 1e-06);  getitem_500 = None
        rsqrt_127 = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
        sub_127 = torch.ops.aten.sub.Tensor(add_381, getitem_501);  getitem_501 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_127, rsqrt_127);  sub_127 = rsqrt_127 = None
        mul_375 = torch.ops.aten.mul.Tensor(mul_374, arg247_1);  mul_374 = arg247_1 = None
        add_383 = torch.ops.aten.add.Tensor(mul_375, arg248_1);  mul_375 = arg248_1 = None
        view_638 = torch.ops.aten.view.default(add_383, [1568, 320]);  add_383 = None
        permute_436 = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_204 = torch.ops.aten.addmm.default(arg250_1, view_638, permute_436);  arg250_1 = view_638 = permute_436 = None
        view_639 = torch.ops.aten.view.default(addmm_204, [8, 196, 1280]);  addmm_204 = None
        mul_376 = torch.ops.aten.mul.Tensor(view_639, 0.5)
        mul_377 = torch.ops.aten.mul.Tensor(view_639, 0.7071067811865476);  view_639 = None
        erf_40 = torch.ops.aten.erf.default(mul_377);  mul_377 = None
        add_384 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_378 = torch.ops.aten.mul.Tensor(mul_376, add_384);  mul_376 = add_384 = None
        view_640 = torch.ops.aten.view.default(mul_378, [1568, 1280]);  mul_378 = None
        permute_437 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_205 = torch.ops.aten.addmm.default(arg252_1, view_640, permute_437);  arg252_1 = view_640 = permute_437 = None
        view_641 = torch.ops.aten.view.default(addmm_205, [8, 196, 320]);  addmm_205 = None
        add_385 = torch.ops.aten.add.Tensor(add_381, view_641);  add_381 = view_641 = None
        var_mean_128 = torch.ops.aten.var_mean.correction(add_385, [2], correction = 0, keepdim = True)
        getitem_502 = var_mean_128[0]
        getitem_503 = var_mean_128[1];  var_mean_128 = None
        add_386 = torch.ops.aten.add.Tensor(getitem_502, 1e-06);  getitem_502 = None
        rsqrt_128 = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
        sub_128 = torch.ops.aten.sub.Tensor(add_385, getitem_503);  getitem_503 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_128, rsqrt_128);  sub_128 = rsqrt_128 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, arg253_1);  mul_379 = arg253_1 = None
        add_387 = torch.ops.aten.add.Tensor(mul_380, arg254_1);  mul_380 = arg254_1 = None
        view_642 = torch.ops.aten.view.default(add_387, [1568, 320])
        permute_438 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_206 = torch.ops.aten.addmm.default(arg256_1, view_642, permute_438);  arg256_1 = view_642 = permute_438 = None
        view_643 = torch.ops.aten.view.default(addmm_206, [8, 196, 320]);  addmm_206 = None
        view_644 = torch.ops.aten.view.default(view_643, [8, 196, 5, 64]);  view_643 = None
        permute_439 = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None
        permute_440 = torch.ops.aten.permute.default(add_387, [0, 2, 1]);  add_387 = None
        view_645 = torch.ops.aten.view.default(permute_440, [8, 320, 14, 14]);  permute_440 = None
        convolution_52 = torch.ops.aten.convolution.default(view_645, arg257_1, arg258_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_645 = arg257_1 = arg258_1 = None
        view_646 = torch.ops.aten.view.default(convolution_52, [8, 320, 49]);  convolution_52 = None
        permute_441 = torch.ops.aten.permute.default(view_646, [0, 2, 1]);  view_646 = None
        var_mean_129 = torch.ops.aten.var_mean.correction(permute_441, [2], correction = 0, keepdim = True)
        getitem_504 = var_mean_129[0]
        getitem_505 = var_mean_129[1];  var_mean_129 = None
        add_388 = torch.ops.aten.add.Tensor(getitem_504, 1e-05);  getitem_504 = None
        rsqrt_129 = torch.ops.aten.rsqrt.default(add_388);  add_388 = None
        sub_129 = torch.ops.aten.sub.Tensor(permute_441, getitem_505);  permute_441 = getitem_505 = None
        mul_381 = torch.ops.aten.mul.Tensor(sub_129, rsqrt_129);  sub_129 = rsqrt_129 = None
        mul_382 = torch.ops.aten.mul.Tensor(mul_381, arg259_1);  mul_381 = arg259_1 = None
        add_389 = torch.ops.aten.add.Tensor(mul_382, arg260_1);  mul_382 = arg260_1 = None
        view_647 = torch.ops.aten.view.default(add_389, [392, 320]);  add_389 = None
        permute_442 = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_207 = torch.ops.aten.addmm.default(arg262_1, view_647, permute_442);  arg262_1 = view_647 = permute_442 = None
        view_648 = torch.ops.aten.view.default(addmm_207, [8, 49, 640]);  addmm_207 = None
        view_649 = torch.ops.aten.view.default(view_648, [8, -1, 2, 5, 64]);  view_648 = None
        permute_443 = torch.ops.aten.permute.default(view_649, [2, 0, 3, 1, 4]);  view_649 = None
        unbind_41 = torch.ops.aten.unbind.int(permute_443);  permute_443 = None
        getitem_506 = unbind_41[0]
        getitem_507 = unbind_41[1];  unbind_41 = None
        _scaled_dot_product_efficient_attention_41 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_439, getitem_506, getitem_507, None, False);  permute_439 = getitem_506 = getitem_507 = None
        getitem_508 = _scaled_dot_product_efficient_attention_41[0];  _scaled_dot_product_efficient_attention_41 = None
        permute_444 = torch.ops.aten.permute.default(getitem_508, [0, 2, 1, 3]);  getitem_508 = None
        view_650 = torch.ops.aten.view.default(permute_444, [8, 196, 320]);  permute_444 = None
        view_651 = torch.ops.aten.view.default(view_650, [1568, 320]);  view_650 = None
        permute_445 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_208 = torch.ops.aten.addmm.default(arg264_1, view_651, permute_445);  arg264_1 = view_651 = permute_445 = None
        view_652 = torch.ops.aten.view.default(addmm_208, [8, 196, 320]);  addmm_208 = None
        add_390 = torch.ops.aten.add.Tensor(add_385, view_652);  add_385 = view_652 = None
        var_mean_130 = torch.ops.aten.var_mean.correction(add_390, [2], correction = 0, keepdim = True)
        getitem_512 = var_mean_130[0]
        getitem_513 = var_mean_130[1];  var_mean_130 = None
        add_391 = torch.ops.aten.add.Tensor(getitem_512, 1e-06);  getitem_512 = None
        rsqrt_130 = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
        sub_130 = torch.ops.aten.sub.Tensor(add_390, getitem_513);  getitem_513 = None
        mul_383 = torch.ops.aten.mul.Tensor(sub_130, rsqrt_130);  sub_130 = rsqrt_130 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_383, arg265_1);  mul_383 = arg265_1 = None
        add_392 = torch.ops.aten.add.Tensor(mul_384, arg266_1);  mul_384 = arg266_1 = None
        view_653 = torch.ops.aten.view.default(add_392, [1568, 320]);  add_392 = None
        permute_446 = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_209 = torch.ops.aten.addmm.default(arg268_1, view_653, permute_446);  arg268_1 = view_653 = permute_446 = None
        view_654 = torch.ops.aten.view.default(addmm_209, [8, 196, 1280]);  addmm_209 = None
        mul_385 = torch.ops.aten.mul.Tensor(view_654, 0.5)
        mul_386 = torch.ops.aten.mul.Tensor(view_654, 0.7071067811865476);  view_654 = None
        erf_41 = torch.ops.aten.erf.default(mul_386);  mul_386 = None
        add_393 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_385, add_393);  mul_385 = add_393 = None
        view_655 = torch.ops.aten.view.default(mul_387, [1568, 1280]);  mul_387 = None
        permute_447 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_210 = torch.ops.aten.addmm.default(arg270_1, view_655, permute_447);  arg270_1 = view_655 = permute_447 = None
        view_656 = torch.ops.aten.view.default(addmm_210, [8, 196, 320]);  addmm_210 = None
        add_394 = torch.ops.aten.add.Tensor(add_390, view_656);  add_390 = view_656 = None
        var_mean_131 = torch.ops.aten.var_mean.correction(add_394, [2], correction = 0, keepdim = True)
        getitem_514 = var_mean_131[0]
        getitem_515 = var_mean_131[1];  var_mean_131 = None
        add_395 = torch.ops.aten.add.Tensor(getitem_514, 1e-06);  getitem_514 = None
        rsqrt_131 = torch.ops.aten.rsqrt.default(add_395);  add_395 = None
        sub_131 = torch.ops.aten.sub.Tensor(add_394, getitem_515);  getitem_515 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_131, rsqrt_131);  sub_131 = rsqrt_131 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, arg271_1);  mul_388 = arg271_1 = None
        add_396 = torch.ops.aten.add.Tensor(mul_389, arg272_1);  mul_389 = arg272_1 = None
        view_657 = torch.ops.aten.view.default(add_396, [1568, 320])
        permute_448 = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_211 = torch.ops.aten.addmm.default(arg274_1, view_657, permute_448);  arg274_1 = view_657 = permute_448 = None
        view_658 = torch.ops.aten.view.default(addmm_211, [8, 196, 320]);  addmm_211 = None
        view_659 = torch.ops.aten.view.default(view_658, [8, 196, 5, 64]);  view_658 = None
        permute_449 = torch.ops.aten.permute.default(view_659, [0, 2, 1, 3]);  view_659 = None
        permute_450 = torch.ops.aten.permute.default(add_396, [0, 2, 1]);  add_396 = None
        view_660 = torch.ops.aten.view.default(permute_450, [8, 320, 14, 14]);  permute_450 = None
        convolution_53 = torch.ops.aten.convolution.default(view_660, arg275_1, arg276_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_660 = arg275_1 = arg276_1 = None
        view_661 = torch.ops.aten.view.default(convolution_53, [8, 320, 49]);  convolution_53 = None
        permute_451 = torch.ops.aten.permute.default(view_661, [0, 2, 1]);  view_661 = None
        var_mean_132 = torch.ops.aten.var_mean.correction(permute_451, [2], correction = 0, keepdim = True)
        getitem_516 = var_mean_132[0]
        getitem_517 = var_mean_132[1];  var_mean_132 = None
        add_397 = torch.ops.aten.add.Tensor(getitem_516, 1e-05);  getitem_516 = None
        rsqrt_132 = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
        sub_132 = torch.ops.aten.sub.Tensor(permute_451, getitem_517);  permute_451 = getitem_517 = None
        mul_390 = torch.ops.aten.mul.Tensor(sub_132, rsqrt_132);  sub_132 = rsqrt_132 = None
        mul_391 = torch.ops.aten.mul.Tensor(mul_390, arg277_1);  mul_390 = arg277_1 = None
        add_398 = torch.ops.aten.add.Tensor(mul_391, arg278_1);  mul_391 = arg278_1 = None
        view_662 = torch.ops.aten.view.default(add_398, [392, 320]);  add_398 = None
        permute_452 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_212 = torch.ops.aten.addmm.default(arg280_1, view_662, permute_452);  arg280_1 = view_662 = permute_452 = None
        view_663 = torch.ops.aten.view.default(addmm_212, [8, 49, 640]);  addmm_212 = None
        view_664 = torch.ops.aten.view.default(view_663, [8, -1, 2, 5, 64]);  view_663 = None
        permute_453 = torch.ops.aten.permute.default(view_664, [2, 0, 3, 1, 4]);  view_664 = None
        unbind_42 = torch.ops.aten.unbind.int(permute_453);  permute_453 = None
        getitem_518 = unbind_42[0]
        getitem_519 = unbind_42[1];  unbind_42 = None
        _scaled_dot_product_efficient_attention_42 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_449, getitem_518, getitem_519, None, False);  permute_449 = getitem_518 = getitem_519 = None
        getitem_520 = _scaled_dot_product_efficient_attention_42[0];  _scaled_dot_product_efficient_attention_42 = None
        permute_454 = torch.ops.aten.permute.default(getitem_520, [0, 2, 1, 3]);  getitem_520 = None
        view_665 = torch.ops.aten.view.default(permute_454, [8, 196, 320]);  permute_454 = None
        view_666 = torch.ops.aten.view.default(view_665, [1568, 320]);  view_665 = None
        permute_455 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_213 = torch.ops.aten.addmm.default(arg282_1, view_666, permute_455);  arg282_1 = view_666 = permute_455 = None
        view_667 = torch.ops.aten.view.default(addmm_213, [8, 196, 320]);  addmm_213 = None
        add_399 = torch.ops.aten.add.Tensor(add_394, view_667);  add_394 = view_667 = None
        var_mean_133 = torch.ops.aten.var_mean.correction(add_399, [2], correction = 0, keepdim = True)
        getitem_524 = var_mean_133[0]
        getitem_525 = var_mean_133[1];  var_mean_133 = None
        add_400 = torch.ops.aten.add.Tensor(getitem_524, 1e-06);  getitem_524 = None
        rsqrt_133 = torch.ops.aten.rsqrt.default(add_400);  add_400 = None
        sub_133 = torch.ops.aten.sub.Tensor(add_399, getitem_525);  getitem_525 = None
        mul_392 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_133);  sub_133 = rsqrt_133 = None
        mul_393 = torch.ops.aten.mul.Tensor(mul_392, arg283_1);  mul_392 = arg283_1 = None
        add_401 = torch.ops.aten.add.Tensor(mul_393, arg284_1);  mul_393 = arg284_1 = None
        view_668 = torch.ops.aten.view.default(add_401, [1568, 320]);  add_401 = None
        permute_456 = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_214 = torch.ops.aten.addmm.default(arg286_1, view_668, permute_456);  arg286_1 = view_668 = permute_456 = None
        view_669 = torch.ops.aten.view.default(addmm_214, [8, 196, 1280]);  addmm_214 = None
        mul_394 = torch.ops.aten.mul.Tensor(view_669, 0.5)
        mul_395 = torch.ops.aten.mul.Tensor(view_669, 0.7071067811865476);  view_669 = None
        erf_42 = torch.ops.aten.erf.default(mul_395);  mul_395 = None
        add_402 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_394, add_402);  mul_394 = add_402 = None
        view_670 = torch.ops.aten.view.default(mul_396, [1568, 1280]);  mul_396 = None
        permute_457 = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_215 = torch.ops.aten.addmm.default(arg288_1, view_670, permute_457);  arg288_1 = view_670 = permute_457 = None
        view_671 = torch.ops.aten.view.default(addmm_215, [8, 196, 320]);  addmm_215 = None
        add_403 = torch.ops.aten.add.Tensor(add_399, view_671);  add_399 = view_671 = None
        var_mean_134 = torch.ops.aten.var_mean.correction(add_403, [2], correction = 0, keepdim = True)
        getitem_526 = var_mean_134[0]
        getitem_527 = var_mean_134[1];  var_mean_134 = None
        add_404 = torch.ops.aten.add.Tensor(getitem_526, 1e-06);  getitem_526 = None
        rsqrt_134 = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        sub_134 = torch.ops.aten.sub.Tensor(add_403, getitem_527);  getitem_527 = None
        mul_397 = torch.ops.aten.mul.Tensor(sub_134, rsqrt_134);  sub_134 = rsqrt_134 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_397, arg289_1);  mul_397 = arg289_1 = None
        add_405 = torch.ops.aten.add.Tensor(mul_398, arg290_1);  mul_398 = arg290_1 = None
        view_672 = torch.ops.aten.view.default(add_405, [1568, 320])
        permute_458 = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_216 = torch.ops.aten.addmm.default(arg292_1, view_672, permute_458);  arg292_1 = view_672 = permute_458 = None
        view_673 = torch.ops.aten.view.default(addmm_216, [8, 196, 320]);  addmm_216 = None
        view_674 = torch.ops.aten.view.default(view_673, [8, 196, 5, 64]);  view_673 = None
        permute_459 = torch.ops.aten.permute.default(view_674, [0, 2, 1, 3]);  view_674 = None
        permute_460 = torch.ops.aten.permute.default(add_405, [0, 2, 1]);  add_405 = None
        view_675 = torch.ops.aten.view.default(permute_460, [8, 320, 14, 14]);  permute_460 = None
        convolution_54 = torch.ops.aten.convolution.default(view_675, arg293_1, arg294_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_675 = arg293_1 = arg294_1 = None
        view_676 = torch.ops.aten.view.default(convolution_54, [8, 320, 49]);  convolution_54 = None
        permute_461 = torch.ops.aten.permute.default(view_676, [0, 2, 1]);  view_676 = None
        var_mean_135 = torch.ops.aten.var_mean.correction(permute_461, [2], correction = 0, keepdim = True)
        getitem_528 = var_mean_135[0]
        getitem_529 = var_mean_135[1];  var_mean_135 = None
        add_406 = torch.ops.aten.add.Tensor(getitem_528, 1e-05);  getitem_528 = None
        rsqrt_135 = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
        sub_135 = torch.ops.aten.sub.Tensor(permute_461, getitem_529);  permute_461 = getitem_529 = None
        mul_399 = torch.ops.aten.mul.Tensor(sub_135, rsqrt_135);  sub_135 = rsqrt_135 = None
        mul_400 = torch.ops.aten.mul.Tensor(mul_399, arg295_1);  mul_399 = arg295_1 = None
        add_407 = torch.ops.aten.add.Tensor(mul_400, arg296_1);  mul_400 = arg296_1 = None
        view_677 = torch.ops.aten.view.default(add_407, [392, 320]);  add_407 = None
        permute_462 = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_217 = torch.ops.aten.addmm.default(arg298_1, view_677, permute_462);  arg298_1 = view_677 = permute_462 = None
        view_678 = torch.ops.aten.view.default(addmm_217, [8, 49, 640]);  addmm_217 = None
        view_679 = torch.ops.aten.view.default(view_678, [8, -1, 2, 5, 64]);  view_678 = None
        permute_463 = torch.ops.aten.permute.default(view_679, [2, 0, 3, 1, 4]);  view_679 = None
        unbind_43 = torch.ops.aten.unbind.int(permute_463);  permute_463 = None
        getitem_530 = unbind_43[0]
        getitem_531 = unbind_43[1];  unbind_43 = None
        _scaled_dot_product_efficient_attention_43 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_459, getitem_530, getitem_531, None, False);  permute_459 = getitem_530 = getitem_531 = None
        getitem_532 = _scaled_dot_product_efficient_attention_43[0];  _scaled_dot_product_efficient_attention_43 = None
        permute_464 = torch.ops.aten.permute.default(getitem_532, [0, 2, 1, 3]);  getitem_532 = None
        view_680 = torch.ops.aten.view.default(permute_464, [8, 196, 320]);  permute_464 = None
        view_681 = torch.ops.aten.view.default(view_680, [1568, 320]);  view_680 = None
        permute_465 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_218 = torch.ops.aten.addmm.default(arg300_1, view_681, permute_465);  arg300_1 = view_681 = permute_465 = None
        view_682 = torch.ops.aten.view.default(addmm_218, [8, 196, 320]);  addmm_218 = None
        add_408 = torch.ops.aten.add.Tensor(add_403, view_682);  add_403 = view_682 = None
        var_mean_136 = torch.ops.aten.var_mean.correction(add_408, [2], correction = 0, keepdim = True)
        getitem_536 = var_mean_136[0]
        getitem_537 = var_mean_136[1];  var_mean_136 = None
        add_409 = torch.ops.aten.add.Tensor(getitem_536, 1e-06);  getitem_536 = None
        rsqrt_136 = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
        sub_136 = torch.ops.aten.sub.Tensor(add_408, getitem_537);  getitem_537 = None
        mul_401 = torch.ops.aten.mul.Tensor(sub_136, rsqrt_136);  sub_136 = rsqrt_136 = None
        mul_402 = torch.ops.aten.mul.Tensor(mul_401, arg301_1);  mul_401 = arg301_1 = None
        add_410 = torch.ops.aten.add.Tensor(mul_402, arg302_1);  mul_402 = arg302_1 = None
        view_683 = torch.ops.aten.view.default(add_410, [1568, 320]);  add_410 = None
        permute_466 = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_219 = torch.ops.aten.addmm.default(arg304_1, view_683, permute_466);  arg304_1 = view_683 = permute_466 = None
        view_684 = torch.ops.aten.view.default(addmm_219, [8, 196, 1280]);  addmm_219 = None
        mul_403 = torch.ops.aten.mul.Tensor(view_684, 0.5)
        mul_404 = torch.ops.aten.mul.Tensor(view_684, 0.7071067811865476);  view_684 = None
        erf_43 = torch.ops.aten.erf.default(mul_404);  mul_404 = None
        add_411 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_403, add_411);  mul_403 = add_411 = None
        view_685 = torch.ops.aten.view.default(mul_405, [1568, 1280]);  mul_405 = None
        permute_467 = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_220 = torch.ops.aten.addmm.default(arg306_1, view_685, permute_467);  arg306_1 = view_685 = permute_467 = None
        view_686 = torch.ops.aten.view.default(addmm_220, [8, 196, 320]);  addmm_220 = None
        add_412 = torch.ops.aten.add.Tensor(add_408, view_686);  add_408 = view_686 = None
        var_mean_137 = torch.ops.aten.var_mean.correction(add_412, [2], correction = 0, keepdim = True)
        getitem_538 = var_mean_137[0]
        getitem_539 = var_mean_137[1];  var_mean_137 = None
        add_413 = torch.ops.aten.add.Tensor(getitem_538, 1e-06);  getitem_538 = None
        rsqrt_137 = torch.ops.aten.rsqrt.default(add_413);  add_413 = None
        sub_137 = torch.ops.aten.sub.Tensor(add_412, getitem_539);  getitem_539 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_137, rsqrt_137);  sub_137 = rsqrt_137 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, arg307_1);  mul_406 = arg307_1 = None
        add_414 = torch.ops.aten.add.Tensor(mul_407, arg308_1);  mul_407 = arg308_1 = None
        view_687 = torch.ops.aten.view.default(add_414, [1568, 320])
        permute_468 = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_221 = torch.ops.aten.addmm.default(arg310_1, view_687, permute_468);  arg310_1 = view_687 = permute_468 = None
        view_688 = torch.ops.aten.view.default(addmm_221, [8, 196, 320]);  addmm_221 = None
        view_689 = torch.ops.aten.view.default(view_688, [8, 196, 5, 64]);  view_688 = None
        permute_469 = torch.ops.aten.permute.default(view_689, [0, 2, 1, 3]);  view_689 = None
        permute_470 = torch.ops.aten.permute.default(add_414, [0, 2, 1]);  add_414 = None
        view_690 = torch.ops.aten.view.default(permute_470, [8, 320, 14, 14]);  permute_470 = None
        convolution_55 = torch.ops.aten.convolution.default(view_690, arg311_1, arg312_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_690 = arg311_1 = arg312_1 = None
        view_691 = torch.ops.aten.view.default(convolution_55, [8, 320, 49]);  convolution_55 = None
        permute_471 = torch.ops.aten.permute.default(view_691, [0, 2, 1]);  view_691 = None
        var_mean_138 = torch.ops.aten.var_mean.correction(permute_471, [2], correction = 0, keepdim = True)
        getitem_540 = var_mean_138[0]
        getitem_541 = var_mean_138[1];  var_mean_138 = None
        add_415 = torch.ops.aten.add.Tensor(getitem_540, 1e-05);  getitem_540 = None
        rsqrt_138 = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
        sub_138 = torch.ops.aten.sub.Tensor(permute_471, getitem_541);  permute_471 = getitem_541 = None
        mul_408 = torch.ops.aten.mul.Tensor(sub_138, rsqrt_138);  sub_138 = rsqrt_138 = None
        mul_409 = torch.ops.aten.mul.Tensor(mul_408, arg313_1);  mul_408 = arg313_1 = None
        add_416 = torch.ops.aten.add.Tensor(mul_409, arg314_1);  mul_409 = arg314_1 = None
        view_692 = torch.ops.aten.view.default(add_416, [392, 320]);  add_416 = None
        permute_472 = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_222 = torch.ops.aten.addmm.default(arg316_1, view_692, permute_472);  arg316_1 = view_692 = permute_472 = None
        view_693 = torch.ops.aten.view.default(addmm_222, [8, 49, 640]);  addmm_222 = None
        view_694 = torch.ops.aten.view.default(view_693, [8, -1, 2, 5, 64]);  view_693 = None
        permute_473 = torch.ops.aten.permute.default(view_694, [2, 0, 3, 1, 4]);  view_694 = None
        unbind_44 = torch.ops.aten.unbind.int(permute_473);  permute_473 = None
        getitem_542 = unbind_44[0]
        getitem_543 = unbind_44[1];  unbind_44 = None
        _scaled_dot_product_efficient_attention_44 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_469, getitem_542, getitem_543, None, False);  permute_469 = getitem_542 = getitem_543 = None
        getitem_544 = _scaled_dot_product_efficient_attention_44[0];  _scaled_dot_product_efficient_attention_44 = None
        permute_474 = torch.ops.aten.permute.default(getitem_544, [0, 2, 1, 3]);  getitem_544 = None
        view_695 = torch.ops.aten.view.default(permute_474, [8, 196, 320]);  permute_474 = None
        view_696 = torch.ops.aten.view.default(view_695, [1568, 320]);  view_695 = None
        permute_475 = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_223 = torch.ops.aten.addmm.default(arg318_1, view_696, permute_475);  arg318_1 = view_696 = permute_475 = None
        view_697 = torch.ops.aten.view.default(addmm_223, [8, 196, 320]);  addmm_223 = None
        add_417 = torch.ops.aten.add.Tensor(add_412, view_697);  add_412 = view_697 = None
        var_mean_139 = torch.ops.aten.var_mean.correction(add_417, [2], correction = 0, keepdim = True)
        getitem_548 = var_mean_139[0]
        getitem_549 = var_mean_139[1];  var_mean_139 = None
        add_418 = torch.ops.aten.add.Tensor(getitem_548, 1e-06);  getitem_548 = None
        rsqrt_139 = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        sub_139 = torch.ops.aten.sub.Tensor(add_417, getitem_549);  getitem_549 = None
        mul_410 = torch.ops.aten.mul.Tensor(sub_139, rsqrt_139);  sub_139 = rsqrt_139 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_410, arg319_1);  mul_410 = arg319_1 = None
        add_419 = torch.ops.aten.add.Tensor(mul_411, arg320_1);  mul_411 = arg320_1 = None
        view_698 = torch.ops.aten.view.default(add_419, [1568, 320]);  add_419 = None
        permute_476 = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_224 = torch.ops.aten.addmm.default(arg322_1, view_698, permute_476);  arg322_1 = view_698 = permute_476 = None
        view_699 = torch.ops.aten.view.default(addmm_224, [8, 196, 1280]);  addmm_224 = None
        mul_412 = torch.ops.aten.mul.Tensor(view_699, 0.5)
        mul_413 = torch.ops.aten.mul.Tensor(view_699, 0.7071067811865476);  view_699 = None
        erf_44 = torch.ops.aten.erf.default(mul_413);  mul_413 = None
        add_420 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_414 = torch.ops.aten.mul.Tensor(mul_412, add_420);  mul_412 = add_420 = None
        view_700 = torch.ops.aten.view.default(mul_414, [1568, 1280]);  mul_414 = None
        permute_477 = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_225 = torch.ops.aten.addmm.default(arg324_1, view_700, permute_477);  arg324_1 = view_700 = permute_477 = None
        view_701 = torch.ops.aten.view.default(addmm_225, [8, 196, 320]);  addmm_225 = None
        add_421 = torch.ops.aten.add.Tensor(add_417, view_701);  add_417 = view_701 = None
        var_mean_140 = torch.ops.aten.var_mean.correction(add_421, [2], correction = 0, keepdim = True)
        getitem_550 = var_mean_140[0]
        getitem_551 = var_mean_140[1];  var_mean_140 = None
        add_422 = torch.ops.aten.add.Tensor(getitem_550, 1e-06);  getitem_550 = None
        rsqrt_140 = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        sub_140 = torch.ops.aten.sub.Tensor(add_421, getitem_551);  getitem_551 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_140, rsqrt_140);  sub_140 = rsqrt_140 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, arg325_1);  mul_415 = arg325_1 = None
        add_423 = torch.ops.aten.add.Tensor(mul_416, arg326_1);  mul_416 = arg326_1 = None
        view_702 = torch.ops.aten.view.default(add_423, [1568, 320])
        permute_478 = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_226 = torch.ops.aten.addmm.default(arg328_1, view_702, permute_478);  arg328_1 = view_702 = permute_478 = None
        view_703 = torch.ops.aten.view.default(addmm_226, [8, 196, 320]);  addmm_226 = None
        view_704 = torch.ops.aten.view.default(view_703, [8, 196, 5, 64]);  view_703 = None
        permute_479 = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
        permute_480 = torch.ops.aten.permute.default(add_423, [0, 2, 1]);  add_423 = None
        view_705 = torch.ops.aten.view.default(permute_480, [8, 320, 14, 14]);  permute_480 = None
        convolution_56 = torch.ops.aten.convolution.default(view_705, arg329_1, arg330_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_705 = arg329_1 = arg330_1 = None
        view_706 = torch.ops.aten.view.default(convolution_56, [8, 320, 49]);  convolution_56 = None
        permute_481 = torch.ops.aten.permute.default(view_706, [0, 2, 1]);  view_706 = None
        var_mean_141 = torch.ops.aten.var_mean.correction(permute_481, [2], correction = 0, keepdim = True)
        getitem_552 = var_mean_141[0]
        getitem_553 = var_mean_141[1];  var_mean_141 = None
        add_424 = torch.ops.aten.add.Tensor(getitem_552, 1e-05);  getitem_552 = None
        rsqrt_141 = torch.ops.aten.rsqrt.default(add_424);  add_424 = None
        sub_141 = torch.ops.aten.sub.Tensor(permute_481, getitem_553);  permute_481 = getitem_553 = None
        mul_417 = torch.ops.aten.mul.Tensor(sub_141, rsqrt_141);  sub_141 = rsqrt_141 = None
        mul_418 = torch.ops.aten.mul.Tensor(mul_417, arg331_1);  mul_417 = arg331_1 = None
        add_425 = torch.ops.aten.add.Tensor(mul_418, arg332_1);  mul_418 = arg332_1 = None
        view_707 = torch.ops.aten.view.default(add_425, [392, 320]);  add_425 = None
        permute_482 = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_227 = torch.ops.aten.addmm.default(arg334_1, view_707, permute_482);  arg334_1 = view_707 = permute_482 = None
        view_708 = torch.ops.aten.view.default(addmm_227, [8, 49, 640]);  addmm_227 = None
        view_709 = torch.ops.aten.view.default(view_708, [8, -1, 2, 5, 64]);  view_708 = None
        permute_483 = torch.ops.aten.permute.default(view_709, [2, 0, 3, 1, 4]);  view_709 = None
        unbind_45 = torch.ops.aten.unbind.int(permute_483);  permute_483 = None
        getitem_554 = unbind_45[0]
        getitem_555 = unbind_45[1];  unbind_45 = None
        _scaled_dot_product_efficient_attention_45 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_479, getitem_554, getitem_555, None, False);  permute_479 = getitem_554 = getitem_555 = None
        getitem_556 = _scaled_dot_product_efficient_attention_45[0];  _scaled_dot_product_efficient_attention_45 = None
        permute_484 = torch.ops.aten.permute.default(getitem_556, [0, 2, 1, 3]);  getitem_556 = None
        view_710 = torch.ops.aten.view.default(permute_484, [8, 196, 320]);  permute_484 = None
        view_711 = torch.ops.aten.view.default(view_710, [1568, 320]);  view_710 = None
        permute_485 = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_228 = torch.ops.aten.addmm.default(arg336_1, view_711, permute_485);  arg336_1 = view_711 = permute_485 = None
        view_712 = torch.ops.aten.view.default(addmm_228, [8, 196, 320]);  addmm_228 = None
        add_426 = torch.ops.aten.add.Tensor(add_421, view_712);  add_421 = view_712 = None
        var_mean_142 = torch.ops.aten.var_mean.correction(add_426, [2], correction = 0, keepdim = True)
        getitem_560 = var_mean_142[0]
        getitem_561 = var_mean_142[1];  var_mean_142 = None
        add_427 = torch.ops.aten.add.Tensor(getitem_560, 1e-06);  getitem_560 = None
        rsqrt_142 = torch.ops.aten.rsqrt.default(add_427);  add_427 = None
        sub_142 = torch.ops.aten.sub.Tensor(add_426, getitem_561);  getitem_561 = None
        mul_419 = torch.ops.aten.mul.Tensor(sub_142, rsqrt_142);  sub_142 = rsqrt_142 = None
        mul_420 = torch.ops.aten.mul.Tensor(mul_419, arg337_1);  mul_419 = arg337_1 = None
        add_428 = torch.ops.aten.add.Tensor(mul_420, arg338_1);  mul_420 = arg338_1 = None
        view_713 = torch.ops.aten.view.default(add_428, [1568, 320]);  add_428 = None
        permute_486 = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_229 = torch.ops.aten.addmm.default(arg340_1, view_713, permute_486);  arg340_1 = view_713 = permute_486 = None
        view_714 = torch.ops.aten.view.default(addmm_229, [8, 196, 1280]);  addmm_229 = None
        mul_421 = torch.ops.aten.mul.Tensor(view_714, 0.5)
        mul_422 = torch.ops.aten.mul.Tensor(view_714, 0.7071067811865476);  view_714 = None
        erf_45 = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_429 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_421, add_429);  mul_421 = add_429 = None
        view_715 = torch.ops.aten.view.default(mul_423, [1568, 1280]);  mul_423 = None
        permute_487 = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_230 = torch.ops.aten.addmm.default(arg342_1, view_715, permute_487);  arg342_1 = view_715 = permute_487 = None
        view_716 = torch.ops.aten.view.default(addmm_230, [8, 196, 320]);  addmm_230 = None
        add_430 = torch.ops.aten.add.Tensor(add_426, view_716);  add_426 = view_716 = None
        var_mean_143 = torch.ops.aten.var_mean.correction(add_430, [2], correction = 0, keepdim = True)
        getitem_562 = var_mean_143[0]
        getitem_563 = var_mean_143[1];  var_mean_143 = None
        add_431 = torch.ops.aten.add.Tensor(getitem_562, 1e-06);  getitem_562 = None
        rsqrt_143 = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
        sub_143 = torch.ops.aten.sub.Tensor(add_430, getitem_563);  getitem_563 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_143, rsqrt_143);  sub_143 = rsqrt_143 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, arg343_1);  mul_424 = arg343_1 = None
        add_432 = torch.ops.aten.add.Tensor(mul_425, arg344_1);  mul_425 = arg344_1 = None
        view_717 = torch.ops.aten.view.default(add_432, [1568, 320])
        permute_488 = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_231 = torch.ops.aten.addmm.default(arg346_1, view_717, permute_488);  arg346_1 = view_717 = permute_488 = None
        view_718 = torch.ops.aten.view.default(addmm_231, [8, 196, 320]);  addmm_231 = None
        view_719 = torch.ops.aten.view.default(view_718, [8, 196, 5, 64]);  view_718 = None
        permute_489 = torch.ops.aten.permute.default(view_719, [0, 2, 1, 3]);  view_719 = None
        permute_490 = torch.ops.aten.permute.default(add_432, [0, 2, 1]);  add_432 = None
        view_720 = torch.ops.aten.view.default(permute_490, [8, 320, 14, 14]);  permute_490 = None
        convolution_57 = torch.ops.aten.convolution.default(view_720, arg347_1, arg348_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_720 = arg347_1 = arg348_1 = None
        view_721 = torch.ops.aten.view.default(convolution_57, [8, 320, 49]);  convolution_57 = None
        permute_491 = torch.ops.aten.permute.default(view_721, [0, 2, 1]);  view_721 = None
        var_mean_144 = torch.ops.aten.var_mean.correction(permute_491, [2], correction = 0, keepdim = True)
        getitem_564 = var_mean_144[0]
        getitem_565 = var_mean_144[1];  var_mean_144 = None
        add_433 = torch.ops.aten.add.Tensor(getitem_564, 1e-05);  getitem_564 = None
        rsqrt_144 = torch.ops.aten.rsqrt.default(add_433);  add_433 = None
        sub_144 = torch.ops.aten.sub.Tensor(permute_491, getitem_565);  permute_491 = getitem_565 = None
        mul_426 = torch.ops.aten.mul.Tensor(sub_144, rsqrt_144);  sub_144 = rsqrt_144 = None
        mul_427 = torch.ops.aten.mul.Tensor(mul_426, arg349_1);  mul_426 = arg349_1 = None
        add_434 = torch.ops.aten.add.Tensor(mul_427, arg350_1);  mul_427 = arg350_1 = None
        view_722 = torch.ops.aten.view.default(add_434, [392, 320]);  add_434 = None
        permute_492 = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_232 = torch.ops.aten.addmm.default(arg352_1, view_722, permute_492);  arg352_1 = view_722 = permute_492 = None
        view_723 = torch.ops.aten.view.default(addmm_232, [8, 49, 640]);  addmm_232 = None
        view_724 = torch.ops.aten.view.default(view_723, [8, -1, 2, 5, 64]);  view_723 = None
        permute_493 = torch.ops.aten.permute.default(view_724, [2, 0, 3, 1, 4]);  view_724 = None
        unbind_46 = torch.ops.aten.unbind.int(permute_493);  permute_493 = None
        getitem_566 = unbind_46[0]
        getitem_567 = unbind_46[1];  unbind_46 = None
        _scaled_dot_product_efficient_attention_46 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_489, getitem_566, getitem_567, None, False);  permute_489 = getitem_566 = getitem_567 = None
        getitem_568 = _scaled_dot_product_efficient_attention_46[0];  _scaled_dot_product_efficient_attention_46 = None
        permute_494 = torch.ops.aten.permute.default(getitem_568, [0, 2, 1, 3]);  getitem_568 = None
        view_725 = torch.ops.aten.view.default(permute_494, [8, 196, 320]);  permute_494 = None
        view_726 = torch.ops.aten.view.default(view_725, [1568, 320]);  view_725 = None
        permute_495 = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_233 = torch.ops.aten.addmm.default(arg354_1, view_726, permute_495);  arg354_1 = view_726 = permute_495 = None
        view_727 = torch.ops.aten.view.default(addmm_233, [8, 196, 320]);  addmm_233 = None
        add_435 = torch.ops.aten.add.Tensor(add_430, view_727);  add_430 = view_727 = None
        var_mean_145 = torch.ops.aten.var_mean.correction(add_435, [2], correction = 0, keepdim = True)
        getitem_572 = var_mean_145[0]
        getitem_573 = var_mean_145[1];  var_mean_145 = None
        add_436 = torch.ops.aten.add.Tensor(getitem_572, 1e-06);  getitem_572 = None
        rsqrt_145 = torch.ops.aten.rsqrt.default(add_436);  add_436 = None
        sub_145 = torch.ops.aten.sub.Tensor(add_435, getitem_573);  getitem_573 = None
        mul_428 = torch.ops.aten.mul.Tensor(sub_145, rsqrt_145);  sub_145 = rsqrt_145 = None
        mul_429 = torch.ops.aten.mul.Tensor(mul_428, arg355_1);  mul_428 = arg355_1 = None
        add_437 = torch.ops.aten.add.Tensor(mul_429, arg356_1);  mul_429 = arg356_1 = None
        view_728 = torch.ops.aten.view.default(add_437, [1568, 320]);  add_437 = None
        permute_496 = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_234 = torch.ops.aten.addmm.default(arg358_1, view_728, permute_496);  arg358_1 = view_728 = permute_496 = None
        view_729 = torch.ops.aten.view.default(addmm_234, [8, 196, 1280]);  addmm_234 = None
        mul_430 = torch.ops.aten.mul.Tensor(view_729, 0.5)
        mul_431 = torch.ops.aten.mul.Tensor(view_729, 0.7071067811865476);  view_729 = None
        erf_46 = torch.ops.aten.erf.default(mul_431);  mul_431 = None
        add_438 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_430, add_438);  mul_430 = add_438 = None
        view_730 = torch.ops.aten.view.default(mul_432, [1568, 1280]);  mul_432 = None
        permute_497 = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_235 = torch.ops.aten.addmm.default(arg360_1, view_730, permute_497);  arg360_1 = view_730 = permute_497 = None
        view_731 = torch.ops.aten.view.default(addmm_235, [8, 196, 320]);  addmm_235 = None
        add_439 = torch.ops.aten.add.Tensor(add_435, view_731);  add_435 = view_731 = None
        var_mean_146 = torch.ops.aten.var_mean.correction(add_439, [2], correction = 0, keepdim = True)
        getitem_574 = var_mean_146[0]
        getitem_575 = var_mean_146[1];  var_mean_146 = None
        add_440 = torch.ops.aten.add.Tensor(getitem_574, 1e-06);  getitem_574 = None
        rsqrt_146 = torch.ops.aten.rsqrt.default(add_440);  add_440 = None
        sub_146 = torch.ops.aten.sub.Tensor(add_439, getitem_575);  getitem_575 = None
        mul_433 = torch.ops.aten.mul.Tensor(sub_146, rsqrt_146);  sub_146 = rsqrt_146 = None
        mul_434 = torch.ops.aten.mul.Tensor(mul_433, arg361_1);  mul_433 = arg361_1 = None
        add_441 = torch.ops.aten.add.Tensor(mul_434, arg362_1);  mul_434 = arg362_1 = None
        view_732 = torch.ops.aten.view.default(add_441, [1568, 320])
        permute_498 = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_236 = torch.ops.aten.addmm.default(arg364_1, view_732, permute_498);  arg364_1 = view_732 = permute_498 = None
        view_733 = torch.ops.aten.view.default(addmm_236, [8, 196, 320]);  addmm_236 = None
        view_734 = torch.ops.aten.view.default(view_733, [8, 196, 5, 64]);  view_733 = None
        permute_499 = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
        permute_500 = torch.ops.aten.permute.default(add_441, [0, 2, 1]);  add_441 = None
        view_735 = torch.ops.aten.view.default(permute_500, [8, 320, 14, 14]);  permute_500 = None
        convolution_58 = torch.ops.aten.convolution.default(view_735, arg365_1, arg366_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_735 = arg365_1 = arg366_1 = None
        view_736 = torch.ops.aten.view.default(convolution_58, [8, 320, 49]);  convolution_58 = None
        permute_501 = torch.ops.aten.permute.default(view_736, [0, 2, 1]);  view_736 = None
        var_mean_147 = torch.ops.aten.var_mean.correction(permute_501, [2], correction = 0, keepdim = True)
        getitem_576 = var_mean_147[0]
        getitem_577 = var_mean_147[1];  var_mean_147 = None
        add_442 = torch.ops.aten.add.Tensor(getitem_576, 1e-05);  getitem_576 = None
        rsqrt_147 = torch.ops.aten.rsqrt.default(add_442);  add_442 = None
        sub_147 = torch.ops.aten.sub.Tensor(permute_501, getitem_577);  permute_501 = getitem_577 = None
        mul_435 = torch.ops.aten.mul.Tensor(sub_147, rsqrt_147);  sub_147 = rsqrt_147 = None
        mul_436 = torch.ops.aten.mul.Tensor(mul_435, arg367_1);  mul_435 = arg367_1 = None
        add_443 = torch.ops.aten.add.Tensor(mul_436, arg368_1);  mul_436 = arg368_1 = None
        view_737 = torch.ops.aten.view.default(add_443, [392, 320]);  add_443 = None
        permute_502 = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_237 = torch.ops.aten.addmm.default(arg370_1, view_737, permute_502);  arg370_1 = view_737 = permute_502 = None
        view_738 = torch.ops.aten.view.default(addmm_237, [8, 49, 640]);  addmm_237 = None
        view_739 = torch.ops.aten.view.default(view_738, [8, -1, 2, 5, 64]);  view_738 = None
        permute_503 = torch.ops.aten.permute.default(view_739, [2, 0, 3, 1, 4]);  view_739 = None
        unbind_47 = torch.ops.aten.unbind.int(permute_503);  permute_503 = None
        getitem_578 = unbind_47[0]
        getitem_579 = unbind_47[1];  unbind_47 = None
        _scaled_dot_product_efficient_attention_47 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_499, getitem_578, getitem_579, None, False);  permute_499 = getitem_578 = getitem_579 = None
        getitem_580 = _scaled_dot_product_efficient_attention_47[0];  _scaled_dot_product_efficient_attention_47 = None
        permute_504 = torch.ops.aten.permute.default(getitem_580, [0, 2, 1, 3]);  getitem_580 = None
        view_740 = torch.ops.aten.view.default(permute_504, [8, 196, 320]);  permute_504 = None
        view_741 = torch.ops.aten.view.default(view_740, [1568, 320]);  view_740 = None
        permute_505 = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        addmm_238 = torch.ops.aten.addmm.default(arg372_1, view_741, permute_505);  arg372_1 = view_741 = permute_505 = None
        view_742 = torch.ops.aten.view.default(addmm_238, [8, 196, 320]);  addmm_238 = None
        add_444 = torch.ops.aten.add.Tensor(add_439, view_742);  add_439 = view_742 = None
        var_mean_148 = torch.ops.aten.var_mean.correction(add_444, [2], correction = 0, keepdim = True)
        getitem_584 = var_mean_148[0]
        getitem_585 = var_mean_148[1];  var_mean_148 = None
        add_445 = torch.ops.aten.add.Tensor(getitem_584, 1e-06);  getitem_584 = None
        rsqrt_148 = torch.ops.aten.rsqrt.default(add_445);  add_445 = None
        sub_148 = torch.ops.aten.sub.Tensor(add_444, getitem_585);  getitem_585 = None
        mul_437 = torch.ops.aten.mul.Tensor(sub_148, rsqrt_148);  sub_148 = rsqrt_148 = None
        mul_438 = torch.ops.aten.mul.Tensor(mul_437, arg373_1);  mul_437 = arg373_1 = None
        add_446 = torch.ops.aten.add.Tensor(mul_438, arg374_1);  mul_438 = arg374_1 = None
        view_743 = torch.ops.aten.view.default(add_446, [1568, 320]);  add_446 = None
        permute_506 = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_239 = torch.ops.aten.addmm.default(arg376_1, view_743, permute_506);  arg376_1 = view_743 = permute_506 = None
        view_744 = torch.ops.aten.view.default(addmm_239, [8, 196, 1280]);  addmm_239 = None
        mul_439 = torch.ops.aten.mul.Tensor(view_744, 0.5)
        mul_440 = torch.ops.aten.mul.Tensor(view_744, 0.7071067811865476);  view_744 = None
        erf_47 = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_447 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_439, add_447);  mul_439 = add_447 = None
        view_745 = torch.ops.aten.view.default(mul_441, [1568, 1280]);  mul_441 = None
        permute_507 = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_240 = torch.ops.aten.addmm.default(arg378_1, view_745, permute_507);  arg378_1 = view_745 = permute_507 = None
        view_746 = torch.ops.aten.view.default(addmm_240, [8, 196, 320]);  addmm_240 = None
        add_448 = torch.ops.aten.add.Tensor(add_444, view_746);  add_444 = view_746 = None
        var_mean_149 = torch.ops.aten.var_mean.correction(add_448, [2], correction = 0, keepdim = True)
        getitem_586 = var_mean_149[0]
        getitem_587 = var_mean_149[1];  var_mean_149 = None
        add_449 = torch.ops.aten.add.Tensor(getitem_586, 1e-06);  getitem_586 = None
        rsqrt_149 = torch.ops.aten.rsqrt.default(add_449);  add_449 = None
        sub_149 = torch.ops.aten.sub.Tensor(add_448, getitem_587);  getitem_587 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_149, rsqrt_149);  sub_149 = rsqrt_149 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, arg379_1);  mul_442 = arg379_1 = None
        add_450 = torch.ops.aten.add.Tensor(mul_443, arg380_1);  mul_443 = arg380_1 = None
        view_747 = torch.ops.aten.view.default(add_450, [1568, 320])
        permute_508 = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        addmm_241 = torch.ops.aten.addmm.default(arg382_1, view_747, permute_508);  arg382_1 = view_747 = permute_508 = None
        view_748 = torch.ops.aten.view.default(addmm_241, [8, 196, 320]);  addmm_241 = None
        view_749 = torch.ops.aten.view.default(view_748, [8, 196, 5, 64]);  view_748 = None
        permute_509 = torch.ops.aten.permute.default(view_749, [0, 2, 1, 3]);  view_749 = None
        permute_510 = torch.ops.aten.permute.default(add_450, [0, 2, 1]);  add_450 = None
        view_750 = torch.ops.aten.view.default(permute_510, [8, 320, 14, 14]);  permute_510 = None
        convolution_59 = torch.ops.aten.convolution.default(view_750, arg383_1, arg384_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_750 = arg383_1 = arg384_1 = None
        view_751 = torch.ops.aten.view.default(convolution_59, [8, 320, 49]);  convolution_59 = None
        permute_511 = torch.ops.aten.permute.default(view_751, [0, 2, 1]);  view_751 = None
        var_mean_150 = torch.ops.aten.var_mean.correction(permute_511, [2], correction = 0, keepdim = True)
        getitem_588 = var_mean_150[0]
        getitem_589 = var_mean_150[1];  var_mean_150 = None
        add_451 = torch.ops.aten.add.Tensor(getitem_588, 1e-05);  getitem_588 = None
        rsqrt_150 = torch.ops.aten.rsqrt.default(add_451);  add_451 = None
        sub_150 = torch.ops.aten.sub.Tensor(permute_511, getitem_589);  permute_511 = getitem_589 = None
        mul_444 = torch.ops.aten.mul.Tensor(sub_150, rsqrt_150);  sub_150 = rsqrt_150 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, arg385_1);  mul_444 = arg385_1 = None
        add_452 = torch.ops.aten.add.Tensor(mul_445, arg386_1);  mul_445 = arg386_1 = None
        view_752 = torch.ops.aten.view.default(add_452, [392, 320]);  add_452 = None
        permute_512 = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_242 = torch.ops.aten.addmm.default(arg388_1, view_752, permute_512);  arg388_1 = view_752 = permute_512 = None
        view_753 = torch.ops.aten.view.default(addmm_242, [8, 49, 640]);  addmm_242 = None
        view_754 = torch.ops.aten.view.default(view_753, [8, -1, 2, 5, 64]);  view_753 = None
        permute_513 = torch.ops.aten.permute.default(view_754, [2, 0, 3, 1, 4]);  view_754 = None
        unbind_48 = torch.ops.aten.unbind.int(permute_513);  permute_513 = None
        getitem_590 = unbind_48[0]
        getitem_591 = unbind_48[1];  unbind_48 = None
        _scaled_dot_product_efficient_attention_48 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_509, getitem_590, getitem_591, None, False);  permute_509 = getitem_590 = getitem_591 = None
        getitem_592 = _scaled_dot_product_efficient_attention_48[0];  _scaled_dot_product_efficient_attention_48 = None
        permute_514 = torch.ops.aten.permute.default(getitem_592, [0, 2, 1, 3]);  getitem_592 = None
        view_755 = torch.ops.aten.view.default(permute_514, [8, 196, 320]);  permute_514 = None
        view_756 = torch.ops.aten.view.default(view_755, [1568, 320]);  view_755 = None
        permute_515 = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_243 = torch.ops.aten.addmm.default(arg390_1, view_756, permute_515);  arg390_1 = view_756 = permute_515 = None
        view_757 = torch.ops.aten.view.default(addmm_243, [8, 196, 320]);  addmm_243 = None
        add_453 = torch.ops.aten.add.Tensor(add_448, view_757);  add_448 = view_757 = None
        var_mean_151 = torch.ops.aten.var_mean.correction(add_453, [2], correction = 0, keepdim = True)
        getitem_596 = var_mean_151[0]
        getitem_597 = var_mean_151[1];  var_mean_151 = None
        add_454 = torch.ops.aten.add.Tensor(getitem_596, 1e-06);  getitem_596 = None
        rsqrt_151 = torch.ops.aten.rsqrt.default(add_454);  add_454 = None
        sub_151 = torch.ops.aten.sub.Tensor(add_453, getitem_597);  getitem_597 = None
        mul_446 = torch.ops.aten.mul.Tensor(sub_151, rsqrt_151);  sub_151 = rsqrt_151 = None
        mul_447 = torch.ops.aten.mul.Tensor(mul_446, arg391_1);  mul_446 = arg391_1 = None
        add_455 = torch.ops.aten.add.Tensor(mul_447, arg392_1);  mul_447 = arg392_1 = None
        view_758 = torch.ops.aten.view.default(add_455, [1568, 320]);  add_455 = None
        permute_516 = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        addmm_244 = torch.ops.aten.addmm.default(arg394_1, view_758, permute_516);  arg394_1 = view_758 = permute_516 = None
        view_759 = torch.ops.aten.view.default(addmm_244, [8, 196, 1280]);  addmm_244 = None
        mul_448 = torch.ops.aten.mul.Tensor(view_759, 0.5)
        mul_449 = torch.ops.aten.mul.Tensor(view_759, 0.7071067811865476);  view_759 = None
        erf_48 = torch.ops.aten.erf.default(mul_449);  mul_449 = None
        add_456 = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_450 = torch.ops.aten.mul.Tensor(mul_448, add_456);  mul_448 = add_456 = None
        view_760 = torch.ops.aten.view.default(mul_450, [1568, 1280]);  mul_450 = None
        permute_517 = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_245 = torch.ops.aten.addmm.default(arg396_1, view_760, permute_517);  arg396_1 = view_760 = permute_517 = None
        view_761 = torch.ops.aten.view.default(addmm_245, [8, 196, 320]);  addmm_245 = None
        add_457 = torch.ops.aten.add.Tensor(add_453, view_761);  add_453 = view_761 = None
        var_mean_152 = torch.ops.aten.var_mean.correction(add_457, [2], correction = 0, keepdim = True)
        getitem_598 = var_mean_152[0]
        getitem_599 = var_mean_152[1];  var_mean_152 = None
        add_458 = torch.ops.aten.add.Tensor(getitem_598, 1e-06);  getitem_598 = None
        rsqrt_152 = torch.ops.aten.rsqrt.default(add_458);  add_458 = None
        sub_152 = torch.ops.aten.sub.Tensor(add_457, getitem_599);  getitem_599 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_152, rsqrt_152);  sub_152 = rsqrt_152 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, arg397_1);  mul_451 = arg397_1 = None
        add_459 = torch.ops.aten.add.Tensor(mul_452, arg398_1);  mul_452 = arg398_1 = None
        view_762 = torch.ops.aten.view.default(add_459, [1568, 320])
        permute_518 = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_246 = torch.ops.aten.addmm.default(arg400_1, view_762, permute_518);  arg400_1 = view_762 = permute_518 = None
        view_763 = torch.ops.aten.view.default(addmm_246, [8, 196, 320]);  addmm_246 = None
        view_764 = torch.ops.aten.view.default(view_763, [8, 196, 5, 64]);  view_763 = None
        permute_519 = torch.ops.aten.permute.default(view_764, [0, 2, 1, 3]);  view_764 = None
        permute_520 = torch.ops.aten.permute.default(add_459, [0, 2, 1]);  add_459 = None
        view_765 = torch.ops.aten.view.default(permute_520, [8, 320, 14, 14]);  permute_520 = None
        convolution_60 = torch.ops.aten.convolution.default(view_765, arg401_1, arg402_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_765 = arg401_1 = arg402_1 = None
        view_766 = torch.ops.aten.view.default(convolution_60, [8, 320, 49]);  convolution_60 = None
        permute_521 = torch.ops.aten.permute.default(view_766, [0, 2, 1]);  view_766 = None
        var_mean_153 = torch.ops.aten.var_mean.correction(permute_521, [2], correction = 0, keepdim = True)
        getitem_600 = var_mean_153[0]
        getitem_601 = var_mean_153[1];  var_mean_153 = None
        add_460 = torch.ops.aten.add.Tensor(getitem_600, 1e-05);  getitem_600 = None
        rsqrt_153 = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
        sub_153 = torch.ops.aten.sub.Tensor(permute_521, getitem_601);  permute_521 = getitem_601 = None
        mul_453 = torch.ops.aten.mul.Tensor(sub_153, rsqrt_153);  sub_153 = rsqrt_153 = None
        mul_454 = torch.ops.aten.mul.Tensor(mul_453, arg403_1);  mul_453 = arg403_1 = None
        add_461 = torch.ops.aten.add.Tensor(mul_454, arg404_1);  mul_454 = arg404_1 = None
        view_767 = torch.ops.aten.view.default(add_461, [392, 320]);  add_461 = None
        permute_522 = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_247 = torch.ops.aten.addmm.default(arg406_1, view_767, permute_522);  arg406_1 = view_767 = permute_522 = None
        view_768 = torch.ops.aten.view.default(addmm_247, [8, 49, 640]);  addmm_247 = None
        view_769 = torch.ops.aten.view.default(view_768, [8, -1, 2, 5, 64]);  view_768 = None
        permute_523 = torch.ops.aten.permute.default(view_769, [2, 0, 3, 1, 4]);  view_769 = None
        unbind_49 = torch.ops.aten.unbind.int(permute_523);  permute_523 = None
        getitem_602 = unbind_49[0]
        getitem_603 = unbind_49[1];  unbind_49 = None
        _scaled_dot_product_efficient_attention_49 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_519, getitem_602, getitem_603, None, False);  permute_519 = getitem_602 = getitem_603 = None
        getitem_604 = _scaled_dot_product_efficient_attention_49[0];  _scaled_dot_product_efficient_attention_49 = None
        permute_524 = torch.ops.aten.permute.default(getitem_604, [0, 2, 1, 3]);  getitem_604 = None
        view_770 = torch.ops.aten.view.default(permute_524, [8, 196, 320]);  permute_524 = None
        view_771 = torch.ops.aten.view.default(view_770, [1568, 320]);  view_770 = None
        permute_525 = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        addmm_248 = torch.ops.aten.addmm.default(arg408_1, view_771, permute_525);  arg408_1 = view_771 = permute_525 = None
        view_772 = torch.ops.aten.view.default(addmm_248, [8, 196, 320]);  addmm_248 = None
        add_462 = torch.ops.aten.add.Tensor(add_457, view_772);  add_457 = view_772 = None
        var_mean_154 = torch.ops.aten.var_mean.correction(add_462, [2], correction = 0, keepdim = True)
        getitem_608 = var_mean_154[0]
        getitem_609 = var_mean_154[1];  var_mean_154 = None
        add_463 = torch.ops.aten.add.Tensor(getitem_608, 1e-06);  getitem_608 = None
        rsqrt_154 = torch.ops.aten.rsqrt.default(add_463);  add_463 = None
        sub_154 = torch.ops.aten.sub.Tensor(add_462, getitem_609);  getitem_609 = None
        mul_455 = torch.ops.aten.mul.Tensor(sub_154, rsqrt_154);  sub_154 = rsqrt_154 = None
        mul_456 = torch.ops.aten.mul.Tensor(mul_455, arg409_1);  mul_455 = arg409_1 = None
        add_464 = torch.ops.aten.add.Tensor(mul_456, arg410_1);  mul_456 = arg410_1 = None
        view_773 = torch.ops.aten.view.default(add_464, [1568, 320]);  add_464 = None
        permute_526 = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_249 = torch.ops.aten.addmm.default(arg412_1, view_773, permute_526);  arg412_1 = view_773 = permute_526 = None
        view_774 = torch.ops.aten.view.default(addmm_249, [8, 196, 1280]);  addmm_249 = None
        mul_457 = torch.ops.aten.mul.Tensor(view_774, 0.5)
        mul_458 = torch.ops.aten.mul.Tensor(view_774, 0.7071067811865476);  view_774 = None
        erf_49 = torch.ops.aten.erf.default(mul_458);  mul_458 = None
        add_465 = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_459 = torch.ops.aten.mul.Tensor(mul_457, add_465);  mul_457 = add_465 = None
        view_775 = torch.ops.aten.view.default(mul_459, [1568, 1280]);  mul_459 = None
        permute_527 = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
        addmm_250 = torch.ops.aten.addmm.default(arg414_1, view_775, permute_527);  arg414_1 = view_775 = permute_527 = None
        view_776 = torch.ops.aten.view.default(addmm_250, [8, 196, 320]);  addmm_250 = None
        add_466 = torch.ops.aten.add.Tensor(add_462, view_776);  add_462 = view_776 = None
        var_mean_155 = torch.ops.aten.var_mean.correction(add_466, [2], correction = 0, keepdim = True)
        getitem_610 = var_mean_155[0]
        getitem_611 = var_mean_155[1];  var_mean_155 = None
        add_467 = torch.ops.aten.add.Tensor(getitem_610, 1e-06);  getitem_610 = None
        rsqrt_155 = torch.ops.aten.rsqrt.default(add_467);  add_467 = None
        sub_155 = torch.ops.aten.sub.Tensor(add_466, getitem_611);  getitem_611 = None
        mul_460 = torch.ops.aten.mul.Tensor(sub_155, rsqrt_155);  sub_155 = rsqrt_155 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, arg415_1);  mul_460 = arg415_1 = None
        add_468 = torch.ops.aten.add.Tensor(mul_461, arg416_1);  mul_461 = arg416_1 = None
        view_777 = torch.ops.aten.view.default(add_468, [1568, 320])
        permute_528 = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
        addmm_251 = torch.ops.aten.addmm.default(arg418_1, view_777, permute_528);  arg418_1 = view_777 = permute_528 = None
        view_778 = torch.ops.aten.view.default(addmm_251, [8, 196, 320]);  addmm_251 = None
        view_779 = torch.ops.aten.view.default(view_778, [8, 196, 5, 64]);  view_778 = None
        permute_529 = torch.ops.aten.permute.default(view_779, [0, 2, 1, 3]);  view_779 = None
        permute_530 = torch.ops.aten.permute.default(add_468, [0, 2, 1]);  add_468 = None
        view_780 = torch.ops.aten.view.default(permute_530, [8, 320, 14, 14]);  permute_530 = None
        convolution_61 = torch.ops.aten.convolution.default(view_780, arg419_1, arg420_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_780 = arg419_1 = arg420_1 = None
        view_781 = torch.ops.aten.view.default(convolution_61, [8, 320, 49]);  convolution_61 = None
        permute_531 = torch.ops.aten.permute.default(view_781, [0, 2, 1]);  view_781 = None
        var_mean_156 = torch.ops.aten.var_mean.correction(permute_531, [2], correction = 0, keepdim = True)
        getitem_612 = var_mean_156[0]
        getitem_613 = var_mean_156[1];  var_mean_156 = None
        add_469 = torch.ops.aten.add.Tensor(getitem_612, 1e-05);  getitem_612 = None
        rsqrt_156 = torch.ops.aten.rsqrt.default(add_469);  add_469 = None
        sub_156 = torch.ops.aten.sub.Tensor(permute_531, getitem_613);  permute_531 = getitem_613 = None
        mul_462 = torch.ops.aten.mul.Tensor(sub_156, rsqrt_156);  sub_156 = rsqrt_156 = None
        mul_463 = torch.ops.aten.mul.Tensor(mul_462, arg421_1);  mul_462 = arg421_1 = None
        add_470 = torch.ops.aten.add.Tensor(mul_463, arg422_1);  mul_463 = arg422_1 = None
        view_782 = torch.ops.aten.view.default(add_470, [392, 320]);  add_470 = None
        permute_532 = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_252 = torch.ops.aten.addmm.default(arg424_1, view_782, permute_532);  arg424_1 = view_782 = permute_532 = None
        view_783 = torch.ops.aten.view.default(addmm_252, [8, 49, 640]);  addmm_252 = None
        view_784 = torch.ops.aten.view.default(view_783, [8, -1, 2, 5, 64]);  view_783 = None
        permute_533 = torch.ops.aten.permute.default(view_784, [2, 0, 3, 1, 4]);  view_784 = None
        unbind_50 = torch.ops.aten.unbind.int(permute_533);  permute_533 = None
        getitem_614 = unbind_50[0]
        getitem_615 = unbind_50[1];  unbind_50 = None
        _scaled_dot_product_efficient_attention_50 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_529, getitem_614, getitem_615, None, False);  permute_529 = getitem_614 = getitem_615 = None
        getitem_616 = _scaled_dot_product_efficient_attention_50[0];  _scaled_dot_product_efficient_attention_50 = None
        permute_534 = torch.ops.aten.permute.default(getitem_616, [0, 2, 1, 3]);  getitem_616 = None
        view_785 = torch.ops.aten.view.default(permute_534, [8, 196, 320]);  permute_534 = None
        view_786 = torch.ops.aten.view.default(view_785, [1568, 320]);  view_785 = None
        permute_535 = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        addmm_253 = torch.ops.aten.addmm.default(arg426_1, view_786, permute_535);  arg426_1 = view_786 = permute_535 = None
        view_787 = torch.ops.aten.view.default(addmm_253, [8, 196, 320]);  addmm_253 = None
        add_471 = torch.ops.aten.add.Tensor(add_466, view_787);  add_466 = view_787 = None
        var_mean_157 = torch.ops.aten.var_mean.correction(add_471, [2], correction = 0, keepdim = True)
        getitem_620 = var_mean_157[0]
        getitem_621 = var_mean_157[1];  var_mean_157 = None
        add_472 = torch.ops.aten.add.Tensor(getitem_620, 1e-06);  getitem_620 = None
        rsqrt_157 = torch.ops.aten.rsqrt.default(add_472);  add_472 = None
        sub_157 = torch.ops.aten.sub.Tensor(add_471, getitem_621);  getitem_621 = None
        mul_464 = torch.ops.aten.mul.Tensor(sub_157, rsqrt_157);  sub_157 = rsqrt_157 = None
        mul_465 = torch.ops.aten.mul.Tensor(mul_464, arg427_1);  mul_464 = arg427_1 = None
        add_473 = torch.ops.aten.add.Tensor(mul_465, arg428_1);  mul_465 = arg428_1 = None
        view_788 = torch.ops.aten.view.default(add_473, [1568, 320]);  add_473 = None
        permute_536 = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
        addmm_254 = torch.ops.aten.addmm.default(arg430_1, view_788, permute_536);  arg430_1 = view_788 = permute_536 = None
        view_789 = torch.ops.aten.view.default(addmm_254, [8, 196, 1280]);  addmm_254 = None
        mul_466 = torch.ops.aten.mul.Tensor(view_789, 0.5)
        mul_467 = torch.ops.aten.mul.Tensor(view_789, 0.7071067811865476);  view_789 = None
        erf_50 = torch.ops.aten.erf.default(mul_467);  mul_467 = None
        add_474 = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_468 = torch.ops.aten.mul.Tensor(mul_466, add_474);  mul_466 = add_474 = None
        view_790 = torch.ops.aten.view.default(mul_468, [1568, 1280]);  mul_468 = None
        permute_537 = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_255 = torch.ops.aten.addmm.default(arg432_1, view_790, permute_537);  arg432_1 = view_790 = permute_537 = None
        view_791 = torch.ops.aten.view.default(addmm_255, [8, 196, 320]);  addmm_255 = None
        add_475 = torch.ops.aten.add.Tensor(add_471, view_791);  add_471 = view_791 = None
        var_mean_158 = torch.ops.aten.var_mean.correction(add_475, [2], correction = 0, keepdim = True)
        getitem_622 = var_mean_158[0]
        getitem_623 = var_mean_158[1];  var_mean_158 = None
        add_476 = torch.ops.aten.add.Tensor(getitem_622, 1e-06);  getitem_622 = None
        rsqrt_158 = torch.ops.aten.rsqrt.default(add_476);  add_476 = None
        sub_158 = torch.ops.aten.sub.Tensor(add_475, getitem_623);  getitem_623 = None
        mul_469 = torch.ops.aten.mul.Tensor(sub_158, rsqrt_158);  sub_158 = rsqrt_158 = None
        mul_470 = torch.ops.aten.mul.Tensor(mul_469, arg433_1);  mul_469 = arg433_1 = None
        add_477 = torch.ops.aten.add.Tensor(mul_470, arg434_1);  mul_470 = arg434_1 = None
        view_792 = torch.ops.aten.view.default(add_477, [1568, 320])
        permute_538 = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_256 = torch.ops.aten.addmm.default(arg436_1, view_792, permute_538);  arg436_1 = view_792 = permute_538 = None
        view_793 = torch.ops.aten.view.default(addmm_256, [8, 196, 320]);  addmm_256 = None
        view_794 = torch.ops.aten.view.default(view_793, [8, 196, 5, 64]);  view_793 = None
        permute_539 = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None
        permute_540 = torch.ops.aten.permute.default(add_477, [0, 2, 1]);  add_477 = None
        view_795 = torch.ops.aten.view.default(permute_540, [8, 320, 14, 14]);  permute_540 = None
        convolution_62 = torch.ops.aten.convolution.default(view_795, arg437_1, arg438_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_795 = arg437_1 = arg438_1 = None
        view_796 = torch.ops.aten.view.default(convolution_62, [8, 320, 49]);  convolution_62 = None
        permute_541 = torch.ops.aten.permute.default(view_796, [0, 2, 1]);  view_796 = None
        var_mean_159 = torch.ops.aten.var_mean.correction(permute_541, [2], correction = 0, keepdim = True)
        getitem_624 = var_mean_159[0]
        getitem_625 = var_mean_159[1];  var_mean_159 = None
        add_478 = torch.ops.aten.add.Tensor(getitem_624, 1e-05);  getitem_624 = None
        rsqrt_159 = torch.ops.aten.rsqrt.default(add_478);  add_478 = None
        sub_159 = torch.ops.aten.sub.Tensor(permute_541, getitem_625);  permute_541 = getitem_625 = None
        mul_471 = torch.ops.aten.mul.Tensor(sub_159, rsqrt_159);  sub_159 = rsqrt_159 = None
        mul_472 = torch.ops.aten.mul.Tensor(mul_471, arg439_1);  mul_471 = arg439_1 = None
        add_479 = torch.ops.aten.add.Tensor(mul_472, arg440_1);  mul_472 = arg440_1 = None
        view_797 = torch.ops.aten.view.default(add_479, [392, 320]);  add_479 = None
        permute_542 = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_257 = torch.ops.aten.addmm.default(arg442_1, view_797, permute_542);  arg442_1 = view_797 = permute_542 = None
        view_798 = torch.ops.aten.view.default(addmm_257, [8, 49, 640]);  addmm_257 = None
        view_799 = torch.ops.aten.view.default(view_798, [8, -1, 2, 5, 64]);  view_798 = None
        permute_543 = torch.ops.aten.permute.default(view_799, [2, 0, 3, 1, 4]);  view_799 = None
        unbind_51 = torch.ops.aten.unbind.int(permute_543);  permute_543 = None
        getitem_626 = unbind_51[0]
        getitem_627 = unbind_51[1];  unbind_51 = None
        _scaled_dot_product_efficient_attention_51 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_539, getitem_626, getitem_627, None, False);  permute_539 = getitem_626 = getitem_627 = None
        getitem_628 = _scaled_dot_product_efficient_attention_51[0];  _scaled_dot_product_efficient_attention_51 = None
        permute_544 = torch.ops.aten.permute.default(getitem_628, [0, 2, 1, 3]);  getitem_628 = None
        view_800 = torch.ops.aten.view.default(permute_544, [8, 196, 320]);  permute_544 = None
        view_801 = torch.ops.aten.view.default(view_800, [1568, 320]);  view_800 = None
        permute_545 = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        addmm_258 = torch.ops.aten.addmm.default(arg444_1, view_801, permute_545);  arg444_1 = view_801 = permute_545 = None
        view_802 = torch.ops.aten.view.default(addmm_258, [8, 196, 320]);  addmm_258 = None
        add_480 = torch.ops.aten.add.Tensor(add_475, view_802);  add_475 = view_802 = None
        var_mean_160 = torch.ops.aten.var_mean.correction(add_480, [2], correction = 0, keepdim = True)
        getitem_632 = var_mean_160[0]
        getitem_633 = var_mean_160[1];  var_mean_160 = None
        add_481 = torch.ops.aten.add.Tensor(getitem_632, 1e-06);  getitem_632 = None
        rsqrt_160 = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        sub_160 = torch.ops.aten.sub.Tensor(add_480, getitem_633);  getitem_633 = None
        mul_473 = torch.ops.aten.mul.Tensor(sub_160, rsqrt_160);  sub_160 = rsqrt_160 = None
        mul_474 = torch.ops.aten.mul.Tensor(mul_473, arg445_1);  mul_473 = arg445_1 = None
        add_482 = torch.ops.aten.add.Tensor(mul_474, arg446_1);  mul_474 = arg446_1 = None
        view_803 = torch.ops.aten.view.default(add_482, [1568, 320]);  add_482 = None
        permute_546 = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_259 = torch.ops.aten.addmm.default(arg448_1, view_803, permute_546);  arg448_1 = view_803 = permute_546 = None
        view_804 = torch.ops.aten.view.default(addmm_259, [8, 196, 1280]);  addmm_259 = None
        mul_475 = torch.ops.aten.mul.Tensor(view_804, 0.5)
        mul_476 = torch.ops.aten.mul.Tensor(view_804, 0.7071067811865476);  view_804 = None
        erf_51 = torch.ops.aten.erf.default(mul_476);  mul_476 = None
        add_483 = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_477 = torch.ops.aten.mul.Tensor(mul_475, add_483);  mul_475 = add_483 = None
        view_805 = torch.ops.aten.view.default(mul_477, [1568, 1280]);  mul_477 = None
        permute_547 = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
        addmm_260 = torch.ops.aten.addmm.default(arg450_1, view_805, permute_547);  arg450_1 = view_805 = permute_547 = None
        view_806 = torch.ops.aten.view.default(addmm_260, [8, 196, 320]);  addmm_260 = None
        add_484 = torch.ops.aten.add.Tensor(add_480, view_806);  add_480 = view_806 = None
        var_mean_161 = torch.ops.aten.var_mean.correction(add_484, [2], correction = 0, keepdim = True)
        getitem_634 = var_mean_161[0]
        getitem_635 = var_mean_161[1];  var_mean_161 = None
        add_485 = torch.ops.aten.add.Tensor(getitem_634, 1e-06);  getitem_634 = None
        rsqrt_161 = torch.ops.aten.rsqrt.default(add_485);  add_485 = None
        sub_161 = torch.ops.aten.sub.Tensor(add_484, getitem_635);  getitem_635 = None
        mul_478 = torch.ops.aten.mul.Tensor(sub_161, rsqrt_161);  sub_161 = rsqrt_161 = None
        mul_479 = torch.ops.aten.mul.Tensor(mul_478, arg451_1);  mul_478 = arg451_1 = None
        add_486 = torch.ops.aten.add.Tensor(mul_479, arg452_1);  mul_479 = arg452_1 = None
        view_807 = torch.ops.aten.view.default(add_486, [1568, 320])
        permute_548 = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
        addmm_261 = torch.ops.aten.addmm.default(arg454_1, view_807, permute_548);  arg454_1 = view_807 = permute_548 = None
        view_808 = torch.ops.aten.view.default(addmm_261, [8, 196, 320]);  addmm_261 = None
        view_809 = torch.ops.aten.view.default(view_808, [8, 196, 5, 64]);  view_808 = None
        permute_549 = torch.ops.aten.permute.default(view_809, [0, 2, 1, 3]);  view_809 = None
        permute_550 = torch.ops.aten.permute.default(add_486, [0, 2, 1]);  add_486 = None
        view_810 = torch.ops.aten.view.default(permute_550, [8, 320, 14, 14]);  permute_550 = None
        convolution_63 = torch.ops.aten.convolution.default(view_810, arg455_1, arg456_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_810 = arg455_1 = arg456_1 = None
        view_811 = torch.ops.aten.view.default(convolution_63, [8, 320, 49]);  convolution_63 = None
        permute_551 = torch.ops.aten.permute.default(view_811, [0, 2, 1]);  view_811 = None
        var_mean_162 = torch.ops.aten.var_mean.correction(permute_551, [2], correction = 0, keepdim = True)
        getitem_636 = var_mean_162[0]
        getitem_637 = var_mean_162[1];  var_mean_162 = None
        add_487 = torch.ops.aten.add.Tensor(getitem_636, 1e-05);  getitem_636 = None
        rsqrt_162 = torch.ops.aten.rsqrt.default(add_487);  add_487 = None
        sub_162 = torch.ops.aten.sub.Tensor(permute_551, getitem_637);  permute_551 = getitem_637 = None
        mul_480 = torch.ops.aten.mul.Tensor(sub_162, rsqrt_162);  sub_162 = rsqrt_162 = None
        mul_481 = torch.ops.aten.mul.Tensor(mul_480, arg457_1);  mul_480 = arg457_1 = None
        add_488 = torch.ops.aten.add.Tensor(mul_481, arg458_1);  mul_481 = arg458_1 = None
        view_812 = torch.ops.aten.view.default(add_488, [392, 320]);  add_488 = None
        permute_552 = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        addmm_262 = torch.ops.aten.addmm.default(arg460_1, view_812, permute_552);  arg460_1 = view_812 = permute_552 = None
        view_813 = torch.ops.aten.view.default(addmm_262, [8, 49, 640]);  addmm_262 = None
        view_814 = torch.ops.aten.view.default(view_813, [8, -1, 2, 5, 64]);  view_813 = None
        permute_553 = torch.ops.aten.permute.default(view_814, [2, 0, 3, 1, 4]);  view_814 = None
        unbind_52 = torch.ops.aten.unbind.int(permute_553);  permute_553 = None
        getitem_638 = unbind_52[0]
        getitem_639 = unbind_52[1];  unbind_52 = None
        _scaled_dot_product_efficient_attention_52 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_549, getitem_638, getitem_639, None, False);  permute_549 = getitem_638 = getitem_639 = None
        getitem_640 = _scaled_dot_product_efficient_attention_52[0];  _scaled_dot_product_efficient_attention_52 = None
        permute_554 = torch.ops.aten.permute.default(getitem_640, [0, 2, 1, 3]);  getitem_640 = None
        view_815 = torch.ops.aten.view.default(permute_554, [8, 196, 320]);  permute_554 = None
        view_816 = torch.ops.aten.view.default(view_815, [1568, 320]);  view_815 = None
        permute_555 = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_263 = torch.ops.aten.addmm.default(arg462_1, view_816, permute_555);  arg462_1 = view_816 = permute_555 = None
        view_817 = torch.ops.aten.view.default(addmm_263, [8, 196, 320]);  addmm_263 = None
        add_489 = torch.ops.aten.add.Tensor(add_484, view_817);  add_484 = view_817 = None
        var_mean_163 = torch.ops.aten.var_mean.correction(add_489, [2], correction = 0, keepdim = True)
        getitem_644 = var_mean_163[0]
        getitem_645 = var_mean_163[1];  var_mean_163 = None
        add_490 = torch.ops.aten.add.Tensor(getitem_644, 1e-06);  getitem_644 = None
        rsqrt_163 = torch.ops.aten.rsqrt.default(add_490);  add_490 = None
        sub_163 = torch.ops.aten.sub.Tensor(add_489, getitem_645);  getitem_645 = None
        mul_482 = torch.ops.aten.mul.Tensor(sub_163, rsqrt_163);  sub_163 = rsqrt_163 = None
        mul_483 = torch.ops.aten.mul.Tensor(mul_482, arg463_1);  mul_482 = arg463_1 = None
        add_491 = torch.ops.aten.add.Tensor(mul_483, arg464_1);  mul_483 = arg464_1 = None
        view_818 = torch.ops.aten.view.default(add_491, [1568, 320]);  add_491 = None
        permute_556 = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        addmm_264 = torch.ops.aten.addmm.default(arg466_1, view_818, permute_556);  arg466_1 = view_818 = permute_556 = None
        view_819 = torch.ops.aten.view.default(addmm_264, [8, 196, 1280]);  addmm_264 = None
        mul_484 = torch.ops.aten.mul.Tensor(view_819, 0.5)
        mul_485 = torch.ops.aten.mul.Tensor(view_819, 0.7071067811865476);  view_819 = None
        erf_52 = torch.ops.aten.erf.default(mul_485);  mul_485 = None
        add_492 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_486 = torch.ops.aten.mul.Tensor(mul_484, add_492);  mul_484 = add_492 = None
        view_820 = torch.ops.aten.view.default(mul_486, [1568, 1280]);  mul_486 = None
        permute_557 = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
        addmm_265 = torch.ops.aten.addmm.default(arg468_1, view_820, permute_557);  arg468_1 = view_820 = permute_557 = None
        view_821 = torch.ops.aten.view.default(addmm_265, [8, 196, 320]);  addmm_265 = None
        add_493 = torch.ops.aten.add.Tensor(add_489, view_821);  add_489 = view_821 = None
        view_822 = torch.ops.aten.view.default(add_493, [8, 14, 14, -1]);  add_493 = None
        permute_558 = torch.ops.aten.permute.default(view_822, [0, 3, 1, 2]);  view_822 = None
        clone_179 = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
        convolution_64 = torch.ops.aten.convolution.default(clone_179, arg469_1, arg470_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_179 = arg469_1 = arg470_1 = None
        view_823 = torch.ops.aten.view.default(convolution_64, [8, 512, 49]);  convolution_64 = None
        permute_559 = torch.ops.aten.permute.default(view_823, [0, 2, 1]);  view_823 = None
        clone_180 = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
        var_mean_164 = torch.ops.aten.var_mean.correction(clone_180, [2], correction = 0, keepdim = True)
        getitem_646 = var_mean_164[0]
        getitem_647 = var_mean_164[1];  var_mean_164 = None
        add_494 = torch.ops.aten.add.Tensor(getitem_646, 1e-05);  getitem_646 = None
        rsqrt_164 = torch.ops.aten.rsqrt.default(add_494);  add_494 = None
        sub_164 = torch.ops.aten.sub.Tensor(clone_180, getitem_647);  clone_180 = getitem_647 = None
        mul_487 = torch.ops.aten.mul.Tensor(sub_164, rsqrt_164);  sub_164 = rsqrt_164 = None
        mul_488 = torch.ops.aten.mul.Tensor(mul_487, arg471_1);  mul_487 = arg471_1 = None
        add_495 = torch.ops.aten.add.Tensor(mul_488, arg472_1);  mul_488 = arg472_1 = None
        var_mean_165 = torch.ops.aten.var_mean.correction(add_495, [2], correction = 0, keepdim = True)
        getitem_648 = var_mean_165[0]
        getitem_649 = var_mean_165[1];  var_mean_165 = None
        add_496 = torch.ops.aten.add.Tensor(getitem_648, 1e-06);  getitem_648 = None
        rsqrt_165 = torch.ops.aten.rsqrt.default(add_496);  add_496 = None
        sub_165 = torch.ops.aten.sub.Tensor(add_495, getitem_649);  getitem_649 = None
        mul_489 = torch.ops.aten.mul.Tensor(sub_165, rsqrt_165);  sub_165 = rsqrt_165 = None
        mul_490 = torch.ops.aten.mul.Tensor(mul_489, arg473_1);  mul_489 = arg473_1 = None
        add_497 = torch.ops.aten.add.Tensor(mul_490, arg474_1);  mul_490 = arg474_1 = None
        view_824 = torch.ops.aten.view.default(add_497, [392, 512])
        permute_560 = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_266 = torch.ops.aten.addmm.default(arg476_1, view_824, permute_560);  arg476_1 = view_824 = permute_560 = None
        view_825 = torch.ops.aten.view.default(addmm_266, [8, 49, 512]);  addmm_266 = None
        view_826 = torch.ops.aten.view.default(view_825, [8, 49, 8, 64]);  view_825 = None
        permute_561 = torch.ops.aten.permute.default(view_826, [0, 2, 1, 3]);  view_826 = None
        view_827 = torch.ops.aten.view.default(add_497, [392, 512]);  add_497 = None
        permute_562 = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_267 = torch.ops.aten.addmm.default(arg478_1, view_827, permute_562);  arg478_1 = view_827 = permute_562 = None
        view_828 = torch.ops.aten.view.default(addmm_267, [8, 49, 1024]);  addmm_267 = None
        view_829 = torch.ops.aten.view.default(view_828, [8, -1, 2, 8, 64]);  view_828 = None
        permute_563 = torch.ops.aten.permute.default(view_829, [2, 0, 3, 1, 4]);  view_829 = None
        unbind_53 = torch.ops.aten.unbind.int(permute_563);  permute_563 = None
        getitem_650 = unbind_53[0]
        getitem_651 = unbind_53[1];  unbind_53 = None
        _scaled_dot_product_efficient_attention_53 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_561, getitem_650, getitem_651, None, False);  permute_561 = getitem_650 = getitem_651 = None
        getitem_652 = _scaled_dot_product_efficient_attention_53[0];  _scaled_dot_product_efficient_attention_53 = None
        permute_564 = torch.ops.aten.permute.default(getitem_652, [0, 2, 1, 3]);  getitem_652 = None
        view_830 = torch.ops.aten.view.default(permute_564, [8, 49, 512]);  permute_564 = None
        view_831 = torch.ops.aten.view.default(view_830, [392, 512]);  view_830 = None
        permute_565 = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        addmm_268 = torch.ops.aten.addmm.default(arg480_1, view_831, permute_565);  arg480_1 = view_831 = permute_565 = None
        view_832 = torch.ops.aten.view.default(addmm_268, [8, 49, 512]);  addmm_268 = None
        add_498 = torch.ops.aten.add.Tensor(add_495, view_832);  add_495 = view_832 = None
        var_mean_166 = torch.ops.aten.var_mean.correction(add_498, [2], correction = 0, keepdim = True)
        getitem_656 = var_mean_166[0]
        getitem_657 = var_mean_166[1];  var_mean_166 = None
        add_499 = torch.ops.aten.add.Tensor(getitem_656, 1e-06);  getitem_656 = None
        rsqrt_166 = torch.ops.aten.rsqrt.default(add_499);  add_499 = None
        sub_166 = torch.ops.aten.sub.Tensor(add_498, getitem_657);  getitem_657 = None
        mul_491 = torch.ops.aten.mul.Tensor(sub_166, rsqrt_166);  sub_166 = rsqrt_166 = None
        mul_492 = torch.ops.aten.mul.Tensor(mul_491, arg481_1);  mul_491 = arg481_1 = None
        add_500 = torch.ops.aten.add.Tensor(mul_492, arg482_1);  mul_492 = arg482_1 = None
        view_833 = torch.ops.aten.view.default(add_500, [392, 512]);  add_500 = None
        permute_566 = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_269 = torch.ops.aten.addmm.default(arg484_1, view_833, permute_566);  arg484_1 = view_833 = permute_566 = None
        view_834 = torch.ops.aten.view.default(addmm_269, [8, 49, 2048]);  addmm_269 = None
        mul_493 = torch.ops.aten.mul.Tensor(view_834, 0.5)
        mul_494 = torch.ops.aten.mul.Tensor(view_834, 0.7071067811865476);  view_834 = None
        erf_53 = torch.ops.aten.erf.default(mul_494);  mul_494 = None
        add_501 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_495 = torch.ops.aten.mul.Tensor(mul_493, add_501);  mul_493 = add_501 = None
        view_835 = torch.ops.aten.view.default(mul_495, [392, 2048]);  mul_495 = None
        permute_567 = torch.ops.aten.permute.default(arg485_1, [1, 0]);  arg485_1 = None
        addmm_270 = torch.ops.aten.addmm.default(arg486_1, view_835, permute_567);  arg486_1 = view_835 = permute_567 = None
        view_836 = torch.ops.aten.view.default(addmm_270, [8, 49, 512]);  addmm_270 = None
        add_502 = torch.ops.aten.add.Tensor(add_498, view_836);  add_498 = view_836 = None
        permute_568 = torch.ops.aten.permute.default(add_502, [0, 2, 1]);  add_502 = None
        view_837 = torch.ops.aten.view.default(permute_568, [8, 512, 7, 7]);  permute_568 = None
        convolution_65 = torch.ops.aten.convolution.default(view_837, arg487_1, arg488_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  arg487_1 = arg488_1 = None
        add_503 = torch.ops.aten.add.Tensor(convolution_65, view_837);  convolution_65 = view_837 = None
        view_839 = torch.ops.aten.view.default(add_503, [8, 512, 49]);  add_503 = None
        permute_570 = torch.ops.aten.permute.default(view_839, [0, 2, 1]);  view_839 = None
        var_mean_167 = torch.ops.aten.var_mean.correction(permute_570, [2], correction = 0, keepdim = True)
        getitem_658 = var_mean_167[0]
        getitem_659 = var_mean_167[1];  var_mean_167 = None
        add_504 = torch.ops.aten.add.Tensor(getitem_658, 1e-06);  getitem_658 = None
        rsqrt_167 = torch.ops.aten.rsqrt.default(add_504);  add_504 = None
        sub_167 = torch.ops.aten.sub.Tensor(permute_570, getitem_659);  getitem_659 = None
        mul_496 = torch.ops.aten.mul.Tensor(sub_167, rsqrt_167);  sub_167 = rsqrt_167 = None
        mul_497 = torch.ops.aten.mul.Tensor(mul_496, arg489_1);  mul_496 = arg489_1 = None
        add_505 = torch.ops.aten.add.Tensor(mul_497, arg490_1);  mul_497 = arg490_1 = None
        view_840 = torch.ops.aten.view.default(add_505, [392, 512])
        permute_571 = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
        addmm_271 = torch.ops.aten.addmm.default(arg492_1, view_840, permute_571);  arg492_1 = view_840 = permute_571 = None
        view_841 = torch.ops.aten.view.default(addmm_271, [8, 49, 512]);  addmm_271 = None
        view_842 = torch.ops.aten.view.default(view_841, [8, 49, 8, 64]);  view_841 = None
        permute_572 = torch.ops.aten.permute.default(view_842, [0, 2, 1, 3]);  view_842 = None
        view_843 = torch.ops.aten.view.default(add_505, [392, 512]);  add_505 = None
        permute_573 = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_272 = torch.ops.aten.addmm.default(arg494_1, view_843, permute_573);  arg494_1 = view_843 = permute_573 = None
        view_844 = torch.ops.aten.view.default(addmm_272, [8, 49, 1024]);  addmm_272 = None
        view_845 = torch.ops.aten.view.default(view_844, [8, -1, 2, 8, 64]);  view_844 = None
        permute_574 = torch.ops.aten.permute.default(view_845, [2, 0, 3, 1, 4]);  view_845 = None
        unbind_54 = torch.ops.aten.unbind.int(permute_574);  permute_574 = None
        getitem_660 = unbind_54[0]
        getitem_661 = unbind_54[1];  unbind_54 = None
        _scaled_dot_product_efficient_attention_54 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_572, getitem_660, getitem_661, None, False);  permute_572 = getitem_660 = getitem_661 = None
        getitem_662 = _scaled_dot_product_efficient_attention_54[0];  _scaled_dot_product_efficient_attention_54 = None
        permute_575 = torch.ops.aten.permute.default(getitem_662, [0, 2, 1, 3]);  getitem_662 = None
        view_846 = torch.ops.aten.view.default(permute_575, [8, 49, 512]);  permute_575 = None
        view_847 = torch.ops.aten.view.default(view_846, [392, 512]);  view_846 = None
        permute_576 = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        addmm_273 = torch.ops.aten.addmm.default(arg496_1, view_847, permute_576);  arg496_1 = view_847 = permute_576 = None
        view_848 = torch.ops.aten.view.default(addmm_273, [8, 49, 512]);  addmm_273 = None
        add_506 = torch.ops.aten.add.Tensor(permute_570, view_848);  permute_570 = view_848 = None
        var_mean_168 = torch.ops.aten.var_mean.correction(add_506, [2], correction = 0, keepdim = True)
        getitem_666 = var_mean_168[0]
        getitem_667 = var_mean_168[1];  var_mean_168 = None
        add_507 = torch.ops.aten.add.Tensor(getitem_666, 1e-06);  getitem_666 = None
        rsqrt_168 = torch.ops.aten.rsqrt.default(add_507);  add_507 = None
        sub_168 = torch.ops.aten.sub.Tensor(add_506, getitem_667);  getitem_667 = None
        mul_498 = torch.ops.aten.mul.Tensor(sub_168, rsqrt_168);  sub_168 = rsqrt_168 = None
        mul_499 = torch.ops.aten.mul.Tensor(mul_498, arg497_1);  mul_498 = arg497_1 = None
        add_508 = torch.ops.aten.add.Tensor(mul_499, arg498_1);  mul_499 = arg498_1 = None
        view_849 = torch.ops.aten.view.default(add_508, [392, 512]);  add_508 = None
        permute_577 = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_274 = torch.ops.aten.addmm.default(arg500_1, view_849, permute_577);  arg500_1 = view_849 = permute_577 = None
        view_850 = torch.ops.aten.view.default(addmm_274, [8, 49, 2048]);  addmm_274 = None
        mul_500 = torch.ops.aten.mul.Tensor(view_850, 0.5)
        mul_501 = torch.ops.aten.mul.Tensor(view_850, 0.7071067811865476);  view_850 = None
        erf_54 = torch.ops.aten.erf.default(mul_501);  mul_501 = None
        add_509 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_502 = torch.ops.aten.mul.Tensor(mul_500, add_509);  mul_500 = add_509 = None
        view_851 = torch.ops.aten.view.default(mul_502, [392, 2048]);  mul_502 = None
        permute_578 = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
        addmm_275 = torch.ops.aten.addmm.default(arg502_1, view_851, permute_578);  arg502_1 = view_851 = permute_578 = None
        view_852 = torch.ops.aten.view.default(addmm_275, [8, 49, 512]);  addmm_275 = None
        add_510 = torch.ops.aten.add.Tensor(add_506, view_852);  add_506 = view_852 = None
        var_mean_169 = torch.ops.aten.var_mean.correction(add_510, [2], correction = 0, keepdim = True)
        getitem_668 = var_mean_169[0]
        getitem_669 = var_mean_169[1];  var_mean_169 = None
        add_511 = torch.ops.aten.add.Tensor(getitem_668, 1e-06);  getitem_668 = None
        rsqrt_169 = torch.ops.aten.rsqrt.default(add_511);  add_511 = None
        sub_169 = torch.ops.aten.sub.Tensor(add_510, getitem_669);  getitem_669 = None
        mul_503 = torch.ops.aten.mul.Tensor(sub_169, rsqrt_169);  sub_169 = rsqrt_169 = None
        mul_504 = torch.ops.aten.mul.Tensor(mul_503, arg503_1);  mul_503 = arg503_1 = None
        add_512 = torch.ops.aten.add.Tensor(mul_504, arg504_1);  mul_504 = arg504_1 = None
        view_853 = torch.ops.aten.view.default(add_512, [392, 512])
        permute_579 = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
        addmm_276 = torch.ops.aten.addmm.default(arg506_1, view_853, permute_579);  arg506_1 = view_853 = permute_579 = None
        view_854 = torch.ops.aten.view.default(addmm_276, [8, 49, 512]);  addmm_276 = None
        view_855 = torch.ops.aten.view.default(view_854, [8, 49, 8, 64]);  view_854 = None
        permute_580 = torch.ops.aten.permute.default(view_855, [0, 2, 1, 3]);  view_855 = None
        view_856 = torch.ops.aten.view.default(add_512, [392, 512]);  add_512 = None
        permute_581 = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
        addmm_277 = torch.ops.aten.addmm.default(arg508_1, view_856, permute_581);  arg508_1 = view_856 = permute_581 = None
        view_857 = torch.ops.aten.view.default(addmm_277, [8, 49, 1024]);  addmm_277 = None
        view_858 = torch.ops.aten.view.default(view_857, [8, -1, 2, 8, 64]);  view_857 = None
        permute_582 = torch.ops.aten.permute.default(view_858, [2, 0, 3, 1, 4]);  view_858 = None
        unbind_55 = torch.ops.aten.unbind.int(permute_582);  permute_582 = None
        getitem_670 = unbind_55[0]
        getitem_671 = unbind_55[1];  unbind_55 = None
        _scaled_dot_product_efficient_attention_55 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_580, getitem_670, getitem_671, None, False);  permute_580 = getitem_670 = getitem_671 = None
        getitem_672 = _scaled_dot_product_efficient_attention_55[0];  _scaled_dot_product_efficient_attention_55 = None
        permute_583 = torch.ops.aten.permute.default(getitem_672, [0, 2, 1, 3]);  getitem_672 = None
        view_859 = torch.ops.aten.view.default(permute_583, [8, 49, 512]);  permute_583 = None
        view_860 = torch.ops.aten.view.default(view_859, [392, 512]);  view_859 = None
        permute_584 = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        addmm_278 = torch.ops.aten.addmm.default(arg510_1, view_860, permute_584);  arg510_1 = view_860 = permute_584 = None
        view_861 = torch.ops.aten.view.default(addmm_278, [8, 49, 512]);  addmm_278 = None
        add_513 = torch.ops.aten.add.Tensor(add_510, view_861);  add_510 = view_861 = None
        var_mean_170 = torch.ops.aten.var_mean.correction(add_513, [2], correction = 0, keepdim = True)
        getitem_676 = var_mean_170[0]
        getitem_677 = var_mean_170[1];  var_mean_170 = None
        add_514 = torch.ops.aten.add.Tensor(getitem_676, 1e-06);  getitem_676 = None
        rsqrt_170 = torch.ops.aten.rsqrt.default(add_514);  add_514 = None
        sub_170 = torch.ops.aten.sub.Tensor(add_513, getitem_677);  getitem_677 = None
        mul_505 = torch.ops.aten.mul.Tensor(sub_170, rsqrt_170);  sub_170 = rsqrt_170 = None
        mul_506 = torch.ops.aten.mul.Tensor(mul_505, arg511_1);  mul_505 = arg511_1 = None
        add_515 = torch.ops.aten.add.Tensor(mul_506, arg512_1);  mul_506 = arg512_1 = None
        view_862 = torch.ops.aten.view.default(add_515, [392, 512]);  add_515 = None
        permute_585 = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
        addmm_279 = torch.ops.aten.addmm.default(arg514_1, view_862, permute_585);  arg514_1 = view_862 = permute_585 = None
        view_863 = torch.ops.aten.view.default(addmm_279, [8, 49, 2048]);  addmm_279 = None
        mul_507 = torch.ops.aten.mul.Tensor(view_863, 0.5)
        mul_508 = torch.ops.aten.mul.Tensor(view_863, 0.7071067811865476);  view_863 = None
        erf_55 = torch.ops.aten.erf.default(mul_508);  mul_508 = None
        add_516 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_509 = torch.ops.aten.mul.Tensor(mul_507, add_516);  mul_507 = add_516 = None
        view_864 = torch.ops.aten.view.default(mul_509, [392, 2048]);  mul_509 = None
        permute_586 = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        addmm_280 = torch.ops.aten.addmm.default(arg516_1, view_864, permute_586);  arg516_1 = view_864 = permute_586 = None
        view_865 = torch.ops.aten.view.default(addmm_280, [8, 49, 512]);  addmm_280 = None
        add_517 = torch.ops.aten.add.Tensor(add_513, view_865);  add_513 = view_865 = None
        var_mean_171 = torch.ops.aten.var_mean.correction(add_517, [2], correction = 0, keepdim = True)
        getitem_678 = var_mean_171[0]
        getitem_679 = var_mean_171[1];  var_mean_171 = None
        add_518 = torch.ops.aten.add.Tensor(getitem_678, 1e-06);  getitem_678 = None
        rsqrt_171 = torch.ops.aten.rsqrt.default(add_518);  add_518 = None
        sub_171 = torch.ops.aten.sub.Tensor(add_517, getitem_679);  add_517 = getitem_679 = None
        mul_510 = torch.ops.aten.mul.Tensor(sub_171, rsqrt_171);  sub_171 = rsqrt_171 = None
        mul_511 = torch.ops.aten.mul.Tensor(mul_510, arg517_1);  mul_510 = arg517_1 = None
        add_519 = torch.ops.aten.add.Tensor(mul_511, arg518_1);  mul_511 = arg518_1 = None
        mean_1 = torch.ops.aten.mean.dim(add_519, [1]);  add_519 = None
        permute_587 = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
        addmm_281 = torch.ops.aten.addmm.default(arg520_1, mean_1, permute_587);  arg520_1 = mean_1 = permute_587 = None
        return (addmm_281,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64, 3, 4, 4), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64, 64), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64, 64, 8, 8), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128, 64), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64, 64), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf18, (64,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512, 64), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64, 512), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf23, (64, 1, 3, 3), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf26, (64,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf27, (64, 64), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf28, (64,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf29, (64, 64, 8, 8), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf30, (64,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf32, (64,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128, 64), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf34, (128,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf35, (64, 64), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf36, (64,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf37, (64,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf38, (64,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512, 64), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf40, (512,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (64, 512), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64, 64), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64, 64, 8, 8), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf51, (128, 64), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf52, (128,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64, 64), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf54, (64,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf55, (64,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf56, (64,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512, 64), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (64, 512), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf60, (64,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 64, 2, 2), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf66, (128,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128, 128), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf69, (128, 128, 4, 4), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf71, (128,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf72, (128,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256, 128), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf74, (256,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128, 128), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1024, 128), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf80, (1024,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf81, (128, 1024), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf82, (128,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128, 1, 3, 3), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (128, 128), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf88, (128,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf89, (128, 128, 4, 4), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256, 128), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf94, (256,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf95, (128, 128), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf96, (128,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1024, 128), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1024,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf101, (128, 1024), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf102, (128,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf103, (128,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf104, (128,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (128, 128), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128, 128, 4, 4), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (128,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf109, (128,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf110, (128,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256, 128), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (128, 128), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf114, (128,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf116, (128,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1024, 128), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1024,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf119, (128, 1024), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf120, (128,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf121, (128,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf122, (128,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (128, 128), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf124, (128,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf125, (128, 128, 4, 4), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf126, (128,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf127, (128,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf128, (128,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256, 128), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf131, (128, 128), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf132, (128,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf133, (128,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf134, (128,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1024, 128), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf137, (128, 1024), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf138, (128,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 655360, device=device(type='cuda', index=0))
    reader.tensor(buf139, (320, 128, 2, 2), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf140, (320,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf141, (320,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf142, (320,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf143, (320,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf144, (320,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf145, (320, 320), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf146, (320,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf147, (320, 320, 2, 2), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf148, (320,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf149, (320,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf150, (320,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf151, (640, 320), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf152, (640,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf153, (320, 320), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf154, (320,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf155, (320,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf156, (320,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1280, 320), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1280,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf159, (320, 1280), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf160, (320,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 11520, device=device(type='cuda', index=0))
    reader.tensor(buf161, (320, 1, 3, 3), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf162, (320,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf163, (320,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf164, (320,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf165, (320, 320), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf166, (320,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf167, (320, 320, 2, 2), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf168, (320,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf169, (320,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf170, (320,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf171, (640, 320), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf172, (640,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf173, (320, 320), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf174, (320,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf175, (320,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf176, (320,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1280, 320), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf178, (1280,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf179, (320, 1280), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf180, (320,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf181, (320,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf182, (320,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf183, (320, 320), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf184, (320,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf185, (320, 320, 2, 2), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf186, (320,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf187, (320,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf188, (320,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf189, (640, 320), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf190, (640,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf191, (320, 320), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf192, (320,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf193, (320,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf194, (320,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1280, 320), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1280,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf197, (320, 1280), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf198, (320,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf199, (320,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf200, (320,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf201, (320, 320), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf202, (320,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf203, (320, 320, 2, 2), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf204, (320,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf205, (320,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf206, (320,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf207, (640, 320), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf208, (640,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf209, (320, 320), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf210, (320,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf211, (320,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf212, (320,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1280, 320), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1280,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf215, (320, 1280), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf216, (320,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf217, (320,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf218, (320,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf219, (320, 320), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf220, (320,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf221, (320, 320, 2, 2), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf222, (320,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf223, (320,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf224, (320,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf225, (640, 320), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf226, (640,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf227, (320, 320), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf228, (320,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf229, (320,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf230, (320,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1280, 320), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1280,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf233, (320, 1280), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf234, (320,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf235, (320,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf236, (320,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf237, (320, 320), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf238, (320,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf239, (320, 320, 2, 2), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf240, (320,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf241, (320,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf242, (320,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf243, (640, 320), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf244, (640,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf245, (320, 320), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf246, (320,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf247, (320,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf248, (320,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf249, (1280, 320), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf250, (1280,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf251, (320, 1280), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf252, (320,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf253, (320,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf254, (320,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf255, (320, 320), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf256, (320,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf257, (320, 320, 2, 2), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf258, (320,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf259, (320,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf260, (320,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf261, (640, 320), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf262, (640,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf263, (320, 320), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf264, (320,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf265, (320,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf266, (320,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1280, 320), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1280,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf269, (320, 1280), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf270, (320,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf271, (320,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf272, (320,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf273, (320, 320), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf274, (320,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf275, (320, 320, 2, 2), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf276, (320,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf277, (320,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf278, (320,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf279, (640, 320), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf280, (640,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf281, (320, 320), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf282, (320,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf283, (320,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf284, (320,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf285, (1280, 320), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf286, (1280,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf287, (320, 1280), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf288, (320,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf289, (320,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf290, (320,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf291, (320, 320), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf292, (320,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf293, (320, 320, 2, 2), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf294, (320,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf295, (320,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf296, (320,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf297, (640, 320), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf298, (640,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf299, (320, 320), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf300, (320,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf301, (320,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf302, (320,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1280, 320), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf304, (1280,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf305, (320, 1280), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf306, (320,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf307, (320,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf308, (320,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf309, (320, 320), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf310, (320,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf311, (320, 320, 2, 2), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf312, (320,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf313, (320,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf314, (320,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf315, (640, 320), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf316, (640,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf317, (320, 320), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf318, (320,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf319, (320,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf320, (320,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf321, (1280, 320), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1280,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf323, (320, 1280), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf324, (320,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf325, (320,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf326, (320,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf327, (320, 320), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf328, (320,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf329, (320, 320, 2, 2), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf330, (320,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf331, (320,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf332, (320,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf333, (640, 320), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf334, (640,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf335, (320, 320), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf336, (320,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf337, (320,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf338, (320,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1280, 320), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf340, (1280,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf341, (320, 1280), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf342, (320,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf343, (320,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf344, (320,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf345, (320, 320), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf346, (320,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf347, (320, 320, 2, 2), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf348, (320,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf349, (320,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf350, (320,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf351, (640, 320), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf352, (640,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf353, (320, 320), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf354, (320,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf355, (320,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf356, (320,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf357, (1280, 320), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf358, (1280,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf359, (320, 1280), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf360, (320,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf361, (320,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf362, (320,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf363, (320, 320), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf364, (320,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf365, (320, 320, 2, 2), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf366, (320,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf367, (320,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf368, (320,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf369, (640, 320), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf370, (640,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf371, (320, 320), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf372, (320,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf373, (320,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf374, (320,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf375, (1280, 320), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf376, (1280,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf377, (320, 1280), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf378, (320,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf379, (320,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf380, (320,), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf381, (320, 320), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf382, (320,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf383, (320, 320, 2, 2), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf384, (320,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf385, (320,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf386, (320,), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf387, (640, 320), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf388, (640,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf389, (320, 320), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf390, (320,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf391, (320,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf392, (320,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf393, (1280, 320), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf394, (1280,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf395, (320, 1280), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf396, (320,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf397, (320,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf398, (320,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf399, (320, 320), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf400, (320,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf401, (320, 320, 2, 2), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf402, (320,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf403, (320,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf404, (320,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf405, (640, 320), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf406, (640,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf407, (320, 320), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf408, (320,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf409, (320,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf410, (320,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf411, (1280, 320), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf412, (1280,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf413, (320, 1280), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf414, (320,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf415, (320,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf416, (320,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf417, (320, 320), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf418, (320,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf419, (320, 320, 2, 2), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf420, (320,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf421, (320,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf422, (320,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf423, (640, 320), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf424, (640,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf425, (320, 320), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf426, (320,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf427, (320,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf428, (320,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf429, (1280, 320), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf430, (1280,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf431, (320, 1280), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf432, (320,), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf433, (320,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf434, (320,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf435, (320, 320), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf436, (320,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf437, (320, 320, 2, 2), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf438, (320,), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf439, (320,), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf440, (320,), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf441, (640, 320), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf442, (640,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf443, (320, 320), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf444, (320,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf445, (320,), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf446, (320,), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf447, (1280, 320), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf448, (1280,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf449, (320, 1280), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf450, (320,), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf451, (320,), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf452, (320,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf453, (320, 320), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf454, (320,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf455, (320, 320, 2, 2), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf456, (320,), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf457, (320,), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf458, (320,), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf459, (640, 320), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf460, (640,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf461, (320, 320), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf462, (320,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf463, (320,), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf464, (320,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf465, (1280, 320), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf466, (1280,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf467, (320, 1280), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf468, (320,), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 2621440, device=device(type='cuda', index=0))
    reader.tensor(buf469, (512, 320, 2, 2), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf470, (512,), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf471, (512,), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf472, (512,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf473, (512,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf474, (512,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf475, (512, 512), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf476, (512,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf477, (1024, 512), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf478, (1024,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf479, (512, 512), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf480, (512,), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf481, (512,), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf482, (512,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf483, (2048, 512), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf484, (2048,), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf485, (512, 2048), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf486, (512,), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf487, (512, 1, 3, 3), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf488, (512,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf489, (512,), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf490, (512,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf491, (512, 512), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf492, (512,), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf493, (1024, 512), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf494, (1024,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf495, (512, 512), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf496, (512,), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf497, (512,), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf498, (512,), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf499, (2048, 512), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf500, (2048,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf501, (512, 2048), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf502, (512,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf503, (512,), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf504, (512,), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf505, (512, 512), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf506, (512,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf507, (1024, 512), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf508, (1024,), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf509, (512, 512), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf510, (512,), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf511, (512,), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf512, (512,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf513, (2048, 512), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf514, (2048,), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf515, (512, 2048), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf516, (512,), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf517, (512,), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf518, (512,), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf519, (1000, 512), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf520, (1000,), is_leaf=True)  # arg520_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)