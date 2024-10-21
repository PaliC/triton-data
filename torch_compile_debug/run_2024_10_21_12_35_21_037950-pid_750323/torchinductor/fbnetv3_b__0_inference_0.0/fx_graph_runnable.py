
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1):
        convolution_124 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_292 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_292);  add_292 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_356 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_356, -1);  mul_356 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_697);  convolution_124 = unsqueeze_697 = None
        mul_357 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_358 = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_701);  mul_357 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_293 = torch.ops.aten.add.Tensor(mul_358, unsqueeze_703);  mul_358 = unsqueeze_703 = None
        add_294 = torch.ops.aten.add.Tensor(add_293, 3)
        clamp_min_95 = torch.ops.aten.clamp_min.default(add_294, 0);  add_294 = None
        clamp_max_95 = torch.ops.aten.clamp_max.default(clamp_min_95, 6);  clamp_min_95 = None
        mul_359 = torch.ops.aten.mul.Tensor(add_293, clamp_max_95);  add_293 = clamp_max_95 = None
        div_95 = torch.ops.aten.div.Tensor(mul_359, 6);  mul_359 = None
        convolution_125 = torch.ops.aten.convolution.default(div_95, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg6_1 = None
        add_295 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_705);  convolution_125 = unsqueeze_705 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_709);  mul_361 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_296 = torch.ops.aten.add.Tensor(mul_362, unsqueeze_711);  mul_362 = unsqueeze_711 = None
        add_297 = torch.ops.aten.add.Tensor(add_296, 3)
        clamp_min_96 = torch.ops.aten.clamp_min.default(add_297, 0);  add_297 = None
        clamp_max_96 = torch.ops.aten.clamp_max.default(clamp_min_96, 6);  clamp_min_96 = None
        mul_363 = torch.ops.aten.mul.Tensor(add_296, clamp_max_96);  add_296 = clamp_max_96 = None
        div_96 = torch.ops.aten.div.Tensor(mul_363, 6);  mul_363 = None
        convolution_126 = torch.ops.aten.convolution.default(div_96, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_96 = arg11_1 = None
        add_298 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_298);  add_298 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_364 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_364, -1);  mul_364 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_713);  convolution_126 = unsqueeze_713 = None
        mul_365 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_365, unsqueeze_717);  mul_365 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_299 = torch.ops.aten.add.Tensor(mul_366, unsqueeze_719);  mul_366 = unsqueeze_719 = None
        add_300 = torch.ops.aten.add.Tensor(add_299, div_95);  add_299 = div_95 = None
        convolution_127 = torch.ops.aten.convolution.default(add_300, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg16_1 = None
        add_301 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_301);  add_301 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_367 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_367, -1);  mul_367 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_721);  convolution_127 = unsqueeze_721 = None
        mul_368 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_368, unsqueeze_725);  mul_368 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_302 = torch.ops.aten.add.Tensor(mul_369, unsqueeze_727);  mul_369 = unsqueeze_727 = None
        add_303 = torch.ops.aten.add.Tensor(add_302, 3)
        clamp_min_97 = torch.ops.aten.clamp_min.default(add_303, 0);  add_303 = None
        clamp_max_97 = torch.ops.aten.clamp_max.default(clamp_min_97, 6);  clamp_min_97 = None
        mul_370 = torch.ops.aten.mul.Tensor(add_302, clamp_max_97);  add_302 = clamp_max_97 = None
        div_97 = torch.ops.aten.div.Tensor(mul_370, 6);  mul_370 = None
        convolution_128 = torch.ops.aten.convolution.default(div_97, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_97 = arg21_1 = None
        add_304 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_304);  add_304 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_371 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_371, -1);  mul_371 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_729);  convolution_128 = unsqueeze_729 = None
        mul_372 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_373 = torch.ops.aten.mul.Tensor(mul_372, unsqueeze_733);  mul_372 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_305 = torch.ops.aten.add.Tensor(mul_373, unsqueeze_735);  mul_373 = unsqueeze_735 = None
        add_306 = torch.ops.aten.add.Tensor(add_305, add_300);  add_305 = add_300 = None
        convolution_129 = torch.ops.aten.convolution.default(add_306, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_306 = arg26_1 = None
        add_307 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_374 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_374, -1);  mul_374 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_737);  convolution_129 = unsqueeze_737 = None
        mul_375 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_376 = torch.ops.aten.mul.Tensor(mul_375, unsqueeze_741);  mul_375 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_308 = torch.ops.aten.add.Tensor(mul_376, unsqueeze_743);  mul_376 = unsqueeze_743 = None
        add_309 = torch.ops.aten.add.Tensor(add_308, 3)
        clamp_min_98 = torch.ops.aten.clamp_min.default(add_309, 0);  add_309 = None
        clamp_max_98 = torch.ops.aten.clamp_max.default(clamp_min_98, 6);  clamp_min_98 = None
        mul_377 = torch.ops.aten.mul.Tensor(add_308, clamp_max_98);  add_308 = clamp_max_98 = None
        div_98 = torch.ops.aten.div.Tensor(mul_377, 6);  mul_377 = None
        convolution_130 = torch.ops.aten.convolution.default(div_98, arg31_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 64);  div_98 = arg31_1 = None
        add_310 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_310);  add_310 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_745);  convolution_130 = unsqueeze_745 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_749);  mul_379 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_311 = torch.ops.aten.add.Tensor(mul_380, unsqueeze_751);  mul_380 = unsqueeze_751 = None
        add_312 = torch.ops.aten.add.Tensor(add_311, 3)
        clamp_min_99 = torch.ops.aten.clamp_min.default(add_312, 0);  add_312 = None
        clamp_max_99 = torch.ops.aten.clamp_max.default(clamp_min_99, 6);  clamp_min_99 = None
        mul_381 = torch.ops.aten.mul.Tensor(add_311, clamp_max_99);  add_311 = clamp_max_99 = None
        div_99 = torch.ops.aten.div.Tensor(mul_381, 6);  mul_381 = None
        convolution_131 = torch.ops.aten.convolution.default(div_99, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_99 = arg36_1 = None
        add_313 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_382 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_382, -1);  mul_382 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_753);  convolution_131 = unsqueeze_753 = None
        mul_383 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_383, unsqueeze_757);  mul_383 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_314 = torch.ops.aten.add.Tensor(mul_384, unsqueeze_759);  mul_384 = unsqueeze_759 = None
        convolution_132 = torch.ops.aten.convolution.default(add_314, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = None
        add_315 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_315);  add_315 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_385 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_385, -1);  mul_385 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_761);  convolution_132 = unsqueeze_761 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_765);  mul_386 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_316 = torch.ops.aten.add.Tensor(mul_387, unsqueeze_767);  mul_387 = unsqueeze_767 = None
        add_317 = torch.ops.aten.add.Tensor(add_316, 3)
        clamp_min_100 = torch.ops.aten.clamp_min.default(add_317, 0);  add_317 = None
        clamp_max_100 = torch.ops.aten.clamp_max.default(clamp_min_100, 6);  clamp_min_100 = None
        mul_388 = torch.ops.aten.mul.Tensor(add_316, clamp_max_100);  add_316 = clamp_max_100 = None
        div_100 = torch.ops.aten.div.Tensor(mul_388, 6);  mul_388 = None
        convolution_133 = torch.ops.aten.convolution.default(div_100, arg46_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_100 = arg46_1 = None
        add_318 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_318);  add_318 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_389 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_389, -1);  mul_389 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_769);  convolution_133 = unsqueeze_769 = None
        mul_390 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_391 = torch.ops.aten.mul.Tensor(mul_390, unsqueeze_773);  mul_390 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_319 = torch.ops.aten.add.Tensor(mul_391, unsqueeze_775);  mul_391 = unsqueeze_775 = None
        add_320 = torch.ops.aten.add.Tensor(add_319, 3)
        clamp_min_101 = torch.ops.aten.clamp_min.default(add_320, 0);  add_320 = None
        clamp_max_101 = torch.ops.aten.clamp_max.default(clamp_min_101, 6);  clamp_min_101 = None
        mul_392 = torch.ops.aten.mul.Tensor(add_319, clamp_max_101);  add_319 = clamp_max_101 = None
        div_101 = torch.ops.aten.div.Tensor(mul_392, 6);  mul_392 = None
        convolution_134 = torch.ops.aten.convolution.default(div_101, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_101 = arg51_1 = None
        add_321 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_393 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_777);  convolution_134 = unsqueeze_777 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_781);  mul_394 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_322 = torch.ops.aten.add.Tensor(mul_395, unsqueeze_783);  mul_395 = unsqueeze_783 = None
        add_323 = torch.ops.aten.add.Tensor(add_322, add_314);  add_322 = add_314 = None
        convolution_135 = torch.ops.aten.convolution.default(add_323, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg56_1 = None
        add_324 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_324);  add_324 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_396 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_785);  convolution_135 = unsqueeze_785 = None
        mul_397 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_789);  mul_397 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_325 = torch.ops.aten.add.Tensor(mul_398, unsqueeze_791);  mul_398 = unsqueeze_791 = None
        add_326 = torch.ops.aten.add.Tensor(add_325, 3)
        clamp_min_102 = torch.ops.aten.clamp_min.default(add_326, 0);  add_326 = None
        clamp_max_102 = torch.ops.aten.clamp_max.default(clamp_min_102, 6);  clamp_min_102 = None
        mul_399 = torch.ops.aten.mul.Tensor(add_325, clamp_max_102);  add_325 = clamp_max_102 = None
        div_102 = torch.ops.aten.div.Tensor(mul_399, 6);  mul_399 = None
        convolution_136 = torch.ops.aten.convolution.default(div_102, arg61_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_102 = arg61_1 = None
        add_327 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_327);  add_327 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_400 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_400, -1);  mul_400 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_793);  convolution_136 = unsqueeze_793 = None
        mul_401 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_402 = torch.ops.aten.mul.Tensor(mul_401, unsqueeze_797);  mul_401 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_328 = torch.ops.aten.add.Tensor(mul_402, unsqueeze_799);  mul_402 = unsqueeze_799 = None
        add_329 = torch.ops.aten.add.Tensor(add_328, 3)
        clamp_min_103 = torch.ops.aten.clamp_min.default(add_329, 0);  add_329 = None
        clamp_max_103 = torch.ops.aten.clamp_max.default(clamp_min_103, 6);  clamp_min_103 = None
        mul_403 = torch.ops.aten.mul.Tensor(add_328, clamp_max_103);  add_328 = clamp_max_103 = None
        div_103 = torch.ops.aten.div.Tensor(mul_403, 6);  mul_403 = None
        convolution_137 = torch.ops.aten.convolution.default(div_103, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_103 = arg66_1 = None
        add_330 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_404 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_801);  convolution_137 = unsqueeze_801 = None
        mul_405 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_406 = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_805);  mul_405 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_331 = torch.ops.aten.add.Tensor(mul_406, unsqueeze_807);  mul_406 = unsqueeze_807 = None
        add_332 = torch.ops.aten.add.Tensor(add_331, add_323);  add_331 = add_323 = None
        convolution_138 = torch.ops.aten.convolution.default(add_332, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
        add_333 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_333);  add_333 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_407 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_809);  convolution_138 = unsqueeze_809 = None
        mul_408 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_409 = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_813);  mul_408 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_334 = torch.ops.aten.add.Tensor(mul_409, unsqueeze_815);  mul_409 = unsqueeze_815 = None
        add_335 = torch.ops.aten.add.Tensor(add_334, 3)
        clamp_min_104 = torch.ops.aten.clamp_min.default(add_335, 0);  add_335 = None
        clamp_max_104 = torch.ops.aten.clamp_max.default(clamp_min_104, 6);  clamp_min_104 = None
        mul_410 = torch.ops.aten.mul.Tensor(add_334, clamp_max_104);  add_334 = clamp_max_104 = None
        div_104 = torch.ops.aten.div.Tensor(mul_410, 6);  mul_410 = None
        convolution_139 = torch.ops.aten.convolution.default(div_104, arg76_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_104 = arg76_1 = None
        add_336 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_336);  add_336 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_411 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_817);  convolution_139 = unsqueeze_817 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_821);  mul_412 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_337 = torch.ops.aten.add.Tensor(mul_413, unsqueeze_823);  mul_413 = unsqueeze_823 = None
        add_338 = torch.ops.aten.add.Tensor(add_337, 3)
        clamp_min_105 = torch.ops.aten.clamp_min.default(add_338, 0);  add_338 = None
        clamp_max_105 = torch.ops.aten.clamp_max.default(clamp_min_105, 6);  clamp_min_105 = None
        mul_414 = torch.ops.aten.mul.Tensor(add_337, clamp_max_105);  add_337 = clamp_max_105 = None
        div_105 = torch.ops.aten.div.Tensor(mul_414, 6);  mul_414 = None
        convolution_140 = torch.ops.aten.convolution.default(div_105, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_105 = arg81_1 = None
        add_339 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_339);  add_339 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_415 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_415, -1);  mul_415 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_825);  convolution_140 = unsqueeze_825 = None
        mul_416 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_417 = torch.ops.aten.mul.Tensor(mul_416, unsqueeze_829);  mul_416 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_340 = torch.ops.aten.add.Tensor(mul_417, unsqueeze_831);  mul_417 = unsqueeze_831 = None
        add_341 = torch.ops.aten.add.Tensor(add_340, add_332);  add_340 = add_332 = None
        convolution_141 = torch.ops.aten.convolution.default(add_341, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_341 = arg86_1 = None
        add_342 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_342);  add_342 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_418 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_418, -1);  mul_418 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_833);  convolution_141 = unsqueeze_833 = None
        mul_419 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_420 = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_837);  mul_419 = unsqueeze_837 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_343 = torch.ops.aten.add.Tensor(mul_420, unsqueeze_839);  mul_420 = unsqueeze_839 = None
        add_344 = torch.ops.aten.add.Tensor(add_343, 3)
        clamp_min_106 = torch.ops.aten.clamp_min.default(add_344, 0);  add_344 = None
        clamp_max_106 = torch.ops.aten.clamp_max.default(clamp_min_106, 6);  clamp_min_106 = None
        mul_421 = torch.ops.aten.mul.Tensor(add_343, clamp_max_106);  add_343 = clamp_max_106 = None
        div_106 = torch.ops.aten.div.Tensor(mul_421, 6);  mul_421 = None
        convolution_142 = torch.ops.aten.convolution.default(div_106, arg91_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 120);  div_106 = arg91_1 = None
        add_345 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_345);  add_345 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_422 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(mul_422, -1);  mul_422 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_841);  convolution_142 = unsqueeze_841 = None
        mul_423 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_424 = torch.ops.aten.mul.Tensor(mul_423, unsqueeze_845);  mul_423 = unsqueeze_845 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_346 = torch.ops.aten.add.Tensor(mul_424, unsqueeze_847);  mul_424 = unsqueeze_847 = None
        add_347 = torch.ops.aten.add.Tensor(add_346, 3)
        clamp_min_107 = torch.ops.aten.clamp_min.default(add_347, 0);  add_347 = None
        clamp_max_107 = torch.ops.aten.clamp_max.default(clamp_min_107, 6);  clamp_min_107 = None
        mul_425 = torch.ops.aten.mul.Tensor(add_346, clamp_max_107);  add_346 = clamp_max_107 = None
        div_107 = torch.ops.aten.div.Tensor(mul_425, 6);  mul_425 = None
        mean_19 = torch.ops.aten.mean.dim(div_107, [2, 3], True)
        convolution_143 = torch.ops.aten.convolution.default(mean_19, arg96_1, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg96_1 = arg97_1 = None
        add_348 = torch.ops.aten.add.Tensor(convolution_143, 3)
        clamp_min_108 = torch.ops.aten.clamp_min.default(add_348, 0);  add_348 = None
        clamp_max_108 = torch.ops.aten.clamp_max.default(clamp_min_108, 6);  clamp_min_108 = None
        mul_426 = torch.ops.aten.mul.Tensor(convolution_143, clamp_max_108);  convolution_143 = clamp_max_108 = None
        div_108 = torch.ops.aten.div.Tensor(mul_426, 6);  mul_426 = None
        convolution_144 = torch.ops.aten.convolution.default(div_108, arg98_1, arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_108 = arg98_1 = arg99_1 = None
        add_349 = torch.ops.aten.add.Tensor(convolution_144, 3);  convolution_144 = None
        clamp_min_109 = torch.ops.aten.clamp_min.default(add_349, 0);  add_349 = None
        clamp_max_109 = torch.ops.aten.clamp_max.default(clamp_min_109, 6);  clamp_min_109 = None
        div_109 = torch.ops.aten.div.Tensor(clamp_max_109, 6);  clamp_max_109 = None
        mul_427 = torch.ops.aten.mul.Tensor(div_107, div_109);  div_107 = div_109 = None
        convolution_145 = torch.ops.aten.convolution.default(mul_427, arg100_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_427 = arg100_1 = None
        add_350 = torch.ops.aten.add.Tensor(arg102_1, 1e-05);  arg102_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_428 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(mul_428, -1);  mul_428 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106 = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_849);  convolution_145 = unsqueeze_849 = None
        mul_429 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_430 = torch.ops.aten.mul.Tensor(mul_429, unsqueeze_853);  mul_429 = unsqueeze_853 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_351 = torch.ops.aten.add.Tensor(mul_430, unsqueeze_855);  mul_430 = unsqueeze_855 = None
        convolution_146 = torch.ops.aten.convolution.default(add_351, arg105_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg105_1 = None
        add_352 = torch.ops.aten.add.Tensor(arg107_1, 1e-05);  arg107_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_352);  add_352 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_431 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_431, -1);  mul_431 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107 = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_857);  convolution_146 = unsqueeze_857 = None
        mul_432 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_433 = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_861);  mul_432 = unsqueeze_861 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_353 = torch.ops.aten.add.Tensor(mul_433, unsqueeze_863);  mul_433 = unsqueeze_863 = None
        add_354 = torch.ops.aten.add.Tensor(add_353, 3)
        clamp_min_110 = torch.ops.aten.clamp_min.default(add_354, 0);  add_354 = None
        clamp_max_110 = torch.ops.aten.clamp_max.default(clamp_min_110, 6);  clamp_min_110 = None
        mul_434 = torch.ops.aten.mul.Tensor(add_353, clamp_max_110);  add_353 = clamp_max_110 = None
        div_110 = torch.ops.aten.div.Tensor(mul_434, 6);  mul_434 = None
        convolution_147 = torch.ops.aten.convolution.default(div_110, arg110_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_110 = arg110_1 = None
        add_355 = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_355);  add_355 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_435 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108 = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_865);  convolution_147 = unsqueeze_865 = None
        mul_436 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_437 = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_869);  mul_436 = unsqueeze_869 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_356 = torch.ops.aten.add.Tensor(mul_437, unsqueeze_871);  mul_437 = unsqueeze_871 = None
        add_357 = torch.ops.aten.add.Tensor(add_356, 3)
        clamp_min_111 = torch.ops.aten.clamp_min.default(add_357, 0);  add_357 = None
        clamp_max_111 = torch.ops.aten.clamp_max.default(clamp_min_111, 6);  clamp_min_111 = None
        mul_438 = torch.ops.aten.mul.Tensor(add_356, clamp_max_111);  add_356 = clamp_max_111 = None
        div_111 = torch.ops.aten.div.Tensor(mul_438, 6);  mul_438 = None
        mean_20 = torch.ops.aten.mean.dim(div_111, [2, 3], True)
        convolution_148 = torch.ops.aten.convolution.default(mean_20, arg115_1, arg116_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg115_1 = arg116_1 = None
        add_358 = torch.ops.aten.add.Tensor(convolution_148, 3)
        clamp_min_112 = torch.ops.aten.clamp_min.default(add_358, 0);  add_358 = None
        clamp_max_112 = torch.ops.aten.clamp_max.default(clamp_min_112, 6);  clamp_min_112 = None
        mul_439 = torch.ops.aten.mul.Tensor(convolution_148, clamp_max_112);  convolution_148 = clamp_max_112 = None
        div_112 = torch.ops.aten.div.Tensor(mul_439, 6);  mul_439 = None
        convolution_149 = torch.ops.aten.convolution.default(div_112, arg117_1, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_112 = arg117_1 = arg118_1 = None
        add_359 = torch.ops.aten.add.Tensor(convolution_149, 3);  convolution_149 = None
        clamp_min_113 = torch.ops.aten.clamp_min.default(add_359, 0);  add_359 = None
        clamp_max_113 = torch.ops.aten.clamp_max.default(clamp_min_113, 6);  clamp_min_113 = None
        div_113 = torch.ops.aten.div.Tensor(clamp_max_113, 6);  clamp_max_113 = None
        mul_440 = torch.ops.aten.mul.Tensor(div_111, div_113);  div_111 = div_113 = None
        convolution_150 = torch.ops.aten.convolution.default(mul_440, arg119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_440 = arg119_1 = None
        add_360 = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_441 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109 = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_873);  convolution_150 = unsqueeze_873 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_877);  mul_442 = unsqueeze_877 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_879 = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_361 = torch.ops.aten.add.Tensor(mul_443, unsqueeze_879);  mul_443 = unsqueeze_879 = None
        add_362 = torch.ops.aten.add.Tensor(add_361, add_351);  add_361 = add_351 = None
        convolution_151 = torch.ops.aten.convolution.default(add_362, arg124_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg124_1 = None
        add_363 = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_444 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_881 = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110 = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_881);  convolution_151 = unsqueeze_881 = None
        mul_445 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_885);  mul_445 = unsqueeze_885 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_364 = torch.ops.aten.add.Tensor(mul_446, unsqueeze_887);  mul_446 = unsqueeze_887 = None
        add_365 = torch.ops.aten.add.Tensor(add_364, 3)
        clamp_min_114 = torch.ops.aten.clamp_min.default(add_365, 0);  add_365 = None
        clamp_max_114 = torch.ops.aten.clamp_max.default(clamp_min_114, 6);  clamp_min_114 = None
        mul_447 = torch.ops.aten.mul.Tensor(add_364, clamp_max_114);  add_364 = clamp_max_114 = None
        div_114 = torch.ops.aten.div.Tensor(mul_447, 6);  mul_447 = None
        convolution_152 = torch.ops.aten.convolution.default(div_114, arg129_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_114 = arg129_1 = None
        add_366 = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_366);  add_366 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_448 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_448, -1);  mul_448 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_889);  convolution_152 = unsqueeze_889 = None
        mul_449 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_450 = torch.ops.aten.mul.Tensor(mul_449, unsqueeze_893);  mul_449 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_367 = torch.ops.aten.add.Tensor(mul_450, unsqueeze_895);  mul_450 = unsqueeze_895 = None
        add_368 = torch.ops.aten.add.Tensor(add_367, 3)
        clamp_min_115 = torch.ops.aten.clamp_min.default(add_368, 0);  add_368 = None
        clamp_max_115 = torch.ops.aten.clamp_max.default(clamp_min_115, 6);  clamp_min_115 = None
        mul_451 = torch.ops.aten.mul.Tensor(add_367, clamp_max_115);  add_367 = clamp_max_115 = None
        div_115 = torch.ops.aten.div.Tensor(mul_451, 6);  mul_451 = None
        mean_21 = torch.ops.aten.mean.dim(div_115, [2, 3], True)
        convolution_153 = torch.ops.aten.convolution.default(mean_21, arg134_1, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg134_1 = arg135_1 = None
        add_369 = torch.ops.aten.add.Tensor(convolution_153, 3)
        clamp_min_116 = torch.ops.aten.clamp_min.default(add_369, 0);  add_369 = None
        clamp_max_116 = torch.ops.aten.clamp_max.default(clamp_min_116, 6);  clamp_min_116 = None
        mul_452 = torch.ops.aten.mul.Tensor(convolution_153, clamp_max_116);  convolution_153 = clamp_max_116 = None
        div_116 = torch.ops.aten.div.Tensor(mul_452, 6);  mul_452 = None
        convolution_154 = torch.ops.aten.convolution.default(div_116, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_116 = arg136_1 = arg137_1 = None
        add_370 = torch.ops.aten.add.Tensor(convolution_154, 3);  convolution_154 = None
        clamp_min_117 = torch.ops.aten.clamp_min.default(add_370, 0);  add_370 = None
        clamp_max_117 = torch.ops.aten.clamp_max.default(clamp_min_117, 6);  clamp_min_117 = None
        div_117 = torch.ops.aten.div.Tensor(clamp_max_117, 6);  clamp_max_117 = None
        mul_453 = torch.ops.aten.mul.Tensor(div_115, div_117);  div_115 = div_117 = None
        convolution_155 = torch.ops.aten.convolution.default(mul_453, arg138_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_453 = arg138_1 = None
        add_371 = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_371);  add_371 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_454 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_454, -1);  mul_454 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_897);  convolution_155 = unsqueeze_897 = None
        mul_455 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_456 = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_901);  mul_455 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_372 = torch.ops.aten.add.Tensor(mul_456, unsqueeze_903);  mul_456 = unsqueeze_903 = None
        add_373 = torch.ops.aten.add.Tensor(add_372, add_362);  add_372 = add_362 = None
        convolution_156 = torch.ops.aten.convolution.default(add_373, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg143_1 = None
        add_374 = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_374);  add_374 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_457 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_457, -1);  mul_457 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_905);  convolution_156 = unsqueeze_905 = None
        mul_458 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_459 = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_909);  mul_458 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_375 = torch.ops.aten.add.Tensor(mul_459, unsqueeze_911);  mul_459 = unsqueeze_911 = None
        add_376 = torch.ops.aten.add.Tensor(add_375, 3)
        clamp_min_118 = torch.ops.aten.clamp_min.default(add_376, 0);  add_376 = None
        clamp_max_118 = torch.ops.aten.clamp_max.default(clamp_min_118, 6);  clamp_min_118 = None
        mul_460 = torch.ops.aten.mul.Tensor(add_375, clamp_max_118);  add_375 = clamp_max_118 = None
        div_118 = torch.ops.aten.div.Tensor(mul_460, 6);  mul_460 = None
        convolution_157 = torch.ops.aten.convolution.default(div_118, arg148_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_118 = arg148_1 = None
        add_377 = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_377);  add_377 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_461 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_461, -1);  mul_461 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_913);  convolution_157 = unsqueeze_913 = None
        mul_462 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_463 = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_917);  mul_462 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_378 = torch.ops.aten.add.Tensor(mul_463, unsqueeze_919);  mul_463 = unsqueeze_919 = None
        add_379 = torch.ops.aten.add.Tensor(add_378, 3)
        clamp_min_119 = torch.ops.aten.clamp_min.default(add_379, 0);  add_379 = None
        clamp_max_119 = torch.ops.aten.clamp_max.default(clamp_min_119, 6);  clamp_min_119 = None
        mul_464 = torch.ops.aten.mul.Tensor(add_378, clamp_max_119);  add_378 = clamp_max_119 = None
        div_119 = torch.ops.aten.div.Tensor(mul_464, 6);  mul_464 = None
        mean_22 = torch.ops.aten.mean.dim(div_119, [2, 3], True)
        convolution_158 = torch.ops.aten.convolution.default(mean_22, arg153_1, arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg153_1 = arg154_1 = None
        add_380 = torch.ops.aten.add.Tensor(convolution_158, 3)
        clamp_min_120 = torch.ops.aten.clamp_min.default(add_380, 0);  add_380 = None
        clamp_max_120 = torch.ops.aten.clamp_max.default(clamp_min_120, 6);  clamp_min_120 = None
        mul_465 = torch.ops.aten.mul.Tensor(convolution_158, clamp_max_120);  convolution_158 = clamp_max_120 = None
        div_120 = torch.ops.aten.div.Tensor(mul_465, 6);  mul_465 = None
        convolution_159 = torch.ops.aten.convolution.default(div_120, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_120 = arg155_1 = arg156_1 = None
        add_381 = torch.ops.aten.add.Tensor(convolution_159, 3);  convolution_159 = None
        clamp_min_121 = torch.ops.aten.clamp_min.default(add_381, 0);  add_381 = None
        clamp_max_121 = torch.ops.aten.clamp_max.default(clamp_min_121, 6);  clamp_min_121 = None
        div_121 = torch.ops.aten.div.Tensor(clamp_max_121, 6);  clamp_max_121 = None
        mul_466 = torch.ops.aten.mul.Tensor(div_119, div_121);  div_119 = div_121 = None
        convolution_160 = torch.ops.aten.convolution.default(mul_466, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_466 = arg157_1 = None
        add_382 = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_382);  add_382 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_467 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_467, -1);  mul_467 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_921);  convolution_160 = unsqueeze_921 = None
        mul_468 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_469 = torch.ops.aten.mul.Tensor(mul_468, unsqueeze_925);  mul_468 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_383 = torch.ops.aten.add.Tensor(mul_469, unsqueeze_927);  mul_469 = unsqueeze_927 = None
        add_384 = torch.ops.aten.add.Tensor(add_383, add_373);  add_383 = add_373 = None
        convolution_161 = torch.ops.aten.convolution.default(add_384, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg162_1 = None
        add_385 = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_116 = torch.ops.aten.sqrt.default(add_385);  add_385 = None
        reciprocal_116 = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_470 = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_929 = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930 = torch.ops.aten.unsqueeze.default(mul_470, -1);  mul_470 = None
        unsqueeze_931 = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116 = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_929);  convolution_161 = unsqueeze_929 = None
        mul_471 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_933 = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_472 = torch.ops.aten.mul.Tensor(mul_471, unsqueeze_933);  mul_471 = unsqueeze_933 = None
        unsqueeze_934 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_935 = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_386 = torch.ops.aten.add.Tensor(mul_472, unsqueeze_935);  mul_472 = unsqueeze_935 = None
        add_387 = torch.ops.aten.add.Tensor(add_386, 3)
        clamp_min_122 = torch.ops.aten.clamp_min.default(add_387, 0);  add_387 = None
        clamp_max_122 = torch.ops.aten.clamp_max.default(clamp_min_122, 6);  clamp_min_122 = None
        mul_473 = torch.ops.aten.mul.Tensor(add_386, clamp_max_122);  add_386 = clamp_max_122 = None
        div_122 = torch.ops.aten.div.Tensor(mul_473, 6);  mul_473 = None
        convolution_162 = torch.ops.aten.convolution.default(div_122, arg167_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_122 = arg167_1 = None
        add_388 = torch.ops.aten.add.Tensor(arg169_1, 1e-05);  arg169_1 = None
        sqrt_117 = torch.ops.aten.sqrt.default(add_388);  add_388 = None
        reciprocal_117 = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_474 = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_937 = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938 = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
        unsqueeze_939 = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117 = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_937);  convolution_162 = unsqueeze_937 = None
        mul_475 = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_941 = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_476 = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_941);  mul_475 = unsqueeze_941 = None
        unsqueeze_942 = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_943 = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_389 = torch.ops.aten.add.Tensor(mul_476, unsqueeze_943);  mul_476 = unsqueeze_943 = None
        add_390 = torch.ops.aten.add.Tensor(add_389, 3)
        clamp_min_123 = torch.ops.aten.clamp_min.default(add_390, 0);  add_390 = None
        clamp_max_123 = torch.ops.aten.clamp_max.default(clamp_min_123, 6);  clamp_min_123 = None
        mul_477 = torch.ops.aten.mul.Tensor(add_389, clamp_max_123);  add_389 = clamp_max_123 = None
        div_123 = torch.ops.aten.div.Tensor(mul_477, 6);  mul_477 = None
        mean_23 = torch.ops.aten.mean.dim(div_123, [2, 3], True)
        convolution_163 = torch.ops.aten.convolution.default(mean_23, arg172_1, arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg172_1 = arg173_1 = None
        add_391 = torch.ops.aten.add.Tensor(convolution_163, 3)
        clamp_min_124 = torch.ops.aten.clamp_min.default(add_391, 0);  add_391 = None
        clamp_max_124 = torch.ops.aten.clamp_max.default(clamp_min_124, 6);  clamp_min_124 = None
        mul_478 = torch.ops.aten.mul.Tensor(convolution_163, clamp_max_124);  convolution_163 = clamp_max_124 = None
        div_124 = torch.ops.aten.div.Tensor(mul_478, 6);  mul_478 = None
        convolution_164 = torch.ops.aten.convolution.default(div_124, arg174_1, arg175_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_124 = arg174_1 = arg175_1 = None
        add_392 = torch.ops.aten.add.Tensor(convolution_164, 3);  convolution_164 = None
        clamp_min_125 = torch.ops.aten.clamp_min.default(add_392, 0);  add_392 = None
        clamp_max_125 = torch.ops.aten.clamp_max.default(clamp_min_125, 6);  clamp_min_125 = None
        div_125 = torch.ops.aten.div.Tensor(clamp_max_125, 6);  clamp_max_125 = None
        mul_479 = torch.ops.aten.mul.Tensor(div_123, div_125);  div_123 = div_125 = None
        convolution_165 = torch.ops.aten.convolution.default(mul_479, arg176_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_479 = arg176_1 = None
        add_393 = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_118 = torch.ops.aten.sqrt.default(add_393);  add_393 = None
        reciprocal_118 = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_480 = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_945 = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946 = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
        unsqueeze_947 = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118 = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_945);  convolution_165 = unsqueeze_945 = None
        mul_481 = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_949 = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_482 = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_949);  mul_481 = unsqueeze_949 = None
        unsqueeze_950 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_951 = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_394 = torch.ops.aten.add.Tensor(mul_482, unsqueeze_951);  mul_482 = unsqueeze_951 = None
        add_395 = torch.ops.aten.add.Tensor(add_394, add_384);  add_394 = add_384 = None
        convolution_166 = torch.ops.aten.convolution.default(add_395, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_395 = arg181_1 = None
        add_396 = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_119 = torch.ops.aten.sqrt.default(add_396);  add_396 = None
        reciprocal_119 = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_483 = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_953 = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954 = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
        unsqueeze_955 = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119 = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_953);  convolution_166 = unsqueeze_953 = None
        mul_484 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_957 = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_485 = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_957);  mul_484 = unsqueeze_957 = None
        unsqueeze_958 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_959 = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_397 = torch.ops.aten.add.Tensor(mul_485, unsqueeze_959);  mul_485 = unsqueeze_959 = None
        add_398 = torch.ops.aten.add.Tensor(add_397, 3)
        clamp_min_126 = torch.ops.aten.clamp_min.default(add_398, 0);  add_398 = None
        clamp_max_126 = torch.ops.aten.clamp_max.default(clamp_min_126, 6);  clamp_min_126 = None
        mul_486 = torch.ops.aten.mul.Tensor(add_397, clamp_max_126);  add_397 = clamp_max_126 = None
        div_126 = torch.ops.aten.div.Tensor(mul_486, 6);  mul_486 = None
        convolution_167 = torch.ops.aten.convolution.default(div_126, arg186_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 200);  div_126 = arg186_1 = None
        add_399 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_120 = torch.ops.aten.sqrt.default(add_399);  add_399 = None
        reciprocal_120 = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_487 = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_961 = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962 = torch.ops.aten.unsqueeze.default(mul_487, -1);  mul_487 = None
        unsqueeze_963 = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120 = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_961);  convolution_167 = unsqueeze_961 = None
        mul_488 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_965 = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_489 = torch.ops.aten.mul.Tensor(mul_488, unsqueeze_965);  mul_488 = unsqueeze_965 = None
        unsqueeze_966 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_967 = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_400 = torch.ops.aten.add.Tensor(mul_489, unsqueeze_967);  mul_489 = unsqueeze_967 = None
        add_401 = torch.ops.aten.add.Tensor(add_400, 3)
        clamp_min_127 = torch.ops.aten.clamp_min.default(add_401, 0);  add_401 = None
        clamp_max_127 = torch.ops.aten.clamp_max.default(clamp_min_127, 6);  clamp_min_127 = None
        mul_490 = torch.ops.aten.mul.Tensor(add_400, clamp_max_127);  add_400 = clamp_max_127 = None
        div_127 = torch.ops.aten.div.Tensor(mul_490, 6);  mul_490 = None
        convolution_168 = torch.ops.aten.convolution.default(div_127, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_127 = arg191_1 = None
        add_402 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_121 = torch.ops.aten.sqrt.default(add_402);  add_402 = None
        reciprocal_121 = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_491 = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_969 = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970 = torch.ops.aten.unsqueeze.default(mul_491, -1);  mul_491 = None
        unsqueeze_971 = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121 = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_969);  convolution_168 = unsqueeze_969 = None
        mul_492 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_973 = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_493 = torch.ops.aten.mul.Tensor(mul_492, unsqueeze_973);  mul_492 = unsqueeze_973 = None
        unsqueeze_974 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_975 = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_403 = torch.ops.aten.add.Tensor(mul_493, unsqueeze_975);  mul_493 = unsqueeze_975 = None
        convolution_169 = torch.ops.aten.convolution.default(add_403, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg196_1 = None
        add_404 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_122 = torch.ops.aten.sqrt.default(add_404);  add_404 = None
        reciprocal_122 = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_494 = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_977 = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978 = torch.ops.aten.unsqueeze.default(mul_494, -1);  mul_494 = None
        unsqueeze_979 = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122 = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_977);  convolution_169 = unsqueeze_977 = None
        mul_495 = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_981 = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_496 = torch.ops.aten.mul.Tensor(mul_495, unsqueeze_981);  mul_495 = unsqueeze_981 = None
        unsqueeze_982 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_983 = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_405 = torch.ops.aten.add.Tensor(mul_496, unsqueeze_983);  mul_496 = unsqueeze_983 = None
        add_406 = torch.ops.aten.add.Tensor(add_405, 3)
        clamp_min_128 = torch.ops.aten.clamp_min.default(add_406, 0);  add_406 = None
        clamp_max_128 = torch.ops.aten.clamp_max.default(clamp_min_128, 6);  clamp_min_128 = None
        mul_497 = torch.ops.aten.mul.Tensor(add_405, clamp_max_128);  add_405 = clamp_max_128 = None
        div_128 = torch.ops.aten.div.Tensor(mul_497, 6);  mul_497 = None
        convolution_170 = torch.ops.aten.convolution.default(div_128, arg201_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_128 = arg201_1 = None
        add_407 = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_123 = torch.ops.aten.sqrt.default(add_407);  add_407 = None
        reciprocal_123 = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_498 = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_985 = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986 = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_987 = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123 = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_985);  convolution_170 = unsqueeze_985 = None
        mul_499 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_989 = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_989);  mul_499 = unsqueeze_989 = None
        unsqueeze_990 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_991 = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_408 = torch.ops.aten.add.Tensor(mul_500, unsqueeze_991);  mul_500 = unsqueeze_991 = None
        add_409 = torch.ops.aten.add.Tensor(add_408, 3)
        clamp_min_129 = torch.ops.aten.clamp_min.default(add_409, 0);  add_409 = None
        clamp_max_129 = torch.ops.aten.clamp_max.default(clamp_min_129, 6);  clamp_min_129 = None
        mul_501 = torch.ops.aten.mul.Tensor(add_408, clamp_max_129);  add_408 = clamp_max_129 = None
        div_129 = torch.ops.aten.div.Tensor(mul_501, 6);  mul_501 = None
        convolution_171 = torch.ops.aten.convolution.default(div_129, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_129 = arg206_1 = None
        add_410 = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_124 = torch.ops.aten.sqrt.default(add_410);  add_410 = None
        reciprocal_124 = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_502 = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_992 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_993 = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        unsqueeze_994 = torch.ops.aten.unsqueeze.default(mul_502, -1);  mul_502 = None
        unsqueeze_995 = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        sub_124 = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_993);  convolution_171 = unsqueeze_993 = None
        mul_503 = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_997 = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_504 = torch.ops.aten.mul.Tensor(mul_503, unsqueeze_997);  mul_503 = unsqueeze_997 = None
        unsqueeze_998 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_999 = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_411 = torch.ops.aten.add.Tensor(mul_504, unsqueeze_999);  mul_504 = unsqueeze_999 = None
        add_412 = torch.ops.aten.add.Tensor(add_411, add_403);  add_411 = add_403 = None
        convolution_172 = torch.ops.aten.convolution.default(add_412, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg211_1 = None
        add_413 = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_125 = torch.ops.aten.sqrt.default(add_413);  add_413 = None
        reciprocal_125 = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_505 = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1000 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_1001 = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        unsqueeze_1002 = torch.ops.aten.unsqueeze.default(mul_505, -1);  mul_505 = None
        unsqueeze_1003 = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        sub_125 = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1001);  convolution_172 = unsqueeze_1001 = None
        mul_506 = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1005 = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_507 = torch.ops.aten.mul.Tensor(mul_506, unsqueeze_1005);  mul_506 = unsqueeze_1005 = None
        unsqueeze_1006 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_1007 = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_414 = torch.ops.aten.add.Tensor(mul_507, unsqueeze_1007);  mul_507 = unsqueeze_1007 = None
        add_415 = torch.ops.aten.add.Tensor(add_414, 3)
        clamp_min_130 = torch.ops.aten.clamp_min.default(add_415, 0);  add_415 = None
        clamp_max_130 = torch.ops.aten.clamp_max.default(clamp_min_130, 6);  clamp_min_130 = None
        mul_508 = torch.ops.aten.mul.Tensor(add_414, clamp_max_130);  add_414 = clamp_max_130 = None
        div_130 = torch.ops.aten.div.Tensor(mul_508, 6);  mul_508 = None
        convolution_173 = torch.ops.aten.convolution.default(div_130, arg216_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_130 = arg216_1 = None
        add_416 = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_126 = torch.ops.aten.sqrt.default(add_416);  add_416 = None
        reciprocal_126 = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_509 = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1008 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_1009 = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        unsqueeze_1010 = torch.ops.aten.unsqueeze.default(mul_509, -1);  mul_509 = None
        unsqueeze_1011 = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        sub_126 = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1009);  convolution_173 = unsqueeze_1009 = None
        mul_510 = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1013 = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_511 = torch.ops.aten.mul.Tensor(mul_510, unsqueeze_1013);  mul_510 = unsqueeze_1013 = None
        unsqueeze_1014 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_1015 = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_417 = torch.ops.aten.add.Tensor(mul_511, unsqueeze_1015);  mul_511 = unsqueeze_1015 = None
        add_418 = torch.ops.aten.add.Tensor(add_417, 3)
        clamp_min_131 = torch.ops.aten.clamp_min.default(add_418, 0);  add_418 = None
        clamp_max_131 = torch.ops.aten.clamp_max.default(clamp_min_131, 6);  clamp_min_131 = None
        mul_512 = torch.ops.aten.mul.Tensor(add_417, clamp_max_131);  add_417 = clamp_max_131 = None
        div_131 = torch.ops.aten.div.Tensor(mul_512, 6);  mul_512 = None
        convolution_174 = torch.ops.aten.convolution.default(div_131, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_131 = arg221_1 = None
        add_419 = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_127 = torch.ops.aten.sqrt.default(add_419);  add_419 = None
        reciprocal_127 = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_513 = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1016 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_1017 = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        unsqueeze_1018 = torch.ops.aten.unsqueeze.default(mul_513, -1);  mul_513 = None
        unsqueeze_1019 = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        sub_127 = torch.ops.aten.sub.Tensor(convolution_174, unsqueeze_1017);  convolution_174 = unsqueeze_1017 = None
        mul_514 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1021 = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_515 = torch.ops.aten.mul.Tensor(mul_514, unsqueeze_1021);  mul_514 = unsqueeze_1021 = None
        unsqueeze_1022 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_1023 = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_420 = torch.ops.aten.add.Tensor(mul_515, unsqueeze_1023);  mul_515 = unsqueeze_1023 = None
        add_421 = torch.ops.aten.add.Tensor(add_420, add_412);  add_420 = add_412 = None
        convolution_175 = torch.ops.aten.convolution.default(add_421, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg226_1 = None
        add_422 = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_128 = torch.ops.aten.sqrt.default(add_422);  add_422 = None
        reciprocal_128 = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_516 = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1024 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1025 = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        unsqueeze_1026 = torch.ops.aten.unsqueeze.default(mul_516, -1);  mul_516 = None
        unsqueeze_1027 = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        sub_128 = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1025);  convolution_175 = unsqueeze_1025 = None
        mul_517 = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1029 = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_518 = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_1029);  mul_517 = unsqueeze_1029 = None
        unsqueeze_1030 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1031 = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_423 = torch.ops.aten.add.Tensor(mul_518, unsqueeze_1031);  mul_518 = unsqueeze_1031 = None
        add_424 = torch.ops.aten.add.Tensor(add_423, 3)
        clamp_min_132 = torch.ops.aten.clamp_min.default(add_424, 0);  add_424 = None
        clamp_max_132 = torch.ops.aten.clamp_max.default(clamp_min_132, 6);  clamp_min_132 = None
        mul_519 = torch.ops.aten.mul.Tensor(add_423, clamp_max_132);  add_423 = clamp_max_132 = None
        div_132 = torch.ops.aten.div.Tensor(mul_519, 6);  mul_519 = None
        convolution_176 = torch.ops.aten.convolution.default(div_132, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_132 = arg231_1 = None
        add_425 = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_129 = torch.ops.aten.sqrt.default(add_425);  add_425 = None
        reciprocal_129 = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_520 = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1032 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_1033 = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        unsqueeze_1034 = torch.ops.aten.unsqueeze.default(mul_520, -1);  mul_520 = None
        unsqueeze_1035 = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        sub_129 = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1033);  convolution_176 = unsqueeze_1033 = None
        mul_521 = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1037 = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_522 = torch.ops.aten.mul.Tensor(mul_521, unsqueeze_1037);  mul_521 = unsqueeze_1037 = None
        unsqueeze_1038 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_1039 = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_426 = torch.ops.aten.add.Tensor(mul_522, unsqueeze_1039);  mul_522 = unsqueeze_1039 = None
        add_427 = torch.ops.aten.add.Tensor(add_426, 3)
        clamp_min_133 = torch.ops.aten.clamp_min.default(add_427, 0);  add_427 = None
        clamp_max_133 = torch.ops.aten.clamp_max.default(clamp_min_133, 6);  clamp_min_133 = None
        mul_523 = torch.ops.aten.mul.Tensor(add_426, clamp_max_133);  add_426 = clamp_max_133 = None
        div_133 = torch.ops.aten.div.Tensor(mul_523, 6);  mul_523 = None
        convolution_177 = torch.ops.aten.convolution.default(div_133, arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_133 = arg236_1 = None
        add_428 = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_130 = torch.ops.aten.sqrt.default(add_428);  add_428 = None
        reciprocal_130 = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_524 = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1040 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1041 = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        unsqueeze_1042 = torch.ops.aten.unsqueeze.default(mul_524, -1);  mul_524 = None
        unsqueeze_1043 = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        sub_130 = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1041);  convolution_177 = unsqueeze_1041 = None
        mul_525 = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1045 = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_526 = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_1045);  mul_525 = unsqueeze_1045 = None
        unsqueeze_1046 = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_1047 = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_429 = torch.ops.aten.add.Tensor(mul_526, unsqueeze_1047);  mul_526 = unsqueeze_1047 = None
        add_430 = torch.ops.aten.add.Tensor(add_429, add_421);  add_429 = add_421 = None
        convolution_178 = torch.ops.aten.convolution.default(add_430, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg241_1 = None
        add_431 = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_131 = torch.ops.aten.sqrt.default(add_431);  add_431 = None
        reciprocal_131 = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_527 = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1048 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1049 = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        unsqueeze_1050 = torch.ops.aten.unsqueeze.default(mul_527, -1);  mul_527 = None
        unsqueeze_1051 = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        sub_131 = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1049);  convolution_178 = unsqueeze_1049 = None
        mul_528 = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1053 = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_529 = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_1053);  mul_528 = unsqueeze_1053 = None
        unsqueeze_1054 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1055 = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_432 = torch.ops.aten.add.Tensor(mul_529, unsqueeze_1055);  mul_529 = unsqueeze_1055 = None
        add_433 = torch.ops.aten.add.Tensor(add_432, 3)
        clamp_min_134 = torch.ops.aten.clamp_min.default(add_433, 0);  add_433 = None
        clamp_max_134 = torch.ops.aten.clamp_max.default(clamp_min_134, 6);  clamp_min_134 = None
        mul_530 = torch.ops.aten.mul.Tensor(add_432, clamp_max_134);  add_432 = clamp_max_134 = None
        div_134 = torch.ops.aten.div.Tensor(mul_530, 6);  mul_530 = None
        convolution_179 = torch.ops.aten.convolution.default(div_134, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_134 = arg246_1 = None
        add_434 = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_132 = torch.ops.aten.sqrt.default(add_434);  add_434 = None
        reciprocal_132 = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_531 = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1056 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1057 = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        unsqueeze_1058 = torch.ops.aten.unsqueeze.default(mul_531, -1);  mul_531 = None
        unsqueeze_1059 = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        sub_132 = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1057);  convolution_179 = unsqueeze_1057 = None
        mul_532 = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_1061 = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_533 = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_1061);  mul_532 = unsqueeze_1061 = None
        unsqueeze_1062 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1063 = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_435 = torch.ops.aten.add.Tensor(mul_533, unsqueeze_1063);  mul_533 = unsqueeze_1063 = None
        add_436 = torch.ops.aten.add.Tensor(add_435, 3)
        clamp_min_135 = torch.ops.aten.clamp_min.default(add_436, 0);  add_436 = None
        clamp_max_135 = torch.ops.aten.clamp_max.default(clamp_min_135, 6);  clamp_min_135 = None
        mul_534 = torch.ops.aten.mul.Tensor(add_435, clamp_max_135);  add_435 = clamp_max_135 = None
        div_135 = torch.ops.aten.div.Tensor(mul_534, 6);  mul_534 = None
        convolution_180 = torch.ops.aten.convolution.default(div_135, arg251_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_135 = arg251_1 = None
        add_437 = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_133 = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_133 = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_535 = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1064 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_1065 = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        unsqueeze_1066 = torch.ops.aten.unsqueeze.default(mul_535, -1);  mul_535 = None
        unsqueeze_1067 = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        sub_133 = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1065);  convolution_180 = unsqueeze_1065 = None
        mul_536 = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1069 = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_537 = torch.ops.aten.mul.Tensor(mul_536, unsqueeze_1069);  mul_536 = unsqueeze_1069 = None
        unsqueeze_1070 = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_1071 = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_438 = torch.ops.aten.add.Tensor(mul_537, unsqueeze_1071);  mul_537 = unsqueeze_1071 = None
        add_439 = torch.ops.aten.add.Tensor(add_438, add_430);  add_438 = add_430 = None
        convolution_181 = torch.ops.aten.convolution.default(add_439, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_439 = arg256_1 = None
        add_440 = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_134 = torch.ops.aten.sqrt.default(add_440);  add_440 = None
        reciprocal_134 = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_538 = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1072 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1073 = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        unsqueeze_1074 = torch.ops.aten.unsqueeze.default(mul_538, -1);  mul_538 = None
        unsqueeze_1075 = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        sub_134 = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1073);  convolution_181 = unsqueeze_1073 = None
        mul_539 = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1077 = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_540 = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_1077);  mul_539 = unsqueeze_1077 = None
        unsqueeze_1078 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1079 = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_441 = torch.ops.aten.add.Tensor(mul_540, unsqueeze_1079);  mul_540 = unsqueeze_1079 = None
        add_442 = torch.ops.aten.add.Tensor(add_441, 3)
        clamp_min_136 = torch.ops.aten.clamp_min.default(add_442, 0);  add_442 = None
        clamp_max_136 = torch.ops.aten.clamp_max.default(clamp_min_136, 6);  clamp_min_136 = None
        mul_541 = torch.ops.aten.mul.Tensor(add_441, clamp_max_136);  add_441 = clamp_max_136 = None
        div_136 = torch.ops.aten.div.Tensor(mul_541, 6);  mul_541 = None
        convolution_182 = torch.ops.aten.convolution.default(div_136, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 360);  div_136 = arg261_1 = None
        add_443 = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_135 = torch.ops.aten.sqrt.default(add_443);  add_443 = None
        reciprocal_135 = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_542 = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1080 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1081 = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        unsqueeze_1082 = torch.ops.aten.unsqueeze.default(mul_542, -1);  mul_542 = None
        unsqueeze_1083 = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        sub_135 = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1081);  convolution_182 = unsqueeze_1081 = None
        mul_543 = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1085 = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_544 = torch.ops.aten.mul.Tensor(mul_543, unsqueeze_1085);  mul_543 = unsqueeze_1085 = None
        unsqueeze_1086 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1087 = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_444 = torch.ops.aten.add.Tensor(mul_544, unsqueeze_1087);  mul_544 = unsqueeze_1087 = None
        add_445 = torch.ops.aten.add.Tensor(add_444, 3)
        clamp_min_137 = torch.ops.aten.clamp_min.default(add_445, 0);  add_445 = None
        clamp_max_137 = torch.ops.aten.clamp_max.default(clamp_min_137, 6);  clamp_min_137 = None
        mul_545 = torch.ops.aten.mul.Tensor(add_444, clamp_max_137);  add_444 = clamp_max_137 = None
        div_137 = torch.ops.aten.div.Tensor(mul_545, 6);  mul_545 = None
        mean_24 = torch.ops.aten.mean.dim(div_137, [2, 3], True)
        convolution_183 = torch.ops.aten.convolution.default(mean_24, arg266_1, arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg266_1 = arg267_1 = None
        add_446 = torch.ops.aten.add.Tensor(convolution_183, 3)
        clamp_min_138 = torch.ops.aten.clamp_min.default(add_446, 0);  add_446 = None
        clamp_max_138 = torch.ops.aten.clamp_max.default(clamp_min_138, 6);  clamp_min_138 = None
        mul_546 = torch.ops.aten.mul.Tensor(convolution_183, clamp_max_138);  convolution_183 = clamp_max_138 = None
        div_138 = torch.ops.aten.div.Tensor(mul_546, 6);  mul_546 = None
        convolution_184 = torch.ops.aten.convolution.default(div_138, arg268_1, arg269_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_138 = arg268_1 = arg269_1 = None
        add_447 = torch.ops.aten.add.Tensor(convolution_184, 3);  convolution_184 = None
        clamp_min_139 = torch.ops.aten.clamp_min.default(add_447, 0);  add_447 = None
        clamp_max_139 = torch.ops.aten.clamp_max.default(clamp_min_139, 6);  clamp_min_139 = None
        div_139 = torch.ops.aten.div.Tensor(clamp_max_139, 6);  clamp_max_139 = None
        mul_547 = torch.ops.aten.mul.Tensor(div_137, div_139);  div_137 = div_139 = None
        convolution_185 = torch.ops.aten.convolution.default(mul_547, arg270_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_547 = arg270_1 = None
        add_448 = torch.ops.aten.add.Tensor(arg272_1, 1e-05);  arg272_1 = None
        sqrt_136 = torch.ops.aten.sqrt.default(add_448);  add_448 = None
        reciprocal_136 = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_548 = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1088 = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_1089 = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        unsqueeze_1090 = torch.ops.aten.unsqueeze.default(mul_548, -1);  mul_548 = None
        unsqueeze_1091 = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        sub_136 = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1089);  convolution_185 = unsqueeze_1089 = None
        mul_549 = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092 = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_1093 = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_550 = torch.ops.aten.mul.Tensor(mul_549, unsqueeze_1093);  mul_549 = unsqueeze_1093 = None
        unsqueeze_1094 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1095 = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_449 = torch.ops.aten.add.Tensor(mul_550, unsqueeze_1095);  mul_550 = unsqueeze_1095 = None
        convolution_186 = torch.ops.aten.convolution.default(add_449, arg275_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg275_1 = None
        add_450 = torch.ops.aten.add.Tensor(arg277_1, 1e-05);  arg277_1 = None
        sqrt_137 = torch.ops.aten.sqrt.default(add_450);  add_450 = None
        reciprocal_137 = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_551 = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1096 = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1097 = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        unsqueeze_1098 = torch.ops.aten.unsqueeze.default(mul_551, -1);  mul_551 = None
        unsqueeze_1099 = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        sub_137 = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1097);  convolution_186 = unsqueeze_1097 = None
        mul_552 = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100 = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_1101 = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_553 = torch.ops.aten.mul.Tensor(mul_552, unsqueeze_1101);  mul_552 = unsqueeze_1101 = None
        unsqueeze_1102 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1103 = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_451 = torch.ops.aten.add.Tensor(mul_553, unsqueeze_1103);  mul_553 = unsqueeze_1103 = None
        add_452 = torch.ops.aten.add.Tensor(add_451, 3)
        clamp_min_140 = torch.ops.aten.clamp_min.default(add_452, 0);  add_452 = None
        clamp_max_140 = torch.ops.aten.clamp_max.default(clamp_min_140, 6);  clamp_min_140 = None
        mul_554 = torch.ops.aten.mul.Tensor(add_451, clamp_max_140);  add_451 = clamp_max_140 = None
        div_140 = torch.ops.aten.div.Tensor(mul_554, 6);  mul_554 = None
        convolution_187 = torch.ops.aten.convolution.default(div_140, arg280_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_140 = arg280_1 = None
        add_453 = torch.ops.aten.add.Tensor(arg282_1, 1e-05);  arg282_1 = None
        sqrt_138 = torch.ops.aten.sqrt.default(add_453);  add_453 = None
        reciprocal_138 = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_555 = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1104 = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_1105 = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        unsqueeze_1106 = torch.ops.aten.unsqueeze.default(mul_555, -1);  mul_555 = None
        unsqueeze_1107 = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        sub_138 = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1105);  convolution_187 = unsqueeze_1105 = None
        mul_556 = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108 = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_1109 = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_557 = torch.ops.aten.mul.Tensor(mul_556, unsqueeze_1109);  mul_556 = unsqueeze_1109 = None
        unsqueeze_1110 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1111 = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_454 = torch.ops.aten.add.Tensor(mul_557, unsqueeze_1111);  mul_557 = unsqueeze_1111 = None
        add_455 = torch.ops.aten.add.Tensor(add_454, 3)
        clamp_min_141 = torch.ops.aten.clamp_min.default(add_455, 0);  add_455 = None
        clamp_max_141 = torch.ops.aten.clamp_max.default(clamp_min_141, 6);  clamp_min_141 = None
        mul_558 = torch.ops.aten.mul.Tensor(add_454, clamp_max_141);  add_454 = clamp_max_141 = None
        div_141 = torch.ops.aten.div.Tensor(mul_558, 6);  mul_558 = None
        mean_25 = torch.ops.aten.mean.dim(div_141, [2, 3], True)
        convolution_188 = torch.ops.aten.convolution.default(mean_25, arg285_1, arg286_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg285_1 = arg286_1 = None
        add_456 = torch.ops.aten.add.Tensor(convolution_188, 3)
        clamp_min_142 = torch.ops.aten.clamp_min.default(add_456, 0);  add_456 = None
        clamp_max_142 = torch.ops.aten.clamp_max.default(clamp_min_142, 6);  clamp_min_142 = None
        mul_559 = torch.ops.aten.mul.Tensor(convolution_188, clamp_max_142);  convolution_188 = clamp_max_142 = None
        div_142 = torch.ops.aten.div.Tensor(mul_559, 6);  mul_559 = None
        convolution_189 = torch.ops.aten.convolution.default(div_142, arg287_1, arg288_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_142 = arg287_1 = arg288_1 = None
        add_457 = torch.ops.aten.add.Tensor(convolution_189, 3);  convolution_189 = None
        clamp_min_143 = torch.ops.aten.clamp_min.default(add_457, 0);  add_457 = None
        clamp_max_143 = torch.ops.aten.clamp_max.default(clamp_min_143, 6);  clamp_min_143 = None
        div_143 = torch.ops.aten.div.Tensor(clamp_max_143, 6);  clamp_max_143 = None
        mul_560 = torch.ops.aten.mul.Tensor(div_141, div_143);  div_141 = div_143 = None
        convolution_190 = torch.ops.aten.convolution.default(mul_560, arg289_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_560 = arg289_1 = None
        add_458 = torch.ops.aten.add.Tensor(arg291_1, 1e-05);  arg291_1 = None
        sqrt_139 = torch.ops.aten.sqrt.default(add_458);  add_458 = None
        reciprocal_139 = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_561 = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1112 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1113 = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        unsqueeze_1114 = torch.ops.aten.unsqueeze.default(mul_561, -1);  mul_561 = None
        unsqueeze_1115 = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        sub_139 = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1113);  convolution_190 = unsqueeze_1113 = None
        mul_562 = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116 = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1117 = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_563 = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_1117);  mul_562 = unsqueeze_1117 = None
        unsqueeze_1118 = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1119 = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_459 = torch.ops.aten.add.Tensor(mul_563, unsqueeze_1119);  mul_563 = unsqueeze_1119 = None
        add_460 = torch.ops.aten.add.Tensor(add_459, add_449);  add_459 = add_449 = None
        convolution_191 = torch.ops.aten.convolution.default(add_460, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg294_1 = None
        add_461 = torch.ops.aten.add.Tensor(arg296_1, 1e-05);  arg296_1 = None
        sqrt_140 = torch.ops.aten.sqrt.default(add_461);  add_461 = None
        reciprocal_140 = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_564 = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1120 = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1121 = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        unsqueeze_1122 = torch.ops.aten.unsqueeze.default(mul_564, -1);  mul_564 = None
        unsqueeze_1123 = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        sub_140 = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1121);  convolution_191 = unsqueeze_1121 = None
        mul_565 = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124 = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_1125 = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_566 = torch.ops.aten.mul.Tensor(mul_565, unsqueeze_1125);  mul_565 = unsqueeze_1125 = None
        unsqueeze_1126 = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1127 = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_462 = torch.ops.aten.add.Tensor(mul_566, unsqueeze_1127);  mul_566 = unsqueeze_1127 = None
        add_463 = torch.ops.aten.add.Tensor(add_462, 3)
        clamp_min_144 = torch.ops.aten.clamp_min.default(add_463, 0);  add_463 = None
        clamp_max_144 = torch.ops.aten.clamp_max.default(clamp_min_144, 6);  clamp_min_144 = None
        mul_567 = torch.ops.aten.mul.Tensor(add_462, clamp_max_144);  add_462 = clamp_max_144 = None
        div_144 = torch.ops.aten.div.Tensor(mul_567, 6);  mul_567 = None
        convolution_192 = torch.ops.aten.convolution.default(div_144, arg299_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_144 = arg299_1 = None
        add_464 = torch.ops.aten.add.Tensor(arg301_1, 1e-05);  arg301_1 = None
        sqrt_141 = torch.ops.aten.sqrt.default(add_464);  add_464 = None
        reciprocal_141 = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_568 = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1128 = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1129 = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        unsqueeze_1130 = torch.ops.aten.unsqueeze.default(mul_568, -1);  mul_568 = None
        unsqueeze_1131 = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        sub_141 = torch.ops.aten.sub.Tensor(convolution_192, unsqueeze_1129);  convolution_192 = unsqueeze_1129 = None
        mul_569 = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132 = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_1133 = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_570 = torch.ops.aten.mul.Tensor(mul_569, unsqueeze_1133);  mul_569 = unsqueeze_1133 = None
        unsqueeze_1134 = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1135 = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_465 = torch.ops.aten.add.Tensor(mul_570, unsqueeze_1135);  mul_570 = unsqueeze_1135 = None
        add_466 = torch.ops.aten.add.Tensor(add_465, 3)
        clamp_min_145 = torch.ops.aten.clamp_min.default(add_466, 0);  add_466 = None
        clamp_max_145 = torch.ops.aten.clamp_max.default(clamp_min_145, 6);  clamp_min_145 = None
        mul_571 = torch.ops.aten.mul.Tensor(add_465, clamp_max_145);  add_465 = clamp_max_145 = None
        div_145 = torch.ops.aten.div.Tensor(mul_571, 6);  mul_571 = None
        mean_26 = torch.ops.aten.mean.dim(div_145, [2, 3], True)
        convolution_193 = torch.ops.aten.convolution.default(mean_26, arg304_1, arg305_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg304_1 = arg305_1 = None
        add_467 = torch.ops.aten.add.Tensor(convolution_193, 3)
        clamp_min_146 = torch.ops.aten.clamp_min.default(add_467, 0);  add_467 = None
        clamp_max_146 = torch.ops.aten.clamp_max.default(clamp_min_146, 6);  clamp_min_146 = None
        mul_572 = torch.ops.aten.mul.Tensor(convolution_193, clamp_max_146);  convolution_193 = clamp_max_146 = None
        div_146 = torch.ops.aten.div.Tensor(mul_572, 6);  mul_572 = None
        convolution_194 = torch.ops.aten.convolution.default(div_146, arg306_1, arg307_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_146 = arg306_1 = arg307_1 = None
        add_468 = torch.ops.aten.add.Tensor(convolution_194, 3);  convolution_194 = None
        clamp_min_147 = torch.ops.aten.clamp_min.default(add_468, 0);  add_468 = None
        clamp_max_147 = torch.ops.aten.clamp_max.default(clamp_min_147, 6);  clamp_min_147 = None
        div_147 = torch.ops.aten.div.Tensor(clamp_max_147, 6);  clamp_max_147 = None
        mul_573 = torch.ops.aten.mul.Tensor(div_145, div_147);  div_145 = div_147 = None
        convolution_195 = torch.ops.aten.convolution.default(mul_573, arg308_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_573 = arg308_1 = None
        add_469 = torch.ops.aten.add.Tensor(arg310_1, 1e-05);  arg310_1 = None
        sqrt_142 = torch.ops.aten.sqrt.default(add_469);  add_469 = None
        reciprocal_142 = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_574 = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1136 = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1137 = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        unsqueeze_1138 = torch.ops.aten.unsqueeze.default(mul_574, -1);  mul_574 = None
        unsqueeze_1139 = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        sub_142 = torch.ops.aten.sub.Tensor(convolution_195, unsqueeze_1137);  convolution_195 = unsqueeze_1137 = None
        mul_575 = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140 = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1141 = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_576 = torch.ops.aten.mul.Tensor(mul_575, unsqueeze_1141);  mul_575 = unsqueeze_1141 = None
        unsqueeze_1142 = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_1143 = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_470 = torch.ops.aten.add.Tensor(mul_576, unsqueeze_1143);  mul_576 = unsqueeze_1143 = None
        add_471 = torch.ops.aten.add.Tensor(add_470, add_460);  add_470 = add_460 = None
        convolution_196 = torch.ops.aten.convolution.default(add_471, arg313_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg313_1 = None
        add_472 = torch.ops.aten.add.Tensor(arg315_1, 1e-05);  arg315_1 = None
        sqrt_143 = torch.ops.aten.sqrt.default(add_472);  add_472 = None
        reciprocal_143 = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_577 = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1144 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1145 = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        unsqueeze_1146 = torch.ops.aten.unsqueeze.default(mul_577, -1);  mul_577 = None
        unsqueeze_1147 = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        sub_143 = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1145);  convolution_196 = unsqueeze_1145 = None
        mul_578 = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148 = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
        unsqueeze_1149 = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_579 = torch.ops.aten.mul.Tensor(mul_578, unsqueeze_1149);  mul_578 = unsqueeze_1149 = None
        unsqueeze_1150 = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1151 = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_473 = torch.ops.aten.add.Tensor(mul_579, unsqueeze_1151);  mul_579 = unsqueeze_1151 = None
        add_474 = torch.ops.aten.add.Tensor(add_473, 3)
        clamp_min_148 = torch.ops.aten.clamp_min.default(add_474, 0);  add_474 = None
        clamp_max_148 = torch.ops.aten.clamp_max.default(clamp_min_148, 6);  clamp_min_148 = None
        mul_580 = torch.ops.aten.mul.Tensor(add_473, clamp_max_148);  add_473 = clamp_max_148 = None
        div_148 = torch.ops.aten.div.Tensor(mul_580, 6);  mul_580 = None
        convolution_197 = torch.ops.aten.convolution.default(div_148, arg318_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_148 = arg318_1 = None
        add_475 = torch.ops.aten.add.Tensor(arg320_1, 1e-05);  arg320_1 = None
        sqrt_144 = torch.ops.aten.sqrt.default(add_475);  add_475 = None
        reciprocal_144 = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_581 = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1152 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1153 = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        unsqueeze_1154 = torch.ops.aten.unsqueeze.default(mul_581, -1);  mul_581 = None
        unsqueeze_1155 = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        sub_144 = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1153);  convolution_197 = unsqueeze_1153 = None
        mul_582 = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156 = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_1157 = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_583 = torch.ops.aten.mul.Tensor(mul_582, unsqueeze_1157);  mul_582 = unsqueeze_1157 = None
        unsqueeze_1158 = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1159 = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_476 = torch.ops.aten.add.Tensor(mul_583, unsqueeze_1159);  mul_583 = unsqueeze_1159 = None
        add_477 = torch.ops.aten.add.Tensor(add_476, 3)
        clamp_min_149 = torch.ops.aten.clamp_min.default(add_477, 0);  add_477 = None
        clamp_max_149 = torch.ops.aten.clamp_max.default(clamp_min_149, 6);  clamp_min_149 = None
        mul_584 = torch.ops.aten.mul.Tensor(add_476, clamp_max_149);  add_476 = clamp_max_149 = None
        div_149 = torch.ops.aten.div.Tensor(mul_584, 6);  mul_584 = None
        mean_27 = torch.ops.aten.mean.dim(div_149, [2, 3], True)
        convolution_198 = torch.ops.aten.convolution.default(mean_27, arg323_1, arg324_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_27 = arg323_1 = arg324_1 = None
        add_478 = torch.ops.aten.add.Tensor(convolution_198, 3)
        clamp_min_150 = torch.ops.aten.clamp_min.default(add_478, 0);  add_478 = None
        clamp_max_150 = torch.ops.aten.clamp_max.default(clamp_min_150, 6);  clamp_min_150 = None
        mul_585 = torch.ops.aten.mul.Tensor(convolution_198, clamp_max_150);  convolution_198 = clamp_max_150 = None
        div_150 = torch.ops.aten.div.Tensor(mul_585, 6);  mul_585 = None
        convolution_199 = torch.ops.aten.convolution.default(div_150, arg325_1, arg326_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_150 = arg325_1 = arg326_1 = None
        add_479 = torch.ops.aten.add.Tensor(convolution_199, 3);  convolution_199 = None
        clamp_min_151 = torch.ops.aten.clamp_min.default(add_479, 0);  add_479 = None
        clamp_max_151 = torch.ops.aten.clamp_max.default(clamp_min_151, 6);  clamp_min_151 = None
        div_151 = torch.ops.aten.div.Tensor(clamp_max_151, 6);  clamp_max_151 = None
        mul_586 = torch.ops.aten.mul.Tensor(div_149, div_151);  div_149 = div_151 = None
        convolution_200 = torch.ops.aten.convolution.default(mul_586, arg327_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_586 = arg327_1 = None
        add_480 = torch.ops.aten.add.Tensor(arg329_1, 1e-05);  arg329_1 = None
        sqrt_145 = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_145 = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_587 = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1160 = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_1161 = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        unsqueeze_1162 = torch.ops.aten.unsqueeze.default(mul_587, -1);  mul_587 = None
        unsqueeze_1163 = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        sub_145 = torch.ops.aten.sub.Tensor(convolution_200, unsqueeze_1161);  convolution_200 = unsqueeze_1161 = None
        mul_588 = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164 = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1165 = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_589 = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_1165);  mul_588 = unsqueeze_1165 = None
        unsqueeze_1166 = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_1167 = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_481 = torch.ops.aten.add.Tensor(mul_589, unsqueeze_1167);  mul_589 = unsqueeze_1167 = None
        add_482 = torch.ops.aten.add.Tensor(add_481, add_471);  add_481 = add_471 = None
        convolution_201 = torch.ops.aten.convolution.default(add_482, arg332_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg332_1 = None
        add_483 = torch.ops.aten.add.Tensor(arg334_1, 1e-05);  arg334_1 = None
        sqrt_146 = torch.ops.aten.sqrt.default(add_483);  add_483 = None
        reciprocal_146 = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_590 = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1168 = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_1169 = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        unsqueeze_1170 = torch.ops.aten.unsqueeze.default(mul_590, -1);  mul_590 = None
        unsqueeze_1171 = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        sub_146 = torch.ops.aten.sub.Tensor(convolution_201, unsqueeze_1169);  convolution_201 = unsqueeze_1169 = None
        mul_591 = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172 = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1173 = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_592 = torch.ops.aten.mul.Tensor(mul_591, unsqueeze_1173);  mul_591 = unsqueeze_1173 = None
        unsqueeze_1174 = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_1175 = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_484 = torch.ops.aten.add.Tensor(mul_592, unsqueeze_1175);  mul_592 = unsqueeze_1175 = None
        add_485 = torch.ops.aten.add.Tensor(add_484, 3)
        clamp_min_152 = torch.ops.aten.clamp_min.default(add_485, 0);  add_485 = None
        clamp_max_152 = torch.ops.aten.clamp_max.default(clamp_min_152, 6);  clamp_min_152 = None
        mul_593 = torch.ops.aten.mul.Tensor(add_484, clamp_max_152);  add_484 = clamp_max_152 = None
        div_152 = torch.ops.aten.div.Tensor(mul_593, 6);  mul_593 = None
        convolution_202 = torch.ops.aten.convolution.default(div_152, arg337_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_152 = arg337_1 = None
        add_486 = torch.ops.aten.add.Tensor(arg339_1, 1e-05);  arg339_1 = None
        sqrt_147 = torch.ops.aten.sqrt.default(add_486);  add_486 = None
        reciprocal_147 = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_594 = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1176 = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
        unsqueeze_1177 = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        unsqueeze_1178 = torch.ops.aten.unsqueeze.default(mul_594, -1);  mul_594 = None
        unsqueeze_1179 = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        sub_147 = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1177);  convolution_202 = unsqueeze_1177 = None
        mul_595 = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180 = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_1181 = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_596 = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_1181);  mul_595 = unsqueeze_1181 = None
        unsqueeze_1182 = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_1183 = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_487 = torch.ops.aten.add.Tensor(mul_596, unsqueeze_1183);  mul_596 = unsqueeze_1183 = None
        add_488 = torch.ops.aten.add.Tensor(add_487, 3)
        clamp_min_153 = torch.ops.aten.clamp_min.default(add_488, 0);  add_488 = None
        clamp_max_153 = torch.ops.aten.clamp_max.default(clamp_min_153, 6);  clamp_min_153 = None
        mul_597 = torch.ops.aten.mul.Tensor(add_487, clamp_max_153);  add_487 = clamp_max_153 = None
        div_153 = torch.ops.aten.div.Tensor(mul_597, 6);  mul_597 = None
        mean_28 = torch.ops.aten.mean.dim(div_153, [2, 3], True)
        convolution_203 = torch.ops.aten.convolution.default(mean_28, arg342_1, arg343_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_28 = arg342_1 = arg343_1 = None
        add_489 = torch.ops.aten.add.Tensor(convolution_203, 3)
        clamp_min_154 = torch.ops.aten.clamp_min.default(add_489, 0);  add_489 = None
        clamp_max_154 = torch.ops.aten.clamp_max.default(clamp_min_154, 6);  clamp_min_154 = None
        mul_598 = torch.ops.aten.mul.Tensor(convolution_203, clamp_max_154);  convolution_203 = clamp_max_154 = None
        div_154 = torch.ops.aten.div.Tensor(mul_598, 6);  mul_598 = None
        convolution_204 = torch.ops.aten.convolution.default(div_154, arg344_1, arg345_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_154 = arg344_1 = arg345_1 = None
        add_490 = torch.ops.aten.add.Tensor(convolution_204, 3);  convolution_204 = None
        clamp_min_155 = torch.ops.aten.clamp_min.default(add_490, 0);  add_490 = None
        clamp_max_155 = torch.ops.aten.clamp_max.default(clamp_min_155, 6);  clamp_min_155 = None
        div_155 = torch.ops.aten.div.Tensor(clamp_max_155, 6);  clamp_max_155 = None
        mul_599 = torch.ops.aten.mul.Tensor(div_153, div_155);  div_153 = div_155 = None
        convolution_205 = torch.ops.aten.convolution.default(mul_599, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_599 = arg346_1 = None
        add_491 = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_148 = torch.ops.aten.sqrt.default(add_491);  add_491 = None
        reciprocal_148 = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_600 = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1184 = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_1185 = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        unsqueeze_1186 = torch.ops.aten.unsqueeze.default(mul_600, -1);  mul_600 = None
        unsqueeze_1187 = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        sub_148 = torch.ops.aten.sub.Tensor(convolution_205, unsqueeze_1185);  convolution_205 = unsqueeze_1185 = None
        mul_601 = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1189 = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_602 = torch.ops.aten.mul.Tensor(mul_601, unsqueeze_1189);  mul_601 = unsqueeze_1189 = None
        unsqueeze_1190 = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_1191 = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_492 = torch.ops.aten.add.Tensor(mul_602, unsqueeze_1191);  mul_602 = unsqueeze_1191 = None
        add_493 = torch.ops.aten.add.Tensor(add_492, add_482);  add_492 = add_482 = None
        convolution_206 = torch.ops.aten.convolution.default(add_493, arg351_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg351_1 = None
        add_494 = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_149 = torch.ops.aten.sqrt.default(add_494);  add_494 = None
        reciprocal_149 = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_603 = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192 = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1193 = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194 = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
        unsqueeze_1195 = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_149 = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1193);  convolution_206 = unsqueeze_1193 = None
        mul_604 = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196 = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1197 = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_605 = torch.ops.aten.mul.Tensor(mul_604, unsqueeze_1197);  mul_604 = unsqueeze_1197 = None
        unsqueeze_1198 = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_1199 = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_495 = torch.ops.aten.add.Tensor(mul_605, unsqueeze_1199);  mul_605 = unsqueeze_1199 = None
        add_496 = torch.ops.aten.add.Tensor(add_495, 3)
        clamp_min_156 = torch.ops.aten.clamp_min.default(add_496, 0);  add_496 = None
        clamp_max_156 = torch.ops.aten.clamp_max.default(clamp_min_156, 6);  clamp_min_156 = None
        mul_606 = torch.ops.aten.mul.Tensor(add_495, clamp_max_156);  add_495 = clamp_max_156 = None
        div_156 = torch.ops.aten.div.Tensor(mul_606, 6);  mul_606 = None
        convolution_207 = torch.ops.aten.convolution.default(div_156, arg356_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_156 = arg356_1 = None
        add_497 = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_150 = torch.ops.aten.sqrt.default(add_497);  add_497 = None
        reciprocal_150 = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_607 = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200 = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1201 = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202 = torch.ops.aten.unsqueeze.default(mul_607, -1);  mul_607 = None
        unsqueeze_1203 = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_150 = torch.ops.aten.sub.Tensor(convolution_207, unsqueeze_1201);  convolution_207 = unsqueeze_1201 = None
        mul_608 = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204 = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1205 = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_609 = torch.ops.aten.mul.Tensor(mul_608, unsqueeze_1205);  mul_608 = unsqueeze_1205 = None
        unsqueeze_1206 = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_1207 = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_498 = torch.ops.aten.add.Tensor(mul_609, unsqueeze_1207);  mul_609 = unsqueeze_1207 = None
        add_499 = torch.ops.aten.add.Tensor(add_498, 3)
        clamp_min_157 = torch.ops.aten.clamp_min.default(add_499, 0);  add_499 = None
        clamp_max_157 = torch.ops.aten.clamp_max.default(clamp_min_157, 6);  clamp_min_157 = None
        mul_610 = torch.ops.aten.mul.Tensor(add_498, clamp_max_157);  add_498 = clamp_max_157 = None
        div_157 = torch.ops.aten.div.Tensor(mul_610, 6);  mul_610 = None
        mean_29 = torch.ops.aten.mean.dim(div_157, [2, 3], True)
        convolution_208 = torch.ops.aten.convolution.default(mean_29, arg361_1, arg362_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_29 = arg361_1 = arg362_1 = None
        add_500 = torch.ops.aten.add.Tensor(convolution_208, 3)
        clamp_min_158 = torch.ops.aten.clamp_min.default(add_500, 0);  add_500 = None
        clamp_max_158 = torch.ops.aten.clamp_max.default(clamp_min_158, 6);  clamp_min_158 = None
        mul_611 = torch.ops.aten.mul.Tensor(convolution_208, clamp_max_158);  convolution_208 = clamp_max_158 = None
        div_158 = torch.ops.aten.div.Tensor(mul_611, 6);  mul_611 = None
        convolution_209 = torch.ops.aten.convolution.default(div_158, arg363_1, arg364_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_158 = arg363_1 = arg364_1 = None
        add_501 = torch.ops.aten.add.Tensor(convolution_209, 3);  convolution_209 = None
        clamp_min_159 = torch.ops.aten.clamp_min.default(add_501, 0);  add_501 = None
        clamp_max_159 = torch.ops.aten.clamp_max.default(clamp_min_159, 6);  clamp_min_159 = None
        div_159 = torch.ops.aten.div.Tensor(clamp_max_159, 6);  clamp_max_159 = None
        mul_612 = torch.ops.aten.mul.Tensor(div_157, div_159);  div_157 = div_159 = None
        convolution_210 = torch.ops.aten.convolution.default(mul_612, arg365_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_612 = arg365_1 = None
        add_502 = torch.ops.aten.add.Tensor(arg367_1, 1e-05);  arg367_1 = None
        sqrt_151 = torch.ops.aten.sqrt.default(add_502);  add_502 = None
        reciprocal_151 = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_613 = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208 = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
        unsqueeze_1209 = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210 = torch.ops.aten.unsqueeze.default(mul_613, -1);  mul_613 = None
        unsqueeze_1211 = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_151 = torch.ops.aten.sub.Tensor(convolution_210, unsqueeze_1209);  convolution_210 = unsqueeze_1209 = None
        mul_614 = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212 = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_1213 = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_615 = torch.ops.aten.mul.Tensor(mul_614, unsqueeze_1213);  mul_614 = unsqueeze_1213 = None
        unsqueeze_1214 = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_1215 = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_503 = torch.ops.aten.add.Tensor(mul_615, unsqueeze_1215);  mul_615 = unsqueeze_1215 = None
        add_504 = torch.ops.aten.add.Tensor(add_503, add_493);  add_503 = add_493 = None
        convolution_211 = torch.ops.aten.convolution.default(add_504, arg370_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_504 = arg370_1 = None
        add_505 = torch.ops.aten.add.Tensor(arg372_1, 1e-05);  arg372_1 = None
        sqrt_152 = torch.ops.aten.sqrt.default(add_505);  add_505 = None
        reciprocal_152 = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_616 = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216 = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
        unsqueeze_1217 = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218 = torch.ops.aten.unsqueeze.default(mul_616, -1);  mul_616 = None
        unsqueeze_1219 = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_152 = torch.ops.aten.sub.Tensor(convolution_211, unsqueeze_1217);  convolution_211 = unsqueeze_1217 = None
        mul_617 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220 = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_1221 = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_618 = torch.ops.aten.mul.Tensor(mul_617, unsqueeze_1221);  mul_617 = unsqueeze_1221 = None
        unsqueeze_1222 = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1223 = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_506 = torch.ops.aten.add.Tensor(mul_618, unsqueeze_1223);  mul_618 = unsqueeze_1223 = None
        add_507 = torch.ops.aten.add.Tensor(add_506, 3)
        clamp_min_160 = torch.ops.aten.clamp_min.default(add_507, 0);  add_507 = None
        clamp_max_160 = torch.ops.aten.clamp_max.default(clamp_min_160, 6);  clamp_min_160 = None
        mul_619 = torch.ops.aten.mul.Tensor(add_506, clamp_max_160);  add_506 = clamp_max_160 = None
        div_160 = torch.ops.aten.div.Tensor(mul_619, 6);  mul_619 = None
        convolution_212 = torch.ops.aten.convolution.default(div_160, arg375_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 720);  div_160 = arg375_1 = None
        add_508 = torch.ops.aten.add.Tensor(arg377_1, 1e-05);  arg377_1 = None
        sqrt_153 = torch.ops.aten.sqrt.default(add_508);  add_508 = None
        reciprocal_153 = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_620 = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224 = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
        unsqueeze_1225 = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226 = torch.ops.aten.unsqueeze.default(mul_620, -1);  mul_620 = None
        unsqueeze_1227 = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_153 = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1225);  convolution_212 = unsqueeze_1225 = None
        mul_621 = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228 = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_1229 = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_622 = torch.ops.aten.mul.Tensor(mul_621, unsqueeze_1229);  mul_621 = unsqueeze_1229 = None
        unsqueeze_1230 = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1231 = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_509 = torch.ops.aten.add.Tensor(mul_622, unsqueeze_1231);  mul_622 = unsqueeze_1231 = None
        add_510 = torch.ops.aten.add.Tensor(add_509, 3)
        clamp_min_161 = torch.ops.aten.clamp_min.default(add_510, 0);  add_510 = None
        clamp_max_161 = torch.ops.aten.clamp_max.default(clamp_min_161, 6);  clamp_min_161 = None
        mul_623 = torch.ops.aten.mul.Tensor(add_509, clamp_max_161);  add_509 = clamp_max_161 = None
        div_161 = torch.ops.aten.div.Tensor(mul_623, 6);  mul_623 = None
        mean_30 = torch.ops.aten.mean.dim(div_161, [2, 3], True)
        convolution_213 = torch.ops.aten.convolution.default(mean_30, arg380_1, arg381_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_30 = arg380_1 = arg381_1 = None
        add_511 = torch.ops.aten.add.Tensor(convolution_213, 3)
        clamp_min_162 = torch.ops.aten.clamp_min.default(add_511, 0);  add_511 = None
        clamp_max_162 = torch.ops.aten.clamp_max.default(clamp_min_162, 6);  clamp_min_162 = None
        mul_624 = torch.ops.aten.mul.Tensor(convolution_213, clamp_max_162);  convolution_213 = clamp_max_162 = None
        div_162 = torch.ops.aten.div.Tensor(mul_624, 6);  mul_624 = None
        convolution_214 = torch.ops.aten.convolution.default(div_162, arg382_1, arg383_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_162 = arg382_1 = arg383_1 = None
        add_512 = torch.ops.aten.add.Tensor(convolution_214, 3);  convolution_214 = None
        clamp_min_163 = torch.ops.aten.clamp_min.default(add_512, 0);  add_512 = None
        clamp_max_163 = torch.ops.aten.clamp_max.default(clamp_min_163, 6);  clamp_min_163 = None
        div_163 = torch.ops.aten.div.Tensor(clamp_max_163, 6);  clamp_max_163 = None
        mul_625 = torch.ops.aten.mul.Tensor(div_161, div_163);  div_161 = div_163 = None
        convolution_215 = torch.ops.aten.convolution.default(mul_625, arg384_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_625 = arg384_1 = None
        add_513 = torch.ops.aten.add.Tensor(arg386_1, 1e-05);  arg386_1 = None
        sqrt_154 = torch.ops.aten.sqrt.default(add_513);  add_513 = None
        reciprocal_154 = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_626 = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232 = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_1233 = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234 = torch.ops.aten.unsqueeze.default(mul_626, -1);  mul_626 = None
        unsqueeze_1235 = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_154 = torch.ops.aten.sub.Tensor(convolution_215, unsqueeze_1233);  convolution_215 = unsqueeze_1233 = None
        mul_627 = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236 = torch.ops.aten.unsqueeze.default(arg387_1, -1);  arg387_1 = None
        unsqueeze_1237 = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_628 = torch.ops.aten.mul.Tensor(mul_627, unsqueeze_1237);  mul_627 = unsqueeze_1237 = None
        unsqueeze_1238 = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1239 = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_514 = torch.ops.aten.add.Tensor(mul_628, unsqueeze_1239);  mul_628 = unsqueeze_1239 = None
        convolution_216 = torch.ops.aten.convolution.default(add_514, arg389_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg389_1 = None
        add_515 = torch.ops.aten.add.Tensor(arg391_1, 1e-05);  arg391_1 = None
        sqrt_155 = torch.ops.aten.sqrt.default(add_515);  add_515 = None
        reciprocal_155 = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_629 = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240 = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_1241 = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242 = torch.ops.aten.unsqueeze.default(mul_629, -1);  mul_629 = None
        unsqueeze_1243 = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_155 = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1241);  convolution_216 = unsqueeze_1241 = None
        mul_630 = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244 = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
        unsqueeze_1245 = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_631 = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_1245);  mul_630 = unsqueeze_1245 = None
        unsqueeze_1246 = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_1247 = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_516 = torch.ops.aten.add.Tensor(mul_631, unsqueeze_1247);  mul_631 = unsqueeze_1247 = None
        add_517 = torch.ops.aten.add.Tensor(add_516, 3)
        clamp_min_164 = torch.ops.aten.clamp_min.default(add_517, 0);  add_517 = None
        clamp_max_164 = torch.ops.aten.clamp_max.default(clamp_min_164, 6);  clamp_min_164 = None
        mul_632 = torch.ops.aten.mul.Tensor(add_516, clamp_max_164);  add_516 = clamp_max_164 = None
        div_164 = torch.ops.aten.div.Tensor(mul_632, 6);  mul_632 = None
        convolution_217 = torch.ops.aten.convolution.default(div_164, arg394_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_164 = arg394_1 = None
        add_518 = torch.ops.aten.add.Tensor(arg396_1, 1e-05);  arg396_1 = None
        sqrt_156 = torch.ops.aten.sqrt.default(add_518);  add_518 = None
        reciprocal_156 = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_633 = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248 = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_1249 = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250 = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_1251 = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_156 = torch.ops.aten.sub.Tensor(convolution_217, unsqueeze_1249);  convolution_217 = unsqueeze_1249 = None
        mul_634 = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252 = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_1253 = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_635 = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_1253);  mul_634 = unsqueeze_1253 = None
        unsqueeze_1254 = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1255 = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_519 = torch.ops.aten.add.Tensor(mul_635, unsqueeze_1255);  mul_635 = unsqueeze_1255 = None
        add_520 = torch.ops.aten.add.Tensor(add_519, 3)
        clamp_min_165 = torch.ops.aten.clamp_min.default(add_520, 0);  add_520 = None
        clamp_max_165 = torch.ops.aten.clamp_max.default(clamp_min_165, 6);  clamp_min_165 = None
        mul_636 = torch.ops.aten.mul.Tensor(add_519, clamp_max_165);  add_519 = clamp_max_165 = None
        div_165 = torch.ops.aten.div.Tensor(mul_636, 6);  mul_636 = None
        mean_31 = torch.ops.aten.mean.dim(div_165, [2, 3], True)
        convolution_218 = torch.ops.aten.convolution.default(mean_31, arg399_1, arg400_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_31 = arg399_1 = arg400_1 = None
        add_521 = torch.ops.aten.add.Tensor(convolution_218, 3)
        clamp_min_166 = torch.ops.aten.clamp_min.default(add_521, 0);  add_521 = None
        clamp_max_166 = torch.ops.aten.clamp_max.default(clamp_min_166, 6);  clamp_min_166 = None
        mul_637 = torch.ops.aten.mul.Tensor(convolution_218, clamp_max_166);  convolution_218 = clamp_max_166 = None
        div_166 = torch.ops.aten.div.Tensor(mul_637, 6);  mul_637 = None
        convolution_219 = torch.ops.aten.convolution.default(div_166, arg401_1, arg402_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_166 = arg401_1 = arg402_1 = None
        add_522 = torch.ops.aten.add.Tensor(convolution_219, 3);  convolution_219 = None
        clamp_min_167 = torch.ops.aten.clamp_min.default(add_522, 0);  add_522 = None
        clamp_max_167 = torch.ops.aten.clamp_max.default(clamp_min_167, 6);  clamp_min_167 = None
        div_167 = torch.ops.aten.div.Tensor(clamp_max_167, 6);  clamp_max_167 = None
        mul_638 = torch.ops.aten.mul.Tensor(div_165, div_167);  div_165 = div_167 = None
        convolution_220 = torch.ops.aten.convolution.default(mul_638, arg403_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_638 = arg403_1 = None
        add_523 = torch.ops.aten.add.Tensor(arg405_1, 1e-05);  arg405_1 = None
        sqrt_157 = torch.ops.aten.sqrt.default(add_523);  add_523 = None
        reciprocal_157 = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_639 = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256 = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1257 = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258 = torch.ops.aten.unsqueeze.default(mul_639, -1);  mul_639 = None
        unsqueeze_1259 = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_157 = torch.ops.aten.sub.Tensor(convolution_220, unsqueeze_1257);  convolution_220 = unsqueeze_1257 = None
        mul_640 = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260 = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
        unsqueeze_1261 = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_641 = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_1261);  mul_640 = unsqueeze_1261 = None
        unsqueeze_1262 = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_1263 = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_524 = torch.ops.aten.add.Tensor(mul_641, unsqueeze_1263);  mul_641 = unsqueeze_1263 = None
        add_525 = torch.ops.aten.add.Tensor(add_524, add_514);  add_524 = add_514 = None
        convolution_221 = torch.ops.aten.convolution.default(add_525, arg408_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg408_1 = None
        add_526 = torch.ops.aten.add.Tensor(arg410_1, 1e-05);  arg410_1 = None
        sqrt_158 = torch.ops.aten.sqrt.default(add_526);  add_526 = None
        reciprocal_158 = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_642 = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264 = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1265 = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266 = torch.ops.aten.unsqueeze.default(mul_642, -1);  mul_642 = None
        unsqueeze_1267 = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_158 = torch.ops.aten.sub.Tensor(convolution_221, unsqueeze_1265);  convolution_221 = unsqueeze_1265 = None
        mul_643 = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268 = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_1269 = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_644 = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_1269);  mul_643 = unsqueeze_1269 = None
        unsqueeze_1270 = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
        unsqueeze_1271 = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_527 = torch.ops.aten.add.Tensor(mul_644, unsqueeze_1271);  mul_644 = unsqueeze_1271 = None
        add_528 = torch.ops.aten.add.Tensor(add_527, 3)
        clamp_min_168 = torch.ops.aten.clamp_min.default(add_528, 0);  add_528 = None
        clamp_max_168 = torch.ops.aten.clamp_max.default(clamp_min_168, 6);  clamp_min_168 = None
        mul_645 = torch.ops.aten.mul.Tensor(add_527, clamp_max_168);  add_527 = clamp_max_168 = None
        div_168 = torch.ops.aten.div.Tensor(mul_645, 6);  mul_645 = None
        convolution_222 = torch.ops.aten.convolution.default(div_168, arg413_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_168 = arg413_1 = None
        add_529 = torch.ops.aten.add.Tensor(arg415_1, 1e-05);  arg415_1 = None
        sqrt_159 = torch.ops.aten.sqrt.default(add_529);  add_529 = None
        reciprocal_159 = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_646 = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272 = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1273 = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274 = torch.ops.aten.unsqueeze.default(mul_646, -1);  mul_646 = None
        unsqueeze_1275 = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_159 = torch.ops.aten.sub.Tensor(convolution_222, unsqueeze_1273);  convolution_222 = unsqueeze_1273 = None
        mul_647 = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276 = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
        unsqueeze_1277 = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_648 = torch.ops.aten.mul.Tensor(mul_647, unsqueeze_1277);  mul_647 = unsqueeze_1277 = None
        unsqueeze_1278 = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_1279 = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_530 = torch.ops.aten.add.Tensor(mul_648, unsqueeze_1279);  mul_648 = unsqueeze_1279 = None
        add_531 = torch.ops.aten.add.Tensor(add_530, 3)
        clamp_min_169 = torch.ops.aten.clamp_min.default(add_531, 0);  add_531 = None
        clamp_max_169 = torch.ops.aten.clamp_max.default(clamp_min_169, 6);  clamp_min_169 = None
        mul_649 = torch.ops.aten.mul.Tensor(add_530, clamp_max_169);  add_530 = clamp_max_169 = None
        div_169 = torch.ops.aten.div.Tensor(mul_649, 6);  mul_649 = None
        mean_32 = torch.ops.aten.mean.dim(div_169, [2, 3], True)
        convolution_223 = torch.ops.aten.convolution.default(mean_32, arg418_1, arg419_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_32 = arg418_1 = arg419_1 = None
        add_532 = torch.ops.aten.add.Tensor(convolution_223, 3)
        clamp_min_170 = torch.ops.aten.clamp_min.default(add_532, 0);  add_532 = None
        clamp_max_170 = torch.ops.aten.clamp_max.default(clamp_min_170, 6);  clamp_min_170 = None
        mul_650 = torch.ops.aten.mul.Tensor(convolution_223, clamp_max_170);  convolution_223 = clamp_max_170 = None
        div_170 = torch.ops.aten.div.Tensor(mul_650, 6);  mul_650 = None
        convolution_224 = torch.ops.aten.convolution.default(div_170, arg420_1, arg421_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_170 = arg420_1 = arg421_1 = None
        add_533 = torch.ops.aten.add.Tensor(convolution_224, 3);  convolution_224 = None
        clamp_min_171 = torch.ops.aten.clamp_min.default(add_533, 0);  add_533 = None
        clamp_max_171 = torch.ops.aten.clamp_max.default(clamp_min_171, 6);  clamp_min_171 = None
        div_171 = torch.ops.aten.div.Tensor(clamp_max_171, 6);  clamp_max_171 = None
        mul_651 = torch.ops.aten.mul.Tensor(div_169, div_171);  div_169 = div_171 = None
        convolution_225 = torch.ops.aten.convolution.default(mul_651, arg422_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_651 = arg422_1 = None
        add_534 = torch.ops.aten.add.Tensor(arg424_1, 1e-05);  arg424_1 = None
        sqrt_160 = torch.ops.aten.sqrt.default(add_534);  add_534 = None
        reciprocal_160 = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_652 = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1280 = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1281 = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        unsqueeze_1282 = torch.ops.aten.unsqueeze.default(mul_652, -1);  mul_652 = None
        unsqueeze_1283 = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        sub_160 = torch.ops.aten.sub.Tensor(convolution_225, unsqueeze_1281);  convolution_225 = unsqueeze_1281 = None
        mul_653 = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_1283);  sub_160 = unsqueeze_1283 = None
        unsqueeze_1284 = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1285 = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_654 = torch.ops.aten.mul.Tensor(mul_653, unsqueeze_1285);  mul_653 = unsqueeze_1285 = None
        unsqueeze_1286 = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
        unsqueeze_1287 = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_535 = torch.ops.aten.add.Tensor(mul_654, unsqueeze_1287);  mul_654 = unsqueeze_1287 = None
        add_536 = torch.ops.aten.add.Tensor(add_535, add_525);  add_535 = add_525 = None
        convolution_226 = torch.ops.aten.convolution.default(add_536, arg427_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg427_1 = None
        add_537 = torch.ops.aten.add.Tensor(arg429_1, 1e-05);  arg429_1 = None
        sqrt_161 = torch.ops.aten.sqrt.default(add_537);  add_537 = None
        reciprocal_161 = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_655 = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1288 = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1289 = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        unsqueeze_1290 = torch.ops.aten.unsqueeze.default(mul_655, -1);  mul_655 = None
        unsqueeze_1291 = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        sub_161 = torch.ops.aten.sub.Tensor(convolution_226, unsqueeze_1289);  convolution_226 = unsqueeze_1289 = None
        mul_656 = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_1291);  sub_161 = unsqueeze_1291 = None
        unsqueeze_1292 = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_1293 = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_657 = torch.ops.aten.mul.Tensor(mul_656, unsqueeze_1293);  mul_656 = unsqueeze_1293 = None
        unsqueeze_1294 = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_1295 = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_538 = torch.ops.aten.add.Tensor(mul_657, unsqueeze_1295);  mul_657 = unsqueeze_1295 = None
        add_539 = torch.ops.aten.add.Tensor(add_538, 3)
        clamp_min_172 = torch.ops.aten.clamp_min.default(add_539, 0);  add_539 = None
        clamp_max_172 = torch.ops.aten.clamp_max.default(clamp_min_172, 6);  clamp_min_172 = None
        mul_658 = torch.ops.aten.mul.Tensor(add_538, clamp_max_172);  add_538 = clamp_max_172 = None
        div_172 = torch.ops.aten.div.Tensor(mul_658, 6);  mul_658 = None
        convolution_227 = torch.ops.aten.convolution.default(div_172, arg432_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_172 = arg432_1 = None
        add_540 = torch.ops.aten.add.Tensor(arg434_1, 1e-05);  arg434_1 = None
        sqrt_162 = torch.ops.aten.sqrt.default(add_540);  add_540 = None
        reciprocal_162 = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_659 = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1296 = torch.ops.aten.unsqueeze.default(arg433_1, -1);  arg433_1 = None
        unsqueeze_1297 = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        unsqueeze_1298 = torch.ops.aten.unsqueeze.default(mul_659, -1);  mul_659 = None
        unsqueeze_1299 = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        sub_162 = torch.ops.aten.sub.Tensor(convolution_227, unsqueeze_1297);  convolution_227 = unsqueeze_1297 = None
        mul_660 = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_1299);  sub_162 = unsqueeze_1299 = None
        unsqueeze_1300 = torch.ops.aten.unsqueeze.default(arg435_1, -1);  arg435_1 = None
        unsqueeze_1301 = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_661 = torch.ops.aten.mul.Tensor(mul_660, unsqueeze_1301);  mul_660 = unsqueeze_1301 = None
        unsqueeze_1302 = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
        unsqueeze_1303 = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_541 = torch.ops.aten.add.Tensor(mul_661, unsqueeze_1303);  mul_661 = unsqueeze_1303 = None
        add_542 = torch.ops.aten.add.Tensor(add_541, 3)
        clamp_min_173 = torch.ops.aten.clamp_min.default(add_542, 0);  add_542 = None
        clamp_max_173 = torch.ops.aten.clamp_max.default(clamp_min_173, 6);  clamp_min_173 = None
        mul_662 = torch.ops.aten.mul.Tensor(add_541, clamp_max_173);  add_541 = clamp_max_173 = None
        div_173 = torch.ops.aten.div.Tensor(mul_662, 6);  mul_662 = None
        mean_33 = torch.ops.aten.mean.dim(div_173, [2, 3], True)
        convolution_228 = torch.ops.aten.convolution.default(mean_33, arg437_1, arg438_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_33 = arg437_1 = arg438_1 = None
        add_543 = torch.ops.aten.add.Tensor(convolution_228, 3)
        clamp_min_174 = torch.ops.aten.clamp_min.default(add_543, 0);  add_543 = None
        clamp_max_174 = torch.ops.aten.clamp_max.default(clamp_min_174, 6);  clamp_min_174 = None
        mul_663 = torch.ops.aten.mul.Tensor(convolution_228, clamp_max_174);  convolution_228 = clamp_max_174 = None
        div_174 = torch.ops.aten.div.Tensor(mul_663, 6);  mul_663 = None
        convolution_229 = torch.ops.aten.convolution.default(div_174, arg439_1, arg440_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_174 = arg439_1 = arg440_1 = None
        add_544 = torch.ops.aten.add.Tensor(convolution_229, 3);  convolution_229 = None
        clamp_min_175 = torch.ops.aten.clamp_min.default(add_544, 0);  add_544 = None
        clamp_max_175 = torch.ops.aten.clamp_max.default(clamp_min_175, 6);  clamp_min_175 = None
        div_175 = torch.ops.aten.div.Tensor(clamp_max_175, 6);  clamp_max_175 = None
        mul_664 = torch.ops.aten.mul.Tensor(div_173, div_175);  div_173 = div_175 = None
        convolution_230 = torch.ops.aten.convolution.default(mul_664, arg441_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_664 = arg441_1 = None
        add_545 = torch.ops.aten.add.Tensor(arg443_1, 1e-05);  arg443_1 = None
        sqrt_163 = torch.ops.aten.sqrt.default(add_545);  add_545 = None
        reciprocal_163 = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_665 = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1304 = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
        unsqueeze_1305 = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        unsqueeze_1306 = torch.ops.aten.unsqueeze.default(mul_665, -1);  mul_665 = None
        unsqueeze_1307 = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        sub_163 = torch.ops.aten.sub.Tensor(convolution_230, unsqueeze_1305);  convolution_230 = unsqueeze_1305 = None
        mul_666 = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_1307);  sub_163 = unsqueeze_1307 = None
        unsqueeze_1308 = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_1309 = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_667 = torch.ops.aten.mul.Tensor(mul_666, unsqueeze_1309);  mul_666 = unsqueeze_1309 = None
        unsqueeze_1310 = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_1311 = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_546 = torch.ops.aten.add.Tensor(mul_667, unsqueeze_1311);  mul_667 = unsqueeze_1311 = None
        add_547 = torch.ops.aten.add.Tensor(add_546, add_536);  add_546 = add_536 = None
        convolution_231 = torch.ops.aten.convolution.default(add_547, arg446_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg446_1 = None
        add_548 = torch.ops.aten.add.Tensor(arg448_1, 1e-05);  arg448_1 = None
        sqrt_164 = torch.ops.aten.sqrt.default(add_548);  add_548 = None
        reciprocal_164 = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_668 = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1312 = torch.ops.aten.unsqueeze.default(arg447_1, -1);  arg447_1 = None
        unsqueeze_1313 = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        unsqueeze_1314 = torch.ops.aten.unsqueeze.default(mul_668, -1);  mul_668 = None
        unsqueeze_1315 = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        sub_164 = torch.ops.aten.sub.Tensor(convolution_231, unsqueeze_1313);  convolution_231 = unsqueeze_1313 = None
        mul_669 = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_1315);  sub_164 = unsqueeze_1315 = None
        unsqueeze_1316 = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_1317 = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_670 = torch.ops.aten.mul.Tensor(mul_669, unsqueeze_1317);  mul_669 = unsqueeze_1317 = None
        unsqueeze_1318 = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_1319 = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_549 = torch.ops.aten.add.Tensor(mul_670, unsqueeze_1319);  mul_670 = unsqueeze_1319 = None
        add_550 = torch.ops.aten.add.Tensor(add_549, 3)
        clamp_min_176 = torch.ops.aten.clamp_min.default(add_550, 0);  add_550 = None
        clamp_max_176 = torch.ops.aten.clamp_max.default(clamp_min_176, 6);  clamp_min_176 = None
        mul_671 = torch.ops.aten.mul.Tensor(add_549, clamp_max_176);  add_549 = clamp_max_176 = None
        div_176 = torch.ops.aten.div.Tensor(mul_671, 6);  mul_671 = None
        convolution_232 = torch.ops.aten.convolution.default(div_176, arg451_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_176 = arg451_1 = None
        add_551 = torch.ops.aten.add.Tensor(arg453_1, 1e-05);  arg453_1 = None
        sqrt_165 = torch.ops.aten.sqrt.default(add_551);  add_551 = None
        reciprocal_165 = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_672 = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1320 = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
        unsqueeze_1321 = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        unsqueeze_1322 = torch.ops.aten.unsqueeze.default(mul_672, -1);  mul_672 = None
        unsqueeze_1323 = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        sub_165 = torch.ops.aten.sub.Tensor(convolution_232, unsqueeze_1321);  convolution_232 = unsqueeze_1321 = None
        mul_673 = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_1323);  sub_165 = unsqueeze_1323 = None
        unsqueeze_1324 = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_1325 = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_674 = torch.ops.aten.mul.Tensor(mul_673, unsqueeze_1325);  mul_673 = unsqueeze_1325 = None
        unsqueeze_1326 = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_1327 = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_552 = torch.ops.aten.add.Tensor(mul_674, unsqueeze_1327);  mul_674 = unsqueeze_1327 = None
        add_553 = torch.ops.aten.add.Tensor(add_552, 3)
        clamp_min_177 = torch.ops.aten.clamp_min.default(add_553, 0);  add_553 = None
        clamp_max_177 = torch.ops.aten.clamp_max.default(clamp_min_177, 6);  clamp_min_177 = None
        mul_675 = torch.ops.aten.mul.Tensor(add_552, clamp_max_177);  add_552 = clamp_max_177 = None
        div_177 = torch.ops.aten.div.Tensor(mul_675, 6);  mul_675 = None
        mean_34 = torch.ops.aten.mean.dim(div_177, [2, 3], True)
        convolution_233 = torch.ops.aten.convolution.default(mean_34, arg456_1, arg457_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_34 = arg456_1 = arg457_1 = None
        add_554 = torch.ops.aten.add.Tensor(convolution_233, 3)
        clamp_min_178 = torch.ops.aten.clamp_min.default(add_554, 0);  add_554 = None
        clamp_max_178 = torch.ops.aten.clamp_max.default(clamp_min_178, 6);  clamp_min_178 = None
        mul_676 = torch.ops.aten.mul.Tensor(convolution_233, clamp_max_178);  convolution_233 = clamp_max_178 = None
        div_178 = torch.ops.aten.div.Tensor(mul_676, 6);  mul_676 = None
        convolution_234 = torch.ops.aten.convolution.default(div_178, arg458_1, arg459_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_178 = arg458_1 = arg459_1 = None
        add_555 = torch.ops.aten.add.Tensor(convolution_234, 3);  convolution_234 = None
        clamp_min_179 = torch.ops.aten.clamp_min.default(add_555, 0);  add_555 = None
        clamp_max_179 = torch.ops.aten.clamp_max.default(clamp_min_179, 6);  clamp_min_179 = None
        div_179 = torch.ops.aten.div.Tensor(clamp_max_179, 6);  clamp_max_179 = None
        mul_677 = torch.ops.aten.mul.Tensor(div_177, div_179);  div_177 = div_179 = None
        convolution_235 = torch.ops.aten.convolution.default(mul_677, arg460_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_677 = arg460_1 = None
        add_556 = torch.ops.aten.add.Tensor(arg462_1, 1e-05);  arg462_1 = None
        sqrt_166 = torch.ops.aten.sqrt.default(add_556);  add_556 = None
        reciprocal_166 = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_678 = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1328 = torch.ops.aten.unsqueeze.default(arg461_1, -1);  arg461_1 = None
        unsqueeze_1329 = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        unsqueeze_1330 = torch.ops.aten.unsqueeze.default(mul_678, -1);  mul_678 = None
        unsqueeze_1331 = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        sub_166 = torch.ops.aten.sub.Tensor(convolution_235, unsqueeze_1329);  convolution_235 = unsqueeze_1329 = None
        mul_679 = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_1331);  sub_166 = unsqueeze_1331 = None
        unsqueeze_1332 = torch.ops.aten.unsqueeze.default(arg463_1, -1);  arg463_1 = None
        unsqueeze_1333 = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_680 = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_1333);  mul_679 = unsqueeze_1333 = None
        unsqueeze_1334 = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_1335 = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_557 = torch.ops.aten.add.Tensor(mul_680, unsqueeze_1335);  mul_680 = unsqueeze_1335 = None
        add_558 = torch.ops.aten.add.Tensor(add_557, add_547);  add_557 = add_547 = None
        convolution_236 = torch.ops.aten.convolution.default(add_558, arg465_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg465_1 = None
        add_559 = torch.ops.aten.add.Tensor(arg467_1, 1e-05);  arg467_1 = None
        sqrt_167 = torch.ops.aten.sqrt.default(add_559);  add_559 = None
        reciprocal_167 = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_681 = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1336 = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
        unsqueeze_1337 = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        unsqueeze_1338 = torch.ops.aten.unsqueeze.default(mul_681, -1);  mul_681 = None
        unsqueeze_1339 = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        sub_167 = torch.ops.aten.sub.Tensor(convolution_236, unsqueeze_1337);  convolution_236 = unsqueeze_1337 = None
        mul_682 = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_1339);  sub_167 = unsqueeze_1339 = None
        unsqueeze_1340 = torch.ops.aten.unsqueeze.default(arg468_1, -1);  arg468_1 = None
        unsqueeze_1341 = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_683 = torch.ops.aten.mul.Tensor(mul_682, unsqueeze_1341);  mul_682 = unsqueeze_1341 = None
        unsqueeze_1342 = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_1343 = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_560 = torch.ops.aten.add.Tensor(mul_683, unsqueeze_1343);  mul_683 = unsqueeze_1343 = None
        add_561 = torch.ops.aten.add.Tensor(add_560, 3)
        clamp_min_180 = torch.ops.aten.clamp_min.default(add_561, 0);  add_561 = None
        clamp_max_180 = torch.ops.aten.clamp_max.default(clamp_min_180, 6);  clamp_min_180 = None
        mul_684 = torch.ops.aten.mul.Tensor(add_560, clamp_max_180);  add_560 = clamp_max_180 = None
        div_180 = torch.ops.aten.div.Tensor(mul_684, 6);  mul_684 = None
        convolution_237 = torch.ops.aten.convolution.default(div_180, arg470_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_180 = arg470_1 = None
        add_562 = torch.ops.aten.add.Tensor(arg472_1, 1e-05);  arg472_1 = None
        sqrt_168 = torch.ops.aten.sqrt.default(add_562);  add_562 = None
        reciprocal_168 = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_685 = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1344 = torch.ops.aten.unsqueeze.default(arg471_1, -1);  arg471_1 = None
        unsqueeze_1345 = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        unsqueeze_1346 = torch.ops.aten.unsqueeze.default(mul_685, -1);  mul_685 = None
        unsqueeze_1347 = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        sub_168 = torch.ops.aten.sub.Tensor(convolution_237, unsqueeze_1345);  convolution_237 = unsqueeze_1345 = None
        mul_686 = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_1347);  sub_168 = unsqueeze_1347 = None
        unsqueeze_1348 = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
        unsqueeze_1349 = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_687 = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_1349);  mul_686 = unsqueeze_1349 = None
        unsqueeze_1350 = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_1351 = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_563 = torch.ops.aten.add.Tensor(mul_687, unsqueeze_1351);  mul_687 = unsqueeze_1351 = None
        add_564 = torch.ops.aten.add.Tensor(add_563, 3)
        clamp_min_181 = torch.ops.aten.clamp_min.default(add_564, 0);  add_564 = None
        clamp_max_181 = torch.ops.aten.clamp_max.default(clamp_min_181, 6);  clamp_min_181 = None
        mul_688 = torch.ops.aten.mul.Tensor(add_563, clamp_max_181);  add_563 = clamp_max_181 = None
        div_181 = torch.ops.aten.div.Tensor(mul_688, 6);  mul_688 = None
        mean_35 = torch.ops.aten.mean.dim(div_181, [2, 3], True)
        convolution_238 = torch.ops.aten.convolution.default(mean_35, arg475_1, arg476_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_35 = arg475_1 = arg476_1 = None
        add_565 = torch.ops.aten.add.Tensor(convolution_238, 3)
        clamp_min_182 = torch.ops.aten.clamp_min.default(add_565, 0);  add_565 = None
        clamp_max_182 = torch.ops.aten.clamp_max.default(clamp_min_182, 6);  clamp_min_182 = None
        mul_689 = torch.ops.aten.mul.Tensor(convolution_238, clamp_max_182);  convolution_238 = clamp_max_182 = None
        div_182 = torch.ops.aten.div.Tensor(mul_689, 6);  mul_689 = None
        convolution_239 = torch.ops.aten.convolution.default(div_182, arg477_1, arg478_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_182 = arg477_1 = arg478_1 = None
        add_566 = torch.ops.aten.add.Tensor(convolution_239, 3);  convolution_239 = None
        clamp_min_183 = torch.ops.aten.clamp_min.default(add_566, 0);  add_566 = None
        clamp_max_183 = torch.ops.aten.clamp_max.default(clamp_min_183, 6);  clamp_min_183 = None
        div_183 = torch.ops.aten.div.Tensor(clamp_max_183, 6);  clamp_max_183 = None
        mul_690 = torch.ops.aten.mul.Tensor(div_181, div_183);  div_181 = div_183 = None
        convolution_240 = torch.ops.aten.convolution.default(mul_690, arg479_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_690 = arg479_1 = None
        add_567 = torch.ops.aten.add.Tensor(arg481_1, 1e-05);  arg481_1 = None
        sqrt_169 = torch.ops.aten.sqrt.default(add_567);  add_567 = None
        reciprocal_169 = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_691 = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1352 = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_1353 = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        unsqueeze_1354 = torch.ops.aten.unsqueeze.default(mul_691, -1);  mul_691 = None
        unsqueeze_1355 = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        sub_169 = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_1353);  convolution_240 = unsqueeze_1353 = None
        mul_692 = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_1355);  sub_169 = unsqueeze_1355 = None
        unsqueeze_1356 = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_1357 = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_693 = torch.ops.aten.mul.Tensor(mul_692, unsqueeze_1357);  mul_692 = unsqueeze_1357 = None
        unsqueeze_1358 = torch.ops.aten.unsqueeze.default(arg483_1, -1);  arg483_1 = None
        unsqueeze_1359 = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_568 = torch.ops.aten.add.Tensor(mul_693, unsqueeze_1359);  mul_693 = unsqueeze_1359 = None
        add_569 = torch.ops.aten.add.Tensor(add_568, add_558);  add_568 = add_558 = None
        convolution_241 = torch.ops.aten.convolution.default(add_569, arg484_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_569 = arg484_1 = None
        add_570 = torch.ops.aten.add.Tensor(arg486_1, 1e-05);  arg486_1 = None
        sqrt_170 = torch.ops.aten.sqrt.default(add_570);  add_570 = None
        reciprocal_170 = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_694 = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1360 = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
        unsqueeze_1361 = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        unsqueeze_1362 = torch.ops.aten.unsqueeze.default(mul_694, -1);  mul_694 = None
        unsqueeze_1363 = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        sub_170 = torch.ops.aten.sub.Tensor(convolution_241, unsqueeze_1361);  convolution_241 = unsqueeze_1361 = None
        mul_695 = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_1363);  sub_170 = unsqueeze_1363 = None
        unsqueeze_1364 = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
        unsqueeze_1365 = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_696 = torch.ops.aten.mul.Tensor(mul_695, unsqueeze_1365);  mul_695 = unsqueeze_1365 = None
        unsqueeze_1366 = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
        unsqueeze_1367 = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_571 = torch.ops.aten.add.Tensor(mul_696, unsqueeze_1367);  mul_696 = unsqueeze_1367 = None
        add_572 = torch.ops.aten.add.Tensor(add_571, 3)
        clamp_min_184 = torch.ops.aten.clamp_min.default(add_572, 0);  add_572 = None
        clamp_max_184 = torch.ops.aten.clamp_max.default(clamp_min_184, 6);  clamp_min_184 = None
        mul_697 = torch.ops.aten.mul.Tensor(add_571, clamp_max_184);  add_571 = clamp_max_184 = None
        div_184 = torch.ops.aten.div.Tensor(mul_697, 6);  mul_697 = None
        convolution_242 = torch.ops.aten.convolution.default(div_184, arg489_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1104);  div_184 = arg489_1 = None
        add_573 = torch.ops.aten.add.Tensor(arg491_1, 1e-05);  arg491_1 = None
        sqrt_171 = torch.ops.aten.sqrt.default(add_573);  add_573 = None
        reciprocal_171 = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_698 = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1368 = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_1369 = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        unsqueeze_1370 = torch.ops.aten.unsqueeze.default(mul_698, -1);  mul_698 = None
        unsqueeze_1371 = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        sub_171 = torch.ops.aten.sub.Tensor(convolution_242, unsqueeze_1369);  convolution_242 = unsqueeze_1369 = None
        mul_699 = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_1371);  sub_171 = unsqueeze_1371 = None
        unsqueeze_1372 = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
        unsqueeze_1373 = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_700 = torch.ops.aten.mul.Tensor(mul_699, unsqueeze_1373);  mul_699 = unsqueeze_1373 = None
        unsqueeze_1374 = torch.ops.aten.unsqueeze.default(arg493_1, -1);  arg493_1 = None
        unsqueeze_1375 = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_574 = torch.ops.aten.add.Tensor(mul_700, unsqueeze_1375);  mul_700 = unsqueeze_1375 = None
        add_575 = torch.ops.aten.add.Tensor(add_574, 3)
        clamp_min_185 = torch.ops.aten.clamp_min.default(add_575, 0);  add_575 = None
        clamp_max_185 = torch.ops.aten.clamp_max.default(clamp_min_185, 6);  clamp_min_185 = None
        mul_701 = torch.ops.aten.mul.Tensor(add_574, clamp_max_185);  add_574 = clamp_max_185 = None
        div_185 = torch.ops.aten.div.Tensor(mul_701, 6);  mul_701 = None
        mean_36 = torch.ops.aten.mean.dim(div_185, [2, 3], True)
        convolution_243 = torch.ops.aten.convolution.default(mean_36, arg494_1, arg495_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_36 = arg494_1 = arg495_1 = None
        add_576 = torch.ops.aten.add.Tensor(convolution_243, 3)
        clamp_min_186 = torch.ops.aten.clamp_min.default(add_576, 0);  add_576 = None
        clamp_max_186 = torch.ops.aten.clamp_max.default(clamp_min_186, 6);  clamp_min_186 = None
        mul_702 = torch.ops.aten.mul.Tensor(convolution_243, clamp_max_186);  convolution_243 = clamp_max_186 = None
        div_186 = torch.ops.aten.div.Tensor(mul_702, 6);  mul_702 = None
        convolution_244 = torch.ops.aten.convolution.default(div_186, arg496_1, arg497_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_186 = arg496_1 = arg497_1 = None
        add_577 = torch.ops.aten.add.Tensor(convolution_244, 3);  convolution_244 = None
        clamp_min_187 = torch.ops.aten.clamp_min.default(add_577, 0);  add_577 = None
        clamp_max_187 = torch.ops.aten.clamp_max.default(clamp_min_187, 6);  clamp_min_187 = None
        div_187 = torch.ops.aten.div.Tensor(clamp_max_187, 6);  clamp_max_187 = None
        mul_703 = torch.ops.aten.mul.Tensor(div_185, div_187);  div_185 = div_187 = None
        convolution_245 = torch.ops.aten.convolution.default(mul_703, arg498_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_703 = arg498_1 = None
        add_578 = torch.ops.aten.add.Tensor(arg500_1, 1e-05);  arg500_1 = None
        sqrt_172 = torch.ops.aten.sqrt.default(add_578);  add_578 = None
        reciprocal_172 = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_704 = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1376 = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_1377 = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        unsqueeze_1378 = torch.ops.aten.unsqueeze.default(mul_704, -1);  mul_704 = None
        unsqueeze_1379 = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        sub_172 = torch.ops.aten.sub.Tensor(convolution_245, unsqueeze_1377);  convolution_245 = unsqueeze_1377 = None
        mul_705 = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1379);  sub_172 = unsqueeze_1379 = None
        unsqueeze_1380 = torch.ops.aten.unsqueeze.default(arg501_1, -1);  arg501_1 = None
        unsqueeze_1381 = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_706 = torch.ops.aten.mul.Tensor(mul_705, unsqueeze_1381);  mul_705 = unsqueeze_1381 = None
        unsqueeze_1382 = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_1383 = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_579 = torch.ops.aten.add.Tensor(mul_706, unsqueeze_1383);  mul_706 = unsqueeze_1383 = None
        convolution_246 = torch.ops.aten.convolution.default(add_579, arg503_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_579 = arg503_1 = None
        add_580 = torch.ops.aten.add.Tensor(arg505_1, 1e-05);  arg505_1 = None
        sqrt_173 = torch.ops.aten.sqrt.default(add_580);  add_580 = None
        reciprocal_173 = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_707 = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1384 = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_1385 = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        unsqueeze_1386 = torch.ops.aten.unsqueeze.default(mul_707, -1);  mul_707 = None
        unsqueeze_1387 = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        sub_173 = torch.ops.aten.sub.Tensor(convolution_246, unsqueeze_1385);  convolution_246 = unsqueeze_1385 = None
        mul_708 = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1387);  sub_173 = unsqueeze_1387 = None
        unsqueeze_1388 = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
        unsqueeze_1389 = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_709 = torch.ops.aten.mul.Tensor(mul_708, unsqueeze_1389);  mul_708 = unsqueeze_1389 = None
        unsqueeze_1390 = torch.ops.aten.unsqueeze.default(arg507_1, -1);  arg507_1 = None
        unsqueeze_1391 = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_581 = torch.ops.aten.add.Tensor(mul_709, unsqueeze_1391);  mul_709 = unsqueeze_1391 = None
        add_582 = torch.ops.aten.add.Tensor(add_581, 3)
        clamp_min_188 = torch.ops.aten.clamp_min.default(add_582, 0);  add_582 = None
        clamp_max_188 = torch.ops.aten.clamp_max.default(clamp_min_188, 6);  clamp_min_188 = None
        mul_710 = torch.ops.aten.mul.Tensor(add_581, clamp_max_188);  add_581 = clamp_max_188 = None
        div_188 = torch.ops.aten.div.Tensor(mul_710, 6);  mul_710 = None
        mean_37 = torch.ops.aten.mean.dim(div_188, [-1, -2], True);  div_188 = None
        convolution_247 = torch.ops.aten.convolution.default(mean_37, arg508_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_37 = arg508_1 = None
        add_583 = torch.ops.aten.add.Tensor(convolution_247, 3)
        clamp_min_189 = torch.ops.aten.clamp_min.default(add_583, 0);  add_583 = None
        clamp_max_189 = torch.ops.aten.clamp_max.default(clamp_min_189, 6);  clamp_min_189 = None
        mul_711 = torch.ops.aten.mul.Tensor(convolution_247, clamp_max_189);  convolution_247 = clamp_max_189 = None
        div_189 = torch.ops.aten.div.Tensor(mul_711, 6);  mul_711 = None
        permute_1 = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        view_3 = torch.ops.aten.view.default(div_189, [8, 1984]);  div_189 = None
        addmm_1 = torch.ops.aten.addmm.default(arg510_1, view_3, permute_1);  arg510_1 = view_3 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf0, (16, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 256, 256), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf2, (16,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf6, (16, 1, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf7, (16,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf8, (16,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf9, (16,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf10, (16,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf11, (16, 16, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf14, (16,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf16, (16, 1, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf17, (16,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf18, (16,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf19, (16,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf20, (16,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf21, (16, 16, 1, 1), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf22, (16,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf23, (16,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf24, (16,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf25, (16,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf26, (64, 16, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf27, (64,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf28, (64,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf29, (64,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf30, (64,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64, 1, 5, 5), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf32, (64,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf33, (64,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf34, (64,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf35, (64,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf36, (24, 64, 1, 1), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf37, (24,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf38, (24,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf39, (24,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf40, (24,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf41, (48, 24, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf42, (48,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf43, (48,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf44, (48,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf45, (48,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf46, (48, 1, 5, 5), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf47, (48,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf48, (48,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf49, (48,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf50, (48,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf51, (24, 48, 1, 1), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf52, (24,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf53, (24,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf54, (24,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf55, (24,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf56, (48, 24, 1, 1), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf57, (48,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf58, (48,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf59, (48,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf60, (48,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf61, (48, 1, 5, 5), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf62, (48,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf63, (48,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf64, (48,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf65, (48,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf66, (24, 48, 1, 1), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf67, (24,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf68, (24,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf69, (24,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf70, (24,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf71, (48, 24, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf72, (48,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf73, (48,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf74, (48,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf75, (48,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf76, (48, 1, 5, 5), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf77, (48,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf78, (48,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf79, (48,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf80, (48,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf81, (24, 48, 1, 1), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf82, (24,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf83, (24,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf84, (24,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf85, (24,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 11520, device=device(type='cuda', index=0))
    reader.tensor(buf86, (120, 24, 1, 1), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf87, (120,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf88, (120,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf89, (120,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf90, (120,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf91, (120, 1, 5, 5), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf92, (120,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf93, (120,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf94, (120,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf95, (120,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf96, (8, 120, 1, 1), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf97, (8,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf98, (120, 8, 1, 1), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf99, (120,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf100, (40, 120, 1, 1), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf101, (40,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf102, (40,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf103, (40,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf104, (40,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf105, (120, 40, 1, 1), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf106, (120,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf107, (120,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf108, (120,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf109, (120,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf110, (120, 1, 5, 5), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf111, (120,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf112, (120,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf113, (120,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf114, (120,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf115, (16, 120, 1, 1), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf116, (16,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf117, (120, 16, 1, 1), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf118, (120,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf119, (40, 120, 1, 1), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf120, (40,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf121, (40,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf122, (40,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf123, (40,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf124, (120, 40, 1, 1), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf125, (120,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf126, (120,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf127, (120,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf128, (120,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf129, (120, 1, 5, 5), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf130, (120,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf131, (120,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf132, (120,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf133, (120,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf134, (16, 120, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf135, (16,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf136, (120, 16, 1, 1), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf137, (120,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf138, (40, 120, 1, 1), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf139, (40,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf140, (40,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf141, (40,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf142, (40,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf143, (120, 40, 1, 1), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf144, (120,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf145, (120,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf146, (120,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf147, (120,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf148, (120, 1, 5, 5), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf149, (120,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf150, (120,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf151, (120,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf152, (120,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf153, (16, 120, 1, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf154, (16,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf155, (120, 16, 1, 1), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf156, (120,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf157, (40, 120, 1, 1), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf158, (40,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf159, (40,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf160, (40,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf161, (40,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf162, (120, 40, 1, 1), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf163, (120,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf164, (120,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf165, (120,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf166, (120,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf167, (120, 1, 5, 5), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf168, (120,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf169, (120,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf170, (120,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf171, (120,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf172, (16, 120, 1, 1), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf173, (16,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf174, (120, 16, 1, 1), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf175, (120,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf176, (40, 120, 1, 1), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf177, (40,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf178, (40,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf179, (40,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf180, (40,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 32000, device=device(type='cuda', index=0))
    reader.tensor(buf181, (200, 40, 1, 1), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf182, (200,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf183, (200,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf184, (200,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf185, (200,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 20000, device=device(type='cuda', index=0))
    reader.tensor(buf186, (200, 1, 5, 5), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf187, (200,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf188, (200,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf189, (200,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf190, (200,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 57600, device=device(type='cuda', index=0))
    reader.tensor(buf191, (72, 200, 1, 1), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf192, (72,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf193, (72,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf194, (72,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf195, (72,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf196, (216, 72, 1, 1), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf197, (216,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf198, (216,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf199, (216,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf200, (216,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf201, (216, 1, 3, 3), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf202, (216,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf203, (216,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf204, (216,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf205, (216,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf206, (72, 216, 1, 1), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf207, (72,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf208, (72,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf209, (72,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf210, (72,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf211, (216, 72, 1, 1), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf212, (216,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf213, (216,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf214, (216,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf215, (216,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf216, (216, 1, 3, 3), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf217, (216,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf218, (216,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf219, (216,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf220, (216,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf221, (72, 216, 1, 1), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf222, (72,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf223, (72,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf224, (72,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf225, (72,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf226, (216, 72, 1, 1), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf227, (216,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf228, (216,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf229, (216,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf230, (216,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf231, (216, 1, 3, 3), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf232, (216,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf233, (216,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf234, (216,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf235, (216,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf236, (72, 216, 1, 1), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf237, (72,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf238, (72,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf239, (72,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf240, (72,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf241, (216, 72, 1, 1), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf242, (216,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf243, (216,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf244, (216,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf245, (216,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf246, (216, 1, 3, 3), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf247, (216,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf248, (216,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf249, (216,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf250, (216,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf251, (72, 216, 1, 1), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf252, (72,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf253, (72,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf254, (72,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf255, (72,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 103680, device=device(type='cuda', index=0))
    reader.tensor(buf256, (360, 72, 1, 1), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf257, (360,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf258, (360,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf259, (360,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf260, (360,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 12960, device=device(type='cuda', index=0))
    reader.tensor(buf261, (360, 1, 3, 3), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf262, (360,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf263, (360,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf264, (360,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf265, (360,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 34560, device=device(type='cuda', index=0))
    reader.tensor(buf266, (24, 360, 1, 1), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf267, (24,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 34560, device=device(type='cuda', index=0))
    reader.tensor(buf268, (360, 24, 1, 1), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf269, (360,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf270, (120, 360, 1, 1), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf271, (120,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf272, (120,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf273, (120,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf274, (120,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf275, (360, 120, 1, 1), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf276, (360,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf277, (360,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf278, (360,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf279, (360,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 36000, device=device(type='cuda', index=0))
    reader.tensor(buf280, (360, 1, 5, 5), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf281, (360,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf282, (360,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf283, (360,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf284, (360,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf285, (32, 360, 1, 1), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf286, (32,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf287, (360, 32, 1, 1), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf288, (360,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf289, (120, 360, 1, 1), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf290, (120,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf291, (120,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf292, (120,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf293, (120,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf294, (360, 120, 1, 1), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf295, (360,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf296, (360,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf297, (360,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf298, (360,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 36000, device=device(type='cuda', index=0))
    reader.tensor(buf299, (360, 1, 5, 5), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf300, (360,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf301, (360,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf302, (360,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf303, (360,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf304, (32, 360, 1, 1), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf305, (32,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf306, (360, 32, 1, 1), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf307, (360,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf308, (120, 360, 1, 1), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf309, (120,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf310, (120,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf311, (120,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf312, (120,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf313, (360, 120, 1, 1), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf314, (360,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf315, (360,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf316, (360,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf317, (360,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 36000, device=device(type='cuda', index=0))
    reader.tensor(buf318, (360, 1, 5, 5), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf319, (360,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf320, (360,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf321, (360,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf322, (360,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf323, (32, 360, 1, 1), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf324, (32,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf325, (360, 32, 1, 1), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf326, (360,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf327, (120, 360, 1, 1), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf328, (120,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf329, (120,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf330, (120,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf331, (120,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf332, (360, 120, 1, 1), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf333, (360,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf334, (360,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf335, (360,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf336, (360,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 36000, device=device(type='cuda', index=0))
    reader.tensor(buf337, (360, 1, 5, 5), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf338, (360,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf339, (360,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf340, (360,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf341, (360,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf342, (32, 360, 1, 1), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf343, (32,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf344, (360, 32, 1, 1), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf345, (360,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf346, (120, 360, 1, 1), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf347, (120,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf348, (120,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf349, (120,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf350, (120,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf351, (360, 120, 1, 1), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf352, (360,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf353, (360,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf354, (360,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf355, (360,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 36000, device=device(type='cuda', index=0))
    reader.tensor(buf356, (360, 1, 5, 5), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf357, (360,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf358, (360,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf359, (360,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf360, (360,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf361, (32, 360, 1, 1), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf362, (32,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf363, (360, 32, 1, 1), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf364, (360,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 172800, device=device(type='cuda', index=0))
    reader.tensor(buf365, (120, 360, 1, 1), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf366, (120,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf367, (120,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf368, (120,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf369, (120,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 345600, device=device(type='cuda', index=0))
    reader.tensor(buf370, (720, 120, 1, 1), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf371, (720,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf372, (720,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf373, (720,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf374, (720,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 25920, device=device(type='cuda', index=0))
    reader.tensor(buf375, (720, 1, 3, 3), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf376, (720,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf377, (720,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf378, (720,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf379, (720,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 92160, device=device(type='cuda', index=0))
    reader.tensor(buf380, (32, 720, 1, 1), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf381, (32,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 92160, device=device(type='cuda', index=0))
    reader.tensor(buf382, (720, 32, 1, 1), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf383, (720,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 529920, device=device(type='cuda', index=0))
    reader.tensor(buf384, (184, 720, 1, 1), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf385, (184,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf386, (184,), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf387, (184,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf388, (184,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf389, (736, 184, 1, 1), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf390, (736,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf391, (736,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf392, (736,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf393, (736,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 73600, device=device(type='cuda', index=0))
    reader.tensor(buf394, (736, 1, 5, 5), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf395, (736,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf396, (736,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf397, (736,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf398, (736,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf399, (48, 736, 1, 1), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf400, (48,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf401, (736, 48, 1, 1), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf402, (736,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf403, (184, 736, 1, 1), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf404, (184,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf405, (184,), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf406, (184,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf407, (184,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf408, (736, 184, 1, 1), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf409, (736,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf410, (736,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf411, (736,), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf412, (736,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 73600, device=device(type='cuda', index=0))
    reader.tensor(buf413, (736, 1, 5, 5), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf414, (736,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf415, (736,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf416, (736,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf417, (736,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf418, (48, 736, 1, 1), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf419, (48,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf420, (736, 48, 1, 1), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf421, (736,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf422, (184, 736, 1, 1), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf423, (184,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf424, (184,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf425, (184,), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf426, (184,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf427, (736, 184, 1, 1), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf428, (736,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf429, (736,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf430, (736,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf431, (736,), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 73600, device=device(type='cuda', index=0))
    reader.tensor(buf432, (736, 1, 5, 5), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf433, (736,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf434, (736,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf435, (736,), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf436, (736,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf437, (48, 736, 1, 1), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf438, (48,), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf439, (736, 48, 1, 1), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf440, (736,), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf441, (184, 736, 1, 1), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf442, (184,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf443, (184,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf444, (184,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf445, (184,), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf446, (736, 184, 1, 1), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf447, (736,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf448, (736,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf449, (736,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf450, (736,), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 73600, device=device(type='cuda', index=0))
    reader.tensor(buf451, (736, 1, 5, 5), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf452, (736,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf453, (736,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf454, (736,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf455, (736,), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf456, (48, 736, 1, 1), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf457, (48,), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf458, (736, 48, 1, 1), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf459, (736,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf460, (184, 736, 1, 1), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf461, (184,), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf462, (184,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf463, (184,), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf464, (184,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf465, (736, 184, 1, 1), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf466, (736,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf467, (736,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf468, (736,), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf469, (736,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 73600, device=device(type='cuda', index=0))
    reader.tensor(buf470, (736, 1, 5, 5), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf471, (736,), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf472, (736,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf473, (736,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf474, (736,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf475, (48, 736, 1, 1), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf476, (48,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 141312, device=device(type='cuda', index=0))
    reader.tensor(buf477, (736, 48, 1, 1), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 2944, device=device(type='cuda', index=0))
    reader.tensor(buf478, (736,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 541696, device=device(type='cuda', index=0))
    reader.tensor(buf479, (184, 736, 1, 1), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf480, (184,), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf481, (184,), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf482, (184,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 736, device=device(type='cuda', index=0))
    reader.tensor(buf483, (184,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 812544, device=device(type='cuda', index=0))
    reader.tensor(buf484, (1104, 184, 1, 1), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf485, (1104,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf486, (1104,), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf487, (1104,), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf488, (1104,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 110400, device=device(type='cuda', index=0))
    reader.tensor(buf489, (1104, 1, 5, 5), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf490, (1104,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf491, (1104,), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf492, (1104,), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf493, (1104,), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 211968, device=device(type='cuda', index=0))
    reader.tensor(buf494, (48, 1104, 1, 1), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf495, (48,), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 211968, device=device(type='cuda', index=0))
    reader.tensor(buf496, (1104, 48, 1, 1), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 4416, device=device(type='cuda', index=0))
    reader.tensor(buf497, (1104,), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 989184, device=device(type='cuda', index=0))
    reader.tensor(buf498, (224, 1104, 1, 1), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf499, (224,), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf500, (224,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf501, (224,), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 896, device=device(type='cuda', index=0))
    reader.tensor(buf502, (224,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 1204224, device=device(type='cuda', index=0))
    reader.tensor(buf503, (1344, 224, 1, 1), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf504, (1344,), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf505, (1344,), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf506, (1344,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf507, (1344,), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 10665984, device=device(type='cuda', index=0))
    reader.tensor(buf508, (1984, 1344, 1, 1), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 7936000, device=device(type='cuda', index=0))
    reader.tensor(buf509, (1000, 1984), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf510, (1000,), is_leaf=True)  # arg510_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)