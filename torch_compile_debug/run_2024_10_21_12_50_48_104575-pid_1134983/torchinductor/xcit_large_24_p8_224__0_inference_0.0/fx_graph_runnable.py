
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1):
        convolution_52 = torch.ops.aten.convolution.default(arg0_1, arg1_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = None
        add_365 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_27 = torch.ops.aten.sqrt.default(add_365);  add_365 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_494 = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(unsqueeze_225, -1);  unsqueeze_225 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(mul_494, -1);  mul_494 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(unsqueeze_227, -1);  unsqueeze_227 = None
        sub_128 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_226);  convolution_52 = unsqueeze_226 = None
        mul_495 = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_228);  sub_128 = unsqueeze_228 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(unsqueeze_229, -1);  unsqueeze_229 = None
        mul_496 = torch.ops.aten.mul.Tensor(mul_495, unsqueeze_230);  mul_495 = unsqueeze_230 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(unsqueeze_231, -1);  unsqueeze_231 = None
        add_366 = torch.ops.aten.add.Tensor(mul_496, unsqueeze_232);  mul_496 = unsqueeze_232 = None
        mul_497 = torch.ops.aten.mul.Tensor(add_366, 0.5)
        mul_498 = torch.ops.aten.mul.Tensor(add_366, 0.7071067811865476);  add_366 = None
        erf_52 = torch.ops.aten.erf.default(mul_498);  mul_498 = None
        add_367 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_499 = torch.ops.aten.mul.Tensor(mul_497, add_367);  mul_497 = add_367 = None
        convolution_53 = torch.ops.aten.convolution.default(mul_499, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_499 = arg6_1 = None
        add_368 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_28 = torch.ops.aten.sqrt.default(add_368);  add_368 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_500 = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(unsqueeze_233, -1);  unsqueeze_233 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(mul_500, -1);  mul_500 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(unsqueeze_235, -1);  unsqueeze_235 = None
        sub_129 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_234);  convolution_53 = unsqueeze_234 = None
        mul_501 = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_236);  sub_129 = unsqueeze_236 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(unsqueeze_237, -1);  unsqueeze_237 = None
        mul_502 = torch.ops.aten.mul.Tensor(mul_501, unsqueeze_238);  mul_501 = unsqueeze_238 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(unsqueeze_239, -1);  unsqueeze_239 = None
        add_369 = torch.ops.aten.add.Tensor(mul_502, unsqueeze_240);  mul_502 = unsqueeze_240 = None
        mul_503 = torch.ops.aten.mul.Tensor(add_369, 0.5)
        mul_504 = torch.ops.aten.mul.Tensor(add_369, 0.7071067811865476);  add_369 = None
        erf_53 = torch.ops.aten.erf.default(mul_504);  mul_504 = None
        add_370 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_505 = torch.ops.aten.mul.Tensor(mul_503, add_370);  mul_503 = add_370 = None
        convolution_54 = torch.ops.aten.convolution.default(mul_505, arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_505 = arg11_1 = None
        add_371 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_371);  add_371 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_506 = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(unsqueeze_241, -1);  unsqueeze_241 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(mul_506, -1);  mul_506 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(unsqueeze_243, -1);  unsqueeze_243 = None
        sub_130 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_242);  convolution_54 = unsqueeze_242 = None
        mul_507 = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_244);  sub_130 = unsqueeze_244 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(unsqueeze_245, -1);  unsqueeze_245 = None
        mul_508 = torch.ops.aten.mul.Tensor(mul_507, unsqueeze_246);  mul_507 = unsqueeze_246 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(unsqueeze_247, -1);  unsqueeze_247 = None
        add_372 = torch.ops.aten.add.Tensor(mul_508, unsqueeze_248);  mul_508 = unsqueeze_248 = None
        view_464 = torch.ops.aten.view.default(add_372, [8, 768, 784]);  add_372 = None
        permute_240 = torch.ops.aten.permute.default(view_464, [0, 2, 1]);  view_464 = None
        iota_3 = torch.ops.prims.iota.default(28, start = 1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(iota_3, torch.float32);  iota_3 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(convert_element_type_63, 1);  convert_element_type_63 = None
        repeat_3 = torch.ops.aten.repeat.default(unsqueeze_249, [1, 1, 28]);  unsqueeze_249 = None
        iota_4 = torch.ops.prims.iota.default(28, start = 1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(iota_4, torch.float32);  iota_4 = None
        repeat_4 = torch.ops.aten.repeat.default(convert_element_type_64, [1, 28, 1]);  convert_element_type_64 = None
        slice_45 = torch.ops.aten.slice.Tensor(repeat_3, 1, -1, 9223372036854775807)
        add_373 = torch.ops.aten.add.Tensor(slice_45, 1e-06);  slice_45 = None
        div_78 = torch.ops.aten.div.Tensor(repeat_3, add_373);  repeat_3 = add_373 = None
        mul_509 = torch.ops.aten.mul.Tensor(div_78, 6.283185307179586);  div_78 = None
        slice_49 = torch.ops.aten.slice.Tensor(repeat_4, 2, -1, 9223372036854775807)
        add_374 = torch.ops.aten.add.Tensor(slice_49, 1e-06);  slice_49 = None
        div_79 = torch.ops.aten.div.Tensor(repeat_4, add_374);  repeat_4 = add_374 = None
        mul_510 = torch.ops.aten.mul.Tensor(div_79, 6.283185307179586);  div_79 = None
        iota_5 = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_65 = torch.ops.prims.convert_element_type.default(iota_5, torch.float32);  iota_5 = None
        div_80 = torch.ops.aten.div.Tensor_mode(convert_element_type_65, 2, rounding_mode = 'floor');  convert_element_type_65 = None
        mul_511 = torch.ops.aten.mul.Tensor(div_80, 2);  div_80 = None
        div_81 = torch.ops.aten.div.Tensor(mul_511, 32);  mul_511 = None
        pow_98 = torch.ops.aten.pow.Scalar(10000, div_81);  div_81 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(mul_510, 3);  mul_510 = None
        div_82 = torch.ops.aten.div.Tensor(unsqueeze_250, pow_98);  unsqueeze_250 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(mul_509, 3);  mul_509 = None
        div_83 = torch.ops.aten.div.Tensor(unsqueeze_251, pow_98);  unsqueeze_251 = pow_98 = None
        slice_59 = torch.ops.aten.slice.Tensor(div_82, 3, 0, 9223372036854775807, 2)
        sin_2 = torch.ops.aten.sin.default(slice_59);  slice_59 = None
        slice_63 = torch.ops.aten.slice.Tensor(div_82, 3, 1, 9223372036854775807, 2);  div_82 = None
        cos_2 = torch.ops.aten.cos.default(slice_63);  slice_63 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(sin_2, 4);  sin_2 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(cos_2, 4);  cos_2 = None
        cat_8 = torch.ops.aten.cat.default([unsqueeze_252, unsqueeze_253], 4);  unsqueeze_252 = unsqueeze_253 = None
        view_465 = torch.ops.aten.view.default(cat_8, [1, 28, 28, 32]);  cat_8 = None
        slice_67 = torch.ops.aten.slice.Tensor(div_83, 3, 0, 9223372036854775807, 2)
        sin_3 = torch.ops.aten.sin.default(slice_67);  slice_67 = None
        slice_71 = torch.ops.aten.slice.Tensor(div_83, 3, 1, 9223372036854775807, 2);  div_83 = None
        cos_3 = torch.ops.aten.cos.default(slice_71);  slice_71 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(sin_3, 4);  sin_3 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(cos_3, 4);  cos_3 = None
        cat_9 = torch.ops.aten.cat.default([unsqueeze_254, unsqueeze_255], 4);  unsqueeze_254 = unsqueeze_255 = None
        view_466 = torch.ops.aten.view.default(cat_9, [1, 28, 28, 32]);  cat_9 = None
        cat_10 = torch.ops.aten.cat.default([view_466, view_465], 3);  view_466 = view_465 = None
        permute_241 = torch.ops.aten.permute.default(cat_10, [0, 3, 1, 2]);  cat_10 = None
        convolution_55 = torch.ops.aten.convolution.default(permute_241, arg16_1, arg17_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_241 = arg16_1 = arg17_1 = None
        repeat_5 = torch.ops.aten.repeat.default(convolution_55, [8, 1, 1, 1]);  convolution_55 = None
        view_467 = torch.ops.aten.view.default(repeat_5, [8, -1, 784]);  repeat_5 = None
        permute_242 = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
        add_375 = torch.ops.aten.add.Tensor(permute_240, permute_242);  permute_240 = permute_242 = None
        clone_273 = torch.ops.aten.clone.default(add_375, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_273, [2], correction = 0, keepdim = True)
        getitem_234 = var_mean_77[0]
        getitem_235 = var_mean_77[1];  var_mean_77 = None
        add_376 = torch.ops.aten.add.Tensor(getitem_234, 1e-06);  getitem_234 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_131 = torch.ops.aten.sub.Tensor(clone_273, getitem_235);  clone_273 = getitem_235 = None
        mul_512 = torch.ops.aten.mul.Tensor(sub_131, rsqrt_77);  sub_131 = rsqrt_77 = None
        mul_513 = torch.ops.aten.mul.Tensor(mul_512, arg19_1);  mul_512 = arg19_1 = None
        add_377 = torch.ops.aten.add.Tensor(mul_513, arg20_1);  mul_513 = arg20_1 = None
        view_468 = torch.ops.aten.view.default(add_377, [6272, 768]);  add_377 = None
        permute_243 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg22_1, view_468, permute_243);  arg22_1 = view_468 = permute_243 = None
        view_469 = torch.ops.aten.view.default(addmm_83, [8, 784, 2304]);  addmm_83 = None
        view_470 = torch.ops.aten.view.default(view_469, [8, 784, 3, 16, 48]);  view_469 = None
        permute_244 = torch.ops.aten.permute.default(view_470, [2, 0, 3, 4, 1]);  view_470 = None
        unbind_24 = torch.ops.aten.unbind.int(permute_244);  permute_244 = None
        getitem_236 = unbind_24[0]
        getitem_237 = unbind_24[1]
        getitem_238 = unbind_24[2];  unbind_24 = None
        pow_99 = torch.ops.aten.pow.Tensor_Scalar(getitem_236, 2.0)
        sum_73 = torch.ops.aten.sum.dim_IntList(pow_99, [-1], True);  pow_99 = None
        pow_100 = torch.ops.aten.pow.Tensor_Scalar(sum_73, 0.5);  sum_73 = None
        clamp_min_48 = torch.ops.aten.clamp_min.default(pow_100, 1e-12);  pow_100 = None
        expand_145 = torch.ops.aten.expand.default(clamp_min_48, [8, 16, 48, 784]);  clamp_min_48 = None
        div_84 = torch.ops.aten.div.Tensor(getitem_236, expand_145);  getitem_236 = expand_145 = None
        pow_101 = torch.ops.aten.pow.Tensor_Scalar(getitem_237, 2.0)
        sum_74 = torch.ops.aten.sum.dim_IntList(pow_101, [-1], True);  pow_101 = None
        pow_102 = torch.ops.aten.pow.Tensor_Scalar(sum_74, 0.5);  sum_74 = None
        clamp_min_49 = torch.ops.aten.clamp_min.default(pow_102, 1e-12);  pow_102 = None
        expand_146 = torch.ops.aten.expand.default(clamp_min_49, [8, 16, 48, 784]);  clamp_min_49 = None
        div_85 = torch.ops.aten.div.Tensor(getitem_237, expand_146);  getitem_237 = expand_146 = None
        permute_245 = torch.ops.aten.permute.default(div_85, [0, 1, 3, 2]);  div_85 = None
        expand_147 = torch.ops.aten.expand.default(div_84, [8, 16, 48, 784]);  div_84 = None
        clone_274 = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_471 = torch.ops.aten.view.default(clone_274, [128, 48, 784]);  clone_274 = None
        expand_148 = torch.ops.aten.expand.default(permute_245, [8, 16, 784, 48]);  permute_245 = None
        clone_275 = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_472 = torch.ops.aten.view.default(clone_275, [128, 784, 48]);  clone_275 = None
        bmm_48 = torch.ops.aten.bmm.default(view_471, view_472);  view_471 = view_472 = None
        view_473 = torch.ops.aten.view.default(bmm_48, [8, 16, 48, 48]);  bmm_48 = None
        scalar_tensor_default_23 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_23 = torch.ops.aten.ge.Scalar(arg23_1, 0)
        neg_default_23 = torch.ops.aten.neg.default(scalar_tensor_default_23)
        where_self_23 = torch.ops.aten.where.self(ge_scalar_23, scalar_tensor_default_23, neg_default_23);  ge_scalar_23 = scalar_tensor_default_23 = neg_default_23 = None
        mul_tensor_69 = torch.ops.aten.mul.Tensor(view_473, where_self_23);  view_473 = None
        amax_default_23 = torch.ops.aten.amax.default(mul_tensor_69, [-1], True)
        sub_tensor_23 = torch.ops.aten.sub.Tensor(mul_tensor_69, amax_default_23);  mul_tensor_69 = amax_default_23 = None
        mul_tensor_70 = torch.ops.aten.mul.Tensor(where_self_23, arg23_1);  where_self_23 = arg23_1 = None
        mul_tensor_71 = torch.ops.aten.mul.Tensor(sub_tensor_23, mul_tensor_70);  sub_tensor_23 = mul_tensor_70 = None
        exp_24 = torch.ops.aten.exp.default(mul_tensor_71);  mul_tensor_71 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_86 = torch.ops.aten.div.Tensor(exp_24, sum_75);  exp_24 = sum_75 = None
        expand_149 = torch.ops.aten.expand.default(div_86, [8, 16, 48, 48]);  div_86 = None
        view_474 = torch.ops.aten.view.default(expand_149, [128, 48, 48]);  expand_149 = None
        expand_150 = torch.ops.aten.expand.default(getitem_238, [8, 16, 48, 784]);  getitem_238 = None
        clone_277 = torch.ops.aten.clone.default(expand_150, memory_format = torch.contiguous_format);  expand_150 = None
        view_475 = torch.ops.aten.view.default(clone_277, [128, 48, 784]);  clone_277 = None
        bmm_49 = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
        view_476 = torch.ops.aten.view.default(bmm_49, [8, 16, 48, 784]);  bmm_49 = None
        permute_246 = torch.ops.aten.permute.default(view_476, [0, 3, 1, 2]);  view_476 = None
        view_477 = torch.ops.aten.view.default(permute_246, [8, 784, 768]);  permute_246 = None
        permute_247 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        clone_278 = torch.ops.aten.clone.default(view_477, memory_format = torch.contiguous_format);  view_477 = None
        view_478 = torch.ops.aten.view.default(clone_278, [6272, 768]);  clone_278 = None
        mm_26 = torch.ops.aten.mm.default(view_478, permute_247);  view_478 = permute_247 = None
        view_479 = torch.ops.aten.view.default(mm_26, [8, 784, 768]);  mm_26 = None
        add_378 = torch.ops.aten.add.Tensor(view_479, arg25_1);  view_479 = arg25_1 = None
        mul_515 = torch.ops.aten.mul.Tensor(arg18_1, add_378);  arg18_1 = add_378 = None
        add_379 = torch.ops.aten.add.Tensor(add_375, mul_515);  add_375 = mul_515 = None
        clone_280 = torch.ops.aten.clone.default(add_379, memory_format = torch.contiguous_format)
        var_mean_78 = torch.ops.aten.var_mean.correction(clone_280, [2], correction = 0, keepdim = True)
        getitem_239 = var_mean_78[0]
        getitem_240 = var_mean_78[1];  var_mean_78 = None
        add_380 = torch.ops.aten.add.Tensor(getitem_239, 1e-06);  getitem_239 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_380);  add_380 = None
        sub_133 = torch.ops.aten.sub.Tensor(clone_280, getitem_240);  clone_280 = getitem_240 = None
        mul_516 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_78);  sub_133 = rsqrt_78 = None
        mul_517 = torch.ops.aten.mul.Tensor(mul_516, arg27_1);  mul_516 = arg27_1 = None
        add_381 = torch.ops.aten.add.Tensor(mul_517, arg28_1);  mul_517 = arg28_1 = None
        permute_248 = torch.ops.aten.permute.default(add_381, [0, 2, 1]);  add_381 = None
        view_480 = torch.ops.aten.view.default(permute_248, [8, 768, 28, 28]);  permute_248 = None
        convolution_56 = torch.ops.aten.convolution.default(view_480, arg29_1, arg30_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_480 = arg29_1 = arg30_1 = None
        mul_518 = torch.ops.aten.mul.Tensor(convolution_56, 0.5)
        mul_519 = torch.ops.aten.mul.Tensor(convolution_56, 0.7071067811865476);  convolution_56 = None
        erf_54 = torch.ops.aten.erf.default(mul_519);  mul_519 = None
        add_382 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_520 = torch.ops.aten.mul.Tensor(mul_518, add_382);  mul_518 = add_382 = None
        add_383 = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_383);  add_383 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_521 = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_521, -1);  mul_521 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_134 = torch.ops.aten.sub.Tensor(mul_520, unsqueeze_257);  mul_520 = unsqueeze_257 = None
        mul_522 = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_259);  sub_134 = unsqueeze_259 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_523 = torch.ops.aten.mul.Tensor(mul_522, unsqueeze_261);  mul_522 = unsqueeze_261 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_384 = torch.ops.aten.add.Tensor(mul_523, unsqueeze_263);  mul_523 = unsqueeze_263 = None
        convolution_57 = torch.ops.aten.convolution.default(add_384, arg35_1, arg36_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_384 = arg35_1 = arg36_1 = None
        view_481 = torch.ops.aten.view.default(convolution_57, [8, 768, 784]);  convolution_57 = None
        permute_249 = torch.ops.aten.permute.default(view_481, [0, 2, 1]);  view_481 = None
        mul_524 = torch.ops.aten.mul.Tensor(arg26_1, permute_249);  arg26_1 = permute_249 = None
        add_385 = torch.ops.aten.add.Tensor(add_379, mul_524);  add_379 = mul_524 = None
        clone_281 = torch.ops.aten.clone.default(add_385, memory_format = torch.contiguous_format)
        var_mean_79 = torch.ops.aten.var_mean.correction(clone_281, [2], correction = 0, keepdim = True)
        getitem_241 = var_mean_79[0]
        getitem_242 = var_mean_79[1];  var_mean_79 = None
        add_386 = torch.ops.aten.add.Tensor(getitem_241, 1e-06);  getitem_241 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
        sub_135 = torch.ops.aten.sub.Tensor(clone_281, getitem_242);  clone_281 = getitem_242 = None
        mul_525 = torch.ops.aten.mul.Tensor(sub_135, rsqrt_79);  sub_135 = rsqrt_79 = None
        mul_526 = torch.ops.aten.mul.Tensor(mul_525, arg38_1);  mul_525 = arg38_1 = None
        add_387 = torch.ops.aten.add.Tensor(mul_526, arg39_1);  mul_526 = arg39_1 = None
        view_482 = torch.ops.aten.view.default(add_387, [6272, 768]);  add_387 = None
        permute_250 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg41_1, view_482, permute_250);  arg41_1 = view_482 = permute_250 = None
        view_483 = torch.ops.aten.view.default(addmm_84, [8, 784, 3072]);  addmm_84 = None
        mul_527 = torch.ops.aten.mul.Tensor(view_483, 0.5)
        mul_528 = torch.ops.aten.mul.Tensor(view_483, 0.7071067811865476);  view_483 = None
        erf_55 = torch.ops.aten.erf.default(mul_528);  mul_528 = None
        add_388 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_529 = torch.ops.aten.mul.Tensor(mul_527, add_388);  mul_527 = add_388 = None
        view_484 = torch.ops.aten.view.default(mul_529, [6272, 3072]);  mul_529 = None
        permute_251 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg43_1, view_484, permute_251);  arg43_1 = view_484 = permute_251 = None
        view_485 = torch.ops.aten.view.default(addmm_85, [8, 784, 768]);  addmm_85 = None
        mul_530 = torch.ops.aten.mul.Tensor(arg37_1, view_485);  arg37_1 = view_485 = None
        add_389 = torch.ops.aten.add.Tensor(add_385, mul_530);  add_385 = mul_530 = None
        clone_284 = torch.ops.aten.clone.default(add_389, memory_format = torch.contiguous_format)
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_284, [2], correction = 0, keepdim = True)
        getitem_243 = var_mean_80[0]
        getitem_244 = var_mean_80[1];  var_mean_80 = None
        add_390 = torch.ops.aten.add.Tensor(getitem_243, 1e-06);  getitem_243 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_390);  add_390 = None
        sub_136 = torch.ops.aten.sub.Tensor(clone_284, getitem_244);  clone_284 = getitem_244 = None
        mul_531 = torch.ops.aten.mul.Tensor(sub_136, rsqrt_80);  sub_136 = rsqrt_80 = None
        mul_532 = torch.ops.aten.mul.Tensor(mul_531, arg45_1);  mul_531 = arg45_1 = None
        add_391 = torch.ops.aten.add.Tensor(mul_532, arg46_1);  mul_532 = arg46_1 = None
        view_486 = torch.ops.aten.view.default(add_391, [6272, 768]);  add_391 = None
        permute_252 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg48_1, view_486, permute_252);  arg48_1 = view_486 = permute_252 = None
        view_487 = torch.ops.aten.view.default(addmm_86, [8, 784, 2304]);  addmm_86 = None
        view_488 = torch.ops.aten.view.default(view_487, [8, 784, 3, 16, 48]);  view_487 = None
        permute_253 = torch.ops.aten.permute.default(view_488, [2, 0, 3, 4, 1]);  view_488 = None
        unbind_25 = torch.ops.aten.unbind.int(permute_253);  permute_253 = None
        getitem_245 = unbind_25[0]
        getitem_246 = unbind_25[1]
        getitem_247 = unbind_25[2];  unbind_25 = None
        pow_103 = torch.ops.aten.pow.Tensor_Scalar(getitem_245, 2.0)
        sum_76 = torch.ops.aten.sum.dim_IntList(pow_103, [-1], True);  pow_103 = None
        pow_104 = torch.ops.aten.pow.Tensor_Scalar(sum_76, 0.5);  sum_76 = None
        clamp_min_50 = torch.ops.aten.clamp_min.default(pow_104, 1e-12);  pow_104 = None
        expand_151 = torch.ops.aten.expand.default(clamp_min_50, [8, 16, 48, 784]);  clamp_min_50 = None
        div_87 = torch.ops.aten.div.Tensor(getitem_245, expand_151);  getitem_245 = expand_151 = None
        pow_105 = torch.ops.aten.pow.Tensor_Scalar(getitem_246, 2.0)
        sum_77 = torch.ops.aten.sum.dim_IntList(pow_105, [-1], True);  pow_105 = None
        pow_106 = torch.ops.aten.pow.Tensor_Scalar(sum_77, 0.5);  sum_77 = None
        clamp_min_51 = torch.ops.aten.clamp_min.default(pow_106, 1e-12);  pow_106 = None
        expand_152 = torch.ops.aten.expand.default(clamp_min_51, [8, 16, 48, 784]);  clamp_min_51 = None
        div_88 = torch.ops.aten.div.Tensor(getitem_246, expand_152);  getitem_246 = expand_152 = None
        permute_254 = torch.ops.aten.permute.default(div_88, [0, 1, 3, 2]);  div_88 = None
        expand_153 = torch.ops.aten.expand.default(div_87, [8, 16, 48, 784]);  div_87 = None
        clone_285 = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_489 = torch.ops.aten.view.default(clone_285, [128, 48, 784]);  clone_285 = None
        expand_154 = torch.ops.aten.expand.default(permute_254, [8, 16, 784, 48]);  permute_254 = None
        clone_286 = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_490 = torch.ops.aten.view.default(clone_286, [128, 784, 48]);  clone_286 = None
        bmm_50 = torch.ops.aten.bmm.default(view_489, view_490);  view_489 = view_490 = None
        view_491 = torch.ops.aten.view.default(bmm_50, [8, 16, 48, 48]);  bmm_50 = None
        scalar_tensor_default_22 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_22 = torch.ops.aten.ge.Scalar(arg49_1, 0)
        neg_default_22 = torch.ops.aten.neg.default(scalar_tensor_default_22)
        where_self_22 = torch.ops.aten.where.self(ge_scalar_22, scalar_tensor_default_22, neg_default_22);  ge_scalar_22 = scalar_tensor_default_22 = neg_default_22 = None
        mul_tensor_66 = torch.ops.aten.mul.Tensor(view_491, where_self_22);  view_491 = None
        amax_default_22 = torch.ops.aten.amax.default(mul_tensor_66, [-1], True)
        sub_tensor_22 = torch.ops.aten.sub.Tensor(mul_tensor_66, amax_default_22);  mul_tensor_66 = amax_default_22 = None
        mul_tensor_67 = torch.ops.aten.mul.Tensor(where_self_22, arg49_1);  where_self_22 = arg49_1 = None
        mul_tensor_68 = torch.ops.aten.mul.Tensor(sub_tensor_22, mul_tensor_67);  sub_tensor_22 = mul_tensor_67 = None
        exp_25 = torch.ops.aten.exp.default(mul_tensor_68);  mul_tensor_68 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_89 = torch.ops.aten.div.Tensor(exp_25, sum_78);  exp_25 = sum_78 = None
        expand_155 = torch.ops.aten.expand.default(div_89, [8, 16, 48, 48]);  div_89 = None
        view_492 = torch.ops.aten.view.default(expand_155, [128, 48, 48]);  expand_155 = None
        expand_156 = torch.ops.aten.expand.default(getitem_247, [8, 16, 48, 784]);  getitem_247 = None
        clone_288 = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_493 = torch.ops.aten.view.default(clone_288, [128, 48, 784]);  clone_288 = None
        bmm_51 = torch.ops.aten.bmm.default(view_492, view_493);  view_492 = view_493 = None
        view_494 = torch.ops.aten.view.default(bmm_51, [8, 16, 48, 784]);  bmm_51 = None
        permute_255 = torch.ops.aten.permute.default(view_494, [0, 3, 1, 2]);  view_494 = None
        view_495 = torch.ops.aten.view.default(permute_255, [8, 784, 768]);  permute_255 = None
        permute_256 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        clone_289 = torch.ops.aten.clone.default(view_495, memory_format = torch.contiguous_format);  view_495 = None
        view_496 = torch.ops.aten.view.default(clone_289, [6272, 768]);  clone_289 = None
        mm_27 = torch.ops.aten.mm.default(view_496, permute_256);  view_496 = permute_256 = None
        view_497 = torch.ops.aten.view.default(mm_27, [8, 784, 768]);  mm_27 = None
        add_392 = torch.ops.aten.add.Tensor(view_497, arg51_1);  view_497 = arg51_1 = None
        mul_534 = torch.ops.aten.mul.Tensor(arg44_1, add_392);  arg44_1 = add_392 = None
        add_393 = torch.ops.aten.add.Tensor(add_389, mul_534);  add_389 = mul_534 = None
        clone_291 = torch.ops.aten.clone.default(add_393, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_291, [2], correction = 0, keepdim = True)
        getitem_248 = var_mean_81[0]
        getitem_249 = var_mean_81[1];  var_mean_81 = None
        add_394 = torch.ops.aten.add.Tensor(getitem_248, 1e-06);  getitem_248 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
        sub_138 = torch.ops.aten.sub.Tensor(clone_291, getitem_249);  clone_291 = getitem_249 = None
        mul_535 = torch.ops.aten.mul.Tensor(sub_138, rsqrt_81);  sub_138 = rsqrt_81 = None
        mul_536 = torch.ops.aten.mul.Tensor(mul_535, arg53_1);  mul_535 = arg53_1 = None
        add_395 = torch.ops.aten.add.Tensor(mul_536, arg54_1);  mul_536 = arg54_1 = None
        permute_257 = torch.ops.aten.permute.default(add_395, [0, 2, 1]);  add_395 = None
        view_498 = torch.ops.aten.view.default(permute_257, [8, 768, 28, 28]);  permute_257 = None
        convolution_58 = torch.ops.aten.convolution.default(view_498, arg55_1, arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_498 = arg55_1 = arg56_1 = None
        mul_537 = torch.ops.aten.mul.Tensor(convolution_58, 0.5)
        mul_538 = torch.ops.aten.mul.Tensor(convolution_58, 0.7071067811865476);  convolution_58 = None
        erf_56 = torch.ops.aten.erf.default(mul_538);  mul_538 = None
        add_396 = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_539 = torch.ops.aten.mul.Tensor(mul_537, add_396);  mul_537 = add_396 = None
        add_397 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_397);  add_397 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_540 = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_139 = torch.ops.aten.sub.Tensor(mul_539, unsqueeze_265);  mul_539 = unsqueeze_265 = None
        mul_541 = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_267);  sub_139 = unsqueeze_267 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_542 = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_269);  mul_541 = unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_398 = torch.ops.aten.add.Tensor(mul_542, unsqueeze_271);  mul_542 = unsqueeze_271 = None
        convolution_59 = torch.ops.aten.convolution.default(add_398, arg61_1, arg62_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_398 = arg61_1 = arg62_1 = None
        view_499 = torch.ops.aten.view.default(convolution_59, [8, 768, 784]);  convolution_59 = None
        permute_258 = torch.ops.aten.permute.default(view_499, [0, 2, 1]);  view_499 = None
        mul_543 = torch.ops.aten.mul.Tensor(arg52_1, permute_258);  arg52_1 = permute_258 = None
        add_399 = torch.ops.aten.add.Tensor(add_393, mul_543);  add_393 = mul_543 = None
        clone_292 = torch.ops.aten.clone.default(add_399, memory_format = torch.contiguous_format)
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_292, [2], correction = 0, keepdim = True)
        getitem_250 = var_mean_82[0]
        getitem_251 = var_mean_82[1];  var_mean_82 = None
        add_400 = torch.ops.aten.add.Tensor(getitem_250, 1e-06);  getitem_250 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_400);  add_400 = None
        sub_140 = torch.ops.aten.sub.Tensor(clone_292, getitem_251);  clone_292 = getitem_251 = None
        mul_544 = torch.ops.aten.mul.Tensor(sub_140, rsqrt_82);  sub_140 = rsqrt_82 = None
        mul_545 = torch.ops.aten.mul.Tensor(mul_544, arg64_1);  mul_544 = arg64_1 = None
        add_401 = torch.ops.aten.add.Tensor(mul_545, arg65_1);  mul_545 = arg65_1 = None
        view_500 = torch.ops.aten.view.default(add_401, [6272, 768]);  add_401 = None
        permute_259 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg67_1, view_500, permute_259);  arg67_1 = view_500 = permute_259 = None
        view_501 = torch.ops.aten.view.default(addmm_87, [8, 784, 3072]);  addmm_87 = None
        mul_546 = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_547 = torch.ops.aten.mul.Tensor(view_501, 0.7071067811865476);  view_501 = None
        erf_57 = torch.ops.aten.erf.default(mul_547);  mul_547 = None
        add_402 = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_548 = torch.ops.aten.mul.Tensor(mul_546, add_402);  mul_546 = add_402 = None
        view_502 = torch.ops.aten.view.default(mul_548, [6272, 3072]);  mul_548 = None
        permute_260 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg69_1, view_502, permute_260);  arg69_1 = view_502 = permute_260 = None
        view_503 = torch.ops.aten.view.default(addmm_88, [8, 784, 768]);  addmm_88 = None
        mul_549 = torch.ops.aten.mul.Tensor(arg63_1, view_503);  arg63_1 = view_503 = None
        add_403 = torch.ops.aten.add.Tensor(add_399, mul_549);  add_399 = mul_549 = None
        clone_295 = torch.ops.aten.clone.default(add_403, memory_format = torch.contiguous_format)
        var_mean_83 = torch.ops.aten.var_mean.correction(clone_295, [2], correction = 0, keepdim = True)
        getitem_252 = var_mean_83[0]
        getitem_253 = var_mean_83[1];  var_mean_83 = None
        add_404 = torch.ops.aten.add.Tensor(getitem_252, 1e-06);  getitem_252 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        sub_141 = torch.ops.aten.sub.Tensor(clone_295, getitem_253);  clone_295 = getitem_253 = None
        mul_550 = torch.ops.aten.mul.Tensor(sub_141, rsqrt_83);  sub_141 = rsqrt_83 = None
        mul_551 = torch.ops.aten.mul.Tensor(mul_550, arg71_1);  mul_550 = arg71_1 = None
        add_405 = torch.ops.aten.add.Tensor(mul_551, arg72_1);  mul_551 = arg72_1 = None
        view_504 = torch.ops.aten.view.default(add_405, [6272, 768]);  add_405 = None
        permute_261 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg74_1, view_504, permute_261);  arg74_1 = view_504 = permute_261 = None
        view_505 = torch.ops.aten.view.default(addmm_89, [8, 784, 2304]);  addmm_89 = None
        view_506 = torch.ops.aten.view.default(view_505, [8, 784, 3, 16, 48]);  view_505 = None
        permute_262 = torch.ops.aten.permute.default(view_506, [2, 0, 3, 4, 1]);  view_506 = None
        unbind_26 = torch.ops.aten.unbind.int(permute_262);  permute_262 = None
        getitem_254 = unbind_26[0]
        getitem_255 = unbind_26[1]
        getitem_256 = unbind_26[2];  unbind_26 = None
        pow_107 = torch.ops.aten.pow.Tensor_Scalar(getitem_254, 2.0)
        sum_79 = torch.ops.aten.sum.dim_IntList(pow_107, [-1], True);  pow_107 = None
        pow_108 = torch.ops.aten.pow.Tensor_Scalar(sum_79, 0.5);  sum_79 = None
        clamp_min_52 = torch.ops.aten.clamp_min.default(pow_108, 1e-12);  pow_108 = None
        expand_157 = torch.ops.aten.expand.default(clamp_min_52, [8, 16, 48, 784]);  clamp_min_52 = None
        div_90 = torch.ops.aten.div.Tensor(getitem_254, expand_157);  getitem_254 = expand_157 = None
        pow_109 = torch.ops.aten.pow.Tensor_Scalar(getitem_255, 2.0)
        sum_80 = torch.ops.aten.sum.dim_IntList(pow_109, [-1], True);  pow_109 = None
        pow_110 = torch.ops.aten.pow.Tensor_Scalar(sum_80, 0.5);  sum_80 = None
        clamp_min_53 = torch.ops.aten.clamp_min.default(pow_110, 1e-12);  pow_110 = None
        expand_158 = torch.ops.aten.expand.default(clamp_min_53, [8, 16, 48, 784]);  clamp_min_53 = None
        div_91 = torch.ops.aten.div.Tensor(getitem_255, expand_158);  getitem_255 = expand_158 = None
        permute_263 = torch.ops.aten.permute.default(div_91, [0, 1, 3, 2]);  div_91 = None
        expand_159 = torch.ops.aten.expand.default(div_90, [8, 16, 48, 784]);  div_90 = None
        clone_296 = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_507 = torch.ops.aten.view.default(clone_296, [128, 48, 784]);  clone_296 = None
        expand_160 = torch.ops.aten.expand.default(permute_263, [8, 16, 784, 48]);  permute_263 = None
        clone_297 = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
        view_508 = torch.ops.aten.view.default(clone_297, [128, 784, 48]);  clone_297 = None
        bmm_52 = torch.ops.aten.bmm.default(view_507, view_508);  view_507 = view_508 = None
        view_509 = torch.ops.aten.view.default(bmm_52, [8, 16, 48, 48]);  bmm_52 = None
        scalar_tensor_default_21 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_21 = torch.ops.aten.ge.Scalar(arg75_1, 0)
        neg_default_21 = torch.ops.aten.neg.default(scalar_tensor_default_21)
        where_self_21 = torch.ops.aten.where.self(ge_scalar_21, scalar_tensor_default_21, neg_default_21);  ge_scalar_21 = scalar_tensor_default_21 = neg_default_21 = None
        mul_tensor_63 = torch.ops.aten.mul.Tensor(view_509, where_self_21);  view_509 = None
        amax_default_21 = torch.ops.aten.amax.default(mul_tensor_63, [-1], True)
        sub_tensor_21 = torch.ops.aten.sub.Tensor(mul_tensor_63, amax_default_21);  mul_tensor_63 = amax_default_21 = None
        mul_tensor_64 = torch.ops.aten.mul.Tensor(where_self_21, arg75_1);  where_self_21 = arg75_1 = None
        mul_tensor_65 = torch.ops.aten.mul.Tensor(sub_tensor_21, mul_tensor_64);  sub_tensor_21 = mul_tensor_64 = None
        exp_26 = torch.ops.aten.exp.default(mul_tensor_65);  mul_tensor_65 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_92 = torch.ops.aten.div.Tensor(exp_26, sum_81);  exp_26 = sum_81 = None
        expand_161 = torch.ops.aten.expand.default(div_92, [8, 16, 48, 48]);  div_92 = None
        view_510 = torch.ops.aten.view.default(expand_161, [128, 48, 48]);  expand_161 = None
        expand_162 = torch.ops.aten.expand.default(getitem_256, [8, 16, 48, 784]);  getitem_256 = None
        clone_299 = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_511 = torch.ops.aten.view.default(clone_299, [128, 48, 784]);  clone_299 = None
        bmm_53 = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
        view_512 = torch.ops.aten.view.default(bmm_53, [8, 16, 48, 784]);  bmm_53 = None
        permute_264 = torch.ops.aten.permute.default(view_512, [0, 3, 1, 2]);  view_512 = None
        view_513 = torch.ops.aten.view.default(permute_264, [8, 784, 768]);  permute_264 = None
        permute_265 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        clone_300 = torch.ops.aten.clone.default(view_513, memory_format = torch.contiguous_format);  view_513 = None
        view_514 = torch.ops.aten.view.default(clone_300, [6272, 768]);  clone_300 = None
        mm_28 = torch.ops.aten.mm.default(view_514, permute_265);  view_514 = permute_265 = None
        view_515 = torch.ops.aten.view.default(mm_28, [8, 784, 768]);  mm_28 = None
        add_406 = torch.ops.aten.add.Tensor(view_515, arg77_1);  view_515 = arg77_1 = None
        mul_553 = torch.ops.aten.mul.Tensor(arg70_1, add_406);  arg70_1 = add_406 = None
        add_407 = torch.ops.aten.add.Tensor(add_403, mul_553);  add_403 = mul_553 = None
        clone_302 = torch.ops.aten.clone.default(add_407, memory_format = torch.contiguous_format)
        var_mean_84 = torch.ops.aten.var_mean.correction(clone_302, [2], correction = 0, keepdim = True)
        getitem_257 = var_mean_84[0]
        getitem_258 = var_mean_84[1];  var_mean_84 = None
        add_408 = torch.ops.aten.add.Tensor(getitem_257, 1e-06);  getitem_257 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_408);  add_408 = None
        sub_143 = torch.ops.aten.sub.Tensor(clone_302, getitem_258);  clone_302 = getitem_258 = None
        mul_554 = torch.ops.aten.mul.Tensor(sub_143, rsqrt_84);  sub_143 = rsqrt_84 = None
        mul_555 = torch.ops.aten.mul.Tensor(mul_554, arg79_1);  mul_554 = arg79_1 = None
        add_409 = torch.ops.aten.add.Tensor(mul_555, arg80_1);  mul_555 = arg80_1 = None
        permute_266 = torch.ops.aten.permute.default(add_409, [0, 2, 1]);  add_409 = None
        view_516 = torch.ops.aten.view.default(permute_266, [8, 768, 28, 28]);  permute_266 = None
        convolution_60 = torch.ops.aten.convolution.default(view_516, arg81_1, arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_516 = arg81_1 = arg82_1 = None
        mul_556 = torch.ops.aten.mul.Tensor(convolution_60, 0.5)
        mul_557 = torch.ops.aten.mul.Tensor(convolution_60, 0.7071067811865476);  convolution_60 = None
        erf_58 = torch.ops.aten.erf.default(mul_557);  mul_557 = None
        add_410 = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_558 = torch.ops.aten.mul.Tensor(mul_556, add_410);  mul_556 = add_410 = None
        add_411 = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_411);  add_411 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_559 = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(mul_559, -1);  mul_559 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_144 = torch.ops.aten.sub.Tensor(mul_558, unsqueeze_273);  mul_558 = unsqueeze_273 = None
        mul_560 = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_275);  sub_144 = unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_561 = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_277);  mul_560 = unsqueeze_277 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_412 = torch.ops.aten.add.Tensor(mul_561, unsqueeze_279);  mul_561 = unsqueeze_279 = None
        convolution_61 = torch.ops.aten.convolution.default(add_412, arg87_1, arg88_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_412 = arg87_1 = arg88_1 = None
        view_517 = torch.ops.aten.view.default(convolution_61, [8, 768, 784]);  convolution_61 = None
        permute_267 = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None
        mul_562 = torch.ops.aten.mul.Tensor(arg78_1, permute_267);  arg78_1 = permute_267 = None
        add_413 = torch.ops.aten.add.Tensor(add_407, mul_562);  add_407 = mul_562 = None
        clone_303 = torch.ops.aten.clone.default(add_413, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_303, [2], correction = 0, keepdim = True)
        getitem_259 = var_mean_85[0]
        getitem_260 = var_mean_85[1];  var_mean_85 = None
        add_414 = torch.ops.aten.add.Tensor(getitem_259, 1e-06);  getitem_259 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
        sub_145 = torch.ops.aten.sub.Tensor(clone_303, getitem_260);  clone_303 = getitem_260 = None
        mul_563 = torch.ops.aten.mul.Tensor(sub_145, rsqrt_85);  sub_145 = rsqrt_85 = None
        mul_564 = torch.ops.aten.mul.Tensor(mul_563, arg90_1);  mul_563 = arg90_1 = None
        add_415 = torch.ops.aten.add.Tensor(mul_564, arg91_1);  mul_564 = arg91_1 = None
        view_518 = torch.ops.aten.view.default(add_415, [6272, 768]);  add_415 = None
        permute_268 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg93_1, view_518, permute_268);  arg93_1 = view_518 = permute_268 = None
        view_519 = torch.ops.aten.view.default(addmm_90, [8, 784, 3072]);  addmm_90 = None
        mul_565 = torch.ops.aten.mul.Tensor(view_519, 0.5)
        mul_566 = torch.ops.aten.mul.Tensor(view_519, 0.7071067811865476);  view_519 = None
        erf_59 = torch.ops.aten.erf.default(mul_566);  mul_566 = None
        add_416 = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_567 = torch.ops.aten.mul.Tensor(mul_565, add_416);  mul_565 = add_416 = None
        view_520 = torch.ops.aten.view.default(mul_567, [6272, 3072]);  mul_567 = None
        permute_269 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg95_1, view_520, permute_269);  arg95_1 = view_520 = permute_269 = None
        view_521 = torch.ops.aten.view.default(addmm_91, [8, 784, 768]);  addmm_91 = None
        mul_568 = torch.ops.aten.mul.Tensor(arg89_1, view_521);  arg89_1 = view_521 = None
        add_417 = torch.ops.aten.add.Tensor(add_413, mul_568);  add_413 = mul_568 = None
        clone_306 = torch.ops.aten.clone.default(add_417, memory_format = torch.contiguous_format)
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_306, [2], correction = 0, keepdim = True)
        getitem_261 = var_mean_86[0]
        getitem_262 = var_mean_86[1];  var_mean_86 = None
        add_418 = torch.ops.aten.add.Tensor(getitem_261, 1e-06);  getitem_261 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        sub_146 = torch.ops.aten.sub.Tensor(clone_306, getitem_262);  clone_306 = getitem_262 = None
        mul_569 = torch.ops.aten.mul.Tensor(sub_146, rsqrt_86);  sub_146 = rsqrt_86 = None
        mul_570 = torch.ops.aten.mul.Tensor(mul_569, arg97_1);  mul_569 = arg97_1 = None
        add_419 = torch.ops.aten.add.Tensor(mul_570, arg98_1);  mul_570 = arg98_1 = None
        view_522 = torch.ops.aten.view.default(add_419, [6272, 768]);  add_419 = None
        permute_270 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg100_1, view_522, permute_270);  arg100_1 = view_522 = permute_270 = None
        view_523 = torch.ops.aten.view.default(addmm_92, [8, 784, 2304]);  addmm_92 = None
        view_524 = torch.ops.aten.view.default(view_523, [8, 784, 3, 16, 48]);  view_523 = None
        permute_271 = torch.ops.aten.permute.default(view_524, [2, 0, 3, 4, 1]);  view_524 = None
        unbind_27 = torch.ops.aten.unbind.int(permute_271);  permute_271 = None
        getitem_263 = unbind_27[0]
        getitem_264 = unbind_27[1]
        getitem_265 = unbind_27[2];  unbind_27 = None
        pow_111 = torch.ops.aten.pow.Tensor_Scalar(getitem_263, 2.0)
        sum_82 = torch.ops.aten.sum.dim_IntList(pow_111, [-1], True);  pow_111 = None
        pow_112 = torch.ops.aten.pow.Tensor_Scalar(sum_82, 0.5);  sum_82 = None
        clamp_min_54 = torch.ops.aten.clamp_min.default(pow_112, 1e-12);  pow_112 = None
        expand_163 = torch.ops.aten.expand.default(clamp_min_54, [8, 16, 48, 784]);  clamp_min_54 = None
        div_93 = torch.ops.aten.div.Tensor(getitem_263, expand_163);  getitem_263 = expand_163 = None
        pow_113 = torch.ops.aten.pow.Tensor_Scalar(getitem_264, 2.0)
        sum_83 = torch.ops.aten.sum.dim_IntList(pow_113, [-1], True);  pow_113 = None
        pow_114 = torch.ops.aten.pow.Tensor_Scalar(sum_83, 0.5);  sum_83 = None
        clamp_min_55 = torch.ops.aten.clamp_min.default(pow_114, 1e-12);  pow_114 = None
        expand_164 = torch.ops.aten.expand.default(clamp_min_55, [8, 16, 48, 784]);  clamp_min_55 = None
        div_94 = torch.ops.aten.div.Tensor(getitem_264, expand_164);  getitem_264 = expand_164 = None
        permute_272 = torch.ops.aten.permute.default(div_94, [0, 1, 3, 2]);  div_94 = None
        expand_165 = torch.ops.aten.expand.default(div_93, [8, 16, 48, 784]);  div_93 = None
        clone_307 = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_525 = torch.ops.aten.view.default(clone_307, [128, 48, 784]);  clone_307 = None
        expand_166 = torch.ops.aten.expand.default(permute_272, [8, 16, 784, 48]);  permute_272 = None
        clone_308 = torch.ops.aten.clone.default(expand_166, memory_format = torch.contiguous_format);  expand_166 = None
        view_526 = torch.ops.aten.view.default(clone_308, [128, 784, 48]);  clone_308 = None
        bmm_54 = torch.ops.aten.bmm.default(view_525, view_526);  view_525 = view_526 = None
        view_527 = torch.ops.aten.view.default(bmm_54, [8, 16, 48, 48]);  bmm_54 = None
        scalar_tensor_default_20 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_20 = torch.ops.aten.ge.Scalar(arg101_1, 0)
        neg_default_20 = torch.ops.aten.neg.default(scalar_tensor_default_20)
        where_self_20 = torch.ops.aten.where.self(ge_scalar_20, scalar_tensor_default_20, neg_default_20);  ge_scalar_20 = scalar_tensor_default_20 = neg_default_20 = None
        mul_tensor_60 = torch.ops.aten.mul.Tensor(view_527, where_self_20);  view_527 = None
        amax_default_20 = torch.ops.aten.amax.default(mul_tensor_60, [-1], True)
        sub_tensor_20 = torch.ops.aten.sub.Tensor(mul_tensor_60, amax_default_20);  mul_tensor_60 = amax_default_20 = None
        mul_tensor_61 = torch.ops.aten.mul.Tensor(where_self_20, arg101_1);  where_self_20 = arg101_1 = None
        mul_tensor_62 = torch.ops.aten.mul.Tensor(sub_tensor_20, mul_tensor_61);  sub_tensor_20 = mul_tensor_61 = None
        exp_27 = torch.ops.aten.exp.default(mul_tensor_62);  mul_tensor_62 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_95 = torch.ops.aten.div.Tensor(exp_27, sum_84);  exp_27 = sum_84 = None
        expand_167 = torch.ops.aten.expand.default(div_95, [8, 16, 48, 48]);  div_95 = None
        view_528 = torch.ops.aten.view.default(expand_167, [128, 48, 48]);  expand_167 = None
        expand_168 = torch.ops.aten.expand.default(getitem_265, [8, 16, 48, 784]);  getitem_265 = None
        clone_310 = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
        view_529 = torch.ops.aten.view.default(clone_310, [128, 48, 784]);  clone_310 = None
        bmm_55 = torch.ops.aten.bmm.default(view_528, view_529);  view_528 = view_529 = None
        view_530 = torch.ops.aten.view.default(bmm_55, [8, 16, 48, 784]);  bmm_55 = None
        permute_273 = torch.ops.aten.permute.default(view_530, [0, 3, 1, 2]);  view_530 = None
        view_531 = torch.ops.aten.view.default(permute_273, [8, 784, 768]);  permute_273 = None
        permute_274 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        clone_311 = torch.ops.aten.clone.default(view_531, memory_format = torch.contiguous_format);  view_531 = None
        view_532 = torch.ops.aten.view.default(clone_311, [6272, 768]);  clone_311 = None
        mm_29 = torch.ops.aten.mm.default(view_532, permute_274);  view_532 = permute_274 = None
        view_533 = torch.ops.aten.view.default(mm_29, [8, 784, 768]);  mm_29 = None
        add_420 = torch.ops.aten.add.Tensor(view_533, arg103_1);  view_533 = arg103_1 = None
        mul_572 = torch.ops.aten.mul.Tensor(arg96_1, add_420);  arg96_1 = add_420 = None
        add_421 = torch.ops.aten.add.Tensor(add_417, mul_572);  add_417 = mul_572 = None
        clone_313 = torch.ops.aten.clone.default(add_421, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_313, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_87[0]
        getitem_267 = var_mean_87[1];  var_mean_87 = None
        add_422 = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        sub_148 = torch.ops.aten.sub.Tensor(clone_313, getitem_267);  clone_313 = getitem_267 = None
        mul_573 = torch.ops.aten.mul.Tensor(sub_148, rsqrt_87);  sub_148 = rsqrt_87 = None
        mul_574 = torch.ops.aten.mul.Tensor(mul_573, arg105_1);  mul_573 = arg105_1 = None
        add_423 = torch.ops.aten.add.Tensor(mul_574, arg106_1);  mul_574 = arg106_1 = None
        permute_275 = torch.ops.aten.permute.default(add_423, [0, 2, 1]);  add_423 = None
        view_534 = torch.ops.aten.view.default(permute_275, [8, 768, 28, 28]);  permute_275 = None
        convolution_62 = torch.ops.aten.convolution.default(view_534, arg107_1, arg108_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_534 = arg107_1 = arg108_1 = None
        mul_575 = torch.ops.aten.mul.Tensor(convolution_62, 0.5)
        mul_576 = torch.ops.aten.mul.Tensor(convolution_62, 0.7071067811865476);  convolution_62 = None
        erf_60 = torch.ops.aten.erf.default(mul_576);  mul_576 = None
        add_424 = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_577 = torch.ops.aten.mul.Tensor(mul_575, add_424);  mul_575 = add_424 = None
        add_425 = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_425);  add_425 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_578 = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_578, -1);  mul_578 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_149 = torch.ops.aten.sub.Tensor(mul_577, unsqueeze_281);  mul_577 = unsqueeze_281 = None
        mul_579 = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_283);  sub_149 = unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_580 = torch.ops.aten.mul.Tensor(mul_579, unsqueeze_285);  mul_579 = unsqueeze_285 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_426 = torch.ops.aten.add.Tensor(mul_580, unsqueeze_287);  mul_580 = unsqueeze_287 = None
        convolution_63 = torch.ops.aten.convolution.default(add_426, arg113_1, arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_426 = arg113_1 = arg114_1 = None
        view_535 = torch.ops.aten.view.default(convolution_63, [8, 768, 784]);  convolution_63 = None
        permute_276 = torch.ops.aten.permute.default(view_535, [0, 2, 1]);  view_535 = None
        mul_581 = torch.ops.aten.mul.Tensor(arg104_1, permute_276);  arg104_1 = permute_276 = None
        add_427 = torch.ops.aten.add.Tensor(add_421, mul_581);  add_421 = mul_581 = None
        clone_314 = torch.ops.aten.clone.default(add_427, memory_format = torch.contiguous_format)
        var_mean_88 = torch.ops.aten.var_mean.correction(clone_314, [2], correction = 0, keepdim = True)
        getitem_268 = var_mean_88[0]
        getitem_269 = var_mean_88[1];  var_mean_88 = None
        add_428 = torch.ops.aten.add.Tensor(getitem_268, 1e-06);  getitem_268 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_428);  add_428 = None
        sub_150 = torch.ops.aten.sub.Tensor(clone_314, getitem_269);  clone_314 = getitem_269 = None
        mul_582 = torch.ops.aten.mul.Tensor(sub_150, rsqrt_88);  sub_150 = rsqrt_88 = None
        mul_583 = torch.ops.aten.mul.Tensor(mul_582, arg116_1);  mul_582 = arg116_1 = None
        add_429 = torch.ops.aten.add.Tensor(mul_583, arg117_1);  mul_583 = arg117_1 = None
        view_536 = torch.ops.aten.view.default(add_429, [6272, 768]);  add_429 = None
        permute_277 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg119_1, view_536, permute_277);  arg119_1 = view_536 = permute_277 = None
        view_537 = torch.ops.aten.view.default(addmm_93, [8, 784, 3072]);  addmm_93 = None
        mul_584 = torch.ops.aten.mul.Tensor(view_537, 0.5)
        mul_585 = torch.ops.aten.mul.Tensor(view_537, 0.7071067811865476);  view_537 = None
        erf_61 = torch.ops.aten.erf.default(mul_585);  mul_585 = None
        add_430 = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_586 = torch.ops.aten.mul.Tensor(mul_584, add_430);  mul_584 = add_430 = None
        view_538 = torch.ops.aten.view.default(mul_586, [6272, 3072]);  mul_586 = None
        permute_278 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg121_1, view_538, permute_278);  arg121_1 = view_538 = permute_278 = None
        view_539 = torch.ops.aten.view.default(addmm_94, [8, 784, 768]);  addmm_94 = None
        mul_587 = torch.ops.aten.mul.Tensor(arg115_1, view_539);  arg115_1 = view_539 = None
        add_431 = torch.ops.aten.add.Tensor(add_427, mul_587);  add_427 = mul_587 = None
        clone_317 = torch.ops.aten.clone.default(add_431, memory_format = torch.contiguous_format)
        var_mean_89 = torch.ops.aten.var_mean.correction(clone_317, [2], correction = 0, keepdim = True)
        getitem_270 = var_mean_89[0]
        getitem_271 = var_mean_89[1];  var_mean_89 = None
        add_432 = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_432);  add_432 = None
        sub_151 = torch.ops.aten.sub.Tensor(clone_317, getitem_271);  clone_317 = getitem_271 = None
        mul_588 = torch.ops.aten.mul.Tensor(sub_151, rsqrt_89);  sub_151 = rsqrt_89 = None
        mul_589 = torch.ops.aten.mul.Tensor(mul_588, arg123_1);  mul_588 = arg123_1 = None
        add_433 = torch.ops.aten.add.Tensor(mul_589, arg124_1);  mul_589 = arg124_1 = None
        view_540 = torch.ops.aten.view.default(add_433, [6272, 768]);  add_433 = None
        permute_279 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg126_1, view_540, permute_279);  arg126_1 = view_540 = permute_279 = None
        view_541 = torch.ops.aten.view.default(addmm_95, [8, 784, 2304]);  addmm_95 = None
        view_542 = torch.ops.aten.view.default(view_541, [8, 784, 3, 16, 48]);  view_541 = None
        permute_280 = torch.ops.aten.permute.default(view_542, [2, 0, 3, 4, 1]);  view_542 = None
        unbind_28 = torch.ops.aten.unbind.int(permute_280);  permute_280 = None
        getitem_272 = unbind_28[0]
        getitem_273 = unbind_28[1]
        getitem_274 = unbind_28[2];  unbind_28 = None
        pow_115 = torch.ops.aten.pow.Tensor_Scalar(getitem_272, 2.0)
        sum_85 = torch.ops.aten.sum.dim_IntList(pow_115, [-1], True);  pow_115 = None
        pow_116 = torch.ops.aten.pow.Tensor_Scalar(sum_85, 0.5);  sum_85 = None
        clamp_min_56 = torch.ops.aten.clamp_min.default(pow_116, 1e-12);  pow_116 = None
        expand_169 = torch.ops.aten.expand.default(clamp_min_56, [8, 16, 48, 784]);  clamp_min_56 = None
        div_96 = torch.ops.aten.div.Tensor(getitem_272, expand_169);  getitem_272 = expand_169 = None
        pow_117 = torch.ops.aten.pow.Tensor_Scalar(getitem_273, 2.0)
        sum_86 = torch.ops.aten.sum.dim_IntList(pow_117, [-1], True);  pow_117 = None
        pow_118 = torch.ops.aten.pow.Tensor_Scalar(sum_86, 0.5);  sum_86 = None
        clamp_min_57 = torch.ops.aten.clamp_min.default(pow_118, 1e-12);  pow_118 = None
        expand_170 = torch.ops.aten.expand.default(clamp_min_57, [8, 16, 48, 784]);  clamp_min_57 = None
        div_97 = torch.ops.aten.div.Tensor(getitem_273, expand_170);  getitem_273 = expand_170 = None
        permute_281 = torch.ops.aten.permute.default(div_97, [0, 1, 3, 2]);  div_97 = None
        expand_171 = torch.ops.aten.expand.default(div_96, [8, 16, 48, 784]);  div_96 = None
        clone_318 = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_543 = torch.ops.aten.view.default(clone_318, [128, 48, 784]);  clone_318 = None
        expand_172 = torch.ops.aten.expand.default(permute_281, [8, 16, 784, 48]);  permute_281 = None
        clone_319 = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_544 = torch.ops.aten.view.default(clone_319, [128, 784, 48]);  clone_319 = None
        bmm_56 = torch.ops.aten.bmm.default(view_543, view_544);  view_543 = view_544 = None
        view_545 = torch.ops.aten.view.default(bmm_56, [8, 16, 48, 48]);  bmm_56 = None
        scalar_tensor_default_19 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_19 = torch.ops.aten.ge.Scalar(arg127_1, 0)
        neg_default_19 = torch.ops.aten.neg.default(scalar_tensor_default_19)
        where_self_19 = torch.ops.aten.where.self(ge_scalar_19, scalar_tensor_default_19, neg_default_19);  ge_scalar_19 = scalar_tensor_default_19 = neg_default_19 = None
        mul_tensor_57 = torch.ops.aten.mul.Tensor(view_545, where_self_19);  view_545 = None
        amax_default_19 = torch.ops.aten.amax.default(mul_tensor_57, [-1], True)
        sub_tensor_19 = torch.ops.aten.sub.Tensor(mul_tensor_57, amax_default_19);  mul_tensor_57 = amax_default_19 = None
        mul_tensor_58 = torch.ops.aten.mul.Tensor(where_self_19, arg127_1);  where_self_19 = arg127_1 = None
        mul_tensor_59 = torch.ops.aten.mul.Tensor(sub_tensor_19, mul_tensor_58);  sub_tensor_19 = mul_tensor_58 = None
        exp_28 = torch.ops.aten.exp.default(mul_tensor_59);  mul_tensor_59 = None
        sum_87 = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_98 = torch.ops.aten.div.Tensor(exp_28, sum_87);  exp_28 = sum_87 = None
        expand_173 = torch.ops.aten.expand.default(div_98, [8, 16, 48, 48]);  div_98 = None
        view_546 = torch.ops.aten.view.default(expand_173, [128, 48, 48]);  expand_173 = None
        expand_174 = torch.ops.aten.expand.default(getitem_274, [8, 16, 48, 784]);  getitem_274 = None
        clone_321 = torch.ops.aten.clone.default(expand_174, memory_format = torch.contiguous_format);  expand_174 = None
        view_547 = torch.ops.aten.view.default(clone_321, [128, 48, 784]);  clone_321 = None
        bmm_57 = torch.ops.aten.bmm.default(view_546, view_547);  view_546 = view_547 = None
        view_548 = torch.ops.aten.view.default(bmm_57, [8, 16, 48, 784]);  bmm_57 = None
        permute_282 = torch.ops.aten.permute.default(view_548, [0, 3, 1, 2]);  view_548 = None
        view_549 = torch.ops.aten.view.default(permute_282, [8, 784, 768]);  permute_282 = None
        permute_283 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        clone_322 = torch.ops.aten.clone.default(view_549, memory_format = torch.contiguous_format);  view_549 = None
        view_550 = torch.ops.aten.view.default(clone_322, [6272, 768]);  clone_322 = None
        mm_30 = torch.ops.aten.mm.default(view_550, permute_283);  view_550 = permute_283 = None
        view_551 = torch.ops.aten.view.default(mm_30, [8, 784, 768]);  mm_30 = None
        add_434 = torch.ops.aten.add.Tensor(view_551, arg129_1);  view_551 = arg129_1 = None
        mul_591 = torch.ops.aten.mul.Tensor(arg122_1, add_434);  arg122_1 = add_434 = None
        add_435 = torch.ops.aten.add.Tensor(add_431, mul_591);  add_431 = mul_591 = None
        clone_324 = torch.ops.aten.clone.default(add_435, memory_format = torch.contiguous_format)
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_324, [2], correction = 0, keepdim = True)
        getitem_275 = var_mean_90[0]
        getitem_276 = var_mean_90[1];  var_mean_90 = None
        add_436 = torch.ops.aten.add.Tensor(getitem_275, 1e-06);  getitem_275 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_436);  add_436 = None
        sub_153 = torch.ops.aten.sub.Tensor(clone_324, getitem_276);  clone_324 = getitem_276 = None
        mul_592 = torch.ops.aten.mul.Tensor(sub_153, rsqrt_90);  sub_153 = rsqrt_90 = None
        mul_593 = torch.ops.aten.mul.Tensor(mul_592, arg131_1);  mul_592 = arg131_1 = None
        add_437 = torch.ops.aten.add.Tensor(mul_593, arg132_1);  mul_593 = arg132_1 = None
        permute_284 = torch.ops.aten.permute.default(add_437, [0, 2, 1]);  add_437 = None
        view_552 = torch.ops.aten.view.default(permute_284, [8, 768, 28, 28]);  permute_284 = None
        convolution_64 = torch.ops.aten.convolution.default(view_552, arg133_1, arg134_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_552 = arg133_1 = arg134_1 = None
        mul_594 = torch.ops.aten.mul.Tensor(convolution_64, 0.5)
        mul_595 = torch.ops.aten.mul.Tensor(convolution_64, 0.7071067811865476);  convolution_64 = None
        erf_62 = torch.ops.aten.erf.default(mul_595);  mul_595 = None
        add_438 = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_596 = torch.ops.aten.mul.Tensor(mul_594, add_438);  mul_594 = add_438 = None
        add_439 = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_439);  add_439 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_597 = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_597, -1);  mul_597 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_154 = torch.ops.aten.sub.Tensor(mul_596, unsqueeze_289);  mul_596 = unsqueeze_289 = None
        mul_598 = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_291);  sub_154 = unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_599 = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_293);  mul_598 = unsqueeze_293 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_440 = torch.ops.aten.add.Tensor(mul_599, unsqueeze_295);  mul_599 = unsqueeze_295 = None
        convolution_65 = torch.ops.aten.convolution.default(add_440, arg139_1, arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_440 = arg139_1 = arg140_1 = None
        view_553 = torch.ops.aten.view.default(convolution_65, [8, 768, 784]);  convolution_65 = None
        permute_285 = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None
        mul_600 = torch.ops.aten.mul.Tensor(arg130_1, permute_285);  arg130_1 = permute_285 = None
        add_441 = torch.ops.aten.add.Tensor(add_435, mul_600);  add_435 = mul_600 = None
        clone_325 = torch.ops.aten.clone.default(add_441, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_325, [2], correction = 0, keepdim = True)
        getitem_277 = var_mean_91[0]
        getitem_278 = var_mean_91[1];  var_mean_91 = None
        add_442 = torch.ops.aten.add.Tensor(getitem_277, 1e-06);  getitem_277 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_442);  add_442 = None
        sub_155 = torch.ops.aten.sub.Tensor(clone_325, getitem_278);  clone_325 = getitem_278 = None
        mul_601 = torch.ops.aten.mul.Tensor(sub_155, rsqrt_91);  sub_155 = rsqrt_91 = None
        mul_602 = torch.ops.aten.mul.Tensor(mul_601, arg142_1);  mul_601 = arg142_1 = None
        add_443 = torch.ops.aten.add.Tensor(mul_602, arg143_1);  mul_602 = arg143_1 = None
        view_554 = torch.ops.aten.view.default(add_443, [6272, 768]);  add_443 = None
        permute_286 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg145_1, view_554, permute_286);  arg145_1 = view_554 = permute_286 = None
        view_555 = torch.ops.aten.view.default(addmm_96, [8, 784, 3072]);  addmm_96 = None
        mul_603 = torch.ops.aten.mul.Tensor(view_555, 0.5)
        mul_604 = torch.ops.aten.mul.Tensor(view_555, 0.7071067811865476);  view_555 = None
        erf_63 = torch.ops.aten.erf.default(mul_604);  mul_604 = None
        add_444 = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_605 = torch.ops.aten.mul.Tensor(mul_603, add_444);  mul_603 = add_444 = None
        view_556 = torch.ops.aten.view.default(mul_605, [6272, 3072]);  mul_605 = None
        permute_287 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg147_1, view_556, permute_287);  arg147_1 = view_556 = permute_287 = None
        view_557 = torch.ops.aten.view.default(addmm_97, [8, 784, 768]);  addmm_97 = None
        mul_606 = torch.ops.aten.mul.Tensor(arg141_1, view_557);  arg141_1 = view_557 = None
        add_445 = torch.ops.aten.add.Tensor(add_441, mul_606);  add_441 = mul_606 = None
        clone_328 = torch.ops.aten.clone.default(add_445, memory_format = torch.contiguous_format)
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_328, [2], correction = 0, keepdim = True)
        getitem_279 = var_mean_92[0]
        getitem_280 = var_mean_92[1];  var_mean_92 = None
        add_446 = torch.ops.aten.add.Tensor(getitem_279, 1e-06);  getitem_279 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
        sub_156 = torch.ops.aten.sub.Tensor(clone_328, getitem_280);  clone_328 = getitem_280 = None
        mul_607 = torch.ops.aten.mul.Tensor(sub_156, rsqrt_92);  sub_156 = rsqrt_92 = None
        mul_608 = torch.ops.aten.mul.Tensor(mul_607, arg149_1);  mul_607 = arg149_1 = None
        add_447 = torch.ops.aten.add.Tensor(mul_608, arg150_1);  mul_608 = arg150_1 = None
        view_558 = torch.ops.aten.view.default(add_447, [6272, 768]);  add_447 = None
        permute_288 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg152_1, view_558, permute_288);  arg152_1 = view_558 = permute_288 = None
        view_559 = torch.ops.aten.view.default(addmm_98, [8, 784, 2304]);  addmm_98 = None
        view_560 = torch.ops.aten.view.default(view_559, [8, 784, 3, 16, 48]);  view_559 = None
        permute_289 = torch.ops.aten.permute.default(view_560, [2, 0, 3, 4, 1]);  view_560 = None
        unbind_29 = torch.ops.aten.unbind.int(permute_289);  permute_289 = None
        getitem_281 = unbind_29[0]
        getitem_282 = unbind_29[1]
        getitem_283 = unbind_29[2];  unbind_29 = None
        pow_119 = torch.ops.aten.pow.Tensor_Scalar(getitem_281, 2.0)
        sum_88 = torch.ops.aten.sum.dim_IntList(pow_119, [-1], True);  pow_119 = None
        pow_120 = torch.ops.aten.pow.Tensor_Scalar(sum_88, 0.5);  sum_88 = None
        clamp_min_58 = torch.ops.aten.clamp_min.default(pow_120, 1e-12);  pow_120 = None
        expand_175 = torch.ops.aten.expand.default(clamp_min_58, [8, 16, 48, 784]);  clamp_min_58 = None
        div_99 = torch.ops.aten.div.Tensor(getitem_281, expand_175);  getitem_281 = expand_175 = None
        pow_121 = torch.ops.aten.pow.Tensor_Scalar(getitem_282, 2.0)
        sum_89 = torch.ops.aten.sum.dim_IntList(pow_121, [-1], True);  pow_121 = None
        pow_122 = torch.ops.aten.pow.Tensor_Scalar(sum_89, 0.5);  sum_89 = None
        clamp_min_59 = torch.ops.aten.clamp_min.default(pow_122, 1e-12);  pow_122 = None
        expand_176 = torch.ops.aten.expand.default(clamp_min_59, [8, 16, 48, 784]);  clamp_min_59 = None
        div_100 = torch.ops.aten.div.Tensor(getitem_282, expand_176);  getitem_282 = expand_176 = None
        permute_290 = torch.ops.aten.permute.default(div_100, [0, 1, 3, 2]);  div_100 = None
        expand_177 = torch.ops.aten.expand.default(div_99, [8, 16, 48, 784]);  div_99 = None
        clone_329 = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_561 = torch.ops.aten.view.default(clone_329, [128, 48, 784]);  clone_329 = None
        expand_178 = torch.ops.aten.expand.default(permute_290, [8, 16, 784, 48]);  permute_290 = None
        clone_330 = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_562 = torch.ops.aten.view.default(clone_330, [128, 784, 48]);  clone_330 = None
        bmm_58 = torch.ops.aten.bmm.default(view_561, view_562);  view_561 = view_562 = None
        view_563 = torch.ops.aten.view.default(bmm_58, [8, 16, 48, 48]);  bmm_58 = None
        scalar_tensor_default_18 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_18 = torch.ops.aten.ge.Scalar(arg153_1, 0)
        neg_default_18 = torch.ops.aten.neg.default(scalar_tensor_default_18)
        where_self_18 = torch.ops.aten.where.self(ge_scalar_18, scalar_tensor_default_18, neg_default_18);  ge_scalar_18 = scalar_tensor_default_18 = neg_default_18 = None
        mul_tensor_54 = torch.ops.aten.mul.Tensor(view_563, where_self_18);  view_563 = None
        amax_default_18 = torch.ops.aten.amax.default(mul_tensor_54, [-1], True)
        sub_tensor_18 = torch.ops.aten.sub.Tensor(mul_tensor_54, amax_default_18);  mul_tensor_54 = amax_default_18 = None
        mul_tensor_55 = torch.ops.aten.mul.Tensor(where_self_18, arg153_1);  where_self_18 = arg153_1 = None
        mul_tensor_56 = torch.ops.aten.mul.Tensor(sub_tensor_18, mul_tensor_55);  sub_tensor_18 = mul_tensor_55 = None
        exp_29 = torch.ops.aten.exp.default(mul_tensor_56);  mul_tensor_56 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_101 = torch.ops.aten.div.Tensor(exp_29, sum_90);  exp_29 = sum_90 = None
        expand_179 = torch.ops.aten.expand.default(div_101, [8, 16, 48, 48]);  div_101 = None
        view_564 = torch.ops.aten.view.default(expand_179, [128, 48, 48]);  expand_179 = None
        expand_180 = torch.ops.aten.expand.default(getitem_283, [8, 16, 48, 784]);  getitem_283 = None
        clone_332 = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_565 = torch.ops.aten.view.default(clone_332, [128, 48, 784]);  clone_332 = None
        bmm_59 = torch.ops.aten.bmm.default(view_564, view_565);  view_564 = view_565 = None
        view_566 = torch.ops.aten.view.default(bmm_59, [8, 16, 48, 784]);  bmm_59 = None
        permute_291 = torch.ops.aten.permute.default(view_566, [0, 3, 1, 2]);  view_566 = None
        view_567 = torch.ops.aten.view.default(permute_291, [8, 784, 768]);  permute_291 = None
        permute_292 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        clone_333 = torch.ops.aten.clone.default(view_567, memory_format = torch.contiguous_format);  view_567 = None
        view_568 = torch.ops.aten.view.default(clone_333, [6272, 768]);  clone_333 = None
        mm_31 = torch.ops.aten.mm.default(view_568, permute_292);  view_568 = permute_292 = None
        view_569 = torch.ops.aten.view.default(mm_31, [8, 784, 768]);  mm_31 = None
        add_448 = torch.ops.aten.add.Tensor(view_569, arg155_1);  view_569 = arg155_1 = None
        mul_610 = torch.ops.aten.mul.Tensor(arg148_1, add_448);  arg148_1 = add_448 = None
        add_449 = torch.ops.aten.add.Tensor(add_445, mul_610);  add_445 = mul_610 = None
        clone_335 = torch.ops.aten.clone.default(add_449, memory_format = torch.contiguous_format)
        var_mean_93 = torch.ops.aten.var_mean.correction(clone_335, [2], correction = 0, keepdim = True)
        getitem_284 = var_mean_93[0]
        getitem_285 = var_mean_93[1];  var_mean_93 = None
        add_450 = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_450);  add_450 = None
        sub_158 = torch.ops.aten.sub.Tensor(clone_335, getitem_285);  clone_335 = getitem_285 = None
        mul_611 = torch.ops.aten.mul.Tensor(sub_158, rsqrt_93);  sub_158 = rsqrt_93 = None
        mul_612 = torch.ops.aten.mul.Tensor(mul_611, arg157_1);  mul_611 = arg157_1 = None
        add_451 = torch.ops.aten.add.Tensor(mul_612, arg158_1);  mul_612 = arg158_1 = None
        permute_293 = torch.ops.aten.permute.default(add_451, [0, 2, 1]);  add_451 = None
        view_570 = torch.ops.aten.view.default(permute_293, [8, 768, 28, 28]);  permute_293 = None
        convolution_66 = torch.ops.aten.convolution.default(view_570, arg159_1, arg160_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_570 = arg159_1 = arg160_1 = None
        mul_613 = torch.ops.aten.mul.Tensor(convolution_66, 0.5)
        mul_614 = torch.ops.aten.mul.Tensor(convolution_66, 0.7071067811865476);  convolution_66 = None
        erf_64 = torch.ops.aten.erf.default(mul_614);  mul_614 = None
        add_452 = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_615 = torch.ops.aten.mul.Tensor(mul_613, add_452);  mul_613 = add_452 = None
        add_453 = torch.ops.aten.add.Tensor(arg162_1, 1e-05);  arg162_1 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_453);  add_453 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_616 = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(mul_616, -1);  mul_616 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_159 = torch.ops.aten.sub.Tensor(mul_615, unsqueeze_297);  mul_615 = unsqueeze_297 = None
        mul_617 = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_299);  sub_159 = unsqueeze_299 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_618 = torch.ops.aten.mul.Tensor(mul_617, unsqueeze_301);  mul_617 = unsqueeze_301 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_454 = torch.ops.aten.add.Tensor(mul_618, unsqueeze_303);  mul_618 = unsqueeze_303 = None
        convolution_67 = torch.ops.aten.convolution.default(add_454, arg165_1, arg166_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_454 = arg165_1 = arg166_1 = None
        view_571 = torch.ops.aten.view.default(convolution_67, [8, 768, 784]);  convolution_67 = None
        permute_294 = torch.ops.aten.permute.default(view_571, [0, 2, 1]);  view_571 = None
        mul_619 = torch.ops.aten.mul.Tensor(arg156_1, permute_294);  arg156_1 = permute_294 = None
        add_455 = torch.ops.aten.add.Tensor(add_449, mul_619);  add_449 = mul_619 = None
        clone_336 = torch.ops.aten.clone.default(add_455, memory_format = torch.contiguous_format)
        var_mean_94 = torch.ops.aten.var_mean.correction(clone_336, [2], correction = 0, keepdim = True)
        getitem_286 = var_mean_94[0]
        getitem_287 = var_mean_94[1];  var_mean_94 = None
        add_456 = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_456);  add_456 = None
        sub_160 = torch.ops.aten.sub.Tensor(clone_336, getitem_287);  clone_336 = getitem_287 = None
        mul_620 = torch.ops.aten.mul.Tensor(sub_160, rsqrt_94);  sub_160 = rsqrt_94 = None
        mul_621 = torch.ops.aten.mul.Tensor(mul_620, arg168_1);  mul_620 = arg168_1 = None
        add_457 = torch.ops.aten.add.Tensor(mul_621, arg169_1);  mul_621 = arg169_1 = None
        view_572 = torch.ops.aten.view.default(add_457, [6272, 768]);  add_457 = None
        permute_295 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg171_1, view_572, permute_295);  arg171_1 = view_572 = permute_295 = None
        view_573 = torch.ops.aten.view.default(addmm_99, [8, 784, 3072]);  addmm_99 = None
        mul_622 = torch.ops.aten.mul.Tensor(view_573, 0.5)
        mul_623 = torch.ops.aten.mul.Tensor(view_573, 0.7071067811865476);  view_573 = None
        erf_65 = torch.ops.aten.erf.default(mul_623);  mul_623 = None
        add_458 = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_624 = torch.ops.aten.mul.Tensor(mul_622, add_458);  mul_622 = add_458 = None
        view_574 = torch.ops.aten.view.default(mul_624, [6272, 3072]);  mul_624 = None
        permute_296 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg173_1, view_574, permute_296);  arg173_1 = view_574 = permute_296 = None
        view_575 = torch.ops.aten.view.default(addmm_100, [8, 784, 768]);  addmm_100 = None
        mul_625 = torch.ops.aten.mul.Tensor(arg167_1, view_575);  arg167_1 = view_575 = None
        add_459 = torch.ops.aten.add.Tensor(add_455, mul_625);  add_455 = mul_625 = None
        clone_339 = torch.ops.aten.clone.default(add_459, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_339, [2], correction = 0, keepdim = True)
        getitem_288 = var_mean_95[0]
        getitem_289 = var_mean_95[1];  var_mean_95 = None
        add_460 = torch.ops.aten.add.Tensor(getitem_288, 1e-06);  getitem_288 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
        sub_161 = torch.ops.aten.sub.Tensor(clone_339, getitem_289);  clone_339 = getitem_289 = None
        mul_626 = torch.ops.aten.mul.Tensor(sub_161, rsqrt_95);  sub_161 = rsqrt_95 = None
        mul_627 = torch.ops.aten.mul.Tensor(mul_626, arg175_1);  mul_626 = arg175_1 = None
        add_461 = torch.ops.aten.add.Tensor(mul_627, arg176_1);  mul_627 = arg176_1 = None
        view_576 = torch.ops.aten.view.default(add_461, [6272, 768]);  add_461 = None
        permute_297 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg178_1, view_576, permute_297);  arg178_1 = view_576 = permute_297 = None
        view_577 = torch.ops.aten.view.default(addmm_101, [8, 784, 2304]);  addmm_101 = None
        view_578 = torch.ops.aten.view.default(view_577, [8, 784, 3, 16, 48]);  view_577 = None
        permute_298 = torch.ops.aten.permute.default(view_578, [2, 0, 3, 4, 1]);  view_578 = None
        unbind_30 = torch.ops.aten.unbind.int(permute_298);  permute_298 = None
        getitem_290 = unbind_30[0]
        getitem_291 = unbind_30[1]
        getitem_292 = unbind_30[2];  unbind_30 = None
        pow_123 = torch.ops.aten.pow.Tensor_Scalar(getitem_290, 2.0)
        sum_91 = torch.ops.aten.sum.dim_IntList(pow_123, [-1], True);  pow_123 = None
        pow_124 = torch.ops.aten.pow.Tensor_Scalar(sum_91, 0.5);  sum_91 = None
        clamp_min_60 = torch.ops.aten.clamp_min.default(pow_124, 1e-12);  pow_124 = None
        expand_181 = torch.ops.aten.expand.default(clamp_min_60, [8, 16, 48, 784]);  clamp_min_60 = None
        div_102 = torch.ops.aten.div.Tensor(getitem_290, expand_181);  getitem_290 = expand_181 = None
        pow_125 = torch.ops.aten.pow.Tensor_Scalar(getitem_291, 2.0)
        sum_92 = torch.ops.aten.sum.dim_IntList(pow_125, [-1], True);  pow_125 = None
        pow_126 = torch.ops.aten.pow.Tensor_Scalar(sum_92, 0.5);  sum_92 = None
        clamp_min_61 = torch.ops.aten.clamp_min.default(pow_126, 1e-12);  pow_126 = None
        expand_182 = torch.ops.aten.expand.default(clamp_min_61, [8, 16, 48, 784]);  clamp_min_61 = None
        div_103 = torch.ops.aten.div.Tensor(getitem_291, expand_182);  getitem_291 = expand_182 = None
        permute_299 = torch.ops.aten.permute.default(div_103, [0, 1, 3, 2]);  div_103 = None
        expand_183 = torch.ops.aten.expand.default(div_102, [8, 16, 48, 784]);  div_102 = None
        clone_340 = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_579 = torch.ops.aten.view.default(clone_340, [128, 48, 784]);  clone_340 = None
        expand_184 = torch.ops.aten.expand.default(permute_299, [8, 16, 784, 48]);  permute_299 = None
        clone_341 = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
        view_580 = torch.ops.aten.view.default(clone_341, [128, 784, 48]);  clone_341 = None
        bmm_60 = torch.ops.aten.bmm.default(view_579, view_580);  view_579 = view_580 = None
        view_581 = torch.ops.aten.view.default(bmm_60, [8, 16, 48, 48]);  bmm_60 = None
        scalar_tensor_default_17 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_17 = torch.ops.aten.ge.Scalar(arg179_1, 0)
        neg_default_17 = torch.ops.aten.neg.default(scalar_tensor_default_17)
        where_self_17 = torch.ops.aten.where.self(ge_scalar_17, scalar_tensor_default_17, neg_default_17);  ge_scalar_17 = scalar_tensor_default_17 = neg_default_17 = None
        mul_tensor_51 = torch.ops.aten.mul.Tensor(view_581, where_self_17);  view_581 = None
        amax_default_17 = torch.ops.aten.amax.default(mul_tensor_51, [-1], True)
        sub_tensor_17 = torch.ops.aten.sub.Tensor(mul_tensor_51, amax_default_17);  mul_tensor_51 = amax_default_17 = None
        mul_tensor_52 = torch.ops.aten.mul.Tensor(where_self_17, arg179_1);  where_self_17 = arg179_1 = None
        mul_tensor_53 = torch.ops.aten.mul.Tensor(sub_tensor_17, mul_tensor_52);  sub_tensor_17 = mul_tensor_52 = None
        exp_30 = torch.ops.aten.exp.default(mul_tensor_53);  mul_tensor_53 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_104 = torch.ops.aten.div.Tensor(exp_30, sum_93);  exp_30 = sum_93 = None
        expand_185 = torch.ops.aten.expand.default(div_104, [8, 16, 48, 48]);  div_104 = None
        view_582 = torch.ops.aten.view.default(expand_185, [128, 48, 48]);  expand_185 = None
        expand_186 = torch.ops.aten.expand.default(getitem_292, [8, 16, 48, 784]);  getitem_292 = None
        clone_343 = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_583 = torch.ops.aten.view.default(clone_343, [128, 48, 784]);  clone_343 = None
        bmm_61 = torch.ops.aten.bmm.default(view_582, view_583);  view_582 = view_583 = None
        view_584 = torch.ops.aten.view.default(bmm_61, [8, 16, 48, 784]);  bmm_61 = None
        permute_300 = torch.ops.aten.permute.default(view_584, [0, 3, 1, 2]);  view_584 = None
        view_585 = torch.ops.aten.view.default(permute_300, [8, 784, 768]);  permute_300 = None
        permute_301 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        clone_344 = torch.ops.aten.clone.default(view_585, memory_format = torch.contiguous_format);  view_585 = None
        view_586 = torch.ops.aten.view.default(clone_344, [6272, 768]);  clone_344 = None
        mm_32 = torch.ops.aten.mm.default(view_586, permute_301);  view_586 = permute_301 = None
        view_587 = torch.ops.aten.view.default(mm_32, [8, 784, 768]);  mm_32 = None
        add_462 = torch.ops.aten.add.Tensor(view_587, arg181_1);  view_587 = arg181_1 = None
        mul_629 = torch.ops.aten.mul.Tensor(arg174_1, add_462);  arg174_1 = add_462 = None
        add_463 = torch.ops.aten.add.Tensor(add_459, mul_629);  add_459 = mul_629 = None
        clone_346 = torch.ops.aten.clone.default(add_463, memory_format = torch.contiguous_format)
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_346, [2], correction = 0, keepdim = True)
        getitem_293 = var_mean_96[0]
        getitem_294 = var_mean_96[1];  var_mean_96 = None
        add_464 = torch.ops.aten.add.Tensor(getitem_293, 1e-06);  getitem_293 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_464);  add_464 = None
        sub_163 = torch.ops.aten.sub.Tensor(clone_346, getitem_294);  clone_346 = getitem_294 = None
        mul_630 = torch.ops.aten.mul.Tensor(sub_163, rsqrt_96);  sub_163 = rsqrt_96 = None
        mul_631 = torch.ops.aten.mul.Tensor(mul_630, arg183_1);  mul_630 = arg183_1 = None
        add_465 = torch.ops.aten.add.Tensor(mul_631, arg184_1);  mul_631 = arg184_1 = None
        permute_302 = torch.ops.aten.permute.default(add_465, [0, 2, 1]);  add_465 = None
        view_588 = torch.ops.aten.view.default(permute_302, [8, 768, 28, 28]);  permute_302 = None
        convolution_68 = torch.ops.aten.convolution.default(view_588, arg185_1, arg186_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_588 = arg185_1 = arg186_1 = None
        mul_632 = torch.ops.aten.mul.Tensor(convolution_68, 0.5)
        mul_633 = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476);  convolution_68 = None
        erf_66 = torch.ops.aten.erf.default(mul_633);  mul_633 = None
        add_466 = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_634 = torch.ops.aten.mul.Tensor(mul_632, add_466);  mul_632 = add_466 = None
        add_467 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_467);  add_467 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_635 = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_635, -1);  mul_635 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_164 = torch.ops.aten.sub.Tensor(mul_634, unsqueeze_305);  mul_634 = unsqueeze_305 = None
        mul_636 = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_307);  sub_164 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_637 = torch.ops.aten.mul.Tensor(mul_636, unsqueeze_309);  mul_636 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_468 = torch.ops.aten.add.Tensor(mul_637, unsqueeze_311);  mul_637 = unsqueeze_311 = None
        convolution_69 = torch.ops.aten.convolution.default(add_468, arg191_1, arg192_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_468 = arg191_1 = arg192_1 = None
        view_589 = torch.ops.aten.view.default(convolution_69, [8, 768, 784]);  convolution_69 = None
        permute_303 = torch.ops.aten.permute.default(view_589, [0, 2, 1]);  view_589 = None
        mul_638 = torch.ops.aten.mul.Tensor(arg182_1, permute_303);  arg182_1 = permute_303 = None
        add_469 = torch.ops.aten.add.Tensor(add_463, mul_638);  add_463 = mul_638 = None
        clone_347 = torch.ops.aten.clone.default(add_469, memory_format = torch.contiguous_format)
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_347, [2], correction = 0, keepdim = True)
        getitem_295 = var_mean_97[0]
        getitem_296 = var_mean_97[1];  var_mean_97 = None
        add_470 = torch.ops.aten.add.Tensor(getitem_295, 1e-06);  getitem_295 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_470);  add_470 = None
        sub_165 = torch.ops.aten.sub.Tensor(clone_347, getitem_296);  clone_347 = getitem_296 = None
        mul_639 = torch.ops.aten.mul.Tensor(sub_165, rsqrt_97);  sub_165 = rsqrt_97 = None
        mul_640 = torch.ops.aten.mul.Tensor(mul_639, arg194_1);  mul_639 = arg194_1 = None
        add_471 = torch.ops.aten.add.Tensor(mul_640, arg195_1);  mul_640 = arg195_1 = None
        view_590 = torch.ops.aten.view.default(add_471, [6272, 768]);  add_471 = None
        permute_304 = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg197_1, view_590, permute_304);  arg197_1 = view_590 = permute_304 = None
        view_591 = torch.ops.aten.view.default(addmm_102, [8, 784, 3072]);  addmm_102 = None
        mul_641 = torch.ops.aten.mul.Tensor(view_591, 0.5)
        mul_642 = torch.ops.aten.mul.Tensor(view_591, 0.7071067811865476);  view_591 = None
        erf_67 = torch.ops.aten.erf.default(mul_642);  mul_642 = None
        add_472 = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_643 = torch.ops.aten.mul.Tensor(mul_641, add_472);  mul_641 = add_472 = None
        view_592 = torch.ops.aten.view.default(mul_643, [6272, 3072]);  mul_643 = None
        permute_305 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg199_1, view_592, permute_305);  arg199_1 = view_592 = permute_305 = None
        view_593 = torch.ops.aten.view.default(addmm_103, [8, 784, 768]);  addmm_103 = None
        mul_644 = torch.ops.aten.mul.Tensor(arg193_1, view_593);  arg193_1 = view_593 = None
        add_473 = torch.ops.aten.add.Tensor(add_469, mul_644);  add_469 = mul_644 = None
        clone_350 = torch.ops.aten.clone.default(add_473, memory_format = torch.contiguous_format)
        var_mean_98 = torch.ops.aten.var_mean.correction(clone_350, [2], correction = 0, keepdim = True)
        getitem_297 = var_mean_98[0]
        getitem_298 = var_mean_98[1];  var_mean_98 = None
        add_474 = torch.ops.aten.add.Tensor(getitem_297, 1e-06);  getitem_297 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
        sub_166 = torch.ops.aten.sub.Tensor(clone_350, getitem_298);  clone_350 = getitem_298 = None
        mul_645 = torch.ops.aten.mul.Tensor(sub_166, rsqrt_98);  sub_166 = rsqrt_98 = None
        mul_646 = torch.ops.aten.mul.Tensor(mul_645, arg201_1);  mul_645 = arg201_1 = None
        add_475 = torch.ops.aten.add.Tensor(mul_646, arg202_1);  mul_646 = arg202_1 = None
        view_594 = torch.ops.aten.view.default(add_475, [6272, 768]);  add_475 = None
        permute_306 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg204_1, view_594, permute_306);  arg204_1 = view_594 = permute_306 = None
        view_595 = torch.ops.aten.view.default(addmm_104, [8, 784, 2304]);  addmm_104 = None
        view_596 = torch.ops.aten.view.default(view_595, [8, 784, 3, 16, 48]);  view_595 = None
        permute_307 = torch.ops.aten.permute.default(view_596, [2, 0, 3, 4, 1]);  view_596 = None
        unbind_31 = torch.ops.aten.unbind.int(permute_307);  permute_307 = None
        getitem_299 = unbind_31[0]
        getitem_300 = unbind_31[1]
        getitem_301 = unbind_31[2];  unbind_31 = None
        pow_127 = torch.ops.aten.pow.Tensor_Scalar(getitem_299, 2.0)
        sum_94 = torch.ops.aten.sum.dim_IntList(pow_127, [-1], True);  pow_127 = None
        pow_128 = torch.ops.aten.pow.Tensor_Scalar(sum_94, 0.5);  sum_94 = None
        clamp_min_62 = torch.ops.aten.clamp_min.default(pow_128, 1e-12);  pow_128 = None
        expand_187 = torch.ops.aten.expand.default(clamp_min_62, [8, 16, 48, 784]);  clamp_min_62 = None
        div_105 = torch.ops.aten.div.Tensor(getitem_299, expand_187);  getitem_299 = expand_187 = None
        pow_129 = torch.ops.aten.pow.Tensor_Scalar(getitem_300, 2.0)
        sum_95 = torch.ops.aten.sum.dim_IntList(pow_129, [-1], True);  pow_129 = None
        pow_130 = torch.ops.aten.pow.Tensor_Scalar(sum_95, 0.5);  sum_95 = None
        clamp_min_63 = torch.ops.aten.clamp_min.default(pow_130, 1e-12);  pow_130 = None
        expand_188 = torch.ops.aten.expand.default(clamp_min_63, [8, 16, 48, 784]);  clamp_min_63 = None
        div_106 = torch.ops.aten.div.Tensor(getitem_300, expand_188);  getitem_300 = expand_188 = None
        permute_308 = torch.ops.aten.permute.default(div_106, [0, 1, 3, 2]);  div_106 = None
        expand_189 = torch.ops.aten.expand.default(div_105, [8, 16, 48, 784]);  div_105 = None
        clone_351 = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_597 = torch.ops.aten.view.default(clone_351, [128, 48, 784]);  clone_351 = None
        expand_190 = torch.ops.aten.expand.default(permute_308, [8, 16, 784, 48]);  permute_308 = None
        clone_352 = torch.ops.aten.clone.default(expand_190, memory_format = torch.contiguous_format);  expand_190 = None
        view_598 = torch.ops.aten.view.default(clone_352, [128, 784, 48]);  clone_352 = None
        bmm_62 = torch.ops.aten.bmm.default(view_597, view_598);  view_597 = view_598 = None
        view_599 = torch.ops.aten.view.default(bmm_62, [8, 16, 48, 48]);  bmm_62 = None
        scalar_tensor_default_16 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_16 = torch.ops.aten.ge.Scalar(arg205_1, 0)
        neg_default_16 = torch.ops.aten.neg.default(scalar_tensor_default_16)
        where_self_16 = torch.ops.aten.where.self(ge_scalar_16, scalar_tensor_default_16, neg_default_16);  ge_scalar_16 = scalar_tensor_default_16 = neg_default_16 = None
        mul_tensor_48 = torch.ops.aten.mul.Tensor(view_599, where_self_16);  view_599 = None
        amax_default_16 = torch.ops.aten.amax.default(mul_tensor_48, [-1], True)
        sub_tensor_16 = torch.ops.aten.sub.Tensor(mul_tensor_48, amax_default_16);  mul_tensor_48 = amax_default_16 = None
        mul_tensor_49 = torch.ops.aten.mul.Tensor(where_self_16, arg205_1);  where_self_16 = arg205_1 = None
        mul_tensor_50 = torch.ops.aten.mul.Tensor(sub_tensor_16, mul_tensor_49);  sub_tensor_16 = mul_tensor_49 = None
        exp_31 = torch.ops.aten.exp.default(mul_tensor_50);  mul_tensor_50 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_107 = torch.ops.aten.div.Tensor(exp_31, sum_96);  exp_31 = sum_96 = None
        expand_191 = torch.ops.aten.expand.default(div_107, [8, 16, 48, 48]);  div_107 = None
        view_600 = torch.ops.aten.view.default(expand_191, [128, 48, 48]);  expand_191 = None
        expand_192 = torch.ops.aten.expand.default(getitem_301, [8, 16, 48, 784]);  getitem_301 = None
        clone_354 = torch.ops.aten.clone.default(expand_192, memory_format = torch.contiguous_format);  expand_192 = None
        view_601 = torch.ops.aten.view.default(clone_354, [128, 48, 784]);  clone_354 = None
        bmm_63 = torch.ops.aten.bmm.default(view_600, view_601);  view_600 = view_601 = None
        view_602 = torch.ops.aten.view.default(bmm_63, [8, 16, 48, 784]);  bmm_63 = None
        permute_309 = torch.ops.aten.permute.default(view_602, [0, 3, 1, 2]);  view_602 = None
        view_603 = torch.ops.aten.view.default(permute_309, [8, 784, 768]);  permute_309 = None
        permute_310 = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        clone_355 = torch.ops.aten.clone.default(view_603, memory_format = torch.contiguous_format);  view_603 = None
        view_604 = torch.ops.aten.view.default(clone_355, [6272, 768]);  clone_355 = None
        mm_33 = torch.ops.aten.mm.default(view_604, permute_310);  view_604 = permute_310 = None
        view_605 = torch.ops.aten.view.default(mm_33, [8, 784, 768]);  mm_33 = None
        add_476 = torch.ops.aten.add.Tensor(view_605, arg207_1);  view_605 = arg207_1 = None
        mul_648 = torch.ops.aten.mul.Tensor(arg200_1, add_476);  arg200_1 = add_476 = None
        add_477 = torch.ops.aten.add.Tensor(add_473, mul_648);  add_473 = mul_648 = None
        clone_357 = torch.ops.aten.clone.default(add_477, memory_format = torch.contiguous_format)
        var_mean_99 = torch.ops.aten.var_mean.correction(clone_357, [2], correction = 0, keepdim = True)
        getitem_302 = var_mean_99[0]
        getitem_303 = var_mean_99[1];  var_mean_99 = None
        add_478 = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_478);  add_478 = None
        sub_168 = torch.ops.aten.sub.Tensor(clone_357, getitem_303);  clone_357 = getitem_303 = None
        mul_649 = torch.ops.aten.mul.Tensor(sub_168, rsqrt_99);  sub_168 = rsqrt_99 = None
        mul_650 = torch.ops.aten.mul.Tensor(mul_649, arg209_1);  mul_649 = arg209_1 = None
        add_479 = torch.ops.aten.add.Tensor(mul_650, arg210_1);  mul_650 = arg210_1 = None
        permute_311 = torch.ops.aten.permute.default(add_479, [0, 2, 1]);  add_479 = None
        view_606 = torch.ops.aten.view.default(permute_311, [8, 768, 28, 28]);  permute_311 = None
        convolution_70 = torch.ops.aten.convolution.default(view_606, arg211_1, arg212_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_606 = arg211_1 = arg212_1 = None
        mul_651 = torch.ops.aten.mul.Tensor(convolution_70, 0.5)
        mul_652 = torch.ops.aten.mul.Tensor(convolution_70, 0.7071067811865476);  convolution_70 = None
        erf_68 = torch.ops.aten.erf.default(mul_652);  mul_652 = None
        add_480 = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_653 = torch.ops.aten.mul.Tensor(mul_651, add_480);  mul_651 = add_480 = None
        add_481 = torch.ops.aten.add.Tensor(arg214_1, 1e-05);  arg214_1 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_481);  add_481 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_654 = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_654, -1);  mul_654 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_169 = torch.ops.aten.sub.Tensor(mul_653, unsqueeze_313);  mul_653 = unsqueeze_313 = None
        mul_655 = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_315);  sub_169 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_656 = torch.ops.aten.mul.Tensor(mul_655, unsqueeze_317);  mul_655 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_482 = torch.ops.aten.add.Tensor(mul_656, unsqueeze_319);  mul_656 = unsqueeze_319 = None
        convolution_71 = torch.ops.aten.convolution.default(add_482, arg217_1, arg218_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_482 = arg217_1 = arg218_1 = None
        view_607 = torch.ops.aten.view.default(convolution_71, [8, 768, 784]);  convolution_71 = None
        permute_312 = torch.ops.aten.permute.default(view_607, [0, 2, 1]);  view_607 = None
        mul_657 = torch.ops.aten.mul.Tensor(arg208_1, permute_312);  arg208_1 = permute_312 = None
        add_483 = torch.ops.aten.add.Tensor(add_477, mul_657);  add_477 = mul_657 = None
        clone_358 = torch.ops.aten.clone.default(add_483, memory_format = torch.contiguous_format)
        var_mean_100 = torch.ops.aten.var_mean.correction(clone_358, [2], correction = 0, keepdim = True)
        getitem_304 = var_mean_100[0]
        getitem_305 = var_mean_100[1];  var_mean_100 = None
        add_484 = torch.ops.aten.add.Tensor(getitem_304, 1e-06);  getitem_304 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_484);  add_484 = None
        sub_170 = torch.ops.aten.sub.Tensor(clone_358, getitem_305);  clone_358 = getitem_305 = None
        mul_658 = torch.ops.aten.mul.Tensor(sub_170, rsqrt_100);  sub_170 = rsqrt_100 = None
        mul_659 = torch.ops.aten.mul.Tensor(mul_658, arg220_1);  mul_658 = arg220_1 = None
        add_485 = torch.ops.aten.add.Tensor(mul_659, arg221_1);  mul_659 = arg221_1 = None
        view_608 = torch.ops.aten.view.default(add_485, [6272, 768]);  add_485 = None
        permute_313 = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg223_1, view_608, permute_313);  arg223_1 = view_608 = permute_313 = None
        view_609 = torch.ops.aten.view.default(addmm_105, [8, 784, 3072]);  addmm_105 = None
        mul_660 = torch.ops.aten.mul.Tensor(view_609, 0.5)
        mul_661 = torch.ops.aten.mul.Tensor(view_609, 0.7071067811865476);  view_609 = None
        erf_69 = torch.ops.aten.erf.default(mul_661);  mul_661 = None
        add_486 = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_662 = torch.ops.aten.mul.Tensor(mul_660, add_486);  mul_660 = add_486 = None
        view_610 = torch.ops.aten.view.default(mul_662, [6272, 3072]);  mul_662 = None
        permute_314 = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg225_1, view_610, permute_314);  arg225_1 = view_610 = permute_314 = None
        view_611 = torch.ops.aten.view.default(addmm_106, [8, 784, 768]);  addmm_106 = None
        mul_663 = torch.ops.aten.mul.Tensor(arg219_1, view_611);  arg219_1 = view_611 = None
        add_487 = torch.ops.aten.add.Tensor(add_483, mul_663);  add_483 = mul_663 = None
        clone_361 = torch.ops.aten.clone.default(add_487, memory_format = torch.contiguous_format)
        var_mean_101 = torch.ops.aten.var_mean.correction(clone_361, [2], correction = 0, keepdim = True)
        getitem_306 = var_mean_101[0]
        getitem_307 = var_mean_101[1];  var_mean_101 = None
        add_488 = torch.ops.aten.add.Tensor(getitem_306, 1e-06);  getitem_306 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_488);  add_488 = None
        sub_171 = torch.ops.aten.sub.Tensor(clone_361, getitem_307);  clone_361 = getitem_307 = None
        mul_664 = torch.ops.aten.mul.Tensor(sub_171, rsqrt_101);  sub_171 = rsqrt_101 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, arg227_1);  mul_664 = arg227_1 = None
        add_489 = torch.ops.aten.add.Tensor(mul_665, arg228_1);  mul_665 = arg228_1 = None
        view_612 = torch.ops.aten.view.default(add_489, [6272, 768]);  add_489 = None
        permute_315 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg230_1, view_612, permute_315);  arg230_1 = view_612 = permute_315 = None
        view_613 = torch.ops.aten.view.default(addmm_107, [8, 784, 2304]);  addmm_107 = None
        view_614 = torch.ops.aten.view.default(view_613, [8, 784, 3, 16, 48]);  view_613 = None
        permute_316 = torch.ops.aten.permute.default(view_614, [2, 0, 3, 4, 1]);  view_614 = None
        unbind_32 = torch.ops.aten.unbind.int(permute_316);  permute_316 = None
        getitem_308 = unbind_32[0]
        getitem_309 = unbind_32[1]
        getitem_310 = unbind_32[2];  unbind_32 = None
        pow_131 = torch.ops.aten.pow.Tensor_Scalar(getitem_308, 2.0)
        sum_97 = torch.ops.aten.sum.dim_IntList(pow_131, [-1], True);  pow_131 = None
        pow_132 = torch.ops.aten.pow.Tensor_Scalar(sum_97, 0.5);  sum_97 = None
        clamp_min_64 = torch.ops.aten.clamp_min.default(pow_132, 1e-12);  pow_132 = None
        expand_193 = torch.ops.aten.expand.default(clamp_min_64, [8, 16, 48, 784]);  clamp_min_64 = None
        div_108 = torch.ops.aten.div.Tensor(getitem_308, expand_193);  getitem_308 = expand_193 = None
        pow_133 = torch.ops.aten.pow.Tensor_Scalar(getitem_309, 2.0)
        sum_98 = torch.ops.aten.sum.dim_IntList(pow_133, [-1], True);  pow_133 = None
        pow_134 = torch.ops.aten.pow.Tensor_Scalar(sum_98, 0.5);  sum_98 = None
        clamp_min_65 = torch.ops.aten.clamp_min.default(pow_134, 1e-12);  pow_134 = None
        expand_194 = torch.ops.aten.expand.default(clamp_min_65, [8, 16, 48, 784]);  clamp_min_65 = None
        div_109 = torch.ops.aten.div.Tensor(getitem_309, expand_194);  getitem_309 = expand_194 = None
        permute_317 = torch.ops.aten.permute.default(div_109, [0, 1, 3, 2]);  div_109 = None
        expand_195 = torch.ops.aten.expand.default(div_108, [8, 16, 48, 784]);  div_108 = None
        clone_362 = torch.ops.aten.clone.default(expand_195, memory_format = torch.contiguous_format);  expand_195 = None
        view_615 = torch.ops.aten.view.default(clone_362, [128, 48, 784]);  clone_362 = None
        expand_196 = torch.ops.aten.expand.default(permute_317, [8, 16, 784, 48]);  permute_317 = None
        clone_363 = torch.ops.aten.clone.default(expand_196, memory_format = torch.contiguous_format);  expand_196 = None
        view_616 = torch.ops.aten.view.default(clone_363, [128, 784, 48]);  clone_363 = None
        bmm_64 = torch.ops.aten.bmm.default(view_615, view_616);  view_615 = view_616 = None
        view_617 = torch.ops.aten.view.default(bmm_64, [8, 16, 48, 48]);  bmm_64 = None
        scalar_tensor_default_15 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_15 = torch.ops.aten.ge.Scalar(arg231_1, 0)
        neg_default_15 = torch.ops.aten.neg.default(scalar_tensor_default_15)
        where_self_15 = torch.ops.aten.where.self(ge_scalar_15, scalar_tensor_default_15, neg_default_15);  ge_scalar_15 = scalar_tensor_default_15 = neg_default_15 = None
        mul_tensor_45 = torch.ops.aten.mul.Tensor(view_617, where_self_15);  view_617 = None
        amax_default_15 = torch.ops.aten.amax.default(mul_tensor_45, [-1], True)
        sub_tensor_15 = torch.ops.aten.sub.Tensor(mul_tensor_45, amax_default_15);  mul_tensor_45 = amax_default_15 = None
        mul_tensor_46 = torch.ops.aten.mul.Tensor(where_self_15, arg231_1);  where_self_15 = arg231_1 = None
        mul_tensor_47 = torch.ops.aten.mul.Tensor(sub_tensor_15, mul_tensor_46);  sub_tensor_15 = mul_tensor_46 = None
        exp_32 = torch.ops.aten.exp.default(mul_tensor_47);  mul_tensor_47 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_110 = torch.ops.aten.div.Tensor(exp_32, sum_99);  exp_32 = sum_99 = None
        expand_197 = torch.ops.aten.expand.default(div_110, [8, 16, 48, 48]);  div_110 = None
        view_618 = torch.ops.aten.view.default(expand_197, [128, 48, 48]);  expand_197 = None
        expand_198 = torch.ops.aten.expand.default(getitem_310, [8, 16, 48, 784]);  getitem_310 = None
        clone_365 = torch.ops.aten.clone.default(expand_198, memory_format = torch.contiguous_format);  expand_198 = None
        view_619 = torch.ops.aten.view.default(clone_365, [128, 48, 784]);  clone_365 = None
        bmm_65 = torch.ops.aten.bmm.default(view_618, view_619);  view_618 = view_619 = None
        view_620 = torch.ops.aten.view.default(bmm_65, [8, 16, 48, 784]);  bmm_65 = None
        permute_318 = torch.ops.aten.permute.default(view_620, [0, 3, 1, 2]);  view_620 = None
        view_621 = torch.ops.aten.view.default(permute_318, [8, 784, 768]);  permute_318 = None
        permute_319 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        clone_366 = torch.ops.aten.clone.default(view_621, memory_format = torch.contiguous_format);  view_621 = None
        view_622 = torch.ops.aten.view.default(clone_366, [6272, 768]);  clone_366 = None
        mm_34 = torch.ops.aten.mm.default(view_622, permute_319);  view_622 = permute_319 = None
        view_623 = torch.ops.aten.view.default(mm_34, [8, 784, 768]);  mm_34 = None
        add_490 = torch.ops.aten.add.Tensor(view_623, arg233_1);  view_623 = arg233_1 = None
        mul_667 = torch.ops.aten.mul.Tensor(arg226_1, add_490);  arg226_1 = add_490 = None
        add_491 = torch.ops.aten.add.Tensor(add_487, mul_667);  add_487 = mul_667 = None
        clone_368 = torch.ops.aten.clone.default(add_491, memory_format = torch.contiguous_format)
        var_mean_102 = torch.ops.aten.var_mean.correction(clone_368, [2], correction = 0, keepdim = True)
        getitem_311 = var_mean_102[0]
        getitem_312 = var_mean_102[1];  var_mean_102 = None
        add_492 = torch.ops.aten.add.Tensor(getitem_311, 1e-06);  getitem_311 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_492);  add_492 = None
        sub_173 = torch.ops.aten.sub.Tensor(clone_368, getitem_312);  clone_368 = getitem_312 = None
        mul_668 = torch.ops.aten.mul.Tensor(sub_173, rsqrt_102);  sub_173 = rsqrt_102 = None
        mul_669 = torch.ops.aten.mul.Tensor(mul_668, arg235_1);  mul_668 = arg235_1 = None
        add_493 = torch.ops.aten.add.Tensor(mul_669, arg236_1);  mul_669 = arg236_1 = None
        permute_320 = torch.ops.aten.permute.default(add_493, [0, 2, 1]);  add_493 = None
        view_624 = torch.ops.aten.view.default(permute_320, [8, 768, 28, 28]);  permute_320 = None
        convolution_72 = torch.ops.aten.convolution.default(view_624, arg237_1, arg238_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_624 = arg237_1 = arg238_1 = None
        mul_670 = torch.ops.aten.mul.Tensor(convolution_72, 0.5)
        mul_671 = torch.ops.aten.mul.Tensor(convolution_72, 0.7071067811865476);  convolution_72 = None
        erf_70 = torch.ops.aten.erf.default(mul_671);  mul_671 = None
        add_494 = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_672 = torch.ops.aten.mul.Tensor(mul_670, add_494);  mul_670 = add_494 = None
        add_495 = torch.ops.aten.add.Tensor(arg240_1, 1e-05);  arg240_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_495);  add_495 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_673 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_673, -1);  mul_673 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_174 = torch.ops.aten.sub.Tensor(mul_672, unsqueeze_321);  mul_672 = unsqueeze_321 = None
        mul_674 = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_323);  sub_174 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_675 = torch.ops.aten.mul.Tensor(mul_674, unsqueeze_325);  mul_674 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_496 = torch.ops.aten.add.Tensor(mul_675, unsqueeze_327);  mul_675 = unsqueeze_327 = None
        convolution_73 = torch.ops.aten.convolution.default(add_496, arg243_1, arg244_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_496 = arg243_1 = arg244_1 = None
        view_625 = torch.ops.aten.view.default(convolution_73, [8, 768, 784]);  convolution_73 = None
        permute_321 = torch.ops.aten.permute.default(view_625, [0, 2, 1]);  view_625 = None
        mul_676 = torch.ops.aten.mul.Tensor(arg234_1, permute_321);  arg234_1 = permute_321 = None
        add_497 = torch.ops.aten.add.Tensor(add_491, mul_676);  add_491 = mul_676 = None
        clone_369 = torch.ops.aten.clone.default(add_497, memory_format = torch.contiguous_format)
        var_mean_103 = torch.ops.aten.var_mean.correction(clone_369, [2], correction = 0, keepdim = True)
        getitem_313 = var_mean_103[0]
        getitem_314 = var_mean_103[1];  var_mean_103 = None
        add_498 = torch.ops.aten.add.Tensor(getitem_313, 1e-06);  getitem_313 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_498);  add_498 = None
        sub_175 = torch.ops.aten.sub.Tensor(clone_369, getitem_314);  clone_369 = getitem_314 = None
        mul_677 = torch.ops.aten.mul.Tensor(sub_175, rsqrt_103);  sub_175 = rsqrt_103 = None
        mul_678 = torch.ops.aten.mul.Tensor(mul_677, arg246_1);  mul_677 = arg246_1 = None
        add_499 = torch.ops.aten.add.Tensor(mul_678, arg247_1);  mul_678 = arg247_1 = None
        view_626 = torch.ops.aten.view.default(add_499, [6272, 768]);  add_499 = None
        permute_322 = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg249_1, view_626, permute_322);  arg249_1 = view_626 = permute_322 = None
        view_627 = torch.ops.aten.view.default(addmm_108, [8, 784, 3072]);  addmm_108 = None
        mul_679 = torch.ops.aten.mul.Tensor(view_627, 0.5)
        mul_680 = torch.ops.aten.mul.Tensor(view_627, 0.7071067811865476);  view_627 = None
        erf_71 = torch.ops.aten.erf.default(mul_680);  mul_680 = None
        add_500 = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_681 = torch.ops.aten.mul.Tensor(mul_679, add_500);  mul_679 = add_500 = None
        view_628 = torch.ops.aten.view.default(mul_681, [6272, 3072]);  mul_681 = None
        permute_323 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg251_1, view_628, permute_323);  arg251_1 = view_628 = permute_323 = None
        view_629 = torch.ops.aten.view.default(addmm_109, [8, 784, 768]);  addmm_109 = None
        mul_682 = torch.ops.aten.mul.Tensor(arg245_1, view_629);  arg245_1 = view_629 = None
        add_501 = torch.ops.aten.add.Tensor(add_497, mul_682);  add_497 = mul_682 = None
        clone_372 = torch.ops.aten.clone.default(add_501, memory_format = torch.contiguous_format)
        var_mean_104 = torch.ops.aten.var_mean.correction(clone_372, [2], correction = 0, keepdim = True)
        getitem_315 = var_mean_104[0]
        getitem_316 = var_mean_104[1];  var_mean_104 = None
        add_502 = torch.ops.aten.add.Tensor(getitem_315, 1e-06);  getitem_315 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_502);  add_502 = None
        sub_176 = torch.ops.aten.sub.Tensor(clone_372, getitem_316);  clone_372 = getitem_316 = None
        mul_683 = torch.ops.aten.mul.Tensor(sub_176, rsqrt_104);  sub_176 = rsqrt_104 = None
        mul_684 = torch.ops.aten.mul.Tensor(mul_683, arg253_1);  mul_683 = arg253_1 = None
        add_503 = torch.ops.aten.add.Tensor(mul_684, arg254_1);  mul_684 = arg254_1 = None
        view_630 = torch.ops.aten.view.default(add_503, [6272, 768]);  add_503 = None
        permute_324 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg256_1, view_630, permute_324);  arg256_1 = view_630 = permute_324 = None
        view_631 = torch.ops.aten.view.default(addmm_110, [8, 784, 2304]);  addmm_110 = None
        view_632 = torch.ops.aten.view.default(view_631, [8, 784, 3, 16, 48]);  view_631 = None
        permute_325 = torch.ops.aten.permute.default(view_632, [2, 0, 3, 4, 1]);  view_632 = None
        unbind_33 = torch.ops.aten.unbind.int(permute_325);  permute_325 = None
        getitem_317 = unbind_33[0]
        getitem_318 = unbind_33[1]
        getitem_319 = unbind_33[2];  unbind_33 = None
        pow_135 = torch.ops.aten.pow.Tensor_Scalar(getitem_317, 2.0)
        sum_100 = torch.ops.aten.sum.dim_IntList(pow_135, [-1], True);  pow_135 = None
        pow_136 = torch.ops.aten.pow.Tensor_Scalar(sum_100, 0.5);  sum_100 = None
        clamp_min_66 = torch.ops.aten.clamp_min.default(pow_136, 1e-12);  pow_136 = None
        expand_199 = torch.ops.aten.expand.default(clamp_min_66, [8, 16, 48, 784]);  clamp_min_66 = None
        div_111 = torch.ops.aten.div.Tensor(getitem_317, expand_199);  getitem_317 = expand_199 = None
        pow_137 = torch.ops.aten.pow.Tensor_Scalar(getitem_318, 2.0)
        sum_101 = torch.ops.aten.sum.dim_IntList(pow_137, [-1], True);  pow_137 = None
        pow_138 = torch.ops.aten.pow.Tensor_Scalar(sum_101, 0.5);  sum_101 = None
        clamp_min_67 = torch.ops.aten.clamp_min.default(pow_138, 1e-12);  pow_138 = None
        expand_200 = torch.ops.aten.expand.default(clamp_min_67, [8, 16, 48, 784]);  clamp_min_67 = None
        div_112 = torch.ops.aten.div.Tensor(getitem_318, expand_200);  getitem_318 = expand_200 = None
        permute_326 = torch.ops.aten.permute.default(div_112, [0, 1, 3, 2]);  div_112 = None
        expand_201 = torch.ops.aten.expand.default(div_111, [8, 16, 48, 784]);  div_111 = None
        clone_373 = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_633 = torch.ops.aten.view.default(clone_373, [128, 48, 784]);  clone_373 = None
        expand_202 = torch.ops.aten.expand.default(permute_326, [8, 16, 784, 48]);  permute_326 = None
        clone_374 = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_634 = torch.ops.aten.view.default(clone_374, [128, 784, 48]);  clone_374 = None
        bmm_66 = torch.ops.aten.bmm.default(view_633, view_634);  view_633 = view_634 = None
        view_635 = torch.ops.aten.view.default(bmm_66, [8, 16, 48, 48]);  bmm_66 = None
        scalar_tensor_default_14 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_14 = torch.ops.aten.ge.Scalar(arg257_1, 0)
        neg_default_14 = torch.ops.aten.neg.default(scalar_tensor_default_14)
        where_self_14 = torch.ops.aten.where.self(ge_scalar_14, scalar_tensor_default_14, neg_default_14);  ge_scalar_14 = scalar_tensor_default_14 = neg_default_14 = None
        mul_tensor_42 = torch.ops.aten.mul.Tensor(view_635, where_self_14);  view_635 = None
        amax_default_14 = torch.ops.aten.amax.default(mul_tensor_42, [-1], True)
        sub_tensor_14 = torch.ops.aten.sub.Tensor(mul_tensor_42, amax_default_14);  mul_tensor_42 = amax_default_14 = None
        mul_tensor_43 = torch.ops.aten.mul.Tensor(where_self_14, arg257_1);  where_self_14 = arg257_1 = None
        mul_tensor_44 = torch.ops.aten.mul.Tensor(sub_tensor_14, mul_tensor_43);  sub_tensor_14 = mul_tensor_43 = None
        exp_33 = torch.ops.aten.exp.default(mul_tensor_44);  mul_tensor_44 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_113 = torch.ops.aten.div.Tensor(exp_33, sum_102);  exp_33 = sum_102 = None
        expand_203 = torch.ops.aten.expand.default(div_113, [8, 16, 48, 48]);  div_113 = None
        view_636 = torch.ops.aten.view.default(expand_203, [128, 48, 48]);  expand_203 = None
        expand_204 = torch.ops.aten.expand.default(getitem_319, [8, 16, 48, 784]);  getitem_319 = None
        clone_376 = torch.ops.aten.clone.default(expand_204, memory_format = torch.contiguous_format);  expand_204 = None
        view_637 = torch.ops.aten.view.default(clone_376, [128, 48, 784]);  clone_376 = None
        bmm_67 = torch.ops.aten.bmm.default(view_636, view_637);  view_636 = view_637 = None
        view_638 = torch.ops.aten.view.default(bmm_67, [8, 16, 48, 784]);  bmm_67 = None
        permute_327 = torch.ops.aten.permute.default(view_638, [0, 3, 1, 2]);  view_638 = None
        view_639 = torch.ops.aten.view.default(permute_327, [8, 784, 768]);  permute_327 = None
        permute_328 = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        clone_377 = torch.ops.aten.clone.default(view_639, memory_format = torch.contiguous_format);  view_639 = None
        view_640 = torch.ops.aten.view.default(clone_377, [6272, 768]);  clone_377 = None
        mm_35 = torch.ops.aten.mm.default(view_640, permute_328);  view_640 = permute_328 = None
        view_641 = torch.ops.aten.view.default(mm_35, [8, 784, 768]);  mm_35 = None
        add_504 = torch.ops.aten.add.Tensor(view_641, arg259_1);  view_641 = arg259_1 = None
        mul_686 = torch.ops.aten.mul.Tensor(arg252_1, add_504);  arg252_1 = add_504 = None
        add_505 = torch.ops.aten.add.Tensor(add_501, mul_686);  add_501 = mul_686 = None
        clone_379 = torch.ops.aten.clone.default(add_505, memory_format = torch.contiguous_format)
        var_mean_105 = torch.ops.aten.var_mean.correction(clone_379, [2], correction = 0, keepdim = True)
        getitem_320 = var_mean_105[0]
        getitem_321 = var_mean_105[1];  var_mean_105 = None
        add_506 = torch.ops.aten.add.Tensor(getitem_320, 1e-06);  getitem_320 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_506);  add_506 = None
        sub_178 = torch.ops.aten.sub.Tensor(clone_379, getitem_321);  clone_379 = getitem_321 = None
        mul_687 = torch.ops.aten.mul.Tensor(sub_178, rsqrt_105);  sub_178 = rsqrt_105 = None
        mul_688 = torch.ops.aten.mul.Tensor(mul_687, arg261_1);  mul_687 = arg261_1 = None
        add_507 = torch.ops.aten.add.Tensor(mul_688, arg262_1);  mul_688 = arg262_1 = None
        permute_329 = torch.ops.aten.permute.default(add_507, [0, 2, 1]);  add_507 = None
        view_642 = torch.ops.aten.view.default(permute_329, [8, 768, 28, 28]);  permute_329 = None
        convolution_74 = torch.ops.aten.convolution.default(view_642, arg263_1, arg264_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_642 = arg263_1 = arg264_1 = None
        mul_689 = torch.ops.aten.mul.Tensor(convolution_74, 0.5)
        mul_690 = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476);  convolution_74 = None
        erf_72 = torch.ops.aten.erf.default(mul_690);  mul_690 = None
        add_508 = torch.ops.aten.add.Tensor(erf_72, 1);  erf_72 = None
        mul_691 = torch.ops.aten.mul.Tensor(mul_689, add_508);  mul_689 = add_508 = None
        add_509 = torch.ops.aten.add.Tensor(arg266_1, 1e-05);  arg266_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_509);  add_509 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_692 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_692, -1);  mul_692 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_179 = torch.ops.aten.sub.Tensor(mul_691, unsqueeze_329);  mul_691 = unsqueeze_329 = None
        mul_693 = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_331);  sub_179 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_694 = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_333);  mul_693 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_510 = torch.ops.aten.add.Tensor(mul_694, unsqueeze_335);  mul_694 = unsqueeze_335 = None
        convolution_75 = torch.ops.aten.convolution.default(add_510, arg269_1, arg270_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_510 = arg269_1 = arg270_1 = None
        view_643 = torch.ops.aten.view.default(convolution_75, [8, 768, 784]);  convolution_75 = None
        permute_330 = torch.ops.aten.permute.default(view_643, [0, 2, 1]);  view_643 = None
        mul_695 = torch.ops.aten.mul.Tensor(arg260_1, permute_330);  arg260_1 = permute_330 = None
        add_511 = torch.ops.aten.add.Tensor(add_505, mul_695);  add_505 = mul_695 = None
        clone_380 = torch.ops.aten.clone.default(add_511, memory_format = torch.contiguous_format)
        var_mean_106 = torch.ops.aten.var_mean.correction(clone_380, [2], correction = 0, keepdim = True)
        getitem_322 = var_mean_106[0]
        getitem_323 = var_mean_106[1];  var_mean_106 = None
        add_512 = torch.ops.aten.add.Tensor(getitem_322, 1e-06);  getitem_322 = None
        rsqrt_106 = torch.ops.aten.rsqrt.default(add_512);  add_512 = None
        sub_180 = torch.ops.aten.sub.Tensor(clone_380, getitem_323);  clone_380 = getitem_323 = None
        mul_696 = torch.ops.aten.mul.Tensor(sub_180, rsqrt_106);  sub_180 = rsqrt_106 = None
        mul_697 = torch.ops.aten.mul.Tensor(mul_696, arg272_1);  mul_696 = arg272_1 = None
        add_513 = torch.ops.aten.add.Tensor(mul_697, arg273_1);  mul_697 = arg273_1 = None
        view_644 = torch.ops.aten.view.default(add_513, [6272, 768]);  add_513 = None
        permute_331 = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg275_1, view_644, permute_331);  arg275_1 = view_644 = permute_331 = None
        view_645 = torch.ops.aten.view.default(addmm_111, [8, 784, 3072]);  addmm_111 = None
        mul_698 = torch.ops.aten.mul.Tensor(view_645, 0.5)
        mul_699 = torch.ops.aten.mul.Tensor(view_645, 0.7071067811865476);  view_645 = None
        erf_73 = torch.ops.aten.erf.default(mul_699);  mul_699 = None
        add_514 = torch.ops.aten.add.Tensor(erf_73, 1);  erf_73 = None
        mul_700 = torch.ops.aten.mul.Tensor(mul_698, add_514);  mul_698 = add_514 = None
        view_646 = torch.ops.aten.view.default(mul_700, [6272, 3072]);  mul_700 = None
        permute_332 = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg277_1, view_646, permute_332);  arg277_1 = view_646 = permute_332 = None
        view_647 = torch.ops.aten.view.default(addmm_112, [8, 784, 768]);  addmm_112 = None
        mul_701 = torch.ops.aten.mul.Tensor(arg271_1, view_647);  arg271_1 = view_647 = None
        add_515 = torch.ops.aten.add.Tensor(add_511, mul_701);  add_511 = mul_701 = None
        clone_383 = torch.ops.aten.clone.default(add_515, memory_format = torch.contiguous_format)
        var_mean_107 = torch.ops.aten.var_mean.correction(clone_383, [2], correction = 0, keepdim = True)
        getitem_324 = var_mean_107[0]
        getitem_325 = var_mean_107[1];  var_mean_107 = None
        add_516 = torch.ops.aten.add.Tensor(getitem_324, 1e-06);  getitem_324 = None
        rsqrt_107 = torch.ops.aten.rsqrt.default(add_516);  add_516 = None
        sub_181 = torch.ops.aten.sub.Tensor(clone_383, getitem_325);  clone_383 = getitem_325 = None
        mul_702 = torch.ops.aten.mul.Tensor(sub_181, rsqrt_107);  sub_181 = rsqrt_107 = None
        mul_703 = torch.ops.aten.mul.Tensor(mul_702, arg279_1);  mul_702 = arg279_1 = None
        add_517 = torch.ops.aten.add.Tensor(mul_703, arg280_1);  mul_703 = arg280_1 = None
        view_648 = torch.ops.aten.view.default(add_517, [6272, 768]);  add_517 = None
        permute_333 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg282_1, view_648, permute_333);  arg282_1 = view_648 = permute_333 = None
        view_649 = torch.ops.aten.view.default(addmm_113, [8, 784, 2304]);  addmm_113 = None
        view_650 = torch.ops.aten.view.default(view_649, [8, 784, 3, 16, 48]);  view_649 = None
        permute_334 = torch.ops.aten.permute.default(view_650, [2, 0, 3, 4, 1]);  view_650 = None
        unbind_34 = torch.ops.aten.unbind.int(permute_334);  permute_334 = None
        getitem_326 = unbind_34[0]
        getitem_327 = unbind_34[1]
        getitem_328 = unbind_34[2];  unbind_34 = None
        pow_139 = torch.ops.aten.pow.Tensor_Scalar(getitem_326, 2.0)
        sum_103 = torch.ops.aten.sum.dim_IntList(pow_139, [-1], True);  pow_139 = None
        pow_140 = torch.ops.aten.pow.Tensor_Scalar(sum_103, 0.5);  sum_103 = None
        clamp_min_68 = torch.ops.aten.clamp_min.default(pow_140, 1e-12);  pow_140 = None
        expand_205 = torch.ops.aten.expand.default(clamp_min_68, [8, 16, 48, 784]);  clamp_min_68 = None
        div_114 = torch.ops.aten.div.Tensor(getitem_326, expand_205);  getitem_326 = expand_205 = None
        pow_141 = torch.ops.aten.pow.Tensor_Scalar(getitem_327, 2.0)
        sum_104 = torch.ops.aten.sum.dim_IntList(pow_141, [-1], True);  pow_141 = None
        pow_142 = torch.ops.aten.pow.Tensor_Scalar(sum_104, 0.5);  sum_104 = None
        clamp_min_69 = torch.ops.aten.clamp_min.default(pow_142, 1e-12);  pow_142 = None
        expand_206 = torch.ops.aten.expand.default(clamp_min_69, [8, 16, 48, 784]);  clamp_min_69 = None
        div_115 = torch.ops.aten.div.Tensor(getitem_327, expand_206);  getitem_327 = expand_206 = None
        permute_335 = torch.ops.aten.permute.default(div_115, [0, 1, 3, 2]);  div_115 = None
        expand_207 = torch.ops.aten.expand.default(div_114, [8, 16, 48, 784]);  div_114 = None
        clone_384 = torch.ops.aten.clone.default(expand_207, memory_format = torch.contiguous_format);  expand_207 = None
        view_651 = torch.ops.aten.view.default(clone_384, [128, 48, 784]);  clone_384 = None
        expand_208 = torch.ops.aten.expand.default(permute_335, [8, 16, 784, 48]);  permute_335 = None
        clone_385 = torch.ops.aten.clone.default(expand_208, memory_format = torch.contiguous_format);  expand_208 = None
        view_652 = torch.ops.aten.view.default(clone_385, [128, 784, 48]);  clone_385 = None
        bmm_68 = torch.ops.aten.bmm.default(view_651, view_652);  view_651 = view_652 = None
        view_653 = torch.ops.aten.view.default(bmm_68, [8, 16, 48, 48]);  bmm_68 = None
        scalar_tensor_default_13 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_13 = torch.ops.aten.ge.Scalar(arg283_1, 0)
        neg_default_13 = torch.ops.aten.neg.default(scalar_tensor_default_13)
        where_self_13 = torch.ops.aten.where.self(ge_scalar_13, scalar_tensor_default_13, neg_default_13);  ge_scalar_13 = scalar_tensor_default_13 = neg_default_13 = None
        mul_tensor_39 = torch.ops.aten.mul.Tensor(view_653, where_self_13);  view_653 = None
        amax_default_13 = torch.ops.aten.amax.default(mul_tensor_39, [-1], True)
        sub_tensor_13 = torch.ops.aten.sub.Tensor(mul_tensor_39, amax_default_13);  mul_tensor_39 = amax_default_13 = None
        mul_tensor_40 = torch.ops.aten.mul.Tensor(where_self_13, arg283_1);  where_self_13 = arg283_1 = None
        mul_tensor_41 = torch.ops.aten.mul.Tensor(sub_tensor_13, mul_tensor_40);  sub_tensor_13 = mul_tensor_40 = None
        exp_34 = torch.ops.aten.exp.default(mul_tensor_41);  mul_tensor_41 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_116 = torch.ops.aten.div.Tensor(exp_34, sum_105);  exp_34 = sum_105 = None
        expand_209 = torch.ops.aten.expand.default(div_116, [8, 16, 48, 48]);  div_116 = None
        view_654 = torch.ops.aten.view.default(expand_209, [128, 48, 48]);  expand_209 = None
        expand_210 = torch.ops.aten.expand.default(getitem_328, [8, 16, 48, 784]);  getitem_328 = None
        clone_387 = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_655 = torch.ops.aten.view.default(clone_387, [128, 48, 784]);  clone_387 = None
        bmm_69 = torch.ops.aten.bmm.default(view_654, view_655);  view_654 = view_655 = None
        view_656 = torch.ops.aten.view.default(bmm_69, [8, 16, 48, 784]);  bmm_69 = None
        permute_336 = torch.ops.aten.permute.default(view_656, [0, 3, 1, 2]);  view_656 = None
        view_657 = torch.ops.aten.view.default(permute_336, [8, 784, 768]);  permute_336 = None
        permute_337 = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        clone_388 = torch.ops.aten.clone.default(view_657, memory_format = torch.contiguous_format);  view_657 = None
        view_658 = torch.ops.aten.view.default(clone_388, [6272, 768]);  clone_388 = None
        mm_36 = torch.ops.aten.mm.default(view_658, permute_337);  view_658 = permute_337 = None
        view_659 = torch.ops.aten.view.default(mm_36, [8, 784, 768]);  mm_36 = None
        add_518 = torch.ops.aten.add.Tensor(view_659, arg285_1);  view_659 = arg285_1 = None
        mul_705 = torch.ops.aten.mul.Tensor(arg278_1, add_518);  arg278_1 = add_518 = None
        add_519 = torch.ops.aten.add.Tensor(add_515, mul_705);  add_515 = mul_705 = None
        clone_390 = torch.ops.aten.clone.default(add_519, memory_format = torch.contiguous_format)
        var_mean_108 = torch.ops.aten.var_mean.correction(clone_390, [2], correction = 0, keepdim = True)
        getitem_329 = var_mean_108[0]
        getitem_330 = var_mean_108[1];  var_mean_108 = None
        add_520 = torch.ops.aten.add.Tensor(getitem_329, 1e-06);  getitem_329 = None
        rsqrt_108 = torch.ops.aten.rsqrt.default(add_520);  add_520 = None
        sub_183 = torch.ops.aten.sub.Tensor(clone_390, getitem_330);  clone_390 = getitem_330 = None
        mul_706 = torch.ops.aten.mul.Tensor(sub_183, rsqrt_108);  sub_183 = rsqrt_108 = None
        mul_707 = torch.ops.aten.mul.Tensor(mul_706, arg287_1);  mul_706 = arg287_1 = None
        add_521 = torch.ops.aten.add.Tensor(mul_707, arg288_1);  mul_707 = arg288_1 = None
        permute_338 = torch.ops.aten.permute.default(add_521, [0, 2, 1]);  add_521 = None
        view_660 = torch.ops.aten.view.default(permute_338, [8, 768, 28, 28]);  permute_338 = None
        convolution_76 = torch.ops.aten.convolution.default(view_660, arg289_1, arg290_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_660 = arg289_1 = arg290_1 = None
        mul_708 = torch.ops.aten.mul.Tensor(convolution_76, 0.5)
        mul_709 = torch.ops.aten.mul.Tensor(convolution_76, 0.7071067811865476);  convolution_76 = None
        erf_74 = torch.ops.aten.erf.default(mul_709);  mul_709 = None
        add_522 = torch.ops.aten.add.Tensor(erf_74, 1);  erf_74 = None
        mul_710 = torch.ops.aten.mul.Tensor(mul_708, add_522);  mul_708 = add_522 = None
        add_523 = torch.ops.aten.add.Tensor(arg292_1, 1e-05);  arg292_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_523);  add_523 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_711 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_711, -1);  mul_711 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_184 = torch.ops.aten.sub.Tensor(mul_710, unsqueeze_337);  mul_710 = unsqueeze_337 = None
        mul_712 = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_339);  sub_184 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_713 = torch.ops.aten.mul.Tensor(mul_712, unsqueeze_341);  mul_712 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_524 = torch.ops.aten.add.Tensor(mul_713, unsqueeze_343);  mul_713 = unsqueeze_343 = None
        convolution_77 = torch.ops.aten.convolution.default(add_524, arg295_1, arg296_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_524 = arg295_1 = arg296_1 = None
        view_661 = torch.ops.aten.view.default(convolution_77, [8, 768, 784]);  convolution_77 = None
        permute_339 = torch.ops.aten.permute.default(view_661, [0, 2, 1]);  view_661 = None
        mul_714 = torch.ops.aten.mul.Tensor(arg286_1, permute_339);  arg286_1 = permute_339 = None
        add_525 = torch.ops.aten.add.Tensor(add_519, mul_714);  add_519 = mul_714 = None
        clone_391 = torch.ops.aten.clone.default(add_525, memory_format = torch.contiguous_format)
        var_mean_109 = torch.ops.aten.var_mean.correction(clone_391, [2], correction = 0, keepdim = True)
        getitem_331 = var_mean_109[0]
        getitem_332 = var_mean_109[1];  var_mean_109 = None
        add_526 = torch.ops.aten.add.Tensor(getitem_331, 1e-06);  getitem_331 = None
        rsqrt_109 = torch.ops.aten.rsqrt.default(add_526);  add_526 = None
        sub_185 = torch.ops.aten.sub.Tensor(clone_391, getitem_332);  clone_391 = getitem_332 = None
        mul_715 = torch.ops.aten.mul.Tensor(sub_185, rsqrt_109);  sub_185 = rsqrt_109 = None
        mul_716 = torch.ops.aten.mul.Tensor(mul_715, arg298_1);  mul_715 = arg298_1 = None
        add_527 = torch.ops.aten.add.Tensor(mul_716, arg299_1);  mul_716 = arg299_1 = None
        view_662 = torch.ops.aten.view.default(add_527, [6272, 768]);  add_527 = None
        permute_340 = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg301_1, view_662, permute_340);  arg301_1 = view_662 = permute_340 = None
        view_663 = torch.ops.aten.view.default(addmm_114, [8, 784, 3072]);  addmm_114 = None
        mul_717 = torch.ops.aten.mul.Tensor(view_663, 0.5)
        mul_718 = torch.ops.aten.mul.Tensor(view_663, 0.7071067811865476);  view_663 = None
        erf_75 = torch.ops.aten.erf.default(mul_718);  mul_718 = None
        add_528 = torch.ops.aten.add.Tensor(erf_75, 1);  erf_75 = None
        mul_719 = torch.ops.aten.mul.Tensor(mul_717, add_528);  mul_717 = add_528 = None
        view_664 = torch.ops.aten.view.default(mul_719, [6272, 3072]);  mul_719 = None
        permute_341 = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg303_1, view_664, permute_341);  arg303_1 = view_664 = permute_341 = None
        view_665 = torch.ops.aten.view.default(addmm_115, [8, 784, 768]);  addmm_115 = None
        mul_720 = torch.ops.aten.mul.Tensor(arg297_1, view_665);  arg297_1 = view_665 = None
        add_529 = torch.ops.aten.add.Tensor(add_525, mul_720);  add_525 = mul_720 = None
        clone_394 = torch.ops.aten.clone.default(add_529, memory_format = torch.contiguous_format)
        var_mean_110 = torch.ops.aten.var_mean.correction(clone_394, [2], correction = 0, keepdim = True)
        getitem_333 = var_mean_110[0]
        getitem_334 = var_mean_110[1];  var_mean_110 = None
        add_530 = torch.ops.aten.add.Tensor(getitem_333, 1e-06);  getitem_333 = None
        rsqrt_110 = torch.ops.aten.rsqrt.default(add_530);  add_530 = None
        sub_186 = torch.ops.aten.sub.Tensor(clone_394, getitem_334);  clone_394 = getitem_334 = None
        mul_721 = torch.ops.aten.mul.Tensor(sub_186, rsqrt_110);  sub_186 = rsqrt_110 = None
        mul_722 = torch.ops.aten.mul.Tensor(mul_721, arg305_1);  mul_721 = arg305_1 = None
        add_531 = torch.ops.aten.add.Tensor(mul_722, arg306_1);  mul_722 = arg306_1 = None
        view_666 = torch.ops.aten.view.default(add_531, [6272, 768]);  add_531 = None
        permute_342 = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg308_1, view_666, permute_342);  arg308_1 = view_666 = permute_342 = None
        view_667 = torch.ops.aten.view.default(addmm_116, [8, 784, 2304]);  addmm_116 = None
        view_668 = torch.ops.aten.view.default(view_667, [8, 784, 3, 16, 48]);  view_667 = None
        permute_343 = torch.ops.aten.permute.default(view_668, [2, 0, 3, 4, 1]);  view_668 = None
        unbind_35 = torch.ops.aten.unbind.int(permute_343);  permute_343 = None
        getitem_335 = unbind_35[0]
        getitem_336 = unbind_35[1]
        getitem_337 = unbind_35[2];  unbind_35 = None
        pow_143 = torch.ops.aten.pow.Tensor_Scalar(getitem_335, 2.0)
        sum_106 = torch.ops.aten.sum.dim_IntList(pow_143, [-1], True);  pow_143 = None
        pow_144 = torch.ops.aten.pow.Tensor_Scalar(sum_106, 0.5);  sum_106 = None
        clamp_min_70 = torch.ops.aten.clamp_min.default(pow_144, 1e-12);  pow_144 = None
        expand_211 = torch.ops.aten.expand.default(clamp_min_70, [8, 16, 48, 784]);  clamp_min_70 = None
        div_117 = torch.ops.aten.div.Tensor(getitem_335, expand_211);  getitem_335 = expand_211 = None
        pow_145 = torch.ops.aten.pow.Tensor_Scalar(getitem_336, 2.0)
        sum_107 = torch.ops.aten.sum.dim_IntList(pow_145, [-1], True);  pow_145 = None
        pow_146 = torch.ops.aten.pow.Tensor_Scalar(sum_107, 0.5);  sum_107 = None
        clamp_min_71 = torch.ops.aten.clamp_min.default(pow_146, 1e-12);  pow_146 = None
        expand_212 = torch.ops.aten.expand.default(clamp_min_71, [8, 16, 48, 784]);  clamp_min_71 = None
        div_118 = torch.ops.aten.div.Tensor(getitem_336, expand_212);  getitem_336 = expand_212 = None
        permute_344 = torch.ops.aten.permute.default(div_118, [0, 1, 3, 2]);  div_118 = None
        expand_213 = torch.ops.aten.expand.default(div_117, [8, 16, 48, 784]);  div_117 = None
        clone_395 = torch.ops.aten.clone.default(expand_213, memory_format = torch.contiguous_format);  expand_213 = None
        view_669 = torch.ops.aten.view.default(clone_395, [128, 48, 784]);  clone_395 = None
        expand_214 = torch.ops.aten.expand.default(permute_344, [8, 16, 784, 48]);  permute_344 = None
        clone_396 = torch.ops.aten.clone.default(expand_214, memory_format = torch.contiguous_format);  expand_214 = None
        view_670 = torch.ops.aten.view.default(clone_396, [128, 784, 48]);  clone_396 = None
        bmm_70 = torch.ops.aten.bmm.default(view_669, view_670);  view_669 = view_670 = None
        view_671 = torch.ops.aten.view.default(bmm_70, [8, 16, 48, 48]);  bmm_70 = None
        scalar_tensor_default_12 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_12 = torch.ops.aten.ge.Scalar(arg309_1, 0)
        neg_default_12 = torch.ops.aten.neg.default(scalar_tensor_default_12)
        where_self_12 = torch.ops.aten.where.self(ge_scalar_12, scalar_tensor_default_12, neg_default_12);  ge_scalar_12 = scalar_tensor_default_12 = neg_default_12 = None
        mul_tensor_36 = torch.ops.aten.mul.Tensor(view_671, where_self_12);  view_671 = None
        amax_default_12 = torch.ops.aten.amax.default(mul_tensor_36, [-1], True)
        sub_tensor_12 = torch.ops.aten.sub.Tensor(mul_tensor_36, amax_default_12);  mul_tensor_36 = amax_default_12 = None
        mul_tensor_37 = torch.ops.aten.mul.Tensor(where_self_12, arg309_1);  where_self_12 = arg309_1 = None
        mul_tensor_38 = torch.ops.aten.mul.Tensor(sub_tensor_12, mul_tensor_37);  sub_tensor_12 = mul_tensor_37 = None
        exp_35 = torch.ops.aten.exp.default(mul_tensor_38);  mul_tensor_38 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_119 = torch.ops.aten.div.Tensor(exp_35, sum_108);  exp_35 = sum_108 = None
        expand_215 = torch.ops.aten.expand.default(div_119, [8, 16, 48, 48]);  div_119 = None
        view_672 = torch.ops.aten.view.default(expand_215, [128, 48, 48]);  expand_215 = None
        expand_216 = torch.ops.aten.expand.default(getitem_337, [8, 16, 48, 784]);  getitem_337 = None
        clone_398 = torch.ops.aten.clone.default(expand_216, memory_format = torch.contiguous_format);  expand_216 = None
        view_673 = torch.ops.aten.view.default(clone_398, [128, 48, 784]);  clone_398 = None
        bmm_71 = torch.ops.aten.bmm.default(view_672, view_673);  view_672 = view_673 = None
        view_674 = torch.ops.aten.view.default(bmm_71, [8, 16, 48, 784]);  bmm_71 = None
        permute_345 = torch.ops.aten.permute.default(view_674, [0, 3, 1, 2]);  view_674 = None
        view_675 = torch.ops.aten.view.default(permute_345, [8, 784, 768]);  permute_345 = None
        permute_346 = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        clone_399 = torch.ops.aten.clone.default(view_675, memory_format = torch.contiguous_format);  view_675 = None
        view_676 = torch.ops.aten.view.default(clone_399, [6272, 768]);  clone_399 = None
        mm_37 = torch.ops.aten.mm.default(view_676, permute_346);  view_676 = permute_346 = None
        view_677 = torch.ops.aten.view.default(mm_37, [8, 784, 768]);  mm_37 = None
        add_532 = torch.ops.aten.add.Tensor(view_677, arg311_1);  view_677 = arg311_1 = None
        mul_724 = torch.ops.aten.mul.Tensor(arg304_1, add_532);  arg304_1 = add_532 = None
        add_533 = torch.ops.aten.add.Tensor(add_529, mul_724);  add_529 = mul_724 = None
        clone_401 = torch.ops.aten.clone.default(add_533, memory_format = torch.contiguous_format)
        var_mean_111 = torch.ops.aten.var_mean.correction(clone_401, [2], correction = 0, keepdim = True)
        getitem_338 = var_mean_111[0]
        getitem_339 = var_mean_111[1];  var_mean_111 = None
        add_534 = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_111 = torch.ops.aten.rsqrt.default(add_534);  add_534 = None
        sub_188 = torch.ops.aten.sub.Tensor(clone_401, getitem_339);  clone_401 = getitem_339 = None
        mul_725 = torch.ops.aten.mul.Tensor(sub_188, rsqrt_111);  sub_188 = rsqrt_111 = None
        mul_726 = torch.ops.aten.mul.Tensor(mul_725, arg313_1);  mul_725 = arg313_1 = None
        add_535 = torch.ops.aten.add.Tensor(mul_726, arg314_1);  mul_726 = arg314_1 = None
        permute_347 = torch.ops.aten.permute.default(add_535, [0, 2, 1]);  add_535 = None
        view_678 = torch.ops.aten.view.default(permute_347, [8, 768, 28, 28]);  permute_347 = None
        convolution_78 = torch.ops.aten.convolution.default(view_678, arg315_1, arg316_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_678 = arg315_1 = arg316_1 = None
        mul_727 = torch.ops.aten.mul.Tensor(convolution_78, 0.5)
        mul_728 = torch.ops.aten.mul.Tensor(convolution_78, 0.7071067811865476);  convolution_78 = None
        erf_76 = torch.ops.aten.erf.default(mul_728);  mul_728 = None
        add_536 = torch.ops.aten.add.Tensor(erf_76, 1);  erf_76 = None
        mul_729 = torch.ops.aten.mul.Tensor(mul_727, add_536);  mul_727 = add_536 = None
        add_537 = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_537);  add_537 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_730 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_730, -1);  mul_730 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_189 = torch.ops.aten.sub.Tensor(mul_729, unsqueeze_345);  mul_729 = unsqueeze_345 = None
        mul_731 = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_347);  sub_189 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_732 = torch.ops.aten.mul.Tensor(mul_731, unsqueeze_349);  mul_731 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_538 = torch.ops.aten.add.Tensor(mul_732, unsqueeze_351);  mul_732 = unsqueeze_351 = None
        convolution_79 = torch.ops.aten.convolution.default(add_538, arg321_1, arg322_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_538 = arg321_1 = arg322_1 = None
        view_679 = torch.ops.aten.view.default(convolution_79, [8, 768, 784]);  convolution_79 = None
        permute_348 = torch.ops.aten.permute.default(view_679, [0, 2, 1]);  view_679 = None
        mul_733 = torch.ops.aten.mul.Tensor(arg312_1, permute_348);  arg312_1 = permute_348 = None
        add_539 = torch.ops.aten.add.Tensor(add_533, mul_733);  add_533 = mul_733 = None
        clone_402 = torch.ops.aten.clone.default(add_539, memory_format = torch.contiguous_format)
        var_mean_112 = torch.ops.aten.var_mean.correction(clone_402, [2], correction = 0, keepdim = True)
        getitem_340 = var_mean_112[0]
        getitem_341 = var_mean_112[1];  var_mean_112 = None
        add_540 = torch.ops.aten.add.Tensor(getitem_340, 1e-06);  getitem_340 = None
        rsqrt_112 = torch.ops.aten.rsqrt.default(add_540);  add_540 = None
        sub_190 = torch.ops.aten.sub.Tensor(clone_402, getitem_341);  clone_402 = getitem_341 = None
        mul_734 = torch.ops.aten.mul.Tensor(sub_190, rsqrt_112);  sub_190 = rsqrt_112 = None
        mul_735 = torch.ops.aten.mul.Tensor(mul_734, arg324_1);  mul_734 = arg324_1 = None
        add_541 = torch.ops.aten.add.Tensor(mul_735, arg325_1);  mul_735 = arg325_1 = None
        view_680 = torch.ops.aten.view.default(add_541, [6272, 768]);  add_541 = None
        permute_349 = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg327_1, view_680, permute_349);  arg327_1 = view_680 = permute_349 = None
        view_681 = torch.ops.aten.view.default(addmm_117, [8, 784, 3072]);  addmm_117 = None
        mul_736 = torch.ops.aten.mul.Tensor(view_681, 0.5)
        mul_737 = torch.ops.aten.mul.Tensor(view_681, 0.7071067811865476);  view_681 = None
        erf_77 = torch.ops.aten.erf.default(mul_737);  mul_737 = None
        add_542 = torch.ops.aten.add.Tensor(erf_77, 1);  erf_77 = None
        mul_738 = torch.ops.aten.mul.Tensor(mul_736, add_542);  mul_736 = add_542 = None
        view_682 = torch.ops.aten.view.default(mul_738, [6272, 3072]);  mul_738 = None
        permute_350 = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg329_1, view_682, permute_350);  arg329_1 = view_682 = permute_350 = None
        view_683 = torch.ops.aten.view.default(addmm_118, [8, 784, 768]);  addmm_118 = None
        mul_739 = torch.ops.aten.mul.Tensor(arg323_1, view_683);  arg323_1 = view_683 = None
        add_543 = torch.ops.aten.add.Tensor(add_539, mul_739);  add_539 = mul_739 = None
        clone_405 = torch.ops.aten.clone.default(add_543, memory_format = torch.contiguous_format)
        var_mean_113 = torch.ops.aten.var_mean.correction(clone_405, [2], correction = 0, keepdim = True)
        getitem_342 = var_mean_113[0]
        getitem_343 = var_mean_113[1];  var_mean_113 = None
        add_544 = torch.ops.aten.add.Tensor(getitem_342, 1e-06);  getitem_342 = None
        rsqrt_113 = torch.ops.aten.rsqrt.default(add_544);  add_544 = None
        sub_191 = torch.ops.aten.sub.Tensor(clone_405, getitem_343);  clone_405 = getitem_343 = None
        mul_740 = torch.ops.aten.mul.Tensor(sub_191, rsqrt_113);  sub_191 = rsqrt_113 = None
        mul_741 = torch.ops.aten.mul.Tensor(mul_740, arg331_1);  mul_740 = arg331_1 = None
        add_545 = torch.ops.aten.add.Tensor(mul_741, arg332_1);  mul_741 = arg332_1 = None
        view_684 = torch.ops.aten.view.default(add_545, [6272, 768]);  add_545 = None
        permute_351 = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg334_1, view_684, permute_351);  arg334_1 = view_684 = permute_351 = None
        view_685 = torch.ops.aten.view.default(addmm_119, [8, 784, 2304]);  addmm_119 = None
        view_686 = torch.ops.aten.view.default(view_685, [8, 784, 3, 16, 48]);  view_685 = None
        permute_352 = torch.ops.aten.permute.default(view_686, [2, 0, 3, 4, 1]);  view_686 = None
        unbind_36 = torch.ops.aten.unbind.int(permute_352);  permute_352 = None
        getitem_344 = unbind_36[0]
        getitem_345 = unbind_36[1]
        getitem_346 = unbind_36[2];  unbind_36 = None
        pow_147 = torch.ops.aten.pow.Tensor_Scalar(getitem_344, 2.0)
        sum_109 = torch.ops.aten.sum.dim_IntList(pow_147, [-1], True);  pow_147 = None
        pow_148 = torch.ops.aten.pow.Tensor_Scalar(sum_109, 0.5);  sum_109 = None
        clamp_min_72 = torch.ops.aten.clamp_min.default(pow_148, 1e-12);  pow_148 = None
        expand_217 = torch.ops.aten.expand.default(clamp_min_72, [8, 16, 48, 784]);  clamp_min_72 = None
        div_120 = torch.ops.aten.div.Tensor(getitem_344, expand_217);  getitem_344 = expand_217 = None
        pow_149 = torch.ops.aten.pow.Tensor_Scalar(getitem_345, 2.0)
        sum_110 = torch.ops.aten.sum.dim_IntList(pow_149, [-1], True);  pow_149 = None
        pow_150 = torch.ops.aten.pow.Tensor_Scalar(sum_110, 0.5);  sum_110 = None
        clamp_min_73 = torch.ops.aten.clamp_min.default(pow_150, 1e-12);  pow_150 = None
        expand_218 = torch.ops.aten.expand.default(clamp_min_73, [8, 16, 48, 784]);  clamp_min_73 = None
        div_121 = torch.ops.aten.div.Tensor(getitem_345, expand_218);  getitem_345 = expand_218 = None
        permute_353 = torch.ops.aten.permute.default(div_121, [0, 1, 3, 2]);  div_121 = None
        expand_219 = torch.ops.aten.expand.default(div_120, [8, 16, 48, 784]);  div_120 = None
        clone_406 = torch.ops.aten.clone.default(expand_219, memory_format = torch.contiguous_format);  expand_219 = None
        view_687 = torch.ops.aten.view.default(clone_406, [128, 48, 784]);  clone_406 = None
        expand_220 = torch.ops.aten.expand.default(permute_353, [8, 16, 784, 48]);  permute_353 = None
        clone_407 = torch.ops.aten.clone.default(expand_220, memory_format = torch.contiguous_format);  expand_220 = None
        view_688 = torch.ops.aten.view.default(clone_407, [128, 784, 48]);  clone_407 = None
        bmm_72 = torch.ops.aten.bmm.default(view_687, view_688);  view_687 = view_688 = None
        view_689 = torch.ops.aten.view.default(bmm_72, [8, 16, 48, 48]);  bmm_72 = None
        scalar_tensor_default_11 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_11 = torch.ops.aten.ge.Scalar(arg335_1, 0)
        neg_default_11 = torch.ops.aten.neg.default(scalar_tensor_default_11)
        where_self_11 = torch.ops.aten.where.self(ge_scalar_11, scalar_tensor_default_11, neg_default_11);  ge_scalar_11 = scalar_tensor_default_11 = neg_default_11 = None
        mul_tensor_33 = torch.ops.aten.mul.Tensor(view_689, where_self_11);  view_689 = None
        amax_default_11 = torch.ops.aten.amax.default(mul_tensor_33, [-1], True)
        sub_tensor_11 = torch.ops.aten.sub.Tensor(mul_tensor_33, amax_default_11);  mul_tensor_33 = amax_default_11 = None
        mul_tensor_34 = torch.ops.aten.mul.Tensor(where_self_11, arg335_1);  where_self_11 = arg335_1 = None
        mul_tensor_35 = torch.ops.aten.mul.Tensor(sub_tensor_11, mul_tensor_34);  sub_tensor_11 = mul_tensor_34 = None
        exp_36 = torch.ops.aten.exp.default(mul_tensor_35);  mul_tensor_35 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_122 = torch.ops.aten.div.Tensor(exp_36, sum_111);  exp_36 = sum_111 = None
        expand_221 = torch.ops.aten.expand.default(div_122, [8, 16, 48, 48]);  div_122 = None
        view_690 = torch.ops.aten.view.default(expand_221, [128, 48, 48]);  expand_221 = None
        expand_222 = torch.ops.aten.expand.default(getitem_346, [8, 16, 48, 784]);  getitem_346 = None
        clone_409 = torch.ops.aten.clone.default(expand_222, memory_format = torch.contiguous_format);  expand_222 = None
        view_691 = torch.ops.aten.view.default(clone_409, [128, 48, 784]);  clone_409 = None
        bmm_73 = torch.ops.aten.bmm.default(view_690, view_691);  view_690 = view_691 = None
        view_692 = torch.ops.aten.view.default(bmm_73, [8, 16, 48, 784]);  bmm_73 = None
        permute_354 = torch.ops.aten.permute.default(view_692, [0, 3, 1, 2]);  view_692 = None
        view_693 = torch.ops.aten.view.default(permute_354, [8, 784, 768]);  permute_354 = None
        permute_355 = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        clone_410 = torch.ops.aten.clone.default(view_693, memory_format = torch.contiguous_format);  view_693 = None
        view_694 = torch.ops.aten.view.default(clone_410, [6272, 768]);  clone_410 = None
        mm_38 = torch.ops.aten.mm.default(view_694, permute_355);  view_694 = permute_355 = None
        view_695 = torch.ops.aten.view.default(mm_38, [8, 784, 768]);  mm_38 = None
        add_546 = torch.ops.aten.add.Tensor(view_695, arg337_1);  view_695 = arg337_1 = None
        mul_743 = torch.ops.aten.mul.Tensor(arg330_1, add_546);  arg330_1 = add_546 = None
        add_547 = torch.ops.aten.add.Tensor(add_543, mul_743);  add_543 = mul_743 = None
        clone_412 = torch.ops.aten.clone.default(add_547, memory_format = torch.contiguous_format)
        var_mean_114 = torch.ops.aten.var_mean.correction(clone_412, [2], correction = 0, keepdim = True)
        getitem_347 = var_mean_114[0]
        getitem_348 = var_mean_114[1];  var_mean_114 = None
        add_548 = torch.ops.aten.add.Tensor(getitem_347, 1e-06);  getitem_347 = None
        rsqrt_114 = torch.ops.aten.rsqrt.default(add_548);  add_548 = None
        sub_193 = torch.ops.aten.sub.Tensor(clone_412, getitem_348);  clone_412 = getitem_348 = None
        mul_744 = torch.ops.aten.mul.Tensor(sub_193, rsqrt_114);  sub_193 = rsqrt_114 = None
        mul_745 = torch.ops.aten.mul.Tensor(mul_744, arg339_1);  mul_744 = arg339_1 = None
        add_549 = torch.ops.aten.add.Tensor(mul_745, arg340_1);  mul_745 = arg340_1 = None
        permute_356 = torch.ops.aten.permute.default(add_549, [0, 2, 1]);  add_549 = None
        view_696 = torch.ops.aten.view.default(permute_356, [8, 768, 28, 28]);  permute_356 = None
        convolution_80 = torch.ops.aten.convolution.default(view_696, arg341_1, arg342_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_696 = arg341_1 = arg342_1 = None
        mul_746 = torch.ops.aten.mul.Tensor(convolution_80, 0.5)
        mul_747 = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476);  convolution_80 = None
        erf_78 = torch.ops.aten.erf.default(mul_747);  mul_747 = None
        add_550 = torch.ops.aten.add.Tensor(erf_78, 1);  erf_78 = None
        mul_748 = torch.ops.aten.mul.Tensor(mul_746, add_550);  mul_746 = add_550 = None
        add_551 = torch.ops.aten.add.Tensor(arg344_1, 1e-05);  arg344_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_551);  add_551 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_749 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_749, -1);  mul_749 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_194 = torch.ops.aten.sub.Tensor(mul_748, unsqueeze_353);  mul_748 = unsqueeze_353 = None
        mul_750 = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_355);  sub_194 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_751 = torch.ops.aten.mul.Tensor(mul_750, unsqueeze_357);  mul_750 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_552 = torch.ops.aten.add.Tensor(mul_751, unsqueeze_359);  mul_751 = unsqueeze_359 = None
        convolution_81 = torch.ops.aten.convolution.default(add_552, arg347_1, arg348_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_552 = arg347_1 = arg348_1 = None
        view_697 = torch.ops.aten.view.default(convolution_81, [8, 768, 784]);  convolution_81 = None
        permute_357 = torch.ops.aten.permute.default(view_697, [0, 2, 1]);  view_697 = None
        mul_752 = torch.ops.aten.mul.Tensor(arg338_1, permute_357);  arg338_1 = permute_357 = None
        add_553 = torch.ops.aten.add.Tensor(add_547, mul_752);  add_547 = mul_752 = None
        clone_413 = torch.ops.aten.clone.default(add_553, memory_format = torch.contiguous_format)
        var_mean_115 = torch.ops.aten.var_mean.correction(clone_413, [2], correction = 0, keepdim = True)
        getitem_349 = var_mean_115[0]
        getitem_350 = var_mean_115[1];  var_mean_115 = None
        add_554 = torch.ops.aten.add.Tensor(getitem_349, 1e-06);  getitem_349 = None
        rsqrt_115 = torch.ops.aten.rsqrt.default(add_554);  add_554 = None
        sub_195 = torch.ops.aten.sub.Tensor(clone_413, getitem_350);  clone_413 = getitem_350 = None
        mul_753 = torch.ops.aten.mul.Tensor(sub_195, rsqrt_115);  sub_195 = rsqrt_115 = None
        mul_754 = torch.ops.aten.mul.Tensor(mul_753, arg350_1);  mul_753 = arg350_1 = None
        add_555 = torch.ops.aten.add.Tensor(mul_754, arg351_1);  mul_754 = arg351_1 = None
        view_698 = torch.ops.aten.view.default(add_555, [6272, 768]);  add_555 = None
        permute_358 = torch.ops.aten.permute.default(arg352_1, [1, 0]);  arg352_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg353_1, view_698, permute_358);  arg353_1 = view_698 = permute_358 = None
        view_699 = torch.ops.aten.view.default(addmm_120, [8, 784, 3072]);  addmm_120 = None
        mul_755 = torch.ops.aten.mul.Tensor(view_699, 0.5)
        mul_756 = torch.ops.aten.mul.Tensor(view_699, 0.7071067811865476);  view_699 = None
        erf_79 = torch.ops.aten.erf.default(mul_756);  mul_756 = None
        add_556 = torch.ops.aten.add.Tensor(erf_79, 1);  erf_79 = None
        mul_757 = torch.ops.aten.mul.Tensor(mul_755, add_556);  mul_755 = add_556 = None
        view_700 = torch.ops.aten.view.default(mul_757, [6272, 3072]);  mul_757 = None
        permute_359 = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg355_1, view_700, permute_359);  arg355_1 = view_700 = permute_359 = None
        view_701 = torch.ops.aten.view.default(addmm_121, [8, 784, 768]);  addmm_121 = None
        mul_758 = torch.ops.aten.mul.Tensor(arg349_1, view_701);  arg349_1 = view_701 = None
        add_557 = torch.ops.aten.add.Tensor(add_553, mul_758);  add_553 = mul_758 = None
        clone_416 = torch.ops.aten.clone.default(add_557, memory_format = torch.contiguous_format)
        var_mean_116 = torch.ops.aten.var_mean.correction(clone_416, [2], correction = 0, keepdim = True)
        getitem_351 = var_mean_116[0]
        getitem_352 = var_mean_116[1];  var_mean_116 = None
        add_558 = torch.ops.aten.add.Tensor(getitem_351, 1e-06);  getitem_351 = None
        rsqrt_116 = torch.ops.aten.rsqrt.default(add_558);  add_558 = None
        sub_196 = torch.ops.aten.sub.Tensor(clone_416, getitem_352);  clone_416 = getitem_352 = None
        mul_759 = torch.ops.aten.mul.Tensor(sub_196, rsqrt_116);  sub_196 = rsqrt_116 = None
        mul_760 = torch.ops.aten.mul.Tensor(mul_759, arg357_1);  mul_759 = arg357_1 = None
        add_559 = torch.ops.aten.add.Tensor(mul_760, arg358_1);  mul_760 = arg358_1 = None
        view_702 = torch.ops.aten.view.default(add_559, [6272, 768]);  add_559 = None
        permute_360 = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg360_1, view_702, permute_360);  arg360_1 = view_702 = permute_360 = None
        view_703 = torch.ops.aten.view.default(addmm_122, [8, 784, 2304]);  addmm_122 = None
        view_704 = torch.ops.aten.view.default(view_703, [8, 784, 3, 16, 48]);  view_703 = None
        permute_361 = torch.ops.aten.permute.default(view_704, [2, 0, 3, 4, 1]);  view_704 = None
        unbind_37 = torch.ops.aten.unbind.int(permute_361);  permute_361 = None
        getitem_353 = unbind_37[0]
        getitem_354 = unbind_37[1]
        getitem_355 = unbind_37[2];  unbind_37 = None
        pow_151 = torch.ops.aten.pow.Tensor_Scalar(getitem_353, 2.0)
        sum_112 = torch.ops.aten.sum.dim_IntList(pow_151, [-1], True);  pow_151 = None
        pow_152 = torch.ops.aten.pow.Tensor_Scalar(sum_112, 0.5);  sum_112 = None
        clamp_min_74 = torch.ops.aten.clamp_min.default(pow_152, 1e-12);  pow_152 = None
        expand_223 = torch.ops.aten.expand.default(clamp_min_74, [8, 16, 48, 784]);  clamp_min_74 = None
        div_123 = torch.ops.aten.div.Tensor(getitem_353, expand_223);  getitem_353 = expand_223 = None
        pow_153 = torch.ops.aten.pow.Tensor_Scalar(getitem_354, 2.0)
        sum_113 = torch.ops.aten.sum.dim_IntList(pow_153, [-1], True);  pow_153 = None
        pow_154 = torch.ops.aten.pow.Tensor_Scalar(sum_113, 0.5);  sum_113 = None
        clamp_min_75 = torch.ops.aten.clamp_min.default(pow_154, 1e-12);  pow_154 = None
        expand_224 = torch.ops.aten.expand.default(clamp_min_75, [8, 16, 48, 784]);  clamp_min_75 = None
        div_124 = torch.ops.aten.div.Tensor(getitem_354, expand_224);  getitem_354 = expand_224 = None
        permute_362 = torch.ops.aten.permute.default(div_124, [0, 1, 3, 2]);  div_124 = None
        expand_225 = torch.ops.aten.expand.default(div_123, [8, 16, 48, 784]);  div_123 = None
        clone_417 = torch.ops.aten.clone.default(expand_225, memory_format = torch.contiguous_format);  expand_225 = None
        view_705 = torch.ops.aten.view.default(clone_417, [128, 48, 784]);  clone_417 = None
        expand_226 = torch.ops.aten.expand.default(permute_362, [8, 16, 784, 48]);  permute_362 = None
        clone_418 = torch.ops.aten.clone.default(expand_226, memory_format = torch.contiguous_format);  expand_226 = None
        view_706 = torch.ops.aten.view.default(clone_418, [128, 784, 48]);  clone_418 = None
        bmm_74 = torch.ops.aten.bmm.default(view_705, view_706);  view_705 = view_706 = None
        view_707 = torch.ops.aten.view.default(bmm_74, [8, 16, 48, 48]);  bmm_74 = None
        scalar_tensor_default_10 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_10 = torch.ops.aten.ge.Scalar(arg361_1, 0)
        neg_default_10 = torch.ops.aten.neg.default(scalar_tensor_default_10)
        where_self_10 = torch.ops.aten.where.self(ge_scalar_10, scalar_tensor_default_10, neg_default_10);  ge_scalar_10 = scalar_tensor_default_10 = neg_default_10 = None
        mul_tensor_30 = torch.ops.aten.mul.Tensor(view_707, where_self_10);  view_707 = None
        amax_default_10 = torch.ops.aten.amax.default(mul_tensor_30, [-1], True)
        sub_tensor_10 = torch.ops.aten.sub.Tensor(mul_tensor_30, amax_default_10);  mul_tensor_30 = amax_default_10 = None
        mul_tensor_31 = torch.ops.aten.mul.Tensor(where_self_10, arg361_1);  where_self_10 = arg361_1 = None
        mul_tensor_32 = torch.ops.aten.mul.Tensor(sub_tensor_10, mul_tensor_31);  sub_tensor_10 = mul_tensor_31 = None
        exp_37 = torch.ops.aten.exp.default(mul_tensor_32);  mul_tensor_32 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_125 = torch.ops.aten.div.Tensor(exp_37, sum_114);  exp_37 = sum_114 = None
        expand_227 = torch.ops.aten.expand.default(div_125, [8, 16, 48, 48]);  div_125 = None
        view_708 = torch.ops.aten.view.default(expand_227, [128, 48, 48]);  expand_227 = None
        expand_228 = torch.ops.aten.expand.default(getitem_355, [8, 16, 48, 784]);  getitem_355 = None
        clone_420 = torch.ops.aten.clone.default(expand_228, memory_format = torch.contiguous_format);  expand_228 = None
        view_709 = torch.ops.aten.view.default(clone_420, [128, 48, 784]);  clone_420 = None
        bmm_75 = torch.ops.aten.bmm.default(view_708, view_709);  view_708 = view_709 = None
        view_710 = torch.ops.aten.view.default(bmm_75, [8, 16, 48, 784]);  bmm_75 = None
        permute_363 = torch.ops.aten.permute.default(view_710, [0, 3, 1, 2]);  view_710 = None
        view_711 = torch.ops.aten.view.default(permute_363, [8, 784, 768]);  permute_363 = None
        permute_364 = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        clone_421 = torch.ops.aten.clone.default(view_711, memory_format = torch.contiguous_format);  view_711 = None
        view_712 = torch.ops.aten.view.default(clone_421, [6272, 768]);  clone_421 = None
        mm_39 = torch.ops.aten.mm.default(view_712, permute_364);  view_712 = permute_364 = None
        view_713 = torch.ops.aten.view.default(mm_39, [8, 784, 768]);  mm_39 = None
        add_560 = torch.ops.aten.add.Tensor(view_713, arg363_1);  view_713 = arg363_1 = None
        mul_762 = torch.ops.aten.mul.Tensor(arg356_1, add_560);  arg356_1 = add_560 = None
        add_561 = torch.ops.aten.add.Tensor(add_557, mul_762);  add_557 = mul_762 = None
        clone_423 = torch.ops.aten.clone.default(add_561, memory_format = torch.contiguous_format)
        var_mean_117 = torch.ops.aten.var_mean.correction(clone_423, [2], correction = 0, keepdim = True)
        getitem_356 = var_mean_117[0]
        getitem_357 = var_mean_117[1];  var_mean_117 = None
        add_562 = torch.ops.aten.add.Tensor(getitem_356, 1e-06);  getitem_356 = None
        rsqrt_117 = torch.ops.aten.rsqrt.default(add_562);  add_562 = None
        sub_198 = torch.ops.aten.sub.Tensor(clone_423, getitem_357);  clone_423 = getitem_357 = None
        mul_763 = torch.ops.aten.mul.Tensor(sub_198, rsqrt_117);  sub_198 = rsqrt_117 = None
        mul_764 = torch.ops.aten.mul.Tensor(mul_763, arg365_1);  mul_763 = arg365_1 = None
        add_563 = torch.ops.aten.add.Tensor(mul_764, arg366_1);  mul_764 = arg366_1 = None
        permute_365 = torch.ops.aten.permute.default(add_563, [0, 2, 1]);  add_563 = None
        view_714 = torch.ops.aten.view.default(permute_365, [8, 768, 28, 28]);  permute_365 = None
        convolution_82 = torch.ops.aten.convolution.default(view_714, arg367_1, arg368_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_714 = arg367_1 = arg368_1 = None
        mul_765 = torch.ops.aten.mul.Tensor(convolution_82, 0.5)
        mul_766 = torch.ops.aten.mul.Tensor(convolution_82, 0.7071067811865476);  convolution_82 = None
        erf_80 = torch.ops.aten.erf.default(mul_766);  mul_766 = None
        add_564 = torch.ops.aten.add.Tensor(erf_80, 1);  erf_80 = None
        mul_767 = torch.ops.aten.mul.Tensor(mul_765, add_564);  mul_765 = add_564 = None
        add_565 = torch.ops.aten.add.Tensor(arg370_1, 1e-05);  arg370_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_565);  add_565 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_768 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_768, -1);  mul_768 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_199 = torch.ops.aten.sub.Tensor(mul_767, unsqueeze_361);  mul_767 = unsqueeze_361 = None
        mul_769 = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_363);  sub_199 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_770 = torch.ops.aten.mul.Tensor(mul_769, unsqueeze_365);  mul_769 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_566 = torch.ops.aten.add.Tensor(mul_770, unsqueeze_367);  mul_770 = unsqueeze_367 = None
        convolution_83 = torch.ops.aten.convolution.default(add_566, arg373_1, arg374_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_566 = arg373_1 = arg374_1 = None
        view_715 = torch.ops.aten.view.default(convolution_83, [8, 768, 784]);  convolution_83 = None
        permute_366 = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
        mul_771 = torch.ops.aten.mul.Tensor(arg364_1, permute_366);  arg364_1 = permute_366 = None
        add_567 = torch.ops.aten.add.Tensor(add_561, mul_771);  add_561 = mul_771 = None
        clone_424 = torch.ops.aten.clone.default(add_567, memory_format = torch.contiguous_format)
        var_mean_118 = torch.ops.aten.var_mean.correction(clone_424, [2], correction = 0, keepdim = True)
        getitem_358 = var_mean_118[0]
        getitem_359 = var_mean_118[1];  var_mean_118 = None
        add_568 = torch.ops.aten.add.Tensor(getitem_358, 1e-06);  getitem_358 = None
        rsqrt_118 = torch.ops.aten.rsqrt.default(add_568);  add_568 = None
        sub_200 = torch.ops.aten.sub.Tensor(clone_424, getitem_359);  clone_424 = getitem_359 = None
        mul_772 = torch.ops.aten.mul.Tensor(sub_200, rsqrt_118);  sub_200 = rsqrt_118 = None
        mul_773 = torch.ops.aten.mul.Tensor(mul_772, arg376_1);  mul_772 = arg376_1 = None
        add_569 = torch.ops.aten.add.Tensor(mul_773, arg377_1);  mul_773 = arg377_1 = None
        view_716 = torch.ops.aten.view.default(add_569, [6272, 768]);  add_569 = None
        permute_367 = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg379_1, view_716, permute_367);  arg379_1 = view_716 = permute_367 = None
        view_717 = torch.ops.aten.view.default(addmm_123, [8, 784, 3072]);  addmm_123 = None
        mul_774 = torch.ops.aten.mul.Tensor(view_717, 0.5)
        mul_775 = torch.ops.aten.mul.Tensor(view_717, 0.7071067811865476);  view_717 = None
        erf_81 = torch.ops.aten.erf.default(mul_775);  mul_775 = None
        add_570 = torch.ops.aten.add.Tensor(erf_81, 1);  erf_81 = None
        mul_776 = torch.ops.aten.mul.Tensor(mul_774, add_570);  mul_774 = add_570 = None
        view_718 = torch.ops.aten.view.default(mul_776, [6272, 3072]);  mul_776 = None
        permute_368 = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg381_1, view_718, permute_368);  arg381_1 = view_718 = permute_368 = None
        view_719 = torch.ops.aten.view.default(addmm_124, [8, 784, 768]);  addmm_124 = None
        mul_777 = torch.ops.aten.mul.Tensor(arg375_1, view_719);  arg375_1 = view_719 = None
        add_571 = torch.ops.aten.add.Tensor(add_567, mul_777);  add_567 = mul_777 = None
        clone_427 = torch.ops.aten.clone.default(add_571, memory_format = torch.contiguous_format)
        var_mean_119 = torch.ops.aten.var_mean.correction(clone_427, [2], correction = 0, keepdim = True)
        getitem_360 = var_mean_119[0]
        getitem_361 = var_mean_119[1];  var_mean_119 = None
        add_572 = torch.ops.aten.add.Tensor(getitem_360, 1e-06);  getitem_360 = None
        rsqrt_119 = torch.ops.aten.rsqrt.default(add_572);  add_572 = None
        sub_201 = torch.ops.aten.sub.Tensor(clone_427, getitem_361);  clone_427 = getitem_361 = None
        mul_778 = torch.ops.aten.mul.Tensor(sub_201, rsqrt_119);  sub_201 = rsqrt_119 = None
        mul_779 = torch.ops.aten.mul.Tensor(mul_778, arg383_1);  mul_778 = arg383_1 = None
        add_573 = torch.ops.aten.add.Tensor(mul_779, arg384_1);  mul_779 = arg384_1 = None
        view_720 = torch.ops.aten.view.default(add_573, [6272, 768]);  add_573 = None
        permute_369 = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg386_1, view_720, permute_369);  arg386_1 = view_720 = permute_369 = None
        view_721 = torch.ops.aten.view.default(addmm_125, [8, 784, 2304]);  addmm_125 = None
        view_722 = torch.ops.aten.view.default(view_721, [8, 784, 3, 16, 48]);  view_721 = None
        permute_370 = torch.ops.aten.permute.default(view_722, [2, 0, 3, 4, 1]);  view_722 = None
        unbind_38 = torch.ops.aten.unbind.int(permute_370);  permute_370 = None
        getitem_362 = unbind_38[0]
        getitem_363 = unbind_38[1]
        getitem_364 = unbind_38[2];  unbind_38 = None
        pow_155 = torch.ops.aten.pow.Tensor_Scalar(getitem_362, 2.0)
        sum_115 = torch.ops.aten.sum.dim_IntList(pow_155, [-1], True);  pow_155 = None
        pow_156 = torch.ops.aten.pow.Tensor_Scalar(sum_115, 0.5);  sum_115 = None
        clamp_min_76 = torch.ops.aten.clamp_min.default(pow_156, 1e-12);  pow_156 = None
        expand_229 = torch.ops.aten.expand.default(clamp_min_76, [8, 16, 48, 784]);  clamp_min_76 = None
        div_126 = torch.ops.aten.div.Tensor(getitem_362, expand_229);  getitem_362 = expand_229 = None
        pow_157 = torch.ops.aten.pow.Tensor_Scalar(getitem_363, 2.0)
        sum_116 = torch.ops.aten.sum.dim_IntList(pow_157, [-1], True);  pow_157 = None
        pow_158 = torch.ops.aten.pow.Tensor_Scalar(sum_116, 0.5);  sum_116 = None
        clamp_min_77 = torch.ops.aten.clamp_min.default(pow_158, 1e-12);  pow_158 = None
        expand_230 = torch.ops.aten.expand.default(clamp_min_77, [8, 16, 48, 784]);  clamp_min_77 = None
        div_127 = torch.ops.aten.div.Tensor(getitem_363, expand_230);  getitem_363 = expand_230 = None
        permute_371 = torch.ops.aten.permute.default(div_127, [0, 1, 3, 2]);  div_127 = None
        expand_231 = torch.ops.aten.expand.default(div_126, [8, 16, 48, 784]);  div_126 = None
        clone_428 = torch.ops.aten.clone.default(expand_231, memory_format = torch.contiguous_format);  expand_231 = None
        view_723 = torch.ops.aten.view.default(clone_428, [128, 48, 784]);  clone_428 = None
        expand_232 = torch.ops.aten.expand.default(permute_371, [8, 16, 784, 48]);  permute_371 = None
        clone_429 = torch.ops.aten.clone.default(expand_232, memory_format = torch.contiguous_format);  expand_232 = None
        view_724 = torch.ops.aten.view.default(clone_429, [128, 784, 48]);  clone_429 = None
        bmm_76 = torch.ops.aten.bmm.default(view_723, view_724);  view_723 = view_724 = None
        view_725 = torch.ops.aten.view.default(bmm_76, [8, 16, 48, 48]);  bmm_76 = None
        scalar_tensor_default_9 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_9 = torch.ops.aten.ge.Scalar(arg387_1, 0)
        neg_default_9 = torch.ops.aten.neg.default(scalar_tensor_default_9)
        where_self_9 = torch.ops.aten.where.self(ge_scalar_9, scalar_tensor_default_9, neg_default_9);  ge_scalar_9 = scalar_tensor_default_9 = neg_default_9 = None
        mul_tensor_27 = torch.ops.aten.mul.Tensor(view_725, where_self_9);  view_725 = None
        amax_default_9 = torch.ops.aten.amax.default(mul_tensor_27, [-1], True)
        sub_tensor_9 = torch.ops.aten.sub.Tensor(mul_tensor_27, amax_default_9);  mul_tensor_27 = amax_default_9 = None
        mul_tensor_28 = torch.ops.aten.mul.Tensor(where_self_9, arg387_1);  where_self_9 = arg387_1 = None
        mul_tensor_29 = torch.ops.aten.mul.Tensor(sub_tensor_9, mul_tensor_28);  sub_tensor_9 = mul_tensor_28 = None
        exp_38 = torch.ops.aten.exp.default(mul_tensor_29);  mul_tensor_29 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_128 = torch.ops.aten.div.Tensor(exp_38, sum_117);  exp_38 = sum_117 = None
        expand_233 = torch.ops.aten.expand.default(div_128, [8, 16, 48, 48]);  div_128 = None
        view_726 = torch.ops.aten.view.default(expand_233, [128, 48, 48]);  expand_233 = None
        expand_234 = torch.ops.aten.expand.default(getitem_364, [8, 16, 48, 784]);  getitem_364 = None
        clone_431 = torch.ops.aten.clone.default(expand_234, memory_format = torch.contiguous_format);  expand_234 = None
        view_727 = torch.ops.aten.view.default(clone_431, [128, 48, 784]);  clone_431 = None
        bmm_77 = torch.ops.aten.bmm.default(view_726, view_727);  view_726 = view_727 = None
        view_728 = torch.ops.aten.view.default(bmm_77, [8, 16, 48, 784]);  bmm_77 = None
        permute_372 = torch.ops.aten.permute.default(view_728, [0, 3, 1, 2]);  view_728 = None
        view_729 = torch.ops.aten.view.default(permute_372, [8, 784, 768]);  permute_372 = None
        permute_373 = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
        clone_432 = torch.ops.aten.clone.default(view_729, memory_format = torch.contiguous_format);  view_729 = None
        view_730 = torch.ops.aten.view.default(clone_432, [6272, 768]);  clone_432 = None
        mm_40 = torch.ops.aten.mm.default(view_730, permute_373);  view_730 = permute_373 = None
        view_731 = torch.ops.aten.view.default(mm_40, [8, 784, 768]);  mm_40 = None
        add_574 = torch.ops.aten.add.Tensor(view_731, arg389_1);  view_731 = arg389_1 = None
        mul_781 = torch.ops.aten.mul.Tensor(arg382_1, add_574);  arg382_1 = add_574 = None
        add_575 = torch.ops.aten.add.Tensor(add_571, mul_781);  add_571 = mul_781 = None
        clone_434 = torch.ops.aten.clone.default(add_575, memory_format = torch.contiguous_format)
        var_mean_120 = torch.ops.aten.var_mean.correction(clone_434, [2], correction = 0, keepdim = True)
        getitem_365 = var_mean_120[0]
        getitem_366 = var_mean_120[1];  var_mean_120 = None
        add_576 = torch.ops.aten.add.Tensor(getitem_365, 1e-06);  getitem_365 = None
        rsqrt_120 = torch.ops.aten.rsqrt.default(add_576);  add_576 = None
        sub_203 = torch.ops.aten.sub.Tensor(clone_434, getitem_366);  clone_434 = getitem_366 = None
        mul_782 = torch.ops.aten.mul.Tensor(sub_203, rsqrt_120);  sub_203 = rsqrt_120 = None
        mul_783 = torch.ops.aten.mul.Tensor(mul_782, arg391_1);  mul_782 = arg391_1 = None
        add_577 = torch.ops.aten.add.Tensor(mul_783, arg392_1);  mul_783 = arg392_1 = None
        permute_374 = torch.ops.aten.permute.default(add_577, [0, 2, 1]);  add_577 = None
        view_732 = torch.ops.aten.view.default(permute_374, [8, 768, 28, 28]);  permute_374 = None
        convolution_84 = torch.ops.aten.convolution.default(view_732, arg393_1, arg394_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_732 = arg393_1 = arg394_1 = None
        mul_784 = torch.ops.aten.mul.Tensor(convolution_84, 0.5)
        mul_785 = torch.ops.aten.mul.Tensor(convolution_84, 0.7071067811865476);  convolution_84 = None
        erf_82 = torch.ops.aten.erf.default(mul_785);  mul_785 = None
        add_578 = torch.ops.aten.add.Tensor(erf_82, 1);  erf_82 = None
        mul_786 = torch.ops.aten.mul.Tensor(mul_784, add_578);  mul_784 = add_578 = None
        add_579 = torch.ops.aten.add.Tensor(arg396_1, 1e-05);  arg396_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_579);  add_579 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_787 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_787, -1);  mul_787 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_204 = torch.ops.aten.sub.Tensor(mul_786, unsqueeze_369);  mul_786 = unsqueeze_369 = None
        mul_788 = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_371);  sub_204 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_789 = torch.ops.aten.mul.Tensor(mul_788, unsqueeze_373);  mul_788 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_580 = torch.ops.aten.add.Tensor(mul_789, unsqueeze_375);  mul_789 = unsqueeze_375 = None
        convolution_85 = torch.ops.aten.convolution.default(add_580, arg399_1, arg400_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_580 = arg399_1 = arg400_1 = None
        view_733 = torch.ops.aten.view.default(convolution_85, [8, 768, 784]);  convolution_85 = None
        permute_375 = torch.ops.aten.permute.default(view_733, [0, 2, 1]);  view_733 = None
        mul_790 = torch.ops.aten.mul.Tensor(arg390_1, permute_375);  arg390_1 = permute_375 = None
        add_581 = torch.ops.aten.add.Tensor(add_575, mul_790);  add_575 = mul_790 = None
        clone_435 = torch.ops.aten.clone.default(add_581, memory_format = torch.contiguous_format)
        var_mean_121 = torch.ops.aten.var_mean.correction(clone_435, [2], correction = 0, keepdim = True)
        getitem_367 = var_mean_121[0]
        getitem_368 = var_mean_121[1];  var_mean_121 = None
        add_582 = torch.ops.aten.add.Tensor(getitem_367, 1e-06);  getitem_367 = None
        rsqrt_121 = torch.ops.aten.rsqrt.default(add_582);  add_582 = None
        sub_205 = torch.ops.aten.sub.Tensor(clone_435, getitem_368);  clone_435 = getitem_368 = None
        mul_791 = torch.ops.aten.mul.Tensor(sub_205, rsqrt_121);  sub_205 = rsqrt_121 = None
        mul_792 = torch.ops.aten.mul.Tensor(mul_791, arg402_1);  mul_791 = arg402_1 = None
        add_583 = torch.ops.aten.add.Tensor(mul_792, arg403_1);  mul_792 = arg403_1 = None
        view_734 = torch.ops.aten.view.default(add_583, [6272, 768]);  add_583 = None
        permute_376 = torch.ops.aten.permute.default(arg404_1, [1, 0]);  arg404_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg405_1, view_734, permute_376);  arg405_1 = view_734 = permute_376 = None
        view_735 = torch.ops.aten.view.default(addmm_126, [8, 784, 3072]);  addmm_126 = None
        mul_793 = torch.ops.aten.mul.Tensor(view_735, 0.5)
        mul_794 = torch.ops.aten.mul.Tensor(view_735, 0.7071067811865476);  view_735 = None
        erf_83 = torch.ops.aten.erf.default(mul_794);  mul_794 = None
        add_584 = torch.ops.aten.add.Tensor(erf_83, 1);  erf_83 = None
        mul_795 = torch.ops.aten.mul.Tensor(mul_793, add_584);  mul_793 = add_584 = None
        view_736 = torch.ops.aten.view.default(mul_795, [6272, 3072]);  mul_795 = None
        permute_377 = torch.ops.aten.permute.default(arg406_1, [1, 0]);  arg406_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg407_1, view_736, permute_377);  arg407_1 = view_736 = permute_377 = None
        view_737 = torch.ops.aten.view.default(addmm_127, [8, 784, 768]);  addmm_127 = None
        mul_796 = torch.ops.aten.mul.Tensor(arg401_1, view_737);  arg401_1 = view_737 = None
        add_585 = torch.ops.aten.add.Tensor(add_581, mul_796);  add_581 = mul_796 = None
        clone_438 = torch.ops.aten.clone.default(add_585, memory_format = torch.contiguous_format)
        var_mean_122 = torch.ops.aten.var_mean.correction(clone_438, [2], correction = 0, keepdim = True)
        getitem_369 = var_mean_122[0]
        getitem_370 = var_mean_122[1];  var_mean_122 = None
        add_586 = torch.ops.aten.add.Tensor(getitem_369, 1e-06);  getitem_369 = None
        rsqrt_122 = torch.ops.aten.rsqrt.default(add_586);  add_586 = None
        sub_206 = torch.ops.aten.sub.Tensor(clone_438, getitem_370);  clone_438 = getitem_370 = None
        mul_797 = torch.ops.aten.mul.Tensor(sub_206, rsqrt_122);  sub_206 = rsqrt_122 = None
        mul_798 = torch.ops.aten.mul.Tensor(mul_797, arg409_1);  mul_797 = arg409_1 = None
        add_587 = torch.ops.aten.add.Tensor(mul_798, arg410_1);  mul_798 = arg410_1 = None
        view_738 = torch.ops.aten.view.default(add_587, [6272, 768]);  add_587 = None
        permute_378 = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg412_1, view_738, permute_378);  arg412_1 = view_738 = permute_378 = None
        view_739 = torch.ops.aten.view.default(addmm_128, [8, 784, 2304]);  addmm_128 = None
        view_740 = torch.ops.aten.view.default(view_739, [8, 784, 3, 16, 48]);  view_739 = None
        permute_379 = torch.ops.aten.permute.default(view_740, [2, 0, 3, 4, 1]);  view_740 = None
        unbind_39 = torch.ops.aten.unbind.int(permute_379);  permute_379 = None
        getitem_371 = unbind_39[0]
        getitem_372 = unbind_39[1]
        getitem_373 = unbind_39[2];  unbind_39 = None
        pow_159 = torch.ops.aten.pow.Tensor_Scalar(getitem_371, 2.0)
        sum_118 = torch.ops.aten.sum.dim_IntList(pow_159, [-1], True);  pow_159 = None
        pow_160 = torch.ops.aten.pow.Tensor_Scalar(sum_118, 0.5);  sum_118 = None
        clamp_min_78 = torch.ops.aten.clamp_min.default(pow_160, 1e-12);  pow_160 = None
        expand_235 = torch.ops.aten.expand.default(clamp_min_78, [8, 16, 48, 784]);  clamp_min_78 = None
        div_129 = torch.ops.aten.div.Tensor(getitem_371, expand_235);  getitem_371 = expand_235 = None
        pow_161 = torch.ops.aten.pow.Tensor_Scalar(getitem_372, 2.0)
        sum_119 = torch.ops.aten.sum.dim_IntList(pow_161, [-1], True);  pow_161 = None
        pow_162 = torch.ops.aten.pow.Tensor_Scalar(sum_119, 0.5);  sum_119 = None
        clamp_min_79 = torch.ops.aten.clamp_min.default(pow_162, 1e-12);  pow_162 = None
        expand_236 = torch.ops.aten.expand.default(clamp_min_79, [8, 16, 48, 784]);  clamp_min_79 = None
        div_130 = torch.ops.aten.div.Tensor(getitem_372, expand_236);  getitem_372 = expand_236 = None
        permute_380 = torch.ops.aten.permute.default(div_130, [0, 1, 3, 2]);  div_130 = None
        expand_237 = torch.ops.aten.expand.default(div_129, [8, 16, 48, 784]);  div_129 = None
        clone_439 = torch.ops.aten.clone.default(expand_237, memory_format = torch.contiguous_format);  expand_237 = None
        view_741 = torch.ops.aten.view.default(clone_439, [128, 48, 784]);  clone_439 = None
        expand_238 = torch.ops.aten.expand.default(permute_380, [8, 16, 784, 48]);  permute_380 = None
        clone_440 = torch.ops.aten.clone.default(expand_238, memory_format = torch.contiguous_format);  expand_238 = None
        view_742 = torch.ops.aten.view.default(clone_440, [128, 784, 48]);  clone_440 = None
        bmm_78 = torch.ops.aten.bmm.default(view_741, view_742);  view_741 = view_742 = None
        view_743 = torch.ops.aten.view.default(bmm_78, [8, 16, 48, 48]);  bmm_78 = None
        scalar_tensor_default_8 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_8 = torch.ops.aten.ge.Scalar(arg413_1, 0)
        neg_default_8 = torch.ops.aten.neg.default(scalar_tensor_default_8)
        where_self_8 = torch.ops.aten.where.self(ge_scalar_8, scalar_tensor_default_8, neg_default_8);  ge_scalar_8 = scalar_tensor_default_8 = neg_default_8 = None
        mul_tensor_24 = torch.ops.aten.mul.Tensor(view_743, where_self_8);  view_743 = None
        amax_default_8 = torch.ops.aten.amax.default(mul_tensor_24, [-1], True)
        sub_tensor_8 = torch.ops.aten.sub.Tensor(mul_tensor_24, amax_default_8);  mul_tensor_24 = amax_default_8 = None
        mul_tensor_25 = torch.ops.aten.mul.Tensor(where_self_8, arg413_1);  where_self_8 = arg413_1 = None
        mul_tensor_26 = torch.ops.aten.mul.Tensor(sub_tensor_8, mul_tensor_25);  sub_tensor_8 = mul_tensor_25 = None
        exp_39 = torch.ops.aten.exp.default(mul_tensor_26);  mul_tensor_26 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_131 = torch.ops.aten.div.Tensor(exp_39, sum_120);  exp_39 = sum_120 = None
        expand_239 = torch.ops.aten.expand.default(div_131, [8, 16, 48, 48]);  div_131 = None
        view_744 = torch.ops.aten.view.default(expand_239, [128, 48, 48]);  expand_239 = None
        expand_240 = torch.ops.aten.expand.default(getitem_373, [8, 16, 48, 784]);  getitem_373 = None
        clone_442 = torch.ops.aten.clone.default(expand_240, memory_format = torch.contiguous_format);  expand_240 = None
        view_745 = torch.ops.aten.view.default(clone_442, [128, 48, 784]);  clone_442 = None
        bmm_79 = torch.ops.aten.bmm.default(view_744, view_745);  view_744 = view_745 = None
        view_746 = torch.ops.aten.view.default(bmm_79, [8, 16, 48, 784]);  bmm_79 = None
        permute_381 = torch.ops.aten.permute.default(view_746, [0, 3, 1, 2]);  view_746 = None
        view_747 = torch.ops.aten.view.default(permute_381, [8, 784, 768]);  permute_381 = None
        permute_382 = torch.ops.aten.permute.default(arg414_1, [1, 0]);  arg414_1 = None
        clone_443 = torch.ops.aten.clone.default(view_747, memory_format = torch.contiguous_format);  view_747 = None
        view_748 = torch.ops.aten.view.default(clone_443, [6272, 768]);  clone_443 = None
        mm_41 = torch.ops.aten.mm.default(view_748, permute_382);  view_748 = permute_382 = None
        view_749 = torch.ops.aten.view.default(mm_41, [8, 784, 768]);  mm_41 = None
        add_588 = torch.ops.aten.add.Tensor(view_749, arg415_1);  view_749 = arg415_1 = None
        mul_800 = torch.ops.aten.mul.Tensor(arg408_1, add_588);  arg408_1 = add_588 = None
        add_589 = torch.ops.aten.add.Tensor(add_585, mul_800);  add_585 = mul_800 = None
        clone_445 = torch.ops.aten.clone.default(add_589, memory_format = torch.contiguous_format)
        var_mean_123 = torch.ops.aten.var_mean.correction(clone_445, [2], correction = 0, keepdim = True)
        getitem_374 = var_mean_123[0]
        getitem_375 = var_mean_123[1];  var_mean_123 = None
        add_590 = torch.ops.aten.add.Tensor(getitem_374, 1e-06);  getitem_374 = None
        rsqrt_123 = torch.ops.aten.rsqrt.default(add_590);  add_590 = None
        sub_208 = torch.ops.aten.sub.Tensor(clone_445, getitem_375);  clone_445 = getitem_375 = None
        mul_801 = torch.ops.aten.mul.Tensor(sub_208, rsqrt_123);  sub_208 = rsqrt_123 = None
        mul_802 = torch.ops.aten.mul.Tensor(mul_801, arg417_1);  mul_801 = arg417_1 = None
        add_591 = torch.ops.aten.add.Tensor(mul_802, arg418_1);  mul_802 = arg418_1 = None
        permute_383 = torch.ops.aten.permute.default(add_591, [0, 2, 1]);  add_591 = None
        view_750 = torch.ops.aten.view.default(permute_383, [8, 768, 28, 28]);  permute_383 = None
        convolution_86 = torch.ops.aten.convolution.default(view_750, arg419_1, arg420_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_750 = arg419_1 = arg420_1 = None
        mul_803 = torch.ops.aten.mul.Tensor(convolution_86, 0.5)
        mul_804 = torch.ops.aten.mul.Tensor(convolution_86, 0.7071067811865476);  convolution_86 = None
        erf_84 = torch.ops.aten.erf.default(mul_804);  mul_804 = None
        add_592 = torch.ops.aten.add.Tensor(erf_84, 1);  erf_84 = None
        mul_805 = torch.ops.aten.mul.Tensor(mul_803, add_592);  mul_803 = add_592 = None
        add_593 = torch.ops.aten.add.Tensor(arg422_1, 1e-05);  arg422_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_593);  add_593 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_806 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_806, -1);  mul_806 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_209 = torch.ops.aten.sub.Tensor(mul_805, unsqueeze_377);  mul_805 = unsqueeze_377 = None
        mul_807 = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_379);  sub_209 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_808 = torch.ops.aten.mul.Tensor(mul_807, unsqueeze_381);  mul_807 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_594 = torch.ops.aten.add.Tensor(mul_808, unsqueeze_383);  mul_808 = unsqueeze_383 = None
        convolution_87 = torch.ops.aten.convolution.default(add_594, arg425_1, arg426_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_594 = arg425_1 = arg426_1 = None
        view_751 = torch.ops.aten.view.default(convolution_87, [8, 768, 784]);  convolution_87 = None
        permute_384 = torch.ops.aten.permute.default(view_751, [0, 2, 1]);  view_751 = None
        mul_809 = torch.ops.aten.mul.Tensor(arg416_1, permute_384);  arg416_1 = permute_384 = None
        add_595 = torch.ops.aten.add.Tensor(add_589, mul_809);  add_589 = mul_809 = None
        clone_446 = torch.ops.aten.clone.default(add_595, memory_format = torch.contiguous_format)
        var_mean_124 = torch.ops.aten.var_mean.correction(clone_446, [2], correction = 0, keepdim = True)
        getitem_376 = var_mean_124[0]
        getitem_377 = var_mean_124[1];  var_mean_124 = None
        add_596 = torch.ops.aten.add.Tensor(getitem_376, 1e-06);  getitem_376 = None
        rsqrt_124 = torch.ops.aten.rsqrt.default(add_596);  add_596 = None
        sub_210 = torch.ops.aten.sub.Tensor(clone_446, getitem_377);  clone_446 = getitem_377 = None
        mul_810 = torch.ops.aten.mul.Tensor(sub_210, rsqrt_124);  sub_210 = rsqrt_124 = None
        mul_811 = torch.ops.aten.mul.Tensor(mul_810, arg428_1);  mul_810 = arg428_1 = None
        add_597 = torch.ops.aten.add.Tensor(mul_811, arg429_1);  mul_811 = arg429_1 = None
        view_752 = torch.ops.aten.view.default(add_597, [6272, 768]);  add_597 = None
        permute_385 = torch.ops.aten.permute.default(arg430_1, [1, 0]);  arg430_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg431_1, view_752, permute_385);  arg431_1 = view_752 = permute_385 = None
        view_753 = torch.ops.aten.view.default(addmm_129, [8, 784, 3072]);  addmm_129 = None
        mul_812 = torch.ops.aten.mul.Tensor(view_753, 0.5)
        mul_813 = torch.ops.aten.mul.Tensor(view_753, 0.7071067811865476);  view_753 = None
        erf_85 = torch.ops.aten.erf.default(mul_813);  mul_813 = None
        add_598 = torch.ops.aten.add.Tensor(erf_85, 1);  erf_85 = None
        mul_814 = torch.ops.aten.mul.Tensor(mul_812, add_598);  mul_812 = add_598 = None
        view_754 = torch.ops.aten.view.default(mul_814, [6272, 3072]);  mul_814 = None
        permute_386 = torch.ops.aten.permute.default(arg432_1, [1, 0]);  arg432_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg433_1, view_754, permute_386);  arg433_1 = view_754 = permute_386 = None
        view_755 = torch.ops.aten.view.default(addmm_130, [8, 784, 768]);  addmm_130 = None
        mul_815 = torch.ops.aten.mul.Tensor(arg427_1, view_755);  arg427_1 = view_755 = None
        add_599 = torch.ops.aten.add.Tensor(add_595, mul_815);  add_595 = mul_815 = None
        clone_449 = torch.ops.aten.clone.default(add_599, memory_format = torch.contiguous_format)
        var_mean_125 = torch.ops.aten.var_mean.correction(clone_449, [2], correction = 0, keepdim = True)
        getitem_378 = var_mean_125[0]
        getitem_379 = var_mean_125[1];  var_mean_125 = None
        add_600 = torch.ops.aten.add.Tensor(getitem_378, 1e-06);  getitem_378 = None
        rsqrt_125 = torch.ops.aten.rsqrt.default(add_600);  add_600 = None
        sub_211 = torch.ops.aten.sub.Tensor(clone_449, getitem_379);  clone_449 = getitem_379 = None
        mul_816 = torch.ops.aten.mul.Tensor(sub_211, rsqrt_125);  sub_211 = rsqrt_125 = None
        mul_817 = torch.ops.aten.mul.Tensor(mul_816, arg435_1);  mul_816 = arg435_1 = None
        add_601 = torch.ops.aten.add.Tensor(mul_817, arg436_1);  mul_817 = arg436_1 = None
        view_756 = torch.ops.aten.view.default(add_601, [6272, 768]);  add_601 = None
        permute_387 = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg438_1, view_756, permute_387);  arg438_1 = view_756 = permute_387 = None
        view_757 = torch.ops.aten.view.default(addmm_131, [8, 784, 2304]);  addmm_131 = None
        view_758 = torch.ops.aten.view.default(view_757, [8, 784, 3, 16, 48]);  view_757 = None
        permute_388 = torch.ops.aten.permute.default(view_758, [2, 0, 3, 4, 1]);  view_758 = None
        unbind_40 = torch.ops.aten.unbind.int(permute_388);  permute_388 = None
        getitem_380 = unbind_40[0]
        getitem_381 = unbind_40[1]
        getitem_382 = unbind_40[2];  unbind_40 = None
        pow_163 = torch.ops.aten.pow.Tensor_Scalar(getitem_380, 2.0)
        sum_121 = torch.ops.aten.sum.dim_IntList(pow_163, [-1], True);  pow_163 = None
        pow_164 = torch.ops.aten.pow.Tensor_Scalar(sum_121, 0.5);  sum_121 = None
        clamp_min_80 = torch.ops.aten.clamp_min.default(pow_164, 1e-12);  pow_164 = None
        expand_241 = torch.ops.aten.expand.default(clamp_min_80, [8, 16, 48, 784]);  clamp_min_80 = None
        div_132 = torch.ops.aten.div.Tensor(getitem_380, expand_241);  getitem_380 = expand_241 = None
        pow_165 = torch.ops.aten.pow.Tensor_Scalar(getitem_381, 2.0)
        sum_122 = torch.ops.aten.sum.dim_IntList(pow_165, [-1], True);  pow_165 = None
        pow_166 = torch.ops.aten.pow.Tensor_Scalar(sum_122, 0.5);  sum_122 = None
        clamp_min_81 = torch.ops.aten.clamp_min.default(pow_166, 1e-12);  pow_166 = None
        expand_242 = torch.ops.aten.expand.default(clamp_min_81, [8, 16, 48, 784]);  clamp_min_81 = None
        div_133 = torch.ops.aten.div.Tensor(getitem_381, expand_242);  getitem_381 = expand_242 = None
        permute_389 = torch.ops.aten.permute.default(div_133, [0, 1, 3, 2]);  div_133 = None
        expand_243 = torch.ops.aten.expand.default(div_132, [8, 16, 48, 784]);  div_132 = None
        clone_450 = torch.ops.aten.clone.default(expand_243, memory_format = torch.contiguous_format);  expand_243 = None
        view_759 = torch.ops.aten.view.default(clone_450, [128, 48, 784]);  clone_450 = None
        expand_244 = torch.ops.aten.expand.default(permute_389, [8, 16, 784, 48]);  permute_389 = None
        clone_451 = torch.ops.aten.clone.default(expand_244, memory_format = torch.contiguous_format);  expand_244 = None
        view_760 = torch.ops.aten.view.default(clone_451, [128, 784, 48]);  clone_451 = None
        bmm_80 = torch.ops.aten.bmm.default(view_759, view_760);  view_759 = view_760 = None
        view_761 = torch.ops.aten.view.default(bmm_80, [8, 16, 48, 48]);  bmm_80 = None
        scalar_tensor_default_7 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_7 = torch.ops.aten.ge.Scalar(arg439_1, 0)
        neg_default_7 = torch.ops.aten.neg.default(scalar_tensor_default_7)
        where_self_7 = torch.ops.aten.where.self(ge_scalar_7, scalar_tensor_default_7, neg_default_7);  ge_scalar_7 = scalar_tensor_default_7 = neg_default_7 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(view_761, where_self_7);  view_761 = None
        amax_default_7 = torch.ops.aten.amax.default(mul_tensor_21, [-1], True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_21, amax_default_7);  mul_tensor_21 = amax_default_7 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(where_self_7, arg439_1);  where_self_7 = arg439_1 = None
        mul_tensor_23 = torch.ops.aten.mul.Tensor(sub_tensor_7, mul_tensor_22);  sub_tensor_7 = mul_tensor_22 = None
        exp_40 = torch.ops.aten.exp.default(mul_tensor_23);  mul_tensor_23 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_134 = torch.ops.aten.div.Tensor(exp_40, sum_123);  exp_40 = sum_123 = None
        expand_245 = torch.ops.aten.expand.default(div_134, [8, 16, 48, 48]);  div_134 = None
        view_762 = torch.ops.aten.view.default(expand_245, [128, 48, 48]);  expand_245 = None
        expand_246 = torch.ops.aten.expand.default(getitem_382, [8, 16, 48, 784]);  getitem_382 = None
        clone_453 = torch.ops.aten.clone.default(expand_246, memory_format = torch.contiguous_format);  expand_246 = None
        view_763 = torch.ops.aten.view.default(clone_453, [128, 48, 784]);  clone_453 = None
        bmm_81 = torch.ops.aten.bmm.default(view_762, view_763);  view_762 = view_763 = None
        view_764 = torch.ops.aten.view.default(bmm_81, [8, 16, 48, 784]);  bmm_81 = None
        permute_390 = torch.ops.aten.permute.default(view_764, [0, 3, 1, 2]);  view_764 = None
        view_765 = torch.ops.aten.view.default(permute_390, [8, 784, 768]);  permute_390 = None
        permute_391 = torch.ops.aten.permute.default(arg440_1, [1, 0]);  arg440_1 = None
        clone_454 = torch.ops.aten.clone.default(view_765, memory_format = torch.contiguous_format);  view_765 = None
        view_766 = torch.ops.aten.view.default(clone_454, [6272, 768]);  clone_454 = None
        mm_42 = torch.ops.aten.mm.default(view_766, permute_391);  view_766 = permute_391 = None
        view_767 = torch.ops.aten.view.default(mm_42, [8, 784, 768]);  mm_42 = None
        add_602 = torch.ops.aten.add.Tensor(view_767, arg441_1);  view_767 = arg441_1 = None
        mul_819 = torch.ops.aten.mul.Tensor(arg434_1, add_602);  arg434_1 = add_602 = None
        add_603 = torch.ops.aten.add.Tensor(add_599, mul_819);  add_599 = mul_819 = None
        clone_456 = torch.ops.aten.clone.default(add_603, memory_format = torch.contiguous_format)
        var_mean_126 = torch.ops.aten.var_mean.correction(clone_456, [2], correction = 0, keepdim = True)
        getitem_383 = var_mean_126[0]
        getitem_384 = var_mean_126[1];  var_mean_126 = None
        add_604 = torch.ops.aten.add.Tensor(getitem_383, 1e-06);  getitem_383 = None
        rsqrt_126 = torch.ops.aten.rsqrt.default(add_604);  add_604 = None
        sub_213 = torch.ops.aten.sub.Tensor(clone_456, getitem_384);  clone_456 = getitem_384 = None
        mul_820 = torch.ops.aten.mul.Tensor(sub_213, rsqrt_126);  sub_213 = rsqrt_126 = None
        mul_821 = torch.ops.aten.mul.Tensor(mul_820, arg443_1);  mul_820 = arg443_1 = None
        add_605 = torch.ops.aten.add.Tensor(mul_821, arg444_1);  mul_821 = arg444_1 = None
        permute_392 = torch.ops.aten.permute.default(add_605, [0, 2, 1]);  add_605 = None
        view_768 = torch.ops.aten.view.default(permute_392, [8, 768, 28, 28]);  permute_392 = None
        convolution_88 = torch.ops.aten.convolution.default(view_768, arg445_1, arg446_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_768 = arg445_1 = arg446_1 = None
        mul_822 = torch.ops.aten.mul.Tensor(convolution_88, 0.5)
        mul_823 = torch.ops.aten.mul.Tensor(convolution_88, 0.7071067811865476);  convolution_88 = None
        erf_86 = torch.ops.aten.erf.default(mul_823);  mul_823 = None
        add_606 = torch.ops.aten.add.Tensor(erf_86, 1);  erf_86 = None
        mul_824 = torch.ops.aten.mul.Tensor(mul_822, add_606);  mul_822 = add_606 = None
        add_607 = torch.ops.aten.add.Tensor(arg448_1, 1e-05);  arg448_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_607);  add_607 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_825 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg447_1, -1);  arg447_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_825, -1);  mul_825 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_214 = torch.ops.aten.sub.Tensor(mul_824, unsqueeze_385);  mul_824 = unsqueeze_385 = None
        mul_826 = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_387);  sub_214 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_827 = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_389);  mul_826 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_608 = torch.ops.aten.add.Tensor(mul_827, unsqueeze_391);  mul_827 = unsqueeze_391 = None
        convolution_89 = torch.ops.aten.convolution.default(add_608, arg451_1, arg452_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_608 = arg451_1 = arg452_1 = None
        view_769 = torch.ops.aten.view.default(convolution_89, [8, 768, 784]);  convolution_89 = None
        permute_393 = torch.ops.aten.permute.default(view_769, [0, 2, 1]);  view_769 = None
        mul_828 = torch.ops.aten.mul.Tensor(arg442_1, permute_393);  arg442_1 = permute_393 = None
        add_609 = torch.ops.aten.add.Tensor(add_603, mul_828);  add_603 = mul_828 = None
        clone_457 = torch.ops.aten.clone.default(add_609, memory_format = torch.contiguous_format)
        var_mean_127 = torch.ops.aten.var_mean.correction(clone_457, [2], correction = 0, keepdim = True)
        getitem_385 = var_mean_127[0]
        getitem_386 = var_mean_127[1];  var_mean_127 = None
        add_610 = torch.ops.aten.add.Tensor(getitem_385, 1e-06);  getitem_385 = None
        rsqrt_127 = torch.ops.aten.rsqrt.default(add_610);  add_610 = None
        sub_215 = torch.ops.aten.sub.Tensor(clone_457, getitem_386);  clone_457 = getitem_386 = None
        mul_829 = torch.ops.aten.mul.Tensor(sub_215, rsqrt_127);  sub_215 = rsqrt_127 = None
        mul_830 = torch.ops.aten.mul.Tensor(mul_829, arg454_1);  mul_829 = arg454_1 = None
        add_611 = torch.ops.aten.add.Tensor(mul_830, arg455_1);  mul_830 = arg455_1 = None
        view_770 = torch.ops.aten.view.default(add_611, [6272, 768]);  add_611 = None
        permute_394 = torch.ops.aten.permute.default(arg456_1, [1, 0]);  arg456_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg457_1, view_770, permute_394);  arg457_1 = view_770 = permute_394 = None
        view_771 = torch.ops.aten.view.default(addmm_132, [8, 784, 3072]);  addmm_132 = None
        mul_831 = torch.ops.aten.mul.Tensor(view_771, 0.5)
        mul_832 = torch.ops.aten.mul.Tensor(view_771, 0.7071067811865476);  view_771 = None
        erf_87 = torch.ops.aten.erf.default(mul_832);  mul_832 = None
        add_612 = torch.ops.aten.add.Tensor(erf_87, 1);  erf_87 = None
        mul_833 = torch.ops.aten.mul.Tensor(mul_831, add_612);  mul_831 = add_612 = None
        view_772 = torch.ops.aten.view.default(mul_833, [6272, 3072]);  mul_833 = None
        permute_395 = torch.ops.aten.permute.default(arg458_1, [1, 0]);  arg458_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg459_1, view_772, permute_395);  arg459_1 = view_772 = permute_395 = None
        view_773 = torch.ops.aten.view.default(addmm_133, [8, 784, 768]);  addmm_133 = None
        mul_834 = torch.ops.aten.mul.Tensor(arg453_1, view_773);  arg453_1 = view_773 = None
        add_613 = torch.ops.aten.add.Tensor(add_609, mul_834);  add_609 = mul_834 = None
        clone_460 = torch.ops.aten.clone.default(add_613, memory_format = torch.contiguous_format)
        var_mean_128 = torch.ops.aten.var_mean.correction(clone_460, [2], correction = 0, keepdim = True)
        getitem_387 = var_mean_128[0]
        getitem_388 = var_mean_128[1];  var_mean_128 = None
        add_614 = torch.ops.aten.add.Tensor(getitem_387, 1e-06);  getitem_387 = None
        rsqrt_128 = torch.ops.aten.rsqrt.default(add_614);  add_614 = None
        sub_216 = torch.ops.aten.sub.Tensor(clone_460, getitem_388);  clone_460 = getitem_388 = None
        mul_835 = torch.ops.aten.mul.Tensor(sub_216, rsqrt_128);  sub_216 = rsqrt_128 = None
        mul_836 = torch.ops.aten.mul.Tensor(mul_835, arg461_1);  mul_835 = arg461_1 = None
        add_615 = torch.ops.aten.add.Tensor(mul_836, arg462_1);  mul_836 = arg462_1 = None
        view_774 = torch.ops.aten.view.default(add_615, [6272, 768]);  add_615 = None
        permute_396 = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg464_1, view_774, permute_396);  arg464_1 = view_774 = permute_396 = None
        view_775 = torch.ops.aten.view.default(addmm_134, [8, 784, 2304]);  addmm_134 = None
        view_776 = torch.ops.aten.view.default(view_775, [8, 784, 3, 16, 48]);  view_775 = None
        permute_397 = torch.ops.aten.permute.default(view_776, [2, 0, 3, 4, 1]);  view_776 = None
        unbind_41 = torch.ops.aten.unbind.int(permute_397);  permute_397 = None
        getitem_389 = unbind_41[0]
        getitem_390 = unbind_41[1]
        getitem_391 = unbind_41[2];  unbind_41 = None
        pow_167 = torch.ops.aten.pow.Tensor_Scalar(getitem_389, 2.0)
        sum_124 = torch.ops.aten.sum.dim_IntList(pow_167, [-1], True);  pow_167 = None
        pow_168 = torch.ops.aten.pow.Tensor_Scalar(sum_124, 0.5);  sum_124 = None
        clamp_min_82 = torch.ops.aten.clamp_min.default(pow_168, 1e-12);  pow_168 = None
        expand_247 = torch.ops.aten.expand.default(clamp_min_82, [8, 16, 48, 784]);  clamp_min_82 = None
        div_135 = torch.ops.aten.div.Tensor(getitem_389, expand_247);  getitem_389 = expand_247 = None
        pow_169 = torch.ops.aten.pow.Tensor_Scalar(getitem_390, 2.0)
        sum_125 = torch.ops.aten.sum.dim_IntList(pow_169, [-1], True);  pow_169 = None
        pow_170 = torch.ops.aten.pow.Tensor_Scalar(sum_125, 0.5);  sum_125 = None
        clamp_min_83 = torch.ops.aten.clamp_min.default(pow_170, 1e-12);  pow_170 = None
        expand_248 = torch.ops.aten.expand.default(clamp_min_83, [8, 16, 48, 784]);  clamp_min_83 = None
        div_136 = torch.ops.aten.div.Tensor(getitem_390, expand_248);  getitem_390 = expand_248 = None
        permute_398 = torch.ops.aten.permute.default(div_136, [0, 1, 3, 2]);  div_136 = None
        expand_249 = torch.ops.aten.expand.default(div_135, [8, 16, 48, 784]);  div_135 = None
        clone_461 = torch.ops.aten.clone.default(expand_249, memory_format = torch.contiguous_format);  expand_249 = None
        view_777 = torch.ops.aten.view.default(clone_461, [128, 48, 784]);  clone_461 = None
        expand_250 = torch.ops.aten.expand.default(permute_398, [8, 16, 784, 48]);  permute_398 = None
        clone_462 = torch.ops.aten.clone.default(expand_250, memory_format = torch.contiguous_format);  expand_250 = None
        view_778 = torch.ops.aten.view.default(clone_462, [128, 784, 48]);  clone_462 = None
        bmm_82 = torch.ops.aten.bmm.default(view_777, view_778);  view_777 = view_778 = None
        view_779 = torch.ops.aten.view.default(bmm_82, [8, 16, 48, 48]);  bmm_82 = None
        scalar_tensor_default_6 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_6 = torch.ops.aten.ge.Scalar(arg465_1, 0)
        neg_default_6 = torch.ops.aten.neg.default(scalar_tensor_default_6)
        where_self_6 = torch.ops.aten.where.self(ge_scalar_6, scalar_tensor_default_6, neg_default_6);  ge_scalar_6 = scalar_tensor_default_6 = neg_default_6 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(view_779, where_self_6);  view_779 = None
        amax_default_6 = torch.ops.aten.amax.default(mul_tensor_18, [-1], True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_18, amax_default_6);  mul_tensor_18 = amax_default_6 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(where_self_6, arg465_1);  where_self_6 = arg465_1 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(sub_tensor_6, mul_tensor_19);  sub_tensor_6 = mul_tensor_19 = None
        exp_41 = torch.ops.aten.exp.default(mul_tensor_20);  mul_tensor_20 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_137 = torch.ops.aten.div.Tensor(exp_41, sum_126);  exp_41 = sum_126 = None
        expand_251 = torch.ops.aten.expand.default(div_137, [8, 16, 48, 48]);  div_137 = None
        view_780 = torch.ops.aten.view.default(expand_251, [128, 48, 48]);  expand_251 = None
        expand_252 = torch.ops.aten.expand.default(getitem_391, [8, 16, 48, 784]);  getitem_391 = None
        clone_464 = torch.ops.aten.clone.default(expand_252, memory_format = torch.contiguous_format);  expand_252 = None
        view_781 = torch.ops.aten.view.default(clone_464, [128, 48, 784]);  clone_464 = None
        bmm_83 = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782 = torch.ops.aten.view.default(bmm_83, [8, 16, 48, 784]);  bmm_83 = None
        permute_399 = torch.ops.aten.permute.default(view_782, [0, 3, 1, 2]);  view_782 = None
        view_783 = torch.ops.aten.view.default(permute_399, [8, 784, 768]);  permute_399 = None
        permute_400 = torch.ops.aten.permute.default(arg466_1, [1, 0]);  arg466_1 = None
        clone_465 = torch.ops.aten.clone.default(view_783, memory_format = torch.contiguous_format);  view_783 = None
        view_784 = torch.ops.aten.view.default(clone_465, [6272, 768]);  clone_465 = None
        mm_43 = torch.ops.aten.mm.default(view_784, permute_400);  view_784 = permute_400 = None
        view_785 = torch.ops.aten.view.default(mm_43, [8, 784, 768]);  mm_43 = None
        add_616 = torch.ops.aten.add.Tensor(view_785, arg467_1);  view_785 = arg467_1 = None
        mul_838 = torch.ops.aten.mul.Tensor(arg460_1, add_616);  arg460_1 = add_616 = None
        add_617 = torch.ops.aten.add.Tensor(add_613, mul_838);  add_613 = mul_838 = None
        clone_467 = torch.ops.aten.clone.default(add_617, memory_format = torch.contiguous_format)
        var_mean_129 = torch.ops.aten.var_mean.correction(clone_467, [2], correction = 0, keepdim = True)
        getitem_392 = var_mean_129[0]
        getitem_393 = var_mean_129[1];  var_mean_129 = None
        add_618 = torch.ops.aten.add.Tensor(getitem_392, 1e-06);  getitem_392 = None
        rsqrt_129 = torch.ops.aten.rsqrt.default(add_618);  add_618 = None
        sub_218 = torch.ops.aten.sub.Tensor(clone_467, getitem_393);  clone_467 = getitem_393 = None
        mul_839 = torch.ops.aten.mul.Tensor(sub_218, rsqrt_129);  sub_218 = rsqrt_129 = None
        mul_840 = torch.ops.aten.mul.Tensor(mul_839, arg469_1);  mul_839 = arg469_1 = None
        add_619 = torch.ops.aten.add.Tensor(mul_840, arg470_1);  mul_840 = arg470_1 = None
        permute_401 = torch.ops.aten.permute.default(add_619, [0, 2, 1]);  add_619 = None
        view_786 = torch.ops.aten.view.default(permute_401, [8, 768, 28, 28]);  permute_401 = None
        convolution_90 = torch.ops.aten.convolution.default(view_786, arg471_1, arg472_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_786 = arg471_1 = arg472_1 = None
        mul_841 = torch.ops.aten.mul.Tensor(convolution_90, 0.5)
        mul_842 = torch.ops.aten.mul.Tensor(convolution_90, 0.7071067811865476);  convolution_90 = None
        erf_88 = torch.ops.aten.erf.default(mul_842);  mul_842 = None
        add_620 = torch.ops.aten.add.Tensor(erf_88, 1);  erf_88 = None
        mul_843 = torch.ops.aten.mul.Tensor(mul_841, add_620);  mul_841 = add_620 = None
        add_621 = torch.ops.aten.add.Tensor(arg474_1, 1e-05);  arg474_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_621);  add_621 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_844 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_844, -1);  mul_844 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_219 = torch.ops.aten.sub.Tensor(mul_843, unsqueeze_393);  mul_843 = unsqueeze_393 = None
        mul_845 = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_395);  sub_219 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg475_1, -1);  arg475_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_846 = torch.ops.aten.mul.Tensor(mul_845, unsqueeze_397);  mul_845 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_622 = torch.ops.aten.add.Tensor(mul_846, unsqueeze_399);  mul_846 = unsqueeze_399 = None
        convolution_91 = torch.ops.aten.convolution.default(add_622, arg477_1, arg478_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_622 = arg477_1 = arg478_1 = None
        view_787 = torch.ops.aten.view.default(convolution_91, [8, 768, 784]);  convolution_91 = None
        permute_402 = torch.ops.aten.permute.default(view_787, [0, 2, 1]);  view_787 = None
        mul_847 = torch.ops.aten.mul.Tensor(arg468_1, permute_402);  arg468_1 = permute_402 = None
        add_623 = torch.ops.aten.add.Tensor(add_617, mul_847);  add_617 = mul_847 = None
        clone_468 = torch.ops.aten.clone.default(add_623, memory_format = torch.contiguous_format)
        var_mean_130 = torch.ops.aten.var_mean.correction(clone_468, [2], correction = 0, keepdim = True)
        getitem_394 = var_mean_130[0]
        getitem_395 = var_mean_130[1];  var_mean_130 = None
        add_624 = torch.ops.aten.add.Tensor(getitem_394, 1e-06);  getitem_394 = None
        rsqrt_130 = torch.ops.aten.rsqrt.default(add_624);  add_624 = None
        sub_220 = torch.ops.aten.sub.Tensor(clone_468, getitem_395);  clone_468 = getitem_395 = None
        mul_848 = torch.ops.aten.mul.Tensor(sub_220, rsqrt_130);  sub_220 = rsqrt_130 = None
        mul_849 = torch.ops.aten.mul.Tensor(mul_848, arg480_1);  mul_848 = arg480_1 = None
        add_625 = torch.ops.aten.add.Tensor(mul_849, arg481_1);  mul_849 = arg481_1 = None
        view_788 = torch.ops.aten.view.default(add_625, [6272, 768]);  add_625 = None
        permute_403 = torch.ops.aten.permute.default(arg482_1, [1, 0]);  arg482_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg483_1, view_788, permute_403);  arg483_1 = view_788 = permute_403 = None
        view_789 = torch.ops.aten.view.default(addmm_135, [8, 784, 3072]);  addmm_135 = None
        mul_850 = torch.ops.aten.mul.Tensor(view_789, 0.5)
        mul_851 = torch.ops.aten.mul.Tensor(view_789, 0.7071067811865476);  view_789 = None
        erf_89 = torch.ops.aten.erf.default(mul_851);  mul_851 = None
        add_626 = torch.ops.aten.add.Tensor(erf_89, 1);  erf_89 = None
        mul_852 = torch.ops.aten.mul.Tensor(mul_850, add_626);  mul_850 = add_626 = None
        view_790 = torch.ops.aten.view.default(mul_852, [6272, 3072]);  mul_852 = None
        permute_404 = torch.ops.aten.permute.default(arg484_1, [1, 0]);  arg484_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg485_1, view_790, permute_404);  arg485_1 = view_790 = permute_404 = None
        view_791 = torch.ops.aten.view.default(addmm_136, [8, 784, 768]);  addmm_136 = None
        mul_853 = torch.ops.aten.mul.Tensor(arg479_1, view_791);  arg479_1 = view_791 = None
        add_627 = torch.ops.aten.add.Tensor(add_623, mul_853);  add_623 = mul_853 = None
        clone_471 = torch.ops.aten.clone.default(add_627, memory_format = torch.contiguous_format)
        var_mean_131 = torch.ops.aten.var_mean.correction(clone_471, [2], correction = 0, keepdim = True)
        getitem_396 = var_mean_131[0]
        getitem_397 = var_mean_131[1];  var_mean_131 = None
        add_628 = torch.ops.aten.add.Tensor(getitem_396, 1e-06);  getitem_396 = None
        rsqrt_131 = torch.ops.aten.rsqrt.default(add_628);  add_628 = None
        sub_221 = torch.ops.aten.sub.Tensor(clone_471, getitem_397);  clone_471 = getitem_397 = None
        mul_854 = torch.ops.aten.mul.Tensor(sub_221, rsqrt_131);  sub_221 = rsqrt_131 = None
        mul_855 = torch.ops.aten.mul.Tensor(mul_854, arg487_1);  mul_854 = arg487_1 = None
        add_629 = torch.ops.aten.add.Tensor(mul_855, arg488_1);  mul_855 = arg488_1 = None
        view_792 = torch.ops.aten.view.default(add_629, [6272, 768]);  add_629 = None
        permute_405 = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg490_1, view_792, permute_405);  arg490_1 = view_792 = permute_405 = None
        view_793 = torch.ops.aten.view.default(addmm_137, [8, 784, 2304]);  addmm_137 = None
        view_794 = torch.ops.aten.view.default(view_793, [8, 784, 3, 16, 48]);  view_793 = None
        permute_406 = torch.ops.aten.permute.default(view_794, [2, 0, 3, 4, 1]);  view_794 = None
        unbind_42 = torch.ops.aten.unbind.int(permute_406);  permute_406 = None
        getitem_398 = unbind_42[0]
        getitem_399 = unbind_42[1]
        getitem_400 = unbind_42[2];  unbind_42 = None
        pow_171 = torch.ops.aten.pow.Tensor_Scalar(getitem_398, 2.0)
        sum_127 = torch.ops.aten.sum.dim_IntList(pow_171, [-1], True);  pow_171 = None
        pow_172 = torch.ops.aten.pow.Tensor_Scalar(sum_127, 0.5);  sum_127 = None
        clamp_min_84 = torch.ops.aten.clamp_min.default(pow_172, 1e-12);  pow_172 = None
        expand_253 = torch.ops.aten.expand.default(clamp_min_84, [8, 16, 48, 784]);  clamp_min_84 = None
        div_138 = torch.ops.aten.div.Tensor(getitem_398, expand_253);  getitem_398 = expand_253 = None
        pow_173 = torch.ops.aten.pow.Tensor_Scalar(getitem_399, 2.0)
        sum_128 = torch.ops.aten.sum.dim_IntList(pow_173, [-1], True);  pow_173 = None
        pow_174 = torch.ops.aten.pow.Tensor_Scalar(sum_128, 0.5);  sum_128 = None
        clamp_min_85 = torch.ops.aten.clamp_min.default(pow_174, 1e-12);  pow_174 = None
        expand_254 = torch.ops.aten.expand.default(clamp_min_85, [8, 16, 48, 784]);  clamp_min_85 = None
        div_139 = torch.ops.aten.div.Tensor(getitem_399, expand_254);  getitem_399 = expand_254 = None
        permute_407 = torch.ops.aten.permute.default(div_139, [0, 1, 3, 2]);  div_139 = None
        expand_255 = torch.ops.aten.expand.default(div_138, [8, 16, 48, 784]);  div_138 = None
        clone_472 = torch.ops.aten.clone.default(expand_255, memory_format = torch.contiguous_format);  expand_255 = None
        view_795 = torch.ops.aten.view.default(clone_472, [128, 48, 784]);  clone_472 = None
        expand_256 = torch.ops.aten.expand.default(permute_407, [8, 16, 784, 48]);  permute_407 = None
        clone_473 = torch.ops.aten.clone.default(expand_256, memory_format = torch.contiguous_format);  expand_256 = None
        view_796 = torch.ops.aten.view.default(clone_473, [128, 784, 48]);  clone_473 = None
        bmm_84 = torch.ops.aten.bmm.default(view_795, view_796);  view_795 = view_796 = None
        view_797 = torch.ops.aten.view.default(bmm_84, [8, 16, 48, 48]);  bmm_84 = None
        scalar_tensor_default_5 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_5 = torch.ops.aten.ge.Scalar(arg491_1, 0)
        neg_default_5 = torch.ops.aten.neg.default(scalar_tensor_default_5)
        where_self_5 = torch.ops.aten.where.self(ge_scalar_5, scalar_tensor_default_5, neg_default_5);  ge_scalar_5 = scalar_tensor_default_5 = neg_default_5 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(view_797, where_self_5);  view_797 = None
        amax_default_5 = torch.ops.aten.amax.default(mul_tensor_15, [-1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_15, amax_default_5);  mul_tensor_15 = amax_default_5 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(where_self_5, arg491_1);  where_self_5 = arg491_1 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(sub_tensor_5, mul_tensor_16);  sub_tensor_5 = mul_tensor_16 = None
        exp_42 = torch.ops.aten.exp.default(mul_tensor_17);  mul_tensor_17 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_140 = torch.ops.aten.div.Tensor(exp_42, sum_129);  exp_42 = sum_129 = None
        expand_257 = torch.ops.aten.expand.default(div_140, [8, 16, 48, 48]);  div_140 = None
        view_798 = torch.ops.aten.view.default(expand_257, [128, 48, 48]);  expand_257 = None
        expand_258 = torch.ops.aten.expand.default(getitem_400, [8, 16, 48, 784]);  getitem_400 = None
        clone_475 = torch.ops.aten.clone.default(expand_258, memory_format = torch.contiguous_format);  expand_258 = None
        view_799 = torch.ops.aten.view.default(clone_475, [128, 48, 784]);  clone_475 = None
        bmm_85 = torch.ops.aten.bmm.default(view_798, view_799);  view_798 = view_799 = None
        view_800 = torch.ops.aten.view.default(bmm_85, [8, 16, 48, 784]);  bmm_85 = None
        permute_408 = torch.ops.aten.permute.default(view_800, [0, 3, 1, 2]);  view_800 = None
        view_801 = torch.ops.aten.view.default(permute_408, [8, 784, 768]);  permute_408 = None
        permute_409 = torch.ops.aten.permute.default(arg492_1, [1, 0]);  arg492_1 = None
        clone_476 = torch.ops.aten.clone.default(view_801, memory_format = torch.contiguous_format);  view_801 = None
        view_802 = torch.ops.aten.view.default(clone_476, [6272, 768]);  clone_476 = None
        mm_44 = torch.ops.aten.mm.default(view_802, permute_409);  view_802 = permute_409 = None
        view_803 = torch.ops.aten.view.default(mm_44, [8, 784, 768]);  mm_44 = None
        add_630 = torch.ops.aten.add.Tensor(view_803, arg493_1);  view_803 = arg493_1 = None
        mul_857 = torch.ops.aten.mul.Tensor(arg486_1, add_630);  arg486_1 = add_630 = None
        add_631 = torch.ops.aten.add.Tensor(add_627, mul_857);  add_627 = mul_857 = None
        clone_478 = torch.ops.aten.clone.default(add_631, memory_format = torch.contiguous_format)
        var_mean_132 = torch.ops.aten.var_mean.correction(clone_478, [2], correction = 0, keepdim = True)
        getitem_401 = var_mean_132[0]
        getitem_402 = var_mean_132[1];  var_mean_132 = None
        add_632 = torch.ops.aten.add.Tensor(getitem_401, 1e-06);  getitem_401 = None
        rsqrt_132 = torch.ops.aten.rsqrt.default(add_632);  add_632 = None
        sub_223 = torch.ops.aten.sub.Tensor(clone_478, getitem_402);  clone_478 = getitem_402 = None
        mul_858 = torch.ops.aten.mul.Tensor(sub_223, rsqrt_132);  sub_223 = rsqrt_132 = None
        mul_859 = torch.ops.aten.mul.Tensor(mul_858, arg495_1);  mul_858 = arg495_1 = None
        add_633 = torch.ops.aten.add.Tensor(mul_859, arg496_1);  mul_859 = arg496_1 = None
        permute_410 = torch.ops.aten.permute.default(add_633, [0, 2, 1]);  add_633 = None
        view_804 = torch.ops.aten.view.default(permute_410, [8, 768, 28, 28]);  permute_410 = None
        convolution_92 = torch.ops.aten.convolution.default(view_804, arg497_1, arg498_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_804 = arg497_1 = arg498_1 = None
        mul_860 = torch.ops.aten.mul.Tensor(convolution_92, 0.5)
        mul_861 = torch.ops.aten.mul.Tensor(convolution_92, 0.7071067811865476);  convolution_92 = None
        erf_90 = torch.ops.aten.erf.default(mul_861);  mul_861 = None
        add_634 = torch.ops.aten.add.Tensor(erf_90, 1);  erf_90 = None
        mul_862 = torch.ops.aten.mul.Tensor(mul_860, add_634);  mul_860 = add_634 = None
        add_635 = torch.ops.aten.add.Tensor(arg500_1, 1e-05);  arg500_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_635);  add_635 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_863 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_863, -1);  mul_863 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_224 = torch.ops.aten.sub.Tensor(mul_862, unsqueeze_401);  mul_862 = unsqueeze_401 = None
        mul_864 = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_403);  sub_224 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg501_1, -1);  arg501_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_865 = torch.ops.aten.mul.Tensor(mul_864, unsqueeze_405);  mul_864 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_636 = torch.ops.aten.add.Tensor(mul_865, unsqueeze_407);  mul_865 = unsqueeze_407 = None
        convolution_93 = torch.ops.aten.convolution.default(add_636, arg503_1, arg504_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_636 = arg503_1 = arg504_1 = None
        view_805 = torch.ops.aten.view.default(convolution_93, [8, 768, 784]);  convolution_93 = None
        permute_411 = torch.ops.aten.permute.default(view_805, [0, 2, 1]);  view_805 = None
        mul_866 = torch.ops.aten.mul.Tensor(arg494_1, permute_411);  arg494_1 = permute_411 = None
        add_637 = torch.ops.aten.add.Tensor(add_631, mul_866);  add_631 = mul_866 = None
        clone_479 = torch.ops.aten.clone.default(add_637, memory_format = torch.contiguous_format)
        var_mean_133 = torch.ops.aten.var_mean.correction(clone_479, [2], correction = 0, keepdim = True)
        getitem_403 = var_mean_133[0]
        getitem_404 = var_mean_133[1];  var_mean_133 = None
        add_638 = torch.ops.aten.add.Tensor(getitem_403, 1e-06);  getitem_403 = None
        rsqrt_133 = torch.ops.aten.rsqrt.default(add_638);  add_638 = None
        sub_225 = torch.ops.aten.sub.Tensor(clone_479, getitem_404);  clone_479 = getitem_404 = None
        mul_867 = torch.ops.aten.mul.Tensor(sub_225, rsqrt_133);  sub_225 = rsqrt_133 = None
        mul_868 = torch.ops.aten.mul.Tensor(mul_867, arg506_1);  mul_867 = arg506_1 = None
        add_639 = torch.ops.aten.add.Tensor(mul_868, arg507_1);  mul_868 = arg507_1 = None
        view_806 = torch.ops.aten.view.default(add_639, [6272, 768]);  add_639 = None
        permute_412 = torch.ops.aten.permute.default(arg508_1, [1, 0]);  arg508_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg509_1, view_806, permute_412);  arg509_1 = view_806 = permute_412 = None
        view_807 = torch.ops.aten.view.default(addmm_138, [8, 784, 3072]);  addmm_138 = None
        mul_869 = torch.ops.aten.mul.Tensor(view_807, 0.5)
        mul_870 = torch.ops.aten.mul.Tensor(view_807, 0.7071067811865476);  view_807 = None
        erf_91 = torch.ops.aten.erf.default(mul_870);  mul_870 = None
        add_640 = torch.ops.aten.add.Tensor(erf_91, 1);  erf_91 = None
        mul_871 = torch.ops.aten.mul.Tensor(mul_869, add_640);  mul_869 = add_640 = None
        view_808 = torch.ops.aten.view.default(mul_871, [6272, 3072]);  mul_871 = None
        permute_413 = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg511_1, view_808, permute_413);  arg511_1 = view_808 = permute_413 = None
        view_809 = torch.ops.aten.view.default(addmm_139, [8, 784, 768]);  addmm_139 = None
        mul_872 = torch.ops.aten.mul.Tensor(arg505_1, view_809);  arg505_1 = view_809 = None
        add_641 = torch.ops.aten.add.Tensor(add_637, mul_872);  add_637 = mul_872 = None
        clone_482 = torch.ops.aten.clone.default(add_641, memory_format = torch.contiguous_format)
        var_mean_134 = torch.ops.aten.var_mean.correction(clone_482, [2], correction = 0, keepdim = True)
        getitem_405 = var_mean_134[0]
        getitem_406 = var_mean_134[1];  var_mean_134 = None
        add_642 = torch.ops.aten.add.Tensor(getitem_405, 1e-06);  getitem_405 = None
        rsqrt_134 = torch.ops.aten.rsqrt.default(add_642);  add_642 = None
        sub_226 = torch.ops.aten.sub.Tensor(clone_482, getitem_406);  clone_482 = getitem_406 = None
        mul_873 = torch.ops.aten.mul.Tensor(sub_226, rsqrt_134);  sub_226 = rsqrt_134 = None
        mul_874 = torch.ops.aten.mul.Tensor(mul_873, arg513_1);  mul_873 = arg513_1 = None
        add_643 = torch.ops.aten.add.Tensor(mul_874, arg514_1);  mul_874 = arg514_1 = None
        view_810 = torch.ops.aten.view.default(add_643, [6272, 768]);  add_643 = None
        permute_414 = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg516_1, view_810, permute_414);  arg516_1 = view_810 = permute_414 = None
        view_811 = torch.ops.aten.view.default(addmm_140, [8, 784, 2304]);  addmm_140 = None
        view_812 = torch.ops.aten.view.default(view_811, [8, 784, 3, 16, 48]);  view_811 = None
        permute_415 = torch.ops.aten.permute.default(view_812, [2, 0, 3, 4, 1]);  view_812 = None
        unbind_43 = torch.ops.aten.unbind.int(permute_415);  permute_415 = None
        getitem_407 = unbind_43[0]
        getitem_408 = unbind_43[1]
        getitem_409 = unbind_43[2];  unbind_43 = None
        pow_175 = torch.ops.aten.pow.Tensor_Scalar(getitem_407, 2.0)
        sum_130 = torch.ops.aten.sum.dim_IntList(pow_175, [-1], True);  pow_175 = None
        pow_176 = torch.ops.aten.pow.Tensor_Scalar(sum_130, 0.5);  sum_130 = None
        clamp_min_86 = torch.ops.aten.clamp_min.default(pow_176, 1e-12);  pow_176 = None
        expand_259 = torch.ops.aten.expand.default(clamp_min_86, [8, 16, 48, 784]);  clamp_min_86 = None
        div_141 = torch.ops.aten.div.Tensor(getitem_407, expand_259);  getitem_407 = expand_259 = None
        pow_177 = torch.ops.aten.pow.Tensor_Scalar(getitem_408, 2.0)
        sum_131 = torch.ops.aten.sum.dim_IntList(pow_177, [-1], True);  pow_177 = None
        pow_178 = torch.ops.aten.pow.Tensor_Scalar(sum_131, 0.5);  sum_131 = None
        clamp_min_87 = torch.ops.aten.clamp_min.default(pow_178, 1e-12);  pow_178 = None
        expand_260 = torch.ops.aten.expand.default(clamp_min_87, [8, 16, 48, 784]);  clamp_min_87 = None
        div_142 = torch.ops.aten.div.Tensor(getitem_408, expand_260);  getitem_408 = expand_260 = None
        permute_416 = torch.ops.aten.permute.default(div_142, [0, 1, 3, 2]);  div_142 = None
        expand_261 = torch.ops.aten.expand.default(div_141, [8, 16, 48, 784]);  div_141 = None
        clone_483 = torch.ops.aten.clone.default(expand_261, memory_format = torch.contiguous_format);  expand_261 = None
        view_813 = torch.ops.aten.view.default(clone_483, [128, 48, 784]);  clone_483 = None
        expand_262 = torch.ops.aten.expand.default(permute_416, [8, 16, 784, 48]);  permute_416 = None
        clone_484 = torch.ops.aten.clone.default(expand_262, memory_format = torch.contiguous_format);  expand_262 = None
        view_814 = torch.ops.aten.view.default(clone_484, [128, 784, 48]);  clone_484 = None
        bmm_86 = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
        view_815 = torch.ops.aten.view.default(bmm_86, [8, 16, 48, 48]);  bmm_86 = None
        scalar_tensor_default_4 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_4 = torch.ops.aten.ge.Scalar(arg517_1, 0)
        neg_default_4 = torch.ops.aten.neg.default(scalar_tensor_default_4)
        where_self_4 = torch.ops.aten.where.self(ge_scalar_4, scalar_tensor_default_4, neg_default_4);  ge_scalar_4 = scalar_tensor_default_4 = neg_default_4 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_815, where_self_4);  view_815 = None
        amax_default_4 = torch.ops.aten.amax.default(mul_tensor_12, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_4);  mul_tensor_12 = amax_default_4 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(where_self_4, arg517_1);  where_self_4 = arg517_1 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(sub_tensor_4, mul_tensor_13);  sub_tensor_4 = mul_tensor_13 = None
        exp_43 = torch.ops.aten.exp.default(mul_tensor_14);  mul_tensor_14 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_143 = torch.ops.aten.div.Tensor(exp_43, sum_132);  exp_43 = sum_132 = None
        expand_263 = torch.ops.aten.expand.default(div_143, [8, 16, 48, 48]);  div_143 = None
        view_816 = torch.ops.aten.view.default(expand_263, [128, 48, 48]);  expand_263 = None
        expand_264 = torch.ops.aten.expand.default(getitem_409, [8, 16, 48, 784]);  getitem_409 = None
        clone_486 = torch.ops.aten.clone.default(expand_264, memory_format = torch.contiguous_format);  expand_264 = None
        view_817 = torch.ops.aten.view.default(clone_486, [128, 48, 784]);  clone_486 = None
        bmm_87 = torch.ops.aten.bmm.default(view_816, view_817);  view_816 = view_817 = None
        view_818 = torch.ops.aten.view.default(bmm_87, [8, 16, 48, 784]);  bmm_87 = None
        permute_417 = torch.ops.aten.permute.default(view_818, [0, 3, 1, 2]);  view_818 = None
        view_819 = torch.ops.aten.view.default(permute_417, [8, 784, 768]);  permute_417 = None
        permute_418 = torch.ops.aten.permute.default(arg518_1, [1, 0]);  arg518_1 = None
        clone_487 = torch.ops.aten.clone.default(view_819, memory_format = torch.contiguous_format);  view_819 = None
        view_820 = torch.ops.aten.view.default(clone_487, [6272, 768]);  clone_487 = None
        mm_45 = torch.ops.aten.mm.default(view_820, permute_418);  view_820 = permute_418 = None
        view_821 = torch.ops.aten.view.default(mm_45, [8, 784, 768]);  mm_45 = None
        add_644 = torch.ops.aten.add.Tensor(view_821, arg519_1);  view_821 = arg519_1 = None
        mul_876 = torch.ops.aten.mul.Tensor(arg512_1, add_644);  arg512_1 = add_644 = None
        add_645 = torch.ops.aten.add.Tensor(add_641, mul_876);  add_641 = mul_876 = None
        clone_489 = torch.ops.aten.clone.default(add_645, memory_format = torch.contiguous_format)
        var_mean_135 = torch.ops.aten.var_mean.correction(clone_489, [2], correction = 0, keepdim = True)
        getitem_410 = var_mean_135[0]
        getitem_411 = var_mean_135[1];  var_mean_135 = None
        add_646 = torch.ops.aten.add.Tensor(getitem_410, 1e-06);  getitem_410 = None
        rsqrt_135 = torch.ops.aten.rsqrt.default(add_646);  add_646 = None
        sub_228 = torch.ops.aten.sub.Tensor(clone_489, getitem_411);  clone_489 = getitem_411 = None
        mul_877 = torch.ops.aten.mul.Tensor(sub_228, rsqrt_135);  sub_228 = rsqrt_135 = None
        mul_878 = torch.ops.aten.mul.Tensor(mul_877, arg521_1);  mul_877 = arg521_1 = None
        add_647 = torch.ops.aten.add.Tensor(mul_878, arg522_1);  mul_878 = arg522_1 = None
        permute_419 = torch.ops.aten.permute.default(add_647, [0, 2, 1]);  add_647 = None
        view_822 = torch.ops.aten.view.default(permute_419, [8, 768, 28, 28]);  permute_419 = None
        convolution_94 = torch.ops.aten.convolution.default(view_822, arg523_1, arg524_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_822 = arg523_1 = arg524_1 = None
        mul_879 = torch.ops.aten.mul.Tensor(convolution_94, 0.5)
        mul_880 = torch.ops.aten.mul.Tensor(convolution_94, 0.7071067811865476);  convolution_94 = None
        erf_92 = torch.ops.aten.erf.default(mul_880);  mul_880 = None
        add_648 = torch.ops.aten.add.Tensor(erf_92, 1);  erf_92 = None
        mul_881 = torch.ops.aten.mul.Tensor(mul_879, add_648);  mul_879 = add_648 = None
        add_649 = torch.ops.aten.add.Tensor(arg526_1, 1e-05);  arg526_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_649);  add_649 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_882 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg525_1, -1);  arg525_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_882, -1);  mul_882 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_229 = torch.ops.aten.sub.Tensor(mul_881, unsqueeze_409);  mul_881 = unsqueeze_409 = None
        mul_883 = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_411);  sub_229 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg527_1, -1);  arg527_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_884 = torch.ops.aten.mul.Tensor(mul_883, unsqueeze_413);  mul_883 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg528_1, -1);  arg528_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_650 = torch.ops.aten.add.Tensor(mul_884, unsqueeze_415);  mul_884 = unsqueeze_415 = None
        convolution_95 = torch.ops.aten.convolution.default(add_650, arg529_1, arg530_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_650 = arg529_1 = arg530_1 = None
        view_823 = torch.ops.aten.view.default(convolution_95, [8, 768, 784]);  convolution_95 = None
        permute_420 = torch.ops.aten.permute.default(view_823, [0, 2, 1]);  view_823 = None
        mul_885 = torch.ops.aten.mul.Tensor(arg520_1, permute_420);  arg520_1 = permute_420 = None
        add_651 = torch.ops.aten.add.Tensor(add_645, mul_885);  add_645 = mul_885 = None
        clone_490 = torch.ops.aten.clone.default(add_651, memory_format = torch.contiguous_format)
        var_mean_136 = torch.ops.aten.var_mean.correction(clone_490, [2], correction = 0, keepdim = True)
        getitem_412 = var_mean_136[0]
        getitem_413 = var_mean_136[1];  var_mean_136 = None
        add_652 = torch.ops.aten.add.Tensor(getitem_412, 1e-06);  getitem_412 = None
        rsqrt_136 = torch.ops.aten.rsqrt.default(add_652);  add_652 = None
        sub_230 = torch.ops.aten.sub.Tensor(clone_490, getitem_413);  clone_490 = getitem_413 = None
        mul_886 = torch.ops.aten.mul.Tensor(sub_230, rsqrt_136);  sub_230 = rsqrt_136 = None
        mul_887 = torch.ops.aten.mul.Tensor(mul_886, arg532_1);  mul_886 = arg532_1 = None
        add_653 = torch.ops.aten.add.Tensor(mul_887, arg533_1);  mul_887 = arg533_1 = None
        view_824 = torch.ops.aten.view.default(add_653, [6272, 768]);  add_653 = None
        permute_421 = torch.ops.aten.permute.default(arg534_1, [1, 0]);  arg534_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg535_1, view_824, permute_421);  arg535_1 = view_824 = permute_421 = None
        view_825 = torch.ops.aten.view.default(addmm_141, [8, 784, 3072]);  addmm_141 = None
        mul_888 = torch.ops.aten.mul.Tensor(view_825, 0.5)
        mul_889 = torch.ops.aten.mul.Tensor(view_825, 0.7071067811865476);  view_825 = None
        erf_93 = torch.ops.aten.erf.default(mul_889);  mul_889 = None
        add_654 = torch.ops.aten.add.Tensor(erf_93, 1);  erf_93 = None
        mul_890 = torch.ops.aten.mul.Tensor(mul_888, add_654);  mul_888 = add_654 = None
        view_826 = torch.ops.aten.view.default(mul_890, [6272, 3072]);  mul_890 = None
        permute_422 = torch.ops.aten.permute.default(arg536_1, [1, 0]);  arg536_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg537_1, view_826, permute_422);  arg537_1 = view_826 = permute_422 = None
        view_827 = torch.ops.aten.view.default(addmm_142, [8, 784, 768]);  addmm_142 = None
        mul_891 = torch.ops.aten.mul.Tensor(arg531_1, view_827);  arg531_1 = view_827 = None
        add_655 = torch.ops.aten.add.Tensor(add_651, mul_891);  add_651 = mul_891 = None
        clone_493 = torch.ops.aten.clone.default(add_655, memory_format = torch.contiguous_format)
        var_mean_137 = torch.ops.aten.var_mean.correction(clone_493, [2], correction = 0, keepdim = True)
        getitem_414 = var_mean_137[0]
        getitem_415 = var_mean_137[1];  var_mean_137 = None
        add_656 = torch.ops.aten.add.Tensor(getitem_414, 1e-06);  getitem_414 = None
        rsqrt_137 = torch.ops.aten.rsqrt.default(add_656);  add_656 = None
        sub_231 = torch.ops.aten.sub.Tensor(clone_493, getitem_415);  clone_493 = getitem_415 = None
        mul_892 = torch.ops.aten.mul.Tensor(sub_231, rsqrt_137);  sub_231 = rsqrt_137 = None
        mul_893 = torch.ops.aten.mul.Tensor(mul_892, arg539_1);  mul_892 = arg539_1 = None
        add_657 = torch.ops.aten.add.Tensor(mul_893, arg540_1);  mul_893 = arg540_1 = None
        view_828 = torch.ops.aten.view.default(add_657, [6272, 768]);  add_657 = None
        permute_423 = torch.ops.aten.permute.default(arg541_1, [1, 0]);  arg541_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg542_1, view_828, permute_423);  arg542_1 = view_828 = permute_423 = None
        view_829 = torch.ops.aten.view.default(addmm_143, [8, 784, 2304]);  addmm_143 = None
        view_830 = torch.ops.aten.view.default(view_829, [8, 784, 3, 16, 48]);  view_829 = None
        permute_424 = torch.ops.aten.permute.default(view_830, [2, 0, 3, 4, 1]);  view_830 = None
        unbind_44 = torch.ops.aten.unbind.int(permute_424);  permute_424 = None
        getitem_416 = unbind_44[0]
        getitem_417 = unbind_44[1]
        getitem_418 = unbind_44[2];  unbind_44 = None
        pow_179 = torch.ops.aten.pow.Tensor_Scalar(getitem_416, 2.0)
        sum_133 = torch.ops.aten.sum.dim_IntList(pow_179, [-1], True);  pow_179 = None
        pow_180 = torch.ops.aten.pow.Tensor_Scalar(sum_133, 0.5);  sum_133 = None
        clamp_min_88 = torch.ops.aten.clamp_min.default(pow_180, 1e-12);  pow_180 = None
        expand_265 = torch.ops.aten.expand.default(clamp_min_88, [8, 16, 48, 784]);  clamp_min_88 = None
        div_144 = torch.ops.aten.div.Tensor(getitem_416, expand_265);  getitem_416 = expand_265 = None
        pow_181 = torch.ops.aten.pow.Tensor_Scalar(getitem_417, 2.0)
        sum_134 = torch.ops.aten.sum.dim_IntList(pow_181, [-1], True);  pow_181 = None
        pow_182 = torch.ops.aten.pow.Tensor_Scalar(sum_134, 0.5);  sum_134 = None
        clamp_min_89 = torch.ops.aten.clamp_min.default(pow_182, 1e-12);  pow_182 = None
        expand_266 = torch.ops.aten.expand.default(clamp_min_89, [8, 16, 48, 784]);  clamp_min_89 = None
        div_145 = torch.ops.aten.div.Tensor(getitem_417, expand_266);  getitem_417 = expand_266 = None
        permute_425 = torch.ops.aten.permute.default(div_145, [0, 1, 3, 2]);  div_145 = None
        expand_267 = torch.ops.aten.expand.default(div_144, [8, 16, 48, 784]);  div_144 = None
        clone_494 = torch.ops.aten.clone.default(expand_267, memory_format = torch.contiguous_format);  expand_267 = None
        view_831 = torch.ops.aten.view.default(clone_494, [128, 48, 784]);  clone_494 = None
        expand_268 = torch.ops.aten.expand.default(permute_425, [8, 16, 784, 48]);  permute_425 = None
        clone_495 = torch.ops.aten.clone.default(expand_268, memory_format = torch.contiguous_format);  expand_268 = None
        view_832 = torch.ops.aten.view.default(clone_495, [128, 784, 48]);  clone_495 = None
        bmm_88 = torch.ops.aten.bmm.default(view_831, view_832);  view_831 = view_832 = None
        view_833 = torch.ops.aten.view.default(bmm_88, [8, 16, 48, 48]);  bmm_88 = None
        scalar_tensor_default_3 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_3 = torch.ops.aten.ge.Scalar(arg543_1, 0)
        neg_default_3 = torch.ops.aten.neg.default(scalar_tensor_default_3)
        where_self_3 = torch.ops.aten.where.self(ge_scalar_3, scalar_tensor_default_3, neg_default_3);  ge_scalar_3 = scalar_tensor_default_3 = neg_default_3 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(view_833, where_self_3);  view_833 = None
        amax_default_3 = torch.ops.aten.amax.default(mul_tensor_9, [-1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_9, amax_default_3);  mul_tensor_9 = amax_default_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(where_self_3, arg543_1);  where_self_3 = arg543_1 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_3, mul_tensor_10);  sub_tensor_3 = mul_tensor_10 = None
        exp_44 = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_146 = torch.ops.aten.div.Tensor(exp_44, sum_135);  exp_44 = sum_135 = None
        expand_269 = torch.ops.aten.expand.default(div_146, [8, 16, 48, 48]);  div_146 = None
        view_834 = torch.ops.aten.view.default(expand_269, [128, 48, 48]);  expand_269 = None
        expand_270 = torch.ops.aten.expand.default(getitem_418, [8, 16, 48, 784]);  getitem_418 = None
        clone_497 = torch.ops.aten.clone.default(expand_270, memory_format = torch.contiguous_format);  expand_270 = None
        view_835 = torch.ops.aten.view.default(clone_497, [128, 48, 784]);  clone_497 = None
        bmm_89 = torch.ops.aten.bmm.default(view_834, view_835);  view_834 = view_835 = None
        view_836 = torch.ops.aten.view.default(bmm_89, [8, 16, 48, 784]);  bmm_89 = None
        permute_426 = torch.ops.aten.permute.default(view_836, [0, 3, 1, 2]);  view_836 = None
        view_837 = torch.ops.aten.view.default(permute_426, [8, 784, 768]);  permute_426 = None
        permute_427 = torch.ops.aten.permute.default(arg544_1, [1, 0]);  arg544_1 = None
        clone_498 = torch.ops.aten.clone.default(view_837, memory_format = torch.contiguous_format);  view_837 = None
        view_838 = torch.ops.aten.view.default(clone_498, [6272, 768]);  clone_498 = None
        mm_46 = torch.ops.aten.mm.default(view_838, permute_427);  view_838 = permute_427 = None
        view_839 = torch.ops.aten.view.default(mm_46, [8, 784, 768]);  mm_46 = None
        add_658 = torch.ops.aten.add.Tensor(view_839, arg545_1);  view_839 = arg545_1 = None
        mul_895 = torch.ops.aten.mul.Tensor(arg538_1, add_658);  arg538_1 = add_658 = None
        add_659 = torch.ops.aten.add.Tensor(add_655, mul_895);  add_655 = mul_895 = None
        clone_500 = torch.ops.aten.clone.default(add_659, memory_format = torch.contiguous_format)
        var_mean_138 = torch.ops.aten.var_mean.correction(clone_500, [2], correction = 0, keepdim = True)
        getitem_419 = var_mean_138[0]
        getitem_420 = var_mean_138[1];  var_mean_138 = None
        add_660 = torch.ops.aten.add.Tensor(getitem_419, 1e-06);  getitem_419 = None
        rsqrt_138 = torch.ops.aten.rsqrt.default(add_660);  add_660 = None
        sub_233 = torch.ops.aten.sub.Tensor(clone_500, getitem_420);  clone_500 = getitem_420 = None
        mul_896 = torch.ops.aten.mul.Tensor(sub_233, rsqrt_138);  sub_233 = rsqrt_138 = None
        mul_897 = torch.ops.aten.mul.Tensor(mul_896, arg547_1);  mul_896 = arg547_1 = None
        add_661 = torch.ops.aten.add.Tensor(mul_897, arg548_1);  mul_897 = arg548_1 = None
        permute_428 = torch.ops.aten.permute.default(add_661, [0, 2, 1]);  add_661 = None
        view_840 = torch.ops.aten.view.default(permute_428, [8, 768, 28, 28]);  permute_428 = None
        convolution_96 = torch.ops.aten.convolution.default(view_840, arg549_1, arg550_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_840 = arg549_1 = arg550_1 = None
        mul_898 = torch.ops.aten.mul.Tensor(convolution_96, 0.5)
        mul_899 = torch.ops.aten.mul.Tensor(convolution_96, 0.7071067811865476);  convolution_96 = None
        erf_94 = torch.ops.aten.erf.default(mul_899);  mul_899 = None
        add_662 = torch.ops.aten.add.Tensor(erf_94, 1);  erf_94 = None
        mul_900 = torch.ops.aten.mul.Tensor(mul_898, add_662);  mul_898 = add_662 = None
        add_663 = torch.ops.aten.add.Tensor(arg552_1, 1e-05);  arg552_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_663);  add_663 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_901 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg551_1, -1);  arg551_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_901, -1);  mul_901 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_234 = torch.ops.aten.sub.Tensor(mul_900, unsqueeze_417);  mul_900 = unsqueeze_417 = None
        mul_902 = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_419);  sub_234 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg553_1, -1);  arg553_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_903 = torch.ops.aten.mul.Tensor(mul_902, unsqueeze_421);  mul_902 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_664 = torch.ops.aten.add.Tensor(mul_903, unsqueeze_423);  mul_903 = unsqueeze_423 = None
        convolution_97 = torch.ops.aten.convolution.default(add_664, arg555_1, arg556_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_664 = arg555_1 = arg556_1 = None
        view_841 = torch.ops.aten.view.default(convolution_97, [8, 768, 784]);  convolution_97 = None
        permute_429 = torch.ops.aten.permute.default(view_841, [0, 2, 1]);  view_841 = None
        mul_904 = torch.ops.aten.mul.Tensor(arg546_1, permute_429);  arg546_1 = permute_429 = None
        add_665 = torch.ops.aten.add.Tensor(add_659, mul_904);  add_659 = mul_904 = None
        clone_501 = torch.ops.aten.clone.default(add_665, memory_format = torch.contiguous_format)
        var_mean_139 = torch.ops.aten.var_mean.correction(clone_501, [2], correction = 0, keepdim = True)
        getitem_421 = var_mean_139[0]
        getitem_422 = var_mean_139[1];  var_mean_139 = None
        add_666 = torch.ops.aten.add.Tensor(getitem_421, 1e-06);  getitem_421 = None
        rsqrt_139 = torch.ops.aten.rsqrt.default(add_666);  add_666 = None
        sub_235 = torch.ops.aten.sub.Tensor(clone_501, getitem_422);  clone_501 = getitem_422 = None
        mul_905 = torch.ops.aten.mul.Tensor(sub_235, rsqrt_139);  sub_235 = rsqrt_139 = None
        mul_906 = torch.ops.aten.mul.Tensor(mul_905, arg558_1);  mul_905 = arg558_1 = None
        add_667 = torch.ops.aten.add.Tensor(mul_906, arg559_1);  mul_906 = arg559_1 = None
        view_842 = torch.ops.aten.view.default(add_667, [6272, 768]);  add_667 = None
        permute_430 = torch.ops.aten.permute.default(arg560_1, [1, 0]);  arg560_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg561_1, view_842, permute_430);  arg561_1 = view_842 = permute_430 = None
        view_843 = torch.ops.aten.view.default(addmm_144, [8, 784, 3072]);  addmm_144 = None
        mul_907 = torch.ops.aten.mul.Tensor(view_843, 0.5)
        mul_908 = torch.ops.aten.mul.Tensor(view_843, 0.7071067811865476);  view_843 = None
        erf_95 = torch.ops.aten.erf.default(mul_908);  mul_908 = None
        add_668 = torch.ops.aten.add.Tensor(erf_95, 1);  erf_95 = None
        mul_909 = torch.ops.aten.mul.Tensor(mul_907, add_668);  mul_907 = add_668 = None
        view_844 = torch.ops.aten.view.default(mul_909, [6272, 3072]);  mul_909 = None
        permute_431 = torch.ops.aten.permute.default(arg562_1, [1, 0]);  arg562_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg563_1, view_844, permute_431);  arg563_1 = view_844 = permute_431 = None
        view_845 = torch.ops.aten.view.default(addmm_145, [8, 784, 768]);  addmm_145 = None
        mul_910 = torch.ops.aten.mul.Tensor(arg557_1, view_845);  arg557_1 = view_845 = None
        add_669 = torch.ops.aten.add.Tensor(add_665, mul_910);  add_665 = mul_910 = None
        clone_504 = torch.ops.aten.clone.default(add_669, memory_format = torch.contiguous_format)
        var_mean_140 = torch.ops.aten.var_mean.correction(clone_504, [2], correction = 0, keepdim = True)
        getitem_423 = var_mean_140[0]
        getitem_424 = var_mean_140[1];  var_mean_140 = None
        add_670 = torch.ops.aten.add.Tensor(getitem_423, 1e-06);  getitem_423 = None
        rsqrt_140 = torch.ops.aten.rsqrt.default(add_670);  add_670 = None
        sub_236 = torch.ops.aten.sub.Tensor(clone_504, getitem_424);  clone_504 = getitem_424 = None
        mul_911 = torch.ops.aten.mul.Tensor(sub_236, rsqrt_140);  sub_236 = rsqrt_140 = None
        mul_912 = torch.ops.aten.mul.Tensor(mul_911, arg565_1);  mul_911 = arg565_1 = None
        add_671 = torch.ops.aten.add.Tensor(mul_912, arg566_1);  mul_912 = arg566_1 = None
        view_846 = torch.ops.aten.view.default(add_671, [6272, 768]);  add_671 = None
        permute_432 = torch.ops.aten.permute.default(arg567_1, [1, 0]);  arg567_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg568_1, view_846, permute_432);  arg568_1 = view_846 = permute_432 = None
        view_847 = torch.ops.aten.view.default(addmm_146, [8, 784, 2304]);  addmm_146 = None
        view_848 = torch.ops.aten.view.default(view_847, [8, 784, 3, 16, 48]);  view_847 = None
        permute_433 = torch.ops.aten.permute.default(view_848, [2, 0, 3, 4, 1]);  view_848 = None
        unbind_45 = torch.ops.aten.unbind.int(permute_433);  permute_433 = None
        getitem_425 = unbind_45[0]
        getitem_426 = unbind_45[1]
        getitem_427 = unbind_45[2];  unbind_45 = None
        pow_183 = torch.ops.aten.pow.Tensor_Scalar(getitem_425, 2.0)
        sum_136 = torch.ops.aten.sum.dim_IntList(pow_183, [-1], True);  pow_183 = None
        pow_184 = torch.ops.aten.pow.Tensor_Scalar(sum_136, 0.5);  sum_136 = None
        clamp_min_90 = torch.ops.aten.clamp_min.default(pow_184, 1e-12);  pow_184 = None
        expand_271 = torch.ops.aten.expand.default(clamp_min_90, [8, 16, 48, 784]);  clamp_min_90 = None
        div_147 = torch.ops.aten.div.Tensor(getitem_425, expand_271);  getitem_425 = expand_271 = None
        pow_185 = torch.ops.aten.pow.Tensor_Scalar(getitem_426, 2.0)
        sum_137 = torch.ops.aten.sum.dim_IntList(pow_185, [-1], True);  pow_185 = None
        pow_186 = torch.ops.aten.pow.Tensor_Scalar(sum_137, 0.5);  sum_137 = None
        clamp_min_91 = torch.ops.aten.clamp_min.default(pow_186, 1e-12);  pow_186 = None
        expand_272 = torch.ops.aten.expand.default(clamp_min_91, [8, 16, 48, 784]);  clamp_min_91 = None
        div_148 = torch.ops.aten.div.Tensor(getitem_426, expand_272);  getitem_426 = expand_272 = None
        permute_434 = torch.ops.aten.permute.default(div_148, [0, 1, 3, 2]);  div_148 = None
        expand_273 = torch.ops.aten.expand.default(div_147, [8, 16, 48, 784]);  div_147 = None
        clone_505 = torch.ops.aten.clone.default(expand_273, memory_format = torch.contiguous_format);  expand_273 = None
        view_849 = torch.ops.aten.view.default(clone_505, [128, 48, 784]);  clone_505 = None
        expand_274 = torch.ops.aten.expand.default(permute_434, [8, 16, 784, 48]);  permute_434 = None
        clone_506 = torch.ops.aten.clone.default(expand_274, memory_format = torch.contiguous_format);  expand_274 = None
        view_850 = torch.ops.aten.view.default(clone_506, [128, 784, 48]);  clone_506 = None
        bmm_90 = torch.ops.aten.bmm.default(view_849, view_850);  view_849 = view_850 = None
        view_851 = torch.ops.aten.view.default(bmm_90, [8, 16, 48, 48]);  bmm_90 = None
        scalar_tensor_default_2 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_2 = torch.ops.aten.ge.Scalar(arg569_1, 0)
        neg_default_2 = torch.ops.aten.neg.default(scalar_tensor_default_2)
        where_self_2 = torch.ops.aten.where.self(ge_scalar_2, scalar_tensor_default_2, neg_default_2);  ge_scalar_2 = scalar_tensor_default_2 = neg_default_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_851, where_self_2);  view_851 = None
        amax_default_2 = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_2);  mul_tensor_6 = amax_default_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(where_self_2, arg569_1);  where_self_2 = arg569_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_2, mul_tensor_7);  sub_tensor_2 = mul_tensor_7 = None
        exp_45 = torch.ops.aten.exp.default(mul_tensor_8);  mul_tensor_8 = None
        sum_138 = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_149 = torch.ops.aten.div.Tensor(exp_45, sum_138);  exp_45 = sum_138 = None
        expand_275 = torch.ops.aten.expand.default(div_149, [8, 16, 48, 48]);  div_149 = None
        view_852 = torch.ops.aten.view.default(expand_275, [128, 48, 48]);  expand_275 = None
        expand_276 = torch.ops.aten.expand.default(getitem_427, [8, 16, 48, 784]);  getitem_427 = None
        clone_508 = torch.ops.aten.clone.default(expand_276, memory_format = torch.contiguous_format);  expand_276 = None
        view_853 = torch.ops.aten.view.default(clone_508, [128, 48, 784]);  clone_508 = None
        bmm_91 = torch.ops.aten.bmm.default(view_852, view_853);  view_852 = view_853 = None
        view_854 = torch.ops.aten.view.default(bmm_91, [8, 16, 48, 784]);  bmm_91 = None
        permute_435 = torch.ops.aten.permute.default(view_854, [0, 3, 1, 2]);  view_854 = None
        view_855 = torch.ops.aten.view.default(permute_435, [8, 784, 768]);  permute_435 = None
        permute_436 = torch.ops.aten.permute.default(arg570_1, [1, 0]);  arg570_1 = None
        clone_509 = torch.ops.aten.clone.default(view_855, memory_format = torch.contiguous_format);  view_855 = None
        view_856 = torch.ops.aten.view.default(clone_509, [6272, 768]);  clone_509 = None
        mm_47 = torch.ops.aten.mm.default(view_856, permute_436);  view_856 = permute_436 = None
        view_857 = torch.ops.aten.view.default(mm_47, [8, 784, 768]);  mm_47 = None
        add_672 = torch.ops.aten.add.Tensor(view_857, arg571_1);  view_857 = arg571_1 = None
        mul_914 = torch.ops.aten.mul.Tensor(arg564_1, add_672);  arg564_1 = add_672 = None
        add_673 = torch.ops.aten.add.Tensor(add_669, mul_914);  add_669 = mul_914 = None
        clone_511 = torch.ops.aten.clone.default(add_673, memory_format = torch.contiguous_format)
        var_mean_141 = torch.ops.aten.var_mean.correction(clone_511, [2], correction = 0, keepdim = True)
        getitem_428 = var_mean_141[0]
        getitem_429 = var_mean_141[1];  var_mean_141 = None
        add_674 = torch.ops.aten.add.Tensor(getitem_428, 1e-06);  getitem_428 = None
        rsqrt_141 = torch.ops.aten.rsqrt.default(add_674);  add_674 = None
        sub_238 = torch.ops.aten.sub.Tensor(clone_511, getitem_429);  clone_511 = getitem_429 = None
        mul_915 = torch.ops.aten.mul.Tensor(sub_238, rsqrt_141);  sub_238 = rsqrt_141 = None
        mul_916 = torch.ops.aten.mul.Tensor(mul_915, arg573_1);  mul_915 = arg573_1 = None
        add_675 = torch.ops.aten.add.Tensor(mul_916, arg574_1);  mul_916 = arg574_1 = None
        permute_437 = torch.ops.aten.permute.default(add_675, [0, 2, 1]);  add_675 = None
        view_858 = torch.ops.aten.view.default(permute_437, [8, 768, 28, 28]);  permute_437 = None
        convolution_98 = torch.ops.aten.convolution.default(view_858, arg575_1, arg576_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_858 = arg575_1 = arg576_1 = None
        mul_917 = torch.ops.aten.mul.Tensor(convolution_98, 0.5)
        mul_918 = torch.ops.aten.mul.Tensor(convolution_98, 0.7071067811865476);  convolution_98 = None
        erf_96 = torch.ops.aten.erf.default(mul_918);  mul_918 = None
        add_676 = torch.ops.aten.add.Tensor(erf_96, 1);  erf_96 = None
        mul_919 = torch.ops.aten.mul.Tensor(mul_917, add_676);  mul_917 = add_676 = None
        add_677 = torch.ops.aten.add.Tensor(arg578_1, 1e-05);  arg578_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_677);  add_677 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_920 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg577_1, -1);  arg577_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_920, -1);  mul_920 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_239 = torch.ops.aten.sub.Tensor(mul_919, unsqueeze_425);  mul_919 = unsqueeze_425 = None
        mul_921 = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_427);  sub_239 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_922 = torch.ops.aten.mul.Tensor(mul_921, unsqueeze_429);  mul_921 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg580_1, -1);  arg580_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_678 = torch.ops.aten.add.Tensor(mul_922, unsqueeze_431);  mul_922 = unsqueeze_431 = None
        convolution_99 = torch.ops.aten.convolution.default(add_678, arg581_1, arg582_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_678 = arg581_1 = arg582_1 = None
        view_859 = torch.ops.aten.view.default(convolution_99, [8, 768, 784]);  convolution_99 = None
        permute_438 = torch.ops.aten.permute.default(view_859, [0, 2, 1]);  view_859 = None
        mul_923 = torch.ops.aten.mul.Tensor(arg572_1, permute_438);  arg572_1 = permute_438 = None
        add_679 = torch.ops.aten.add.Tensor(add_673, mul_923);  add_673 = mul_923 = None
        clone_512 = torch.ops.aten.clone.default(add_679, memory_format = torch.contiguous_format)
        var_mean_142 = torch.ops.aten.var_mean.correction(clone_512, [2], correction = 0, keepdim = True)
        getitem_430 = var_mean_142[0]
        getitem_431 = var_mean_142[1];  var_mean_142 = None
        add_680 = torch.ops.aten.add.Tensor(getitem_430, 1e-06);  getitem_430 = None
        rsqrt_142 = torch.ops.aten.rsqrt.default(add_680);  add_680 = None
        sub_240 = torch.ops.aten.sub.Tensor(clone_512, getitem_431);  clone_512 = getitem_431 = None
        mul_924 = torch.ops.aten.mul.Tensor(sub_240, rsqrt_142);  sub_240 = rsqrt_142 = None
        mul_925 = torch.ops.aten.mul.Tensor(mul_924, arg584_1);  mul_924 = arg584_1 = None
        add_681 = torch.ops.aten.add.Tensor(mul_925, arg585_1);  mul_925 = arg585_1 = None
        view_860 = torch.ops.aten.view.default(add_681, [6272, 768]);  add_681 = None
        permute_439 = torch.ops.aten.permute.default(arg586_1, [1, 0]);  arg586_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg587_1, view_860, permute_439);  arg587_1 = view_860 = permute_439 = None
        view_861 = torch.ops.aten.view.default(addmm_147, [8, 784, 3072]);  addmm_147 = None
        mul_926 = torch.ops.aten.mul.Tensor(view_861, 0.5)
        mul_927 = torch.ops.aten.mul.Tensor(view_861, 0.7071067811865476);  view_861 = None
        erf_97 = torch.ops.aten.erf.default(mul_927);  mul_927 = None
        add_682 = torch.ops.aten.add.Tensor(erf_97, 1);  erf_97 = None
        mul_928 = torch.ops.aten.mul.Tensor(mul_926, add_682);  mul_926 = add_682 = None
        view_862 = torch.ops.aten.view.default(mul_928, [6272, 3072]);  mul_928 = None
        permute_440 = torch.ops.aten.permute.default(arg588_1, [1, 0]);  arg588_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg589_1, view_862, permute_440);  arg589_1 = view_862 = permute_440 = None
        view_863 = torch.ops.aten.view.default(addmm_148, [8, 784, 768]);  addmm_148 = None
        mul_929 = torch.ops.aten.mul.Tensor(arg583_1, view_863);  arg583_1 = view_863 = None
        add_683 = torch.ops.aten.add.Tensor(add_679, mul_929);  add_679 = mul_929 = None
        clone_515 = torch.ops.aten.clone.default(add_683, memory_format = torch.contiguous_format)
        var_mean_143 = torch.ops.aten.var_mean.correction(clone_515, [2], correction = 0, keepdim = True)
        getitem_432 = var_mean_143[0]
        getitem_433 = var_mean_143[1];  var_mean_143 = None
        add_684 = torch.ops.aten.add.Tensor(getitem_432, 1e-06);  getitem_432 = None
        rsqrt_143 = torch.ops.aten.rsqrt.default(add_684);  add_684 = None
        sub_241 = torch.ops.aten.sub.Tensor(clone_515, getitem_433);  clone_515 = getitem_433 = None
        mul_930 = torch.ops.aten.mul.Tensor(sub_241, rsqrt_143);  sub_241 = rsqrt_143 = None
        mul_931 = torch.ops.aten.mul.Tensor(mul_930, arg591_1);  mul_930 = arg591_1 = None
        add_685 = torch.ops.aten.add.Tensor(mul_931, arg592_1);  mul_931 = arg592_1 = None
        view_864 = torch.ops.aten.view.default(add_685, [6272, 768]);  add_685 = None
        permute_441 = torch.ops.aten.permute.default(arg593_1, [1, 0]);  arg593_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg594_1, view_864, permute_441);  arg594_1 = view_864 = permute_441 = None
        view_865 = torch.ops.aten.view.default(addmm_149, [8, 784, 2304]);  addmm_149 = None
        view_866 = torch.ops.aten.view.default(view_865, [8, 784, 3, 16, 48]);  view_865 = None
        permute_442 = torch.ops.aten.permute.default(view_866, [2, 0, 3, 4, 1]);  view_866 = None
        unbind_46 = torch.ops.aten.unbind.int(permute_442);  permute_442 = None
        getitem_434 = unbind_46[0]
        getitem_435 = unbind_46[1]
        getitem_436 = unbind_46[2];  unbind_46 = None
        pow_187 = torch.ops.aten.pow.Tensor_Scalar(getitem_434, 2.0)
        sum_139 = torch.ops.aten.sum.dim_IntList(pow_187, [-1], True);  pow_187 = None
        pow_188 = torch.ops.aten.pow.Tensor_Scalar(sum_139, 0.5);  sum_139 = None
        clamp_min_92 = torch.ops.aten.clamp_min.default(pow_188, 1e-12);  pow_188 = None
        expand_277 = torch.ops.aten.expand.default(clamp_min_92, [8, 16, 48, 784]);  clamp_min_92 = None
        div_150 = torch.ops.aten.div.Tensor(getitem_434, expand_277);  getitem_434 = expand_277 = None
        pow_189 = torch.ops.aten.pow.Tensor_Scalar(getitem_435, 2.0)
        sum_140 = torch.ops.aten.sum.dim_IntList(pow_189, [-1], True);  pow_189 = None
        pow_190 = torch.ops.aten.pow.Tensor_Scalar(sum_140, 0.5);  sum_140 = None
        clamp_min_93 = torch.ops.aten.clamp_min.default(pow_190, 1e-12);  pow_190 = None
        expand_278 = torch.ops.aten.expand.default(clamp_min_93, [8, 16, 48, 784]);  clamp_min_93 = None
        div_151 = torch.ops.aten.div.Tensor(getitem_435, expand_278);  getitem_435 = expand_278 = None
        permute_443 = torch.ops.aten.permute.default(div_151, [0, 1, 3, 2]);  div_151 = None
        expand_279 = torch.ops.aten.expand.default(div_150, [8, 16, 48, 784]);  div_150 = None
        clone_516 = torch.ops.aten.clone.default(expand_279, memory_format = torch.contiguous_format);  expand_279 = None
        view_867 = torch.ops.aten.view.default(clone_516, [128, 48, 784]);  clone_516 = None
        expand_280 = torch.ops.aten.expand.default(permute_443, [8, 16, 784, 48]);  permute_443 = None
        clone_517 = torch.ops.aten.clone.default(expand_280, memory_format = torch.contiguous_format);  expand_280 = None
        view_868 = torch.ops.aten.view.default(clone_517, [128, 784, 48]);  clone_517 = None
        bmm_92 = torch.ops.aten.bmm.default(view_867, view_868);  view_867 = view_868 = None
        view_869 = torch.ops.aten.view.default(bmm_92, [8, 16, 48, 48]);  bmm_92 = None
        scalar_tensor_default_1 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar_1 = torch.ops.aten.ge.Scalar(arg595_1, 0)
        neg_default_1 = torch.ops.aten.neg.default(scalar_tensor_default_1)
        where_self_1 = torch.ops.aten.where.self(ge_scalar_1, scalar_tensor_default_1, neg_default_1);  ge_scalar_1 = scalar_tensor_default_1 = neg_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_869, where_self_1);  view_869 = None
        amax_default_1 = torch.ops.aten.amax.default(mul_tensor_3, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_3, amax_default_1);  mul_tensor_3 = amax_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(where_self_1, arg595_1);  where_self_1 = arg595_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        exp_46 = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_152 = torch.ops.aten.div.Tensor(exp_46, sum_141);  exp_46 = sum_141 = None
        expand_281 = torch.ops.aten.expand.default(div_152, [8, 16, 48, 48]);  div_152 = None
        view_870 = torch.ops.aten.view.default(expand_281, [128, 48, 48]);  expand_281 = None
        expand_282 = torch.ops.aten.expand.default(getitem_436, [8, 16, 48, 784]);  getitem_436 = None
        clone_519 = torch.ops.aten.clone.default(expand_282, memory_format = torch.contiguous_format);  expand_282 = None
        view_871 = torch.ops.aten.view.default(clone_519, [128, 48, 784]);  clone_519 = None
        bmm_93 = torch.ops.aten.bmm.default(view_870, view_871);  view_870 = view_871 = None
        view_872 = torch.ops.aten.view.default(bmm_93, [8, 16, 48, 784]);  bmm_93 = None
        permute_444 = torch.ops.aten.permute.default(view_872, [0, 3, 1, 2]);  view_872 = None
        view_873 = torch.ops.aten.view.default(permute_444, [8, 784, 768]);  permute_444 = None
        permute_445 = torch.ops.aten.permute.default(arg596_1, [1, 0]);  arg596_1 = None
        clone_520 = torch.ops.aten.clone.default(view_873, memory_format = torch.contiguous_format);  view_873 = None
        view_874 = torch.ops.aten.view.default(clone_520, [6272, 768]);  clone_520 = None
        mm_48 = torch.ops.aten.mm.default(view_874, permute_445);  view_874 = permute_445 = None
        view_875 = torch.ops.aten.view.default(mm_48, [8, 784, 768]);  mm_48 = None
        add_686 = torch.ops.aten.add.Tensor(view_875, arg597_1);  view_875 = arg597_1 = None
        mul_933 = torch.ops.aten.mul.Tensor(arg590_1, add_686);  arg590_1 = add_686 = None
        add_687 = torch.ops.aten.add.Tensor(add_683, mul_933);  add_683 = mul_933 = None
        clone_522 = torch.ops.aten.clone.default(add_687, memory_format = torch.contiguous_format)
        var_mean_144 = torch.ops.aten.var_mean.correction(clone_522, [2], correction = 0, keepdim = True)
        getitem_437 = var_mean_144[0]
        getitem_438 = var_mean_144[1];  var_mean_144 = None
        add_688 = torch.ops.aten.add.Tensor(getitem_437, 1e-06);  getitem_437 = None
        rsqrt_144 = torch.ops.aten.rsqrt.default(add_688);  add_688 = None
        sub_243 = torch.ops.aten.sub.Tensor(clone_522, getitem_438);  clone_522 = getitem_438 = None
        mul_934 = torch.ops.aten.mul.Tensor(sub_243, rsqrt_144);  sub_243 = rsqrt_144 = None
        mul_935 = torch.ops.aten.mul.Tensor(mul_934, arg599_1);  mul_934 = arg599_1 = None
        add_689 = torch.ops.aten.add.Tensor(mul_935, arg600_1);  mul_935 = arg600_1 = None
        permute_446 = torch.ops.aten.permute.default(add_689, [0, 2, 1]);  add_689 = None
        view_876 = torch.ops.aten.view.default(permute_446, [8, 768, 28, 28]);  permute_446 = None
        convolution_100 = torch.ops.aten.convolution.default(view_876, arg601_1, arg602_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_876 = arg601_1 = arg602_1 = None
        mul_936 = torch.ops.aten.mul.Tensor(convolution_100, 0.5)
        mul_937 = torch.ops.aten.mul.Tensor(convolution_100, 0.7071067811865476);  convolution_100 = None
        erf_98 = torch.ops.aten.erf.default(mul_937);  mul_937 = None
        add_690 = torch.ops.aten.add.Tensor(erf_98, 1);  erf_98 = None
        mul_938 = torch.ops.aten.mul.Tensor(mul_936, add_690);  mul_936 = add_690 = None
        add_691 = torch.ops.aten.add.Tensor(arg604_1, 1e-05);  arg604_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_691);  add_691 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_939 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg603_1, -1);  arg603_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(mul_939, -1);  mul_939 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_244 = torch.ops.aten.sub.Tensor(mul_938, unsqueeze_433);  mul_938 = unsqueeze_433 = None
        mul_940 = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_435);  sub_244 = unsqueeze_435 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_941 = torch.ops.aten.mul.Tensor(mul_940, unsqueeze_437);  mul_940 = unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg606_1, -1);  arg606_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_692 = torch.ops.aten.add.Tensor(mul_941, unsqueeze_439);  mul_941 = unsqueeze_439 = None
        convolution_101 = torch.ops.aten.convolution.default(add_692, arg607_1, arg608_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_692 = arg607_1 = arg608_1 = None
        view_877 = torch.ops.aten.view.default(convolution_101, [8, 768, 784]);  convolution_101 = None
        permute_447 = torch.ops.aten.permute.default(view_877, [0, 2, 1]);  view_877 = None
        mul_942 = torch.ops.aten.mul.Tensor(arg598_1, permute_447);  arg598_1 = permute_447 = None
        add_693 = torch.ops.aten.add.Tensor(add_687, mul_942);  add_687 = mul_942 = None
        clone_523 = torch.ops.aten.clone.default(add_693, memory_format = torch.contiguous_format)
        var_mean_145 = torch.ops.aten.var_mean.correction(clone_523, [2], correction = 0, keepdim = True)
        getitem_439 = var_mean_145[0]
        getitem_440 = var_mean_145[1];  var_mean_145 = None
        add_694 = torch.ops.aten.add.Tensor(getitem_439, 1e-06);  getitem_439 = None
        rsqrt_145 = torch.ops.aten.rsqrt.default(add_694);  add_694 = None
        sub_245 = torch.ops.aten.sub.Tensor(clone_523, getitem_440);  clone_523 = getitem_440 = None
        mul_943 = torch.ops.aten.mul.Tensor(sub_245, rsqrt_145);  sub_245 = rsqrt_145 = None
        mul_944 = torch.ops.aten.mul.Tensor(mul_943, arg610_1);  mul_943 = arg610_1 = None
        add_695 = torch.ops.aten.add.Tensor(mul_944, arg611_1);  mul_944 = arg611_1 = None
        view_878 = torch.ops.aten.view.default(add_695, [6272, 768]);  add_695 = None
        permute_448 = torch.ops.aten.permute.default(arg612_1, [1, 0]);  arg612_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg613_1, view_878, permute_448);  arg613_1 = view_878 = permute_448 = None
        view_879 = torch.ops.aten.view.default(addmm_150, [8, 784, 3072]);  addmm_150 = None
        mul_945 = torch.ops.aten.mul.Tensor(view_879, 0.5)
        mul_946 = torch.ops.aten.mul.Tensor(view_879, 0.7071067811865476);  view_879 = None
        erf_99 = torch.ops.aten.erf.default(mul_946);  mul_946 = None
        add_696 = torch.ops.aten.add.Tensor(erf_99, 1);  erf_99 = None
        mul_947 = torch.ops.aten.mul.Tensor(mul_945, add_696);  mul_945 = add_696 = None
        view_880 = torch.ops.aten.view.default(mul_947, [6272, 3072]);  mul_947 = None
        permute_449 = torch.ops.aten.permute.default(arg614_1, [1, 0]);  arg614_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg615_1, view_880, permute_449);  arg615_1 = view_880 = permute_449 = None
        view_881 = torch.ops.aten.view.default(addmm_151, [8, 784, 768]);  addmm_151 = None
        mul_948 = torch.ops.aten.mul.Tensor(arg609_1, view_881);  arg609_1 = view_881 = None
        add_697 = torch.ops.aten.add.Tensor(add_693, mul_948);  add_693 = mul_948 = None
        clone_526 = torch.ops.aten.clone.default(add_697, memory_format = torch.contiguous_format)
        var_mean_146 = torch.ops.aten.var_mean.correction(clone_526, [2], correction = 0, keepdim = True)
        getitem_441 = var_mean_146[0]
        getitem_442 = var_mean_146[1];  var_mean_146 = None
        add_698 = torch.ops.aten.add.Tensor(getitem_441, 1e-06);  getitem_441 = None
        rsqrt_146 = torch.ops.aten.rsqrt.default(add_698);  add_698 = None
        sub_246 = torch.ops.aten.sub.Tensor(clone_526, getitem_442);  clone_526 = getitem_442 = None
        mul_949 = torch.ops.aten.mul.Tensor(sub_246, rsqrt_146);  sub_246 = rsqrt_146 = None
        mul_950 = torch.ops.aten.mul.Tensor(mul_949, arg617_1);  mul_949 = arg617_1 = None
        add_699 = torch.ops.aten.add.Tensor(mul_950, arg618_1);  mul_950 = arg618_1 = None
        view_882 = torch.ops.aten.view.default(add_699, [6272, 768]);  add_699 = None
        permute_450 = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg620_1, view_882, permute_450);  arg620_1 = view_882 = permute_450 = None
        view_883 = torch.ops.aten.view.default(addmm_152, [8, 784, 2304]);  addmm_152 = None
        view_884 = torch.ops.aten.view.default(view_883, [8, 784, 3, 16, 48]);  view_883 = None
        permute_451 = torch.ops.aten.permute.default(view_884, [2, 0, 3, 4, 1]);  view_884 = None
        unbind_47 = torch.ops.aten.unbind.int(permute_451);  permute_451 = None
        getitem_443 = unbind_47[0]
        getitem_444 = unbind_47[1]
        getitem_445 = unbind_47[2];  unbind_47 = None
        pow_191 = torch.ops.aten.pow.Tensor_Scalar(getitem_443, 2.0)
        sum_142 = torch.ops.aten.sum.dim_IntList(pow_191, [-1], True);  pow_191 = None
        pow_192 = torch.ops.aten.pow.Tensor_Scalar(sum_142, 0.5);  sum_142 = None
        clamp_min_94 = torch.ops.aten.clamp_min.default(pow_192, 1e-12);  pow_192 = None
        expand_283 = torch.ops.aten.expand.default(clamp_min_94, [8, 16, 48, 784]);  clamp_min_94 = None
        div_153 = torch.ops.aten.div.Tensor(getitem_443, expand_283);  getitem_443 = expand_283 = None
        pow_193 = torch.ops.aten.pow.Tensor_Scalar(getitem_444, 2.0)
        sum_143 = torch.ops.aten.sum.dim_IntList(pow_193, [-1], True);  pow_193 = None
        pow_194 = torch.ops.aten.pow.Tensor_Scalar(sum_143, 0.5);  sum_143 = None
        clamp_min_95 = torch.ops.aten.clamp_min.default(pow_194, 1e-12);  pow_194 = None
        expand_284 = torch.ops.aten.expand.default(clamp_min_95, [8, 16, 48, 784]);  clamp_min_95 = None
        div_154 = torch.ops.aten.div.Tensor(getitem_444, expand_284);  getitem_444 = expand_284 = None
        permute_452 = torch.ops.aten.permute.default(div_154, [0, 1, 3, 2]);  div_154 = None
        expand_285 = torch.ops.aten.expand.default(div_153, [8, 16, 48, 784]);  div_153 = None
        clone_527 = torch.ops.aten.clone.default(expand_285, memory_format = torch.contiguous_format);  expand_285 = None
        view_885 = torch.ops.aten.view.default(clone_527, [128, 48, 784]);  clone_527 = None
        expand_286 = torch.ops.aten.expand.default(permute_452, [8, 16, 784, 48]);  permute_452 = None
        clone_528 = torch.ops.aten.clone.default(expand_286, memory_format = torch.contiguous_format);  expand_286 = None
        view_886 = torch.ops.aten.view.default(clone_528, [128, 784, 48]);  clone_528 = None
        bmm_94 = torch.ops.aten.bmm.default(view_885, view_886);  view_885 = view_886 = None
        view_887 = torch.ops.aten.view.default(bmm_94, [8, 16, 48, 48]);  bmm_94 = None
        scalar_tensor_default = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar = torch.ops.aten.ge.Scalar(arg621_1, 0)
        neg_default = torch.ops.aten.neg.default(scalar_tensor_default)
        where_self = torch.ops.aten.where.self(ge_scalar, scalar_tensor_default, neg_default);  ge_scalar = scalar_tensor_default = neg_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_887, where_self);  view_887 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(where_self, arg621_1);  where_self = arg621_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, mul_tensor_1);  sub_tensor = mul_tensor_1 = None
        exp_47 = torch.ops.aten.exp.default(mul_tensor_2);  mul_tensor_2 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_155 = torch.ops.aten.div.Tensor(exp_47, sum_144);  exp_47 = sum_144 = None
        expand_287 = torch.ops.aten.expand.default(div_155, [8, 16, 48, 48]);  div_155 = None
        view_888 = torch.ops.aten.view.default(expand_287, [128, 48, 48]);  expand_287 = None
        expand_288 = torch.ops.aten.expand.default(getitem_445, [8, 16, 48, 784]);  getitem_445 = None
        clone_530 = torch.ops.aten.clone.default(expand_288, memory_format = torch.contiguous_format);  expand_288 = None
        view_889 = torch.ops.aten.view.default(clone_530, [128, 48, 784]);  clone_530 = None
        bmm_95 = torch.ops.aten.bmm.default(view_888, view_889);  view_888 = view_889 = None
        view_890 = torch.ops.aten.view.default(bmm_95, [8, 16, 48, 784]);  bmm_95 = None
        permute_453 = torch.ops.aten.permute.default(view_890, [0, 3, 1, 2]);  view_890 = None
        view_891 = torch.ops.aten.view.default(permute_453, [8, 784, 768]);  permute_453 = None
        permute_454 = torch.ops.aten.permute.default(arg622_1, [1, 0]);  arg622_1 = None
        clone_531 = torch.ops.aten.clone.default(view_891, memory_format = torch.contiguous_format);  view_891 = None
        view_892 = torch.ops.aten.view.default(clone_531, [6272, 768]);  clone_531 = None
        mm_49 = torch.ops.aten.mm.default(view_892, permute_454);  view_892 = permute_454 = None
        view_893 = torch.ops.aten.view.default(mm_49, [8, 784, 768]);  mm_49 = None
        add_700 = torch.ops.aten.add.Tensor(view_893, arg623_1);  view_893 = arg623_1 = None
        mul_952 = torch.ops.aten.mul.Tensor(arg616_1, add_700);  arg616_1 = add_700 = None
        add_701 = torch.ops.aten.add.Tensor(add_697, mul_952);  add_697 = mul_952 = None
        clone_533 = torch.ops.aten.clone.default(add_701, memory_format = torch.contiguous_format)
        var_mean_147 = torch.ops.aten.var_mean.correction(clone_533, [2], correction = 0, keepdim = True)
        getitem_446 = var_mean_147[0]
        getitem_447 = var_mean_147[1];  var_mean_147 = None
        add_702 = torch.ops.aten.add.Tensor(getitem_446, 1e-06);  getitem_446 = None
        rsqrt_147 = torch.ops.aten.rsqrt.default(add_702);  add_702 = None
        sub_248 = torch.ops.aten.sub.Tensor(clone_533, getitem_447);  clone_533 = getitem_447 = None
        mul_953 = torch.ops.aten.mul.Tensor(sub_248, rsqrt_147);  sub_248 = rsqrt_147 = None
        mul_954 = torch.ops.aten.mul.Tensor(mul_953, arg625_1);  mul_953 = arg625_1 = None
        add_703 = torch.ops.aten.add.Tensor(mul_954, arg626_1);  mul_954 = arg626_1 = None
        permute_455 = torch.ops.aten.permute.default(add_703, [0, 2, 1]);  add_703 = None
        view_894 = torch.ops.aten.view.default(permute_455, [8, 768, 28, 28]);  permute_455 = None
        convolution_102 = torch.ops.aten.convolution.default(view_894, arg627_1, arg628_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  view_894 = arg627_1 = arg628_1 = None
        mul_955 = torch.ops.aten.mul.Tensor(convolution_102, 0.5)
        mul_956 = torch.ops.aten.mul.Tensor(convolution_102, 0.7071067811865476);  convolution_102 = None
        erf_100 = torch.ops.aten.erf.default(mul_956);  mul_956 = None
        add_704 = torch.ops.aten.add.Tensor(erf_100, 1);  erf_100 = None
        mul_957 = torch.ops.aten.mul.Tensor(mul_955, add_704);  mul_955 = add_704 = None
        add_705 = torch.ops.aten.add.Tensor(arg630_1, 1e-05);  arg630_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_705);  add_705 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_958 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(arg629_1, -1);  arg629_1 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(mul_958, -1);  mul_958 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_249 = torch.ops.aten.sub.Tensor(mul_957, unsqueeze_441);  mul_957 = unsqueeze_441 = None
        mul_959 = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_443);  sub_249 = unsqueeze_443 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg631_1, -1);  arg631_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_960 = torch.ops.aten.mul.Tensor(mul_959, unsqueeze_445);  mul_959 = unsqueeze_445 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(arg632_1, -1);  arg632_1 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_706 = torch.ops.aten.add.Tensor(mul_960, unsqueeze_447);  mul_960 = unsqueeze_447 = None
        convolution_103 = torch.ops.aten.convolution.default(add_706, arg633_1, arg634_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  add_706 = arg633_1 = arg634_1 = None
        view_895 = torch.ops.aten.view.default(convolution_103, [8, 768, 784]);  convolution_103 = None
        permute_456 = torch.ops.aten.permute.default(view_895, [0, 2, 1]);  view_895 = None
        mul_961 = torch.ops.aten.mul.Tensor(arg624_1, permute_456);  arg624_1 = permute_456 = None
        add_707 = torch.ops.aten.add.Tensor(add_701, mul_961);  add_701 = mul_961 = None
        clone_534 = torch.ops.aten.clone.default(add_707, memory_format = torch.contiguous_format)
        var_mean_148 = torch.ops.aten.var_mean.correction(clone_534, [2], correction = 0, keepdim = True)
        getitem_448 = var_mean_148[0]
        getitem_449 = var_mean_148[1];  var_mean_148 = None
        add_708 = torch.ops.aten.add.Tensor(getitem_448, 1e-06);  getitem_448 = None
        rsqrt_148 = torch.ops.aten.rsqrt.default(add_708);  add_708 = None
        sub_250 = torch.ops.aten.sub.Tensor(clone_534, getitem_449);  clone_534 = getitem_449 = None
        mul_962 = torch.ops.aten.mul.Tensor(sub_250, rsqrt_148);  sub_250 = rsqrt_148 = None
        mul_963 = torch.ops.aten.mul.Tensor(mul_962, arg636_1);  mul_962 = arg636_1 = None
        add_709 = torch.ops.aten.add.Tensor(mul_963, arg637_1);  mul_963 = arg637_1 = None
        view_896 = torch.ops.aten.view.default(add_709, [6272, 768]);  add_709 = None
        permute_457 = torch.ops.aten.permute.default(arg638_1, [1, 0]);  arg638_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg639_1, view_896, permute_457);  arg639_1 = view_896 = permute_457 = None
        view_897 = torch.ops.aten.view.default(addmm_153, [8, 784, 3072]);  addmm_153 = None
        mul_964 = torch.ops.aten.mul.Tensor(view_897, 0.5)
        mul_965 = torch.ops.aten.mul.Tensor(view_897, 0.7071067811865476);  view_897 = None
        erf_101 = torch.ops.aten.erf.default(mul_965);  mul_965 = None
        add_710 = torch.ops.aten.add.Tensor(erf_101, 1);  erf_101 = None
        mul_966 = torch.ops.aten.mul.Tensor(mul_964, add_710);  mul_964 = add_710 = None
        view_898 = torch.ops.aten.view.default(mul_966, [6272, 3072]);  mul_966 = None
        permute_458 = torch.ops.aten.permute.default(arg640_1, [1, 0]);  arg640_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg641_1, view_898, permute_458);  arg641_1 = view_898 = permute_458 = None
        view_899 = torch.ops.aten.view.default(addmm_154, [8, 784, 768]);  addmm_154 = None
        mul_967 = torch.ops.aten.mul.Tensor(arg635_1, view_899);  arg635_1 = view_899 = None
        add_711 = torch.ops.aten.add.Tensor(add_707, mul_967);  add_707 = mul_967 = None
        expand_289 = torch.ops.aten.expand.default(arg642_1, [8, -1, -1]);  arg642_1 = None
        cat_11 = torch.ops.aten.cat.default([expand_289, add_711], 1);  expand_289 = add_711 = None
        var_mean_149 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
        getitem_450 = var_mean_149[0]
        getitem_451 = var_mean_149[1];  var_mean_149 = None
        add_712 = torch.ops.aten.add.Tensor(getitem_450, 1e-06);  getitem_450 = None
        rsqrt_149 = torch.ops.aten.rsqrt.default(add_712);  add_712 = None
        sub_251 = torch.ops.aten.sub.Tensor(cat_11, getitem_451);  getitem_451 = None
        mul_968 = torch.ops.aten.mul.Tensor(sub_251, rsqrt_149);  sub_251 = rsqrt_149 = None
        mul_969 = torch.ops.aten.mul.Tensor(mul_968, arg643_1);  mul_968 = arg643_1 = None
        add_713 = torch.ops.aten.add.Tensor(mul_969, arg644_1);  mul_969 = arg644_1 = None
        select_3 = torch.ops.aten.select.int(add_713, 1, 0)
        permute_459 = torch.ops.aten.permute.default(arg645_1, [1, 0]);  arg645_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg646_1, select_3, permute_459);  arg646_1 = select_3 = permute_459 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(addmm_155, 1);  addmm_155 = None
        view_900 = torch.ops.aten.view.default(unsqueeze_448, [8, 1, 16, 48]);  unsqueeze_448 = None
        permute_460 = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None
        view_901 = torch.ops.aten.view.default(add_713, [6280, 768])
        permute_461 = torch.ops.aten.permute.default(arg647_1, [1, 0]);  arg647_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg648_1, view_901, permute_461);  arg648_1 = view_901 = permute_461 = None
        view_902 = torch.ops.aten.view.default(addmm_156, [8, 785, 768]);  addmm_156 = None
        view_903 = torch.ops.aten.view.default(view_902, [8, 785, 16, 48]);  view_902 = None
        permute_462 = torch.ops.aten.permute.default(view_903, [0, 2, 1, 3]);  view_903 = None
        view_904 = torch.ops.aten.view.default(add_713, [6280, 768])
        permute_463 = torch.ops.aten.permute.default(arg649_1, [1, 0]);  arg649_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg650_1, view_904, permute_463);  arg650_1 = view_904 = permute_463 = None
        view_905 = torch.ops.aten.view.default(addmm_157, [8, 785, 768]);  addmm_157 = None
        view_906 = torch.ops.aten.view.default(view_905, [8, 785, 16, 48]);  view_905 = None
        permute_464 = torch.ops.aten.permute.default(view_906, [0, 2, 1, 3]);  view_906 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_460, permute_462, permute_464, None, False);  permute_460 = permute_462 = permute_464 = None
        getitem_452 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_465 = torch.ops.aten.permute.default(getitem_452, [0, 2, 1, 3]);  getitem_452 = None
        view_907 = torch.ops.aten.view.default(permute_465, [8, 1, 768]);  permute_465 = None
        view_908 = torch.ops.aten.view.default(view_907, [8, 768]);  view_907 = None
        permute_466 = torch.ops.aten.permute.default(arg651_1, [1, 0]);  arg651_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg652_1, view_908, permute_466);  arg652_1 = view_908 = permute_466 = None
        view_909 = torch.ops.aten.view.default(addmm_158, [8, 1, 768]);  addmm_158 = None
        slice_74 = torch.ops.aten.slice.Tensor(add_713, 1, 1, 9223372036854775807);  add_713 = None
        cat_12 = torch.ops.aten.cat.default([view_909, slice_74], 1);  view_909 = slice_74 = None
        mul_970 = torch.ops.aten.mul.Tensor(arg653_1, cat_12);  arg653_1 = cat_12 = None
        add_714 = torch.ops.aten.add.Tensor(cat_11, mul_970);  cat_11 = mul_970 = None
        var_mean_150 = torch.ops.aten.var_mean.correction(add_714, [2], correction = 0, keepdim = True)
        getitem_456 = var_mean_150[0]
        getitem_457 = var_mean_150[1];  var_mean_150 = None
        add_715 = torch.ops.aten.add.Tensor(getitem_456, 1e-06);  getitem_456 = None
        rsqrt_150 = torch.ops.aten.rsqrt.default(add_715);  add_715 = None
        sub_252 = torch.ops.aten.sub.Tensor(add_714, getitem_457);  add_714 = getitem_457 = None
        mul_971 = torch.ops.aten.mul.Tensor(sub_252, rsqrt_150);  sub_252 = rsqrt_150 = None
        mul_972 = torch.ops.aten.mul.Tensor(mul_971, arg654_1);  mul_971 = arg654_1 = None
        add_716 = torch.ops.aten.add.Tensor(mul_972, arg655_1);  mul_972 = arg655_1 = None
        slice_76 = torch.ops.aten.slice.Tensor(add_716, 1, 0, 1)
        permute_467 = torch.ops.aten.permute.default(arg657_1, [1, 0]);  arg657_1 = None
        view_910 = torch.ops.aten.view.default(slice_76, [8, 768]);  slice_76 = None
        mm_50 = torch.ops.aten.mm.default(view_910, permute_467);  view_910 = permute_467 = None
        view_911 = torch.ops.aten.view.default(mm_50, [8, 1, 3072]);  mm_50 = None
        add_717 = torch.ops.aten.add.Tensor(view_911, arg658_1);  view_911 = arg658_1 = None
        mul_973 = torch.ops.aten.mul.Tensor(add_717, 0.5)
        mul_974 = torch.ops.aten.mul.Tensor(add_717, 0.7071067811865476);  add_717 = None
        erf_102 = torch.ops.aten.erf.default(mul_974);  mul_974 = None
        add_718 = torch.ops.aten.add.Tensor(erf_102, 1);  erf_102 = None
        mul_975 = torch.ops.aten.mul.Tensor(mul_973, add_718);  mul_973 = add_718 = None
        view_912 = torch.ops.aten.view.default(mul_975, [8, 3072]);  mul_975 = None
        permute_468 = torch.ops.aten.permute.default(arg659_1, [1, 0]);  arg659_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg660_1, view_912, permute_468);  arg660_1 = view_912 = permute_468 = None
        view_913 = torch.ops.aten.view.default(addmm_159, [8, 1, 768]);  addmm_159 = None
        mul_976 = torch.ops.aten.mul.Tensor(arg656_1, view_913);  arg656_1 = view_913 = None
        slice_78 = torch.ops.aten.slice.Tensor(add_716, 1, 1, 9223372036854775807)
        cat_13 = torch.ops.aten.cat.default([mul_976, slice_78], 1);  mul_976 = slice_78 = None
        add_719 = torch.ops.aten.add.Tensor(add_716, cat_13);  add_716 = cat_13 = None
        var_mean_151 = torch.ops.aten.var_mean.correction(add_719, [2], correction = 0, keepdim = True)
        getitem_458 = var_mean_151[0]
        getitem_459 = var_mean_151[1];  var_mean_151 = None
        add_720 = torch.ops.aten.add.Tensor(getitem_458, 1e-06);  getitem_458 = None
        rsqrt_151 = torch.ops.aten.rsqrt.default(add_720);  add_720 = None
        sub_253 = torch.ops.aten.sub.Tensor(add_719, getitem_459);  getitem_459 = None
        mul_977 = torch.ops.aten.mul.Tensor(sub_253, rsqrt_151);  sub_253 = rsqrt_151 = None
        mul_978 = torch.ops.aten.mul.Tensor(mul_977, arg661_1);  mul_977 = arg661_1 = None
        add_721 = torch.ops.aten.add.Tensor(mul_978, arg662_1);  mul_978 = arg662_1 = None
        select_4 = torch.ops.aten.select.int(add_721, 1, 0)
        permute_469 = torch.ops.aten.permute.default(arg663_1, [1, 0]);  arg663_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg664_1, select_4, permute_469);  arg664_1 = select_4 = permute_469 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(addmm_160, 1);  addmm_160 = None
        view_914 = torch.ops.aten.view.default(unsqueeze_449, [8, 1, 16, 48]);  unsqueeze_449 = None
        permute_470 = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None
        view_915 = torch.ops.aten.view.default(add_721, [6280, 768])
        permute_471 = torch.ops.aten.permute.default(arg665_1, [1, 0]);  arg665_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg666_1, view_915, permute_471);  arg666_1 = view_915 = permute_471 = None
        view_916 = torch.ops.aten.view.default(addmm_161, [8, 785, 768]);  addmm_161 = None
        view_917 = torch.ops.aten.view.default(view_916, [8, 785, 16, 48]);  view_916 = None
        permute_472 = torch.ops.aten.permute.default(view_917, [0, 2, 1, 3]);  view_917 = None
        view_918 = torch.ops.aten.view.default(add_721, [6280, 768])
        permute_473 = torch.ops.aten.permute.default(arg667_1, [1, 0]);  arg667_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg668_1, view_918, permute_473);  arg668_1 = view_918 = permute_473 = None
        view_919 = torch.ops.aten.view.default(addmm_162, [8, 785, 768]);  addmm_162 = None
        view_920 = torch.ops.aten.view.default(view_919, [8, 785, 16, 48]);  view_919 = None
        permute_474 = torch.ops.aten.permute.default(view_920, [0, 2, 1, 3]);  view_920 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_470, permute_472, permute_474, None, False);  permute_470 = permute_472 = permute_474 = None
        getitem_460 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_475 = torch.ops.aten.permute.default(getitem_460, [0, 2, 1, 3]);  getitem_460 = None
        view_921 = torch.ops.aten.view.default(permute_475, [8, 1, 768]);  permute_475 = None
        view_922 = torch.ops.aten.view.default(view_921, [8, 768]);  view_921 = None
        permute_476 = torch.ops.aten.permute.default(arg669_1, [1, 0]);  arg669_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg670_1, view_922, permute_476);  arg670_1 = view_922 = permute_476 = None
        view_923 = torch.ops.aten.view.default(addmm_163, [8, 1, 768]);  addmm_163 = None
        slice_81 = torch.ops.aten.slice.Tensor(add_721, 1, 1, 9223372036854775807);  add_721 = None
        cat_14 = torch.ops.aten.cat.default([view_923, slice_81], 1);  view_923 = slice_81 = None
        mul_979 = torch.ops.aten.mul.Tensor(arg671_1, cat_14);  arg671_1 = cat_14 = None
        add_722 = torch.ops.aten.add.Tensor(add_719, mul_979);  add_719 = mul_979 = None
        var_mean_152 = torch.ops.aten.var_mean.correction(add_722, [2], correction = 0, keepdim = True)
        getitem_464 = var_mean_152[0]
        getitem_465 = var_mean_152[1];  var_mean_152 = None
        add_723 = torch.ops.aten.add.Tensor(getitem_464, 1e-06);  getitem_464 = None
        rsqrt_152 = torch.ops.aten.rsqrt.default(add_723);  add_723 = None
        sub_254 = torch.ops.aten.sub.Tensor(add_722, getitem_465);  add_722 = getitem_465 = None
        mul_980 = torch.ops.aten.mul.Tensor(sub_254, rsqrt_152);  sub_254 = rsqrt_152 = None
        mul_981 = torch.ops.aten.mul.Tensor(mul_980, arg672_1);  mul_980 = arg672_1 = None
        add_724 = torch.ops.aten.add.Tensor(mul_981, arg673_1);  mul_981 = arg673_1 = None
        slice_83 = torch.ops.aten.slice.Tensor(add_724, 1, 0, 1)
        permute_477 = torch.ops.aten.permute.default(arg675_1, [1, 0]);  arg675_1 = None
        view_924 = torch.ops.aten.view.default(slice_83, [8, 768]);  slice_83 = None
        mm_51 = torch.ops.aten.mm.default(view_924, permute_477);  view_924 = permute_477 = None
        view_925 = torch.ops.aten.view.default(mm_51, [8, 1, 3072]);  mm_51 = None
        add_725 = torch.ops.aten.add.Tensor(view_925, arg676_1);  view_925 = arg676_1 = None
        mul_982 = torch.ops.aten.mul.Tensor(add_725, 0.5)
        mul_983 = torch.ops.aten.mul.Tensor(add_725, 0.7071067811865476);  add_725 = None
        erf_103 = torch.ops.aten.erf.default(mul_983);  mul_983 = None
        add_726 = torch.ops.aten.add.Tensor(erf_103, 1);  erf_103 = None
        mul_984 = torch.ops.aten.mul.Tensor(mul_982, add_726);  mul_982 = add_726 = None
        view_926 = torch.ops.aten.view.default(mul_984, [8, 3072]);  mul_984 = None
        permute_478 = torch.ops.aten.permute.default(arg677_1, [1, 0]);  arg677_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg678_1, view_926, permute_478);  arg678_1 = view_926 = permute_478 = None
        view_927 = torch.ops.aten.view.default(addmm_164, [8, 1, 768]);  addmm_164 = None
        mul_985 = torch.ops.aten.mul.Tensor(arg674_1, view_927);  arg674_1 = view_927 = None
        slice_85 = torch.ops.aten.slice.Tensor(add_724, 1, 1, 9223372036854775807)
        cat_15 = torch.ops.aten.cat.default([mul_985, slice_85], 1);  mul_985 = slice_85 = None
        add_727 = torch.ops.aten.add.Tensor(add_724, cat_15);  add_724 = cat_15 = None
        var_mean_153 = torch.ops.aten.var_mean.correction(add_727, [2], correction = 0, keepdim = True)
        getitem_466 = var_mean_153[0]
        getitem_467 = var_mean_153[1];  var_mean_153 = None
        add_728 = torch.ops.aten.add.Tensor(getitem_466, 1e-06);  getitem_466 = None
        rsqrt_153 = torch.ops.aten.rsqrt.default(add_728);  add_728 = None
        sub_255 = torch.ops.aten.sub.Tensor(add_727, getitem_467);  add_727 = getitem_467 = None
        mul_986 = torch.ops.aten.mul.Tensor(sub_255, rsqrt_153);  sub_255 = rsqrt_153 = None
        mul_987 = torch.ops.aten.mul.Tensor(mul_986, arg679_1);  mul_986 = arg679_1 = None
        add_729 = torch.ops.aten.add.Tensor(mul_987, arg680_1);  mul_987 = arg680_1 = None
        select_5 = torch.ops.aten.select.int(add_729, 1, 0);  add_729 = None
        clone_543 = torch.ops.aten.clone.default(select_5);  select_5 = None
        permute_479 = torch.ops.aten.permute.default(arg681_1, [1, 0]);  arg681_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg682_1, clone_543, permute_479);  arg682_1 = clone_543 = permute_479 = None
        return (addmm_165,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf1, (192, 3, 3, 3), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf2, (192,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf3, (192,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf4, (192,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf5, (192,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 2654208, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384, 192, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf7, (384,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (384,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf9, (384,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf10, (384,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 10616832, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768, 384, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768, 64, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf21, (2304, 768), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf22, (2304,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf23, (16, 1, 1), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768, 768), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768, 1, 3, 3), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768, 1, 3, 3), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf40, (3072, 768), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf41, (3072,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768, 3072), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf47, (2304, 768), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf48, (2304,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf49, (16, 1, 1), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768, 768), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768, 1, 3, 3), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768, 1, 3, 3), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf66, (3072, 768), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf67, (3072,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768, 3072), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf73, (2304, 768), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf74, (2304,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf75, (16, 1, 1), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768, 768), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768, 1, 3, 3), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768, 1, 3, 3), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf92, (3072, 768), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf93, (3072,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768, 3072), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf99, (2304, 768), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf100, (2304,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf101, (16, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 768), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768, 1, 3, 3), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768, 1, 3, 3), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf118, (3072, 768), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf119, (3072,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768, 3072), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf125, (2304, 768), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf126, (2304,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf127, (16, 1, 1), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768, 768), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768, 1, 3, 3), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768, 1, 3, 3), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf144, (3072, 768), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf145, (3072,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768, 3072), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf151, (2304, 768), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf152, (2304,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf153, (16, 1, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768, 768), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768, 1, 3, 3), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 1, 3, 3), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf170, (3072, 768), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf171, (3072,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768, 3072), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf174, (768,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf177, (2304, 768), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf178, (2304,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf179, (16, 1, 1), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf180, (768, 768), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768, 1, 3, 3), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf189, (768,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768, 1, 3, 3), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf192, (768,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf195, (768,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf196, (3072, 768), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf197, (3072,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768, 3072), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf199, (768,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf200, (768,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf201, (768,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf202, (768,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf203, (2304, 768), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf204, (2304,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf205, (16, 1, 1), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf206, (768, 768), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf207, (768,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf208, (768,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf209, (768,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf210, (768,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf211, (768, 1, 3, 3), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf212, (768,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf213, (768,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf214, (768,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf215, (768,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf216, (768,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf217, (768, 1, 3, 3), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf218, (768,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf219, (768,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf220, (768,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf221, (768,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf222, (3072, 768), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf223, (3072,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf224, (768, 3072), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf225, (768,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf226, (768,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf227, (768,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf228, (768,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf229, (2304, 768), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf230, (2304,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf231, (16, 1, 1), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf232, (768, 768), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf233, (768,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf234, (768,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf235, (768,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf236, (768,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf237, (768, 1, 3, 3), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf238, (768,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf239, (768,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf240, (768,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf241, (768,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf242, (768,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf243, (768, 1, 3, 3), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf244, (768,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf245, (768,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf246, (768,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf247, (768,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf248, (3072, 768), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf249, (3072,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf250, (768, 3072), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf251, (768,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf252, (768,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf253, (768,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf254, (768,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf255, (2304, 768), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf256, (2304,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf257, (16, 1, 1), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf258, (768, 768), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf259, (768,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf260, (768,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf261, (768,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf262, (768,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf263, (768, 1, 3, 3), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf264, (768,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf265, (768,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf266, (768,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf267, (768,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf268, (768,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf269, (768, 1, 3, 3), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf270, (768,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf271, (768,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf272, (768,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf273, (768,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf274, (3072, 768), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf275, (3072,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf276, (768, 3072), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf277, (768,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf278, (768,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf279, (768,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf280, (768,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf281, (2304, 768), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf282, (2304,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf283, (16, 1, 1), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf284, (768, 768), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf285, (768,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf286, (768,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf287, (768,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf288, (768,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf289, (768, 1, 3, 3), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf290, (768,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf291, (768,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf292, (768,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf293, (768,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf294, (768,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf295, (768, 1, 3, 3), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf296, (768,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf297, (768,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf298, (768,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf299, (768,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf300, (3072, 768), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf301, (3072,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf302, (768, 3072), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf303, (768,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf304, (768,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf305, (768,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf306, (768,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf307, (2304, 768), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf308, (2304,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf309, (16, 1, 1), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf310, (768, 768), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf311, (768,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf312, (768,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf313, (768,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf314, (768,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf315, (768, 1, 3, 3), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf316, (768,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf317, (768,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf318, (768,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf319, (768,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf320, (768,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf321, (768, 1, 3, 3), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf322, (768,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf323, (768,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf324, (768,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf325, (768,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf326, (3072, 768), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf327, (3072,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf328, (768, 3072), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf329, (768,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf330, (768,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf331, (768,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf332, (768,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf333, (2304, 768), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf334, (2304,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf335, (16, 1, 1), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf336, (768, 768), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf337, (768,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf338, (768,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf339, (768,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf340, (768,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf341, (768, 1, 3, 3), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf342, (768,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf343, (768,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf344, (768,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf345, (768,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf346, (768,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf347, (768, 1, 3, 3), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf348, (768,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf349, (768,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf350, (768,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf351, (768,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf352, (3072, 768), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf353, (3072,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf354, (768, 3072), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf355, (768,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf356, (768,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf357, (768,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf358, (768,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf359, (2304, 768), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf360, (2304,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf361, (16, 1, 1), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf362, (768, 768), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf363, (768,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf364, (768,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf365, (768,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf366, (768,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf367, (768, 1, 3, 3), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf368, (768,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf369, (768,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf370, (768,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf371, (768,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf372, (768,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf373, (768, 1, 3, 3), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf374, (768,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf375, (768,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf376, (768,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf377, (768,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf378, (3072, 768), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf379, (3072,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf380, (768, 3072), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf381, (768,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf382, (768,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf383, (768,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf384, (768,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf385, (2304, 768), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf386, (2304,), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf387, (16, 1, 1), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf388, (768, 768), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf389, (768,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf390, (768,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf391, (768,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf392, (768,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf393, (768, 1, 3, 3), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf394, (768,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf395, (768,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf396, (768,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf397, (768,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf398, (768,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf399, (768, 1, 3, 3), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf400, (768,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf401, (768,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf402, (768,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf403, (768,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf404, (3072, 768), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf405, (3072,), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf406, (768, 3072), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf407, (768,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf408, (768,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf409, (768,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf410, (768,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf411, (2304, 768), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf412, (2304,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf413, (16, 1, 1), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf414, (768, 768), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf415, (768,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf416, (768,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf417, (768,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf418, (768,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf419, (768, 1, 3, 3), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf420, (768,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf421, (768,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf422, (768,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf423, (768,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf424, (768,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf425, (768, 1, 3, 3), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf426, (768,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf427, (768,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf428, (768,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf429, (768,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf430, (3072, 768), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf431, (3072,), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf432, (768, 3072), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf433, (768,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf434, (768,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf435, (768,), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf436, (768,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf437, (2304, 768), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf438, (2304,), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf439, (16, 1, 1), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf440, (768, 768), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf441, (768,), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf442, (768,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf443, (768,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf444, (768,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf445, (768, 1, 3, 3), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf446, (768,), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf447, (768,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf448, (768,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf449, (768,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf450, (768,), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf451, (768, 1, 3, 3), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf452, (768,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf453, (768,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf454, (768,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf455, (768,), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf456, (3072, 768), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf457, (3072,), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf458, (768, 3072), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf459, (768,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf460, (768,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf461, (768,), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf462, (768,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf463, (2304, 768), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf464, (2304,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf465, (16, 1, 1), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf466, (768, 768), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf467, (768,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf468, (768,), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf469, (768,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf470, (768,), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf471, (768, 1, 3, 3), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf472, (768,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf473, (768,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf474, (768,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf475, (768,), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf476, (768,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf477, (768, 1, 3, 3), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf478, (768,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf479, (768,), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf480, (768,), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf481, (768,), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf482, (3072, 768), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf483, (3072,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf484, (768, 3072), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf485, (768,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf486, (768,), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf487, (768,), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf488, (768,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf489, (2304, 768), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf490, (2304,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf491, (16, 1, 1), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf492, (768, 768), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf493, (768,), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf494, (768,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf495, (768,), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf496, (768,), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf497, (768, 1, 3, 3), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf498, (768,), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf499, (768,), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf500, (768,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf501, (768,), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf502, (768,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf503, (768, 1, 3, 3), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf504, (768,), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf505, (768,), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf506, (768,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf507, (768,), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf508, (3072, 768), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf509, (3072,), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf510, (768, 3072), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf511, (768,), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf512, (768,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf513, (768,), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf514, (768,), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf515, (2304, 768), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf516, (2304,), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf517, (16, 1, 1), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf518, (768, 768), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf519, (768,), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf520, (768,), is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf521, (768,), is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf522, (768,), is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf523, (768, 1, 3, 3), is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf524, (768,), is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf525, (768,), is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf526, (768,), is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf527, (768,), is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf528, (768,), is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf529, (768, 1, 3, 3), is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf530, (768,), is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf531, (768,), is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf532, (768,), is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf533, (768,), is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf534, (3072, 768), is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf535, (3072,), is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf536, (768, 3072), is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf537, (768,), is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf538, (768,), is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf539, (768,), is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf540, (768,), is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf541, (2304, 768), is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf542, (2304,), is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf543, (16, 1, 1), is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf544, (768, 768), is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf545, (768,), is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf546, (768,), is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf547, (768,), is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf548, (768,), is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf549, (768, 1, 3, 3), is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf550, (768,), is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf551, (768,), is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf552, (768,), is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf553, (768,), is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf554, (768,), is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf555, (768, 1, 3, 3), is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf556, (768,), is_leaf=True)  # arg556_1
    buf557 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf557, (768,), is_leaf=True)  # arg557_1
    buf558 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf558, (768,), is_leaf=True)  # arg558_1
    buf559 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf559, (768,), is_leaf=True)  # arg559_1
    buf560 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf560, (3072, 768), is_leaf=True)  # arg560_1
    buf561 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf561, (3072,), is_leaf=True)  # arg561_1
    buf562 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf562, (768, 3072), is_leaf=True)  # arg562_1
    buf563 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf563, (768,), is_leaf=True)  # arg563_1
    buf564 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf564, (768,), is_leaf=True)  # arg564_1
    buf565 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf565, (768,), is_leaf=True)  # arg565_1
    buf566 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf566, (768,), is_leaf=True)  # arg566_1
    buf567 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf567, (2304, 768), is_leaf=True)  # arg567_1
    buf568 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf568, (2304,), is_leaf=True)  # arg568_1
    buf569 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf569, (16, 1, 1), is_leaf=True)  # arg569_1
    buf570 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf570, (768, 768), is_leaf=True)  # arg570_1
    buf571 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf571, (768,), is_leaf=True)  # arg571_1
    buf572 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf572, (768,), is_leaf=True)  # arg572_1
    buf573 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf573, (768,), is_leaf=True)  # arg573_1
    buf574 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf574, (768,), is_leaf=True)  # arg574_1
    buf575 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf575, (768, 1, 3, 3), is_leaf=True)  # arg575_1
    buf576 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf576, (768,), is_leaf=True)  # arg576_1
    buf577 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf577, (768,), is_leaf=True)  # arg577_1
    buf578 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf578, (768,), is_leaf=True)  # arg578_1
    buf579 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf579, (768,), is_leaf=True)  # arg579_1
    buf580 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf580, (768,), is_leaf=True)  # arg580_1
    buf581 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf581, (768, 1, 3, 3), is_leaf=True)  # arg581_1
    buf582 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf582, (768,), is_leaf=True)  # arg582_1
    buf583 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf583, (768,), is_leaf=True)  # arg583_1
    buf584 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf584, (768,), is_leaf=True)  # arg584_1
    buf585 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf585, (768,), is_leaf=True)  # arg585_1
    buf586 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf586, (3072, 768), is_leaf=True)  # arg586_1
    buf587 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf587, (3072,), is_leaf=True)  # arg587_1
    buf588 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf588, (768, 3072), is_leaf=True)  # arg588_1
    buf589 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf589, (768,), is_leaf=True)  # arg589_1
    buf590 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf590, (768,), is_leaf=True)  # arg590_1
    buf591 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf591, (768,), is_leaf=True)  # arg591_1
    buf592 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf592, (768,), is_leaf=True)  # arg592_1
    buf593 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf593, (2304, 768), is_leaf=True)  # arg593_1
    buf594 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf594, (2304,), is_leaf=True)  # arg594_1
    buf595 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf595, (16, 1, 1), is_leaf=True)  # arg595_1
    buf596 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf596, (768, 768), is_leaf=True)  # arg596_1
    buf597 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf597, (768,), is_leaf=True)  # arg597_1
    buf598 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf598, (768,), is_leaf=True)  # arg598_1
    buf599 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf599, (768,), is_leaf=True)  # arg599_1
    buf600 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf600, (768,), is_leaf=True)  # arg600_1
    buf601 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf601, (768, 1, 3, 3), is_leaf=True)  # arg601_1
    buf602 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf602, (768,), is_leaf=True)  # arg602_1
    buf603 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf603, (768,), is_leaf=True)  # arg603_1
    buf604 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf604, (768,), is_leaf=True)  # arg604_1
    buf605 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf605, (768,), is_leaf=True)  # arg605_1
    buf606 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf606, (768,), is_leaf=True)  # arg606_1
    buf607 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf607, (768, 1, 3, 3), is_leaf=True)  # arg607_1
    buf608 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf608, (768,), is_leaf=True)  # arg608_1
    buf609 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf609, (768,), is_leaf=True)  # arg609_1
    buf610 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf610, (768,), is_leaf=True)  # arg610_1
    buf611 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf611, (768,), is_leaf=True)  # arg611_1
    buf612 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf612, (3072, 768), is_leaf=True)  # arg612_1
    buf613 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf613, (3072,), is_leaf=True)  # arg613_1
    buf614 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf614, (768, 3072), is_leaf=True)  # arg614_1
    buf615 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf615, (768,), is_leaf=True)  # arg615_1
    buf616 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf616, (768,), is_leaf=True)  # arg616_1
    buf617 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf617, (768,), is_leaf=True)  # arg617_1
    buf618 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf618, (768,), is_leaf=True)  # arg618_1
    buf619 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf619, (2304, 768), is_leaf=True)  # arg619_1
    buf620 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf620, (2304,), is_leaf=True)  # arg620_1
    buf621 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf621, (16, 1, 1), is_leaf=True)  # arg621_1
    buf622 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf622, (768, 768), is_leaf=True)  # arg622_1
    buf623 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf623, (768,), is_leaf=True)  # arg623_1
    buf624 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf624, (768,), is_leaf=True)  # arg624_1
    buf625 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf625, (768,), is_leaf=True)  # arg625_1
    buf626 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf626, (768,), is_leaf=True)  # arg626_1
    buf627 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf627, (768, 1, 3, 3), is_leaf=True)  # arg627_1
    buf628 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf628, (768,), is_leaf=True)  # arg628_1
    buf629 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf629, (768,), is_leaf=True)  # arg629_1
    buf630 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf630, (768,), is_leaf=True)  # arg630_1
    buf631 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf631, (768,), is_leaf=True)  # arg631_1
    buf632 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf632, (768,), is_leaf=True)  # arg632_1
    buf633 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf633, (768, 1, 3, 3), is_leaf=True)  # arg633_1
    buf634 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf634, (768,), is_leaf=True)  # arg634_1
    buf635 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf635, (768,), is_leaf=True)  # arg635_1
    buf636 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf636, (768,), is_leaf=True)  # arg636_1
    buf637 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf637, (768,), is_leaf=True)  # arg637_1
    buf638 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf638, (3072, 768), is_leaf=True)  # arg638_1
    buf639 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf639, (3072,), is_leaf=True)  # arg639_1
    buf640 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf640, (768, 3072), is_leaf=True)  # arg640_1
    buf641 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf641, (768,), is_leaf=True)  # arg641_1
    buf642 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf642, (1, 1, 768), is_leaf=True)  # arg642_1
    buf643 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf643, (768,), is_leaf=True)  # arg643_1
    buf644 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf644, (768,), is_leaf=True)  # arg644_1
    buf645 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf645, (768, 768), is_leaf=True)  # arg645_1
    buf646 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf646, (768,), is_leaf=True)  # arg646_1
    buf647 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf647, (768, 768), is_leaf=True)  # arg647_1
    buf648 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf648, (768,), is_leaf=True)  # arg648_1
    buf649 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf649, (768, 768), is_leaf=True)  # arg649_1
    buf650 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf650, (768,), is_leaf=True)  # arg650_1
    buf651 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf651, (768, 768), is_leaf=True)  # arg651_1
    buf652 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf652, (768,), is_leaf=True)  # arg652_1
    buf653 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf653, (768,), is_leaf=True)  # arg653_1
    buf654 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf654, (768,), is_leaf=True)  # arg654_1
    buf655 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf655, (768,), is_leaf=True)  # arg655_1
    buf656 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf656, (768,), is_leaf=True)  # arg656_1
    buf657 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf657, (3072, 768), is_leaf=True)  # arg657_1
    buf658 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf658, (3072,), is_leaf=True)  # arg658_1
    buf659 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf659, (768, 3072), is_leaf=True)  # arg659_1
    buf660 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf660, (768,), is_leaf=True)  # arg660_1
    buf661 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf661, (768,), is_leaf=True)  # arg661_1
    buf662 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf662, (768,), is_leaf=True)  # arg662_1
    buf663 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf663, (768, 768), is_leaf=True)  # arg663_1
    buf664 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf664, (768,), is_leaf=True)  # arg664_1
    buf665 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf665, (768, 768), is_leaf=True)  # arg665_1
    buf666 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf666, (768,), is_leaf=True)  # arg666_1
    buf667 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf667, (768, 768), is_leaf=True)  # arg667_1
    buf668 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf668, (768,), is_leaf=True)  # arg668_1
    buf669 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf669, (768, 768), is_leaf=True)  # arg669_1
    buf670 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf670, (768,), is_leaf=True)  # arg670_1
    buf671 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf671, (768,), is_leaf=True)  # arg671_1
    buf672 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf672, (768,), is_leaf=True)  # arg672_1
    buf673 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf673, (768,), is_leaf=True)  # arg673_1
    buf674 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf674, (768,), is_leaf=True)  # arg674_1
    buf675 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf675, (3072, 768), is_leaf=True)  # arg675_1
    buf676 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf676, (3072,), is_leaf=True)  # arg676_1
    buf677 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf677, (768, 3072), is_leaf=True)  # arg677_1
    buf678 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf678, (768,), is_leaf=True)  # arg678_1
    buf679 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf679, (768,), is_leaf=True)  # arg679_1
    buf680 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf680, (768,), is_leaf=True)  # arg680_1
    buf681 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf681, (1000, 768), is_leaf=True)  # arg681_1
    buf682 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf682, (1000,), is_leaf=True)  # arg682_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)