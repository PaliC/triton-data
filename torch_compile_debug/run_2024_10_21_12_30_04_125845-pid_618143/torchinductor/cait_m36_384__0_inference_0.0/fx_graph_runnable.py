
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1):
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_749 = torch.ops.aten.view.default(convolution_1, [8, 768, 576]);  convolution_1 = None
        permute_490 = torch.ops.aten.permute.default(view_749, [0, 2, 1]);  view_749 = None
        add_341 = torch.ops.aten.add.Tensor(permute_490, arg3_1);  permute_490 = arg3_1 = None
        clone_513 = torch.ops.aten.clone.default(add_341, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_513, [2], correction = 0, keepdim = True)
        getitem_162 = var_mean_77[0]
        getitem_163 = var_mean_77[1];  var_mean_77 = None
        add_342 = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_342);  add_342 = None
        sub_113 = torch.ops.aten.sub.Tensor(clone_513, getitem_163);  clone_513 = getitem_163 = None
        mul_380 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_77);  sub_113 = rsqrt_77 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_380, arg5_1);  mul_380 = arg5_1 = None
        add_343 = torch.ops.aten.add.Tensor(mul_381, arg6_1);  mul_381 = arg6_1 = None
        view_750 = torch.ops.aten.view.default(add_343, [4608, 768]);  add_343 = None
        permute_491 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg8_1, view_750, permute_491);  arg8_1 = view_750 = permute_491 = None
        view_751 = torch.ops.aten.view.default(addmm_157, [8, 576, 2304]);  addmm_157 = None
        view_752 = torch.ops.aten.view.default(view_751, [8, 576, 3, 16, 48]);  view_751 = None
        permute_492 = torch.ops.aten.permute.default(view_752, [2, 0, 3, 1, 4]);  view_752 = None
        select_111 = torch.ops.aten.select.int(permute_492, 0, 0)
        mul_382 = torch.ops.aten.mul.Tensor(select_111, 0.14433756729740643);  select_111 = None
        select_112 = torch.ops.aten.select.int(permute_492, 0, 1)
        select_113 = torch.ops.aten.select.int(permute_492, 0, 2);  permute_492 = None
        permute_493 = torch.ops.aten.permute.default(select_112, [0, 1, 3, 2]);  select_112 = None
        expand_145 = torch.ops.aten.expand.default(mul_382, [8, 16, 576, 48]);  mul_382 = None
        clone_514 = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_753 = torch.ops.aten.view.default(clone_514, [128, 576, 48]);  clone_514 = None
        expand_146 = torch.ops.aten.expand.default(permute_493, [8, 16, 48, 576]);  permute_493 = None
        clone_515 = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_754 = torch.ops.aten.view.default(clone_515, [128, 48, 576]);  clone_515 = None
        bmm_72 = torch.ops.aten.bmm.default(view_753, view_754);  view_753 = view_754 = None
        view_755 = torch.ops.aten.view.default(bmm_72, [8, 16, 576, 576]);  bmm_72 = None
        permute_494 = torch.ops.aten.permute.default(view_755, [0, 2, 3, 1]);  view_755 = None
        permute_495 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        clone_516 = torch.ops.aten.clone.default(permute_494, memory_format = torch.contiguous_format);  permute_494 = None
        view_756 = torch.ops.aten.view.default(clone_516, [2654208, 16]);  clone_516 = None
        mm_72 = torch.ops.aten.mm.default(view_756, permute_495);  view_756 = permute_495 = None
        view_757 = torch.ops.aten.view.default(mm_72, [8, 576, 576, 16]);  mm_72 = None
        add_344 = torch.ops.aten.add.Tensor(view_757, arg10_1);  view_757 = arg10_1 = None
        permute_496 = torch.ops.aten.permute.default(add_344, [0, 3, 1, 2]);  add_344 = None
        clone_517 = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
        amax_36 = torch.ops.aten.amax.default(clone_517, [-1], True)
        sub_114 = torch.ops.aten.sub.Tensor(clone_517, amax_36);  clone_517 = amax_36 = None
        exp_36 = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36 = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        permute_497 = torch.ops.aten.permute.default(div_36, [0, 2, 3, 1]);  div_36 = None
        permute_498 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        clone_518 = torch.ops.aten.clone.default(permute_497, memory_format = torch.contiguous_format);  permute_497 = None
        view_758 = torch.ops.aten.view.default(clone_518, [2654208, 16]);  clone_518 = None
        mm_73 = torch.ops.aten.mm.default(view_758, permute_498);  view_758 = permute_498 = None
        view_759 = torch.ops.aten.view.default(mm_73, [8, 576, 576, 16]);  mm_73 = None
        add_345 = torch.ops.aten.add.Tensor(view_759, arg12_1);  view_759 = arg12_1 = None
        permute_499 = torch.ops.aten.permute.default(add_345, [0, 3, 1, 2]);  add_345 = None
        expand_147 = torch.ops.aten.expand.default(permute_499, [8, 16, 576, 576]);  permute_499 = None
        clone_520 = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_760 = torch.ops.aten.view.default(clone_520, [128, 576, 576]);  clone_520 = None
        expand_148 = torch.ops.aten.expand.default(select_113, [8, 16, 576, 48]);  select_113 = None
        clone_521 = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_761 = torch.ops.aten.view.default(clone_521, [128, 576, 48]);  clone_521 = None
        bmm_73 = torch.ops.aten.bmm.default(view_760, view_761);  view_760 = view_761 = None
        view_762 = torch.ops.aten.view.default(bmm_73, [8, 16, 576, 48]);  bmm_73 = None
        permute_500 = torch.ops.aten.permute.default(view_762, [0, 2, 1, 3]);  view_762 = None
        clone_522 = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_763 = torch.ops.aten.view.default(clone_522, [8, 576, 768]);  clone_522 = None
        view_764 = torch.ops.aten.view.default(view_763, [4608, 768]);  view_763 = None
        permute_501 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg14_1, view_764, permute_501);  arg14_1 = view_764 = permute_501 = None
        view_765 = torch.ops.aten.view.default(addmm_158, [8, 576, 768]);  addmm_158 = None
        mul_383 = torch.ops.aten.mul.Tensor(arg4_1, view_765);  arg4_1 = view_765 = None
        add_346 = torch.ops.aten.add.Tensor(add_341, mul_383);  add_341 = mul_383 = None
        clone_524 = torch.ops.aten.clone.default(add_346, memory_format = torch.contiguous_format)
        var_mean_78 = torch.ops.aten.var_mean.correction(clone_524, [2], correction = 0, keepdim = True)
        getitem_164 = var_mean_78[0]
        getitem_165 = var_mean_78[1];  var_mean_78 = None
        add_347 = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_347);  add_347 = None
        sub_115 = torch.ops.aten.sub.Tensor(clone_524, getitem_165);  clone_524 = getitem_165 = None
        mul_384 = torch.ops.aten.mul.Tensor(sub_115, rsqrt_78);  sub_115 = rsqrt_78 = None
        mul_385 = torch.ops.aten.mul.Tensor(mul_384, arg16_1);  mul_384 = arg16_1 = None
        add_348 = torch.ops.aten.add.Tensor(mul_385, arg17_1);  mul_385 = arg17_1 = None
        view_766 = torch.ops.aten.view.default(add_348, [4608, 768]);  add_348 = None
        permute_502 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg19_1, view_766, permute_502);  arg19_1 = view_766 = permute_502 = None
        view_767 = torch.ops.aten.view.default(addmm_159, [8, 576, 3072]);  addmm_159 = None
        mul_386 = torch.ops.aten.mul.Tensor(view_767, 0.5)
        mul_387 = torch.ops.aten.mul.Tensor(view_767, 0.7071067811865476);  view_767 = None
        erf_38 = torch.ops.aten.erf.default(mul_387);  mul_387 = None
        add_349 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_388 = torch.ops.aten.mul.Tensor(mul_386, add_349);  mul_386 = add_349 = None
        view_768 = torch.ops.aten.view.default(mul_388, [4608, 3072]);  mul_388 = None
        permute_503 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg21_1, view_768, permute_503);  arg21_1 = view_768 = permute_503 = None
        view_769 = torch.ops.aten.view.default(addmm_160, [8, 576, 768]);  addmm_160 = None
        mul_389 = torch.ops.aten.mul.Tensor(arg15_1, view_769);  arg15_1 = view_769 = None
        add_350 = torch.ops.aten.add.Tensor(add_346, mul_389);  add_346 = mul_389 = None
        clone_527 = torch.ops.aten.clone.default(add_350, memory_format = torch.contiguous_format)
        var_mean_79 = torch.ops.aten.var_mean.correction(clone_527, [2], correction = 0, keepdim = True)
        getitem_166 = var_mean_79[0]
        getitem_167 = var_mean_79[1];  var_mean_79 = None
        add_351 = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_116 = torch.ops.aten.sub.Tensor(clone_527, getitem_167);  clone_527 = getitem_167 = None
        mul_390 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_79);  sub_116 = rsqrt_79 = None
        mul_391 = torch.ops.aten.mul.Tensor(mul_390, arg23_1);  mul_390 = arg23_1 = None
        add_352 = torch.ops.aten.add.Tensor(mul_391, arg24_1);  mul_391 = arg24_1 = None
        view_770 = torch.ops.aten.view.default(add_352, [4608, 768]);  add_352 = None
        permute_504 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg26_1, view_770, permute_504);  arg26_1 = view_770 = permute_504 = None
        view_771 = torch.ops.aten.view.default(addmm_161, [8, 576, 2304]);  addmm_161 = None
        view_772 = torch.ops.aten.view.default(view_771, [8, 576, 3, 16, 48]);  view_771 = None
        permute_505 = torch.ops.aten.permute.default(view_772, [2, 0, 3, 1, 4]);  view_772 = None
        select_114 = torch.ops.aten.select.int(permute_505, 0, 0)
        mul_392 = torch.ops.aten.mul.Tensor(select_114, 0.14433756729740643);  select_114 = None
        select_115 = torch.ops.aten.select.int(permute_505, 0, 1)
        select_116 = torch.ops.aten.select.int(permute_505, 0, 2);  permute_505 = None
        permute_506 = torch.ops.aten.permute.default(select_115, [0, 1, 3, 2]);  select_115 = None
        expand_149 = torch.ops.aten.expand.default(mul_392, [8, 16, 576, 48]);  mul_392 = None
        clone_528 = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_773 = torch.ops.aten.view.default(clone_528, [128, 576, 48]);  clone_528 = None
        expand_150 = torch.ops.aten.expand.default(permute_506, [8, 16, 48, 576]);  permute_506 = None
        clone_529 = torch.ops.aten.clone.default(expand_150, memory_format = torch.contiguous_format);  expand_150 = None
        view_774 = torch.ops.aten.view.default(clone_529, [128, 48, 576]);  clone_529 = None
        bmm_74 = torch.ops.aten.bmm.default(view_773, view_774);  view_773 = view_774 = None
        view_775 = torch.ops.aten.view.default(bmm_74, [8, 16, 576, 576]);  bmm_74 = None
        permute_507 = torch.ops.aten.permute.default(view_775, [0, 2, 3, 1]);  view_775 = None
        permute_508 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        clone_530 = torch.ops.aten.clone.default(permute_507, memory_format = torch.contiguous_format);  permute_507 = None
        view_776 = torch.ops.aten.view.default(clone_530, [2654208, 16]);  clone_530 = None
        mm_74 = torch.ops.aten.mm.default(view_776, permute_508);  view_776 = permute_508 = None
        view_777 = torch.ops.aten.view.default(mm_74, [8, 576, 576, 16]);  mm_74 = None
        add_353 = torch.ops.aten.add.Tensor(view_777, arg28_1);  view_777 = arg28_1 = None
        permute_509 = torch.ops.aten.permute.default(add_353, [0, 3, 1, 2]);  add_353 = None
        clone_531 = torch.ops.aten.clone.default(permute_509, memory_format = torch.contiguous_format);  permute_509 = None
        amax_37 = torch.ops.aten.amax.default(clone_531, [-1], True)
        sub_117 = torch.ops.aten.sub.Tensor(clone_531, amax_37);  clone_531 = amax_37 = None
        exp_37 = torch.ops.aten.exp.default(sub_117);  sub_117 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37 = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        permute_510 = torch.ops.aten.permute.default(div_37, [0, 2, 3, 1]);  div_37 = None
        permute_511 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        clone_532 = torch.ops.aten.clone.default(permute_510, memory_format = torch.contiguous_format);  permute_510 = None
        view_778 = torch.ops.aten.view.default(clone_532, [2654208, 16]);  clone_532 = None
        mm_75 = torch.ops.aten.mm.default(view_778, permute_511);  view_778 = permute_511 = None
        view_779 = torch.ops.aten.view.default(mm_75, [8, 576, 576, 16]);  mm_75 = None
        add_354 = torch.ops.aten.add.Tensor(view_779, arg30_1);  view_779 = arg30_1 = None
        permute_512 = torch.ops.aten.permute.default(add_354, [0, 3, 1, 2]);  add_354 = None
        expand_151 = torch.ops.aten.expand.default(permute_512, [8, 16, 576, 576]);  permute_512 = None
        clone_534 = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_780 = torch.ops.aten.view.default(clone_534, [128, 576, 576]);  clone_534 = None
        expand_152 = torch.ops.aten.expand.default(select_116, [8, 16, 576, 48]);  select_116 = None
        clone_535 = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
        view_781 = torch.ops.aten.view.default(clone_535, [128, 576, 48]);  clone_535 = None
        bmm_75 = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782 = torch.ops.aten.view.default(bmm_75, [8, 16, 576, 48]);  bmm_75 = None
        permute_513 = torch.ops.aten.permute.default(view_782, [0, 2, 1, 3]);  view_782 = None
        clone_536 = torch.ops.aten.clone.default(permute_513, memory_format = torch.contiguous_format);  permute_513 = None
        view_783 = torch.ops.aten.view.default(clone_536, [8, 576, 768]);  clone_536 = None
        view_784 = torch.ops.aten.view.default(view_783, [4608, 768]);  view_783 = None
        permute_514 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg32_1, view_784, permute_514);  arg32_1 = view_784 = permute_514 = None
        view_785 = torch.ops.aten.view.default(addmm_162, [8, 576, 768]);  addmm_162 = None
        mul_393 = torch.ops.aten.mul.Tensor(arg22_1, view_785);  arg22_1 = view_785 = None
        add_355 = torch.ops.aten.add.Tensor(add_350, mul_393);  add_350 = mul_393 = None
        clone_538 = torch.ops.aten.clone.default(add_355, memory_format = torch.contiguous_format)
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_538, [2], correction = 0, keepdim = True)
        getitem_168 = var_mean_80[0]
        getitem_169 = var_mean_80[1];  var_mean_80 = None
        add_356 = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
        sub_118 = torch.ops.aten.sub.Tensor(clone_538, getitem_169);  clone_538 = getitem_169 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_118, rsqrt_80);  sub_118 = rsqrt_80 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, arg34_1);  mul_394 = arg34_1 = None
        add_357 = torch.ops.aten.add.Tensor(mul_395, arg35_1);  mul_395 = arg35_1 = None
        view_786 = torch.ops.aten.view.default(add_357, [4608, 768]);  add_357 = None
        permute_515 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg37_1, view_786, permute_515);  arg37_1 = view_786 = permute_515 = None
        view_787 = torch.ops.aten.view.default(addmm_163, [8, 576, 3072]);  addmm_163 = None
        mul_396 = torch.ops.aten.mul.Tensor(view_787, 0.5)
        mul_397 = torch.ops.aten.mul.Tensor(view_787, 0.7071067811865476);  view_787 = None
        erf_39 = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_358 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_396, add_358);  mul_396 = add_358 = None
        view_788 = torch.ops.aten.view.default(mul_398, [4608, 3072]);  mul_398 = None
        permute_516 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg39_1, view_788, permute_516);  arg39_1 = view_788 = permute_516 = None
        view_789 = torch.ops.aten.view.default(addmm_164, [8, 576, 768]);  addmm_164 = None
        mul_399 = torch.ops.aten.mul.Tensor(arg33_1, view_789);  arg33_1 = view_789 = None
        add_359 = torch.ops.aten.add.Tensor(add_355, mul_399);  add_355 = mul_399 = None
        clone_541 = torch.ops.aten.clone.default(add_359, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_541, [2], correction = 0, keepdim = True)
        getitem_170 = var_mean_81[0]
        getitem_171 = var_mean_81[1];  var_mean_81 = None
        add_360 = torch.ops.aten.add.Tensor(getitem_170, 1e-06);  getitem_170 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_360);  add_360 = None
        sub_119 = torch.ops.aten.sub.Tensor(clone_541, getitem_171);  clone_541 = getitem_171 = None
        mul_400 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_81);  sub_119 = rsqrt_81 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_400, arg41_1);  mul_400 = arg41_1 = None
        add_361 = torch.ops.aten.add.Tensor(mul_401, arg42_1);  mul_401 = arg42_1 = None
        view_790 = torch.ops.aten.view.default(add_361, [4608, 768]);  add_361 = None
        permute_517 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg44_1, view_790, permute_517);  arg44_1 = view_790 = permute_517 = None
        view_791 = torch.ops.aten.view.default(addmm_165, [8, 576, 2304]);  addmm_165 = None
        view_792 = torch.ops.aten.view.default(view_791, [8, 576, 3, 16, 48]);  view_791 = None
        permute_518 = torch.ops.aten.permute.default(view_792, [2, 0, 3, 1, 4]);  view_792 = None
        select_117 = torch.ops.aten.select.int(permute_518, 0, 0)
        mul_402 = torch.ops.aten.mul.Tensor(select_117, 0.14433756729740643);  select_117 = None
        select_118 = torch.ops.aten.select.int(permute_518, 0, 1)
        select_119 = torch.ops.aten.select.int(permute_518, 0, 2);  permute_518 = None
        permute_519 = torch.ops.aten.permute.default(select_118, [0, 1, 3, 2]);  select_118 = None
        expand_153 = torch.ops.aten.expand.default(mul_402, [8, 16, 576, 48]);  mul_402 = None
        clone_542 = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_793 = torch.ops.aten.view.default(clone_542, [128, 576, 48]);  clone_542 = None
        expand_154 = torch.ops.aten.expand.default(permute_519, [8, 16, 48, 576]);  permute_519 = None
        clone_543 = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_794 = torch.ops.aten.view.default(clone_543, [128, 48, 576]);  clone_543 = None
        bmm_76 = torch.ops.aten.bmm.default(view_793, view_794);  view_793 = view_794 = None
        view_795 = torch.ops.aten.view.default(bmm_76, [8, 16, 576, 576]);  bmm_76 = None
        permute_520 = torch.ops.aten.permute.default(view_795, [0, 2, 3, 1]);  view_795 = None
        permute_521 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        clone_544 = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_796 = torch.ops.aten.view.default(clone_544, [2654208, 16]);  clone_544 = None
        mm_76 = torch.ops.aten.mm.default(view_796, permute_521);  view_796 = permute_521 = None
        view_797 = torch.ops.aten.view.default(mm_76, [8, 576, 576, 16]);  mm_76 = None
        add_362 = torch.ops.aten.add.Tensor(view_797, arg46_1);  view_797 = arg46_1 = None
        permute_522 = torch.ops.aten.permute.default(add_362, [0, 3, 1, 2]);  add_362 = None
        clone_545 = torch.ops.aten.clone.default(permute_522, memory_format = torch.contiguous_format);  permute_522 = None
        amax_38 = torch.ops.aten.amax.default(clone_545, [-1], True)
        sub_120 = torch.ops.aten.sub.Tensor(clone_545, amax_38);  clone_545 = amax_38 = None
        exp_38 = torch.ops.aten.exp.default(sub_120);  sub_120 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38 = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        permute_523 = torch.ops.aten.permute.default(div_38, [0, 2, 3, 1]);  div_38 = None
        permute_524 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        clone_546 = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_798 = torch.ops.aten.view.default(clone_546, [2654208, 16]);  clone_546 = None
        mm_77 = torch.ops.aten.mm.default(view_798, permute_524);  view_798 = permute_524 = None
        view_799 = torch.ops.aten.view.default(mm_77, [8, 576, 576, 16]);  mm_77 = None
        add_363 = torch.ops.aten.add.Tensor(view_799, arg48_1);  view_799 = arg48_1 = None
        permute_525 = torch.ops.aten.permute.default(add_363, [0, 3, 1, 2]);  add_363 = None
        expand_155 = torch.ops.aten.expand.default(permute_525, [8, 16, 576, 576]);  permute_525 = None
        clone_548 = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_800 = torch.ops.aten.view.default(clone_548, [128, 576, 576]);  clone_548 = None
        expand_156 = torch.ops.aten.expand.default(select_119, [8, 16, 576, 48]);  select_119 = None
        clone_549 = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_801 = torch.ops.aten.view.default(clone_549, [128, 576, 48]);  clone_549 = None
        bmm_77 = torch.ops.aten.bmm.default(view_800, view_801);  view_800 = view_801 = None
        view_802 = torch.ops.aten.view.default(bmm_77, [8, 16, 576, 48]);  bmm_77 = None
        permute_526 = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None
        clone_550 = torch.ops.aten.clone.default(permute_526, memory_format = torch.contiguous_format);  permute_526 = None
        view_803 = torch.ops.aten.view.default(clone_550, [8, 576, 768]);  clone_550 = None
        view_804 = torch.ops.aten.view.default(view_803, [4608, 768]);  view_803 = None
        permute_527 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg50_1, view_804, permute_527);  arg50_1 = view_804 = permute_527 = None
        view_805 = torch.ops.aten.view.default(addmm_166, [8, 576, 768]);  addmm_166 = None
        mul_403 = torch.ops.aten.mul.Tensor(arg40_1, view_805);  arg40_1 = view_805 = None
        add_364 = torch.ops.aten.add.Tensor(add_359, mul_403);  add_359 = mul_403 = None
        clone_552 = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_552, [2], correction = 0, keepdim = True)
        getitem_172 = var_mean_82[0]
        getitem_173 = var_mean_82[1];  var_mean_82 = None
        add_365 = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_365);  add_365 = None
        sub_121 = torch.ops.aten.sub.Tensor(clone_552, getitem_173);  clone_552 = getitem_173 = None
        mul_404 = torch.ops.aten.mul.Tensor(sub_121, rsqrt_82);  sub_121 = rsqrt_82 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_404, arg52_1);  mul_404 = arg52_1 = None
        add_366 = torch.ops.aten.add.Tensor(mul_405, arg53_1);  mul_405 = arg53_1 = None
        view_806 = torch.ops.aten.view.default(add_366, [4608, 768]);  add_366 = None
        permute_528 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg55_1, view_806, permute_528);  arg55_1 = view_806 = permute_528 = None
        view_807 = torch.ops.aten.view.default(addmm_167, [8, 576, 3072]);  addmm_167 = None
        mul_406 = torch.ops.aten.mul.Tensor(view_807, 0.5)
        mul_407 = torch.ops.aten.mul.Tensor(view_807, 0.7071067811865476);  view_807 = None
        erf_40 = torch.ops.aten.erf.default(mul_407);  mul_407 = None
        add_367 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_408 = torch.ops.aten.mul.Tensor(mul_406, add_367);  mul_406 = add_367 = None
        view_808 = torch.ops.aten.view.default(mul_408, [4608, 3072]);  mul_408 = None
        permute_529 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg57_1, view_808, permute_529);  arg57_1 = view_808 = permute_529 = None
        view_809 = torch.ops.aten.view.default(addmm_168, [8, 576, 768]);  addmm_168 = None
        mul_409 = torch.ops.aten.mul.Tensor(arg51_1, view_809);  arg51_1 = view_809 = None
        add_368 = torch.ops.aten.add.Tensor(add_364, mul_409);  add_364 = mul_409 = None
        clone_555 = torch.ops.aten.clone.default(add_368, memory_format = torch.contiguous_format)
        var_mean_83 = torch.ops.aten.var_mean.correction(clone_555, [2], correction = 0, keepdim = True)
        getitem_174 = var_mean_83[0]
        getitem_175 = var_mean_83[1];  var_mean_83 = None
        add_369 = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
        sub_122 = torch.ops.aten.sub.Tensor(clone_555, getitem_175);  clone_555 = getitem_175 = None
        mul_410 = torch.ops.aten.mul.Tensor(sub_122, rsqrt_83);  sub_122 = rsqrt_83 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_410, arg59_1);  mul_410 = arg59_1 = None
        add_370 = torch.ops.aten.add.Tensor(mul_411, arg60_1);  mul_411 = arg60_1 = None
        view_810 = torch.ops.aten.view.default(add_370, [4608, 768]);  add_370 = None
        permute_530 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg62_1, view_810, permute_530);  arg62_1 = view_810 = permute_530 = None
        view_811 = torch.ops.aten.view.default(addmm_169, [8, 576, 2304]);  addmm_169 = None
        view_812 = torch.ops.aten.view.default(view_811, [8, 576, 3, 16, 48]);  view_811 = None
        permute_531 = torch.ops.aten.permute.default(view_812, [2, 0, 3, 1, 4]);  view_812 = None
        select_120 = torch.ops.aten.select.int(permute_531, 0, 0)
        mul_412 = torch.ops.aten.mul.Tensor(select_120, 0.14433756729740643);  select_120 = None
        select_121 = torch.ops.aten.select.int(permute_531, 0, 1)
        select_122 = torch.ops.aten.select.int(permute_531, 0, 2);  permute_531 = None
        permute_532 = torch.ops.aten.permute.default(select_121, [0, 1, 3, 2]);  select_121 = None
        expand_157 = torch.ops.aten.expand.default(mul_412, [8, 16, 576, 48]);  mul_412 = None
        clone_556 = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_813 = torch.ops.aten.view.default(clone_556, [128, 576, 48]);  clone_556 = None
        expand_158 = torch.ops.aten.expand.default(permute_532, [8, 16, 48, 576]);  permute_532 = None
        clone_557 = torch.ops.aten.clone.default(expand_158, memory_format = torch.contiguous_format);  expand_158 = None
        view_814 = torch.ops.aten.view.default(clone_557, [128, 48, 576]);  clone_557 = None
        bmm_78 = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
        view_815 = torch.ops.aten.view.default(bmm_78, [8, 16, 576, 576]);  bmm_78 = None
        permute_533 = torch.ops.aten.permute.default(view_815, [0, 2, 3, 1]);  view_815 = None
        permute_534 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        clone_558 = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_816 = torch.ops.aten.view.default(clone_558, [2654208, 16]);  clone_558 = None
        mm_78 = torch.ops.aten.mm.default(view_816, permute_534);  view_816 = permute_534 = None
        view_817 = torch.ops.aten.view.default(mm_78, [8, 576, 576, 16]);  mm_78 = None
        add_371 = torch.ops.aten.add.Tensor(view_817, arg64_1);  view_817 = arg64_1 = None
        permute_535 = torch.ops.aten.permute.default(add_371, [0, 3, 1, 2]);  add_371 = None
        clone_559 = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
        amax_39 = torch.ops.aten.amax.default(clone_559, [-1], True)
        sub_123 = torch.ops.aten.sub.Tensor(clone_559, amax_39);  clone_559 = amax_39 = None
        exp_39 = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39 = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        permute_536 = torch.ops.aten.permute.default(div_39, [0, 2, 3, 1]);  div_39 = None
        permute_537 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        clone_560 = torch.ops.aten.clone.default(permute_536, memory_format = torch.contiguous_format);  permute_536 = None
        view_818 = torch.ops.aten.view.default(clone_560, [2654208, 16]);  clone_560 = None
        mm_79 = torch.ops.aten.mm.default(view_818, permute_537);  view_818 = permute_537 = None
        view_819 = torch.ops.aten.view.default(mm_79, [8, 576, 576, 16]);  mm_79 = None
        add_372 = torch.ops.aten.add.Tensor(view_819, arg66_1);  view_819 = arg66_1 = None
        permute_538 = torch.ops.aten.permute.default(add_372, [0, 3, 1, 2]);  add_372 = None
        expand_159 = torch.ops.aten.expand.default(permute_538, [8, 16, 576, 576]);  permute_538 = None
        clone_562 = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_820 = torch.ops.aten.view.default(clone_562, [128, 576, 576]);  clone_562 = None
        expand_160 = torch.ops.aten.expand.default(select_122, [8, 16, 576, 48]);  select_122 = None
        clone_563 = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
        view_821 = torch.ops.aten.view.default(clone_563, [128, 576, 48]);  clone_563 = None
        bmm_79 = torch.ops.aten.bmm.default(view_820, view_821);  view_820 = view_821 = None
        view_822 = torch.ops.aten.view.default(bmm_79, [8, 16, 576, 48]);  bmm_79 = None
        permute_539 = torch.ops.aten.permute.default(view_822, [0, 2, 1, 3]);  view_822 = None
        clone_564 = torch.ops.aten.clone.default(permute_539, memory_format = torch.contiguous_format);  permute_539 = None
        view_823 = torch.ops.aten.view.default(clone_564, [8, 576, 768]);  clone_564 = None
        view_824 = torch.ops.aten.view.default(view_823, [4608, 768]);  view_823 = None
        permute_540 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg68_1, view_824, permute_540);  arg68_1 = view_824 = permute_540 = None
        view_825 = torch.ops.aten.view.default(addmm_170, [8, 576, 768]);  addmm_170 = None
        mul_413 = torch.ops.aten.mul.Tensor(arg58_1, view_825);  arg58_1 = view_825 = None
        add_373 = torch.ops.aten.add.Tensor(add_368, mul_413);  add_368 = mul_413 = None
        clone_566 = torch.ops.aten.clone.default(add_373, memory_format = torch.contiguous_format)
        var_mean_84 = torch.ops.aten.var_mean.correction(clone_566, [2], correction = 0, keepdim = True)
        getitem_176 = var_mean_84[0]
        getitem_177 = var_mean_84[1];  var_mean_84 = None
        add_374 = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_374);  add_374 = None
        sub_124 = torch.ops.aten.sub.Tensor(clone_566, getitem_177);  clone_566 = getitem_177 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_124, rsqrt_84);  sub_124 = rsqrt_84 = None
        mul_415 = torch.ops.aten.mul.Tensor(mul_414, arg70_1);  mul_414 = arg70_1 = None
        add_375 = torch.ops.aten.add.Tensor(mul_415, arg71_1);  mul_415 = arg71_1 = None
        view_826 = torch.ops.aten.view.default(add_375, [4608, 768]);  add_375 = None
        permute_541 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg73_1, view_826, permute_541);  arg73_1 = view_826 = permute_541 = None
        view_827 = torch.ops.aten.view.default(addmm_171, [8, 576, 3072]);  addmm_171 = None
        mul_416 = torch.ops.aten.mul.Tensor(view_827, 0.5)
        mul_417 = torch.ops.aten.mul.Tensor(view_827, 0.7071067811865476);  view_827 = None
        erf_41 = torch.ops.aten.erf.default(mul_417);  mul_417 = None
        add_376 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_418 = torch.ops.aten.mul.Tensor(mul_416, add_376);  mul_416 = add_376 = None
        view_828 = torch.ops.aten.view.default(mul_418, [4608, 3072]);  mul_418 = None
        permute_542 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_172 = torch.ops.aten.addmm.default(arg75_1, view_828, permute_542);  arg75_1 = view_828 = permute_542 = None
        view_829 = torch.ops.aten.view.default(addmm_172, [8, 576, 768]);  addmm_172 = None
        mul_419 = torch.ops.aten.mul.Tensor(arg69_1, view_829);  arg69_1 = view_829 = None
        add_377 = torch.ops.aten.add.Tensor(add_373, mul_419);  add_373 = mul_419 = None
        clone_569 = torch.ops.aten.clone.default(add_377, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_569, [2], correction = 0, keepdim = True)
        getitem_178 = var_mean_85[0]
        getitem_179 = var_mean_85[1];  var_mean_85 = None
        add_378 = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
        sub_125 = torch.ops.aten.sub.Tensor(clone_569, getitem_179);  clone_569 = getitem_179 = None
        mul_420 = torch.ops.aten.mul.Tensor(sub_125, rsqrt_85);  sub_125 = rsqrt_85 = None
        mul_421 = torch.ops.aten.mul.Tensor(mul_420, arg77_1);  mul_420 = arg77_1 = None
        add_379 = torch.ops.aten.add.Tensor(mul_421, arg78_1);  mul_421 = arg78_1 = None
        view_830 = torch.ops.aten.view.default(add_379, [4608, 768]);  add_379 = None
        permute_543 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_173 = torch.ops.aten.addmm.default(arg80_1, view_830, permute_543);  arg80_1 = view_830 = permute_543 = None
        view_831 = torch.ops.aten.view.default(addmm_173, [8, 576, 2304]);  addmm_173 = None
        view_832 = torch.ops.aten.view.default(view_831, [8, 576, 3, 16, 48]);  view_831 = None
        permute_544 = torch.ops.aten.permute.default(view_832, [2, 0, 3, 1, 4]);  view_832 = None
        select_123 = torch.ops.aten.select.int(permute_544, 0, 0)
        mul_422 = torch.ops.aten.mul.Tensor(select_123, 0.14433756729740643);  select_123 = None
        select_124 = torch.ops.aten.select.int(permute_544, 0, 1)
        select_125 = torch.ops.aten.select.int(permute_544, 0, 2);  permute_544 = None
        permute_545 = torch.ops.aten.permute.default(select_124, [0, 1, 3, 2]);  select_124 = None
        expand_161 = torch.ops.aten.expand.default(mul_422, [8, 16, 576, 48]);  mul_422 = None
        clone_570 = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_833 = torch.ops.aten.view.default(clone_570, [128, 576, 48]);  clone_570 = None
        expand_162 = torch.ops.aten.expand.default(permute_545, [8, 16, 48, 576]);  permute_545 = None
        clone_571 = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_834 = torch.ops.aten.view.default(clone_571, [128, 48, 576]);  clone_571 = None
        bmm_80 = torch.ops.aten.bmm.default(view_833, view_834);  view_833 = view_834 = None
        view_835 = torch.ops.aten.view.default(bmm_80, [8, 16, 576, 576]);  bmm_80 = None
        permute_546 = torch.ops.aten.permute.default(view_835, [0, 2, 3, 1]);  view_835 = None
        permute_547 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        clone_572 = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format);  permute_546 = None
        view_836 = torch.ops.aten.view.default(clone_572, [2654208, 16]);  clone_572 = None
        mm_80 = torch.ops.aten.mm.default(view_836, permute_547);  view_836 = permute_547 = None
        view_837 = torch.ops.aten.view.default(mm_80, [8, 576, 576, 16]);  mm_80 = None
        add_380 = torch.ops.aten.add.Tensor(view_837, arg82_1);  view_837 = arg82_1 = None
        permute_548 = torch.ops.aten.permute.default(add_380, [0, 3, 1, 2]);  add_380 = None
        clone_573 = torch.ops.aten.clone.default(permute_548, memory_format = torch.contiguous_format);  permute_548 = None
        amax_40 = torch.ops.aten.amax.default(clone_573, [-1], True)
        sub_126 = torch.ops.aten.sub.Tensor(clone_573, amax_40);  clone_573 = amax_40 = None
        exp_40 = torch.ops.aten.exp.default(sub_126);  sub_126 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40 = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        permute_549 = torch.ops.aten.permute.default(div_40, [0, 2, 3, 1]);  div_40 = None
        permute_550 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        clone_574 = torch.ops.aten.clone.default(permute_549, memory_format = torch.contiguous_format);  permute_549 = None
        view_838 = torch.ops.aten.view.default(clone_574, [2654208, 16]);  clone_574 = None
        mm_81 = torch.ops.aten.mm.default(view_838, permute_550);  view_838 = permute_550 = None
        view_839 = torch.ops.aten.view.default(mm_81, [8, 576, 576, 16]);  mm_81 = None
        add_381 = torch.ops.aten.add.Tensor(view_839, arg84_1);  view_839 = arg84_1 = None
        permute_551 = torch.ops.aten.permute.default(add_381, [0, 3, 1, 2]);  add_381 = None
        expand_163 = torch.ops.aten.expand.default(permute_551, [8, 16, 576, 576]);  permute_551 = None
        clone_576 = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_840 = torch.ops.aten.view.default(clone_576, [128, 576, 576]);  clone_576 = None
        expand_164 = torch.ops.aten.expand.default(select_125, [8, 16, 576, 48]);  select_125 = None
        clone_577 = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_841 = torch.ops.aten.view.default(clone_577, [128, 576, 48]);  clone_577 = None
        bmm_81 = torch.ops.aten.bmm.default(view_840, view_841);  view_840 = view_841 = None
        view_842 = torch.ops.aten.view.default(bmm_81, [8, 16, 576, 48]);  bmm_81 = None
        permute_552 = torch.ops.aten.permute.default(view_842, [0, 2, 1, 3]);  view_842 = None
        clone_578 = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
        view_843 = torch.ops.aten.view.default(clone_578, [8, 576, 768]);  clone_578 = None
        view_844 = torch.ops.aten.view.default(view_843, [4608, 768]);  view_843 = None
        permute_553 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_174 = torch.ops.aten.addmm.default(arg86_1, view_844, permute_553);  arg86_1 = view_844 = permute_553 = None
        view_845 = torch.ops.aten.view.default(addmm_174, [8, 576, 768]);  addmm_174 = None
        mul_423 = torch.ops.aten.mul.Tensor(arg76_1, view_845);  arg76_1 = view_845 = None
        add_382 = torch.ops.aten.add.Tensor(add_377, mul_423);  add_377 = mul_423 = None
        clone_580 = torch.ops.aten.clone.default(add_382, memory_format = torch.contiguous_format)
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_580, [2], correction = 0, keepdim = True)
        getitem_180 = var_mean_86[0]
        getitem_181 = var_mean_86[1];  var_mean_86 = None
        add_383 = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
        sub_127 = torch.ops.aten.sub.Tensor(clone_580, getitem_181);  clone_580 = getitem_181 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_127, rsqrt_86);  sub_127 = rsqrt_86 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, arg88_1);  mul_424 = arg88_1 = None
        add_384 = torch.ops.aten.add.Tensor(mul_425, arg89_1);  mul_425 = arg89_1 = None
        view_846 = torch.ops.aten.view.default(add_384, [4608, 768]);  add_384 = None
        permute_554 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_175 = torch.ops.aten.addmm.default(arg91_1, view_846, permute_554);  arg91_1 = view_846 = permute_554 = None
        view_847 = torch.ops.aten.view.default(addmm_175, [8, 576, 3072]);  addmm_175 = None
        mul_426 = torch.ops.aten.mul.Tensor(view_847, 0.5)
        mul_427 = torch.ops.aten.mul.Tensor(view_847, 0.7071067811865476);  view_847 = None
        erf_42 = torch.ops.aten.erf.default(mul_427);  mul_427 = None
        add_385 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_426, add_385);  mul_426 = add_385 = None
        view_848 = torch.ops.aten.view.default(mul_428, [4608, 3072]);  mul_428 = None
        permute_555 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_176 = torch.ops.aten.addmm.default(arg93_1, view_848, permute_555);  arg93_1 = view_848 = permute_555 = None
        view_849 = torch.ops.aten.view.default(addmm_176, [8, 576, 768]);  addmm_176 = None
        mul_429 = torch.ops.aten.mul.Tensor(arg87_1, view_849);  arg87_1 = view_849 = None
        add_386 = torch.ops.aten.add.Tensor(add_382, mul_429);  add_382 = mul_429 = None
        clone_583 = torch.ops.aten.clone.default(add_386, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_583, [2], correction = 0, keepdim = True)
        getitem_182 = var_mean_87[0]
        getitem_183 = var_mean_87[1];  var_mean_87 = None
        add_387 = torch.ops.aten.add.Tensor(getitem_182, 1e-06);  getitem_182 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_387);  add_387 = None
        sub_128 = torch.ops.aten.sub.Tensor(clone_583, getitem_183);  clone_583 = getitem_183 = None
        mul_430 = torch.ops.aten.mul.Tensor(sub_128, rsqrt_87);  sub_128 = rsqrt_87 = None
        mul_431 = torch.ops.aten.mul.Tensor(mul_430, arg95_1);  mul_430 = arg95_1 = None
        add_388 = torch.ops.aten.add.Tensor(mul_431, arg96_1);  mul_431 = arg96_1 = None
        view_850 = torch.ops.aten.view.default(add_388, [4608, 768]);  add_388 = None
        permute_556 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_177 = torch.ops.aten.addmm.default(arg98_1, view_850, permute_556);  arg98_1 = view_850 = permute_556 = None
        view_851 = torch.ops.aten.view.default(addmm_177, [8, 576, 2304]);  addmm_177 = None
        view_852 = torch.ops.aten.view.default(view_851, [8, 576, 3, 16, 48]);  view_851 = None
        permute_557 = torch.ops.aten.permute.default(view_852, [2, 0, 3, 1, 4]);  view_852 = None
        select_126 = torch.ops.aten.select.int(permute_557, 0, 0)
        mul_432 = torch.ops.aten.mul.Tensor(select_126, 0.14433756729740643);  select_126 = None
        select_127 = torch.ops.aten.select.int(permute_557, 0, 1)
        select_128 = torch.ops.aten.select.int(permute_557, 0, 2);  permute_557 = None
        permute_558 = torch.ops.aten.permute.default(select_127, [0, 1, 3, 2]);  select_127 = None
        expand_165 = torch.ops.aten.expand.default(mul_432, [8, 16, 576, 48]);  mul_432 = None
        clone_584 = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_853 = torch.ops.aten.view.default(clone_584, [128, 576, 48]);  clone_584 = None
        expand_166 = torch.ops.aten.expand.default(permute_558, [8, 16, 48, 576]);  permute_558 = None
        clone_585 = torch.ops.aten.clone.default(expand_166, memory_format = torch.contiguous_format);  expand_166 = None
        view_854 = torch.ops.aten.view.default(clone_585, [128, 48, 576]);  clone_585 = None
        bmm_82 = torch.ops.aten.bmm.default(view_853, view_854);  view_853 = view_854 = None
        view_855 = torch.ops.aten.view.default(bmm_82, [8, 16, 576, 576]);  bmm_82 = None
        permute_559 = torch.ops.aten.permute.default(view_855, [0, 2, 3, 1]);  view_855 = None
        permute_560 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        clone_586 = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
        view_856 = torch.ops.aten.view.default(clone_586, [2654208, 16]);  clone_586 = None
        mm_82 = torch.ops.aten.mm.default(view_856, permute_560);  view_856 = permute_560 = None
        view_857 = torch.ops.aten.view.default(mm_82, [8, 576, 576, 16]);  mm_82 = None
        add_389 = torch.ops.aten.add.Tensor(view_857, arg100_1);  view_857 = arg100_1 = None
        permute_561 = torch.ops.aten.permute.default(add_389, [0, 3, 1, 2]);  add_389 = None
        clone_587 = torch.ops.aten.clone.default(permute_561, memory_format = torch.contiguous_format);  permute_561 = None
        amax_41 = torch.ops.aten.amax.default(clone_587, [-1], True)
        sub_129 = torch.ops.aten.sub.Tensor(clone_587, amax_41);  clone_587 = amax_41 = None
        exp_41 = torch.ops.aten.exp.default(sub_129);  sub_129 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41 = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        permute_562 = torch.ops.aten.permute.default(div_41, [0, 2, 3, 1]);  div_41 = None
        permute_563 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        clone_588 = torch.ops.aten.clone.default(permute_562, memory_format = torch.contiguous_format);  permute_562 = None
        view_858 = torch.ops.aten.view.default(clone_588, [2654208, 16]);  clone_588 = None
        mm_83 = torch.ops.aten.mm.default(view_858, permute_563);  view_858 = permute_563 = None
        view_859 = torch.ops.aten.view.default(mm_83, [8, 576, 576, 16]);  mm_83 = None
        add_390 = torch.ops.aten.add.Tensor(view_859, arg102_1);  view_859 = arg102_1 = None
        permute_564 = torch.ops.aten.permute.default(add_390, [0, 3, 1, 2]);  add_390 = None
        expand_167 = torch.ops.aten.expand.default(permute_564, [8, 16, 576, 576]);  permute_564 = None
        clone_590 = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_860 = torch.ops.aten.view.default(clone_590, [128, 576, 576]);  clone_590 = None
        expand_168 = torch.ops.aten.expand.default(select_128, [8, 16, 576, 48]);  select_128 = None
        clone_591 = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
        view_861 = torch.ops.aten.view.default(clone_591, [128, 576, 48]);  clone_591 = None
        bmm_83 = torch.ops.aten.bmm.default(view_860, view_861);  view_860 = view_861 = None
        view_862 = torch.ops.aten.view.default(bmm_83, [8, 16, 576, 48]);  bmm_83 = None
        permute_565 = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None
        clone_592 = torch.ops.aten.clone.default(permute_565, memory_format = torch.contiguous_format);  permute_565 = None
        view_863 = torch.ops.aten.view.default(clone_592, [8, 576, 768]);  clone_592 = None
        view_864 = torch.ops.aten.view.default(view_863, [4608, 768]);  view_863 = None
        permute_566 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_178 = torch.ops.aten.addmm.default(arg104_1, view_864, permute_566);  arg104_1 = view_864 = permute_566 = None
        view_865 = torch.ops.aten.view.default(addmm_178, [8, 576, 768]);  addmm_178 = None
        mul_433 = torch.ops.aten.mul.Tensor(arg94_1, view_865);  arg94_1 = view_865 = None
        add_391 = torch.ops.aten.add.Tensor(add_386, mul_433);  add_386 = mul_433 = None
        clone_594 = torch.ops.aten.clone.default(add_391, memory_format = torch.contiguous_format)
        var_mean_88 = torch.ops.aten.var_mean.correction(clone_594, [2], correction = 0, keepdim = True)
        getitem_184 = var_mean_88[0]
        getitem_185 = var_mean_88[1];  var_mean_88 = None
        add_392 = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_392);  add_392 = None
        sub_130 = torch.ops.aten.sub.Tensor(clone_594, getitem_185);  clone_594 = getitem_185 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_130, rsqrt_88);  sub_130 = rsqrt_88 = None
        mul_435 = torch.ops.aten.mul.Tensor(mul_434, arg106_1);  mul_434 = arg106_1 = None
        add_393 = torch.ops.aten.add.Tensor(mul_435, arg107_1);  mul_435 = arg107_1 = None
        view_866 = torch.ops.aten.view.default(add_393, [4608, 768]);  add_393 = None
        permute_567 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_179 = torch.ops.aten.addmm.default(arg109_1, view_866, permute_567);  arg109_1 = view_866 = permute_567 = None
        view_867 = torch.ops.aten.view.default(addmm_179, [8, 576, 3072]);  addmm_179 = None
        mul_436 = torch.ops.aten.mul.Tensor(view_867, 0.5)
        mul_437 = torch.ops.aten.mul.Tensor(view_867, 0.7071067811865476);  view_867 = None
        erf_43 = torch.ops.aten.erf.default(mul_437);  mul_437 = None
        add_394 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_438 = torch.ops.aten.mul.Tensor(mul_436, add_394);  mul_436 = add_394 = None
        view_868 = torch.ops.aten.view.default(mul_438, [4608, 3072]);  mul_438 = None
        permute_568 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_180 = torch.ops.aten.addmm.default(arg111_1, view_868, permute_568);  arg111_1 = view_868 = permute_568 = None
        view_869 = torch.ops.aten.view.default(addmm_180, [8, 576, 768]);  addmm_180 = None
        mul_439 = torch.ops.aten.mul.Tensor(arg105_1, view_869);  arg105_1 = view_869 = None
        add_395 = torch.ops.aten.add.Tensor(add_391, mul_439);  add_391 = mul_439 = None
        clone_597 = torch.ops.aten.clone.default(add_395, memory_format = torch.contiguous_format)
        var_mean_89 = torch.ops.aten.var_mean.correction(clone_597, [2], correction = 0, keepdim = True)
        getitem_186 = var_mean_89[0]
        getitem_187 = var_mean_89[1];  var_mean_89 = None
        add_396 = torch.ops.aten.add.Tensor(getitem_186, 1e-06);  getitem_186 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_396);  add_396 = None
        sub_131 = torch.ops.aten.sub.Tensor(clone_597, getitem_187);  clone_597 = getitem_187 = None
        mul_440 = torch.ops.aten.mul.Tensor(sub_131, rsqrt_89);  sub_131 = rsqrt_89 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_440, arg113_1);  mul_440 = arg113_1 = None
        add_397 = torch.ops.aten.add.Tensor(mul_441, arg114_1);  mul_441 = arg114_1 = None
        view_870 = torch.ops.aten.view.default(add_397, [4608, 768]);  add_397 = None
        permute_569 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg116_1, view_870, permute_569);  arg116_1 = view_870 = permute_569 = None
        view_871 = torch.ops.aten.view.default(addmm_181, [8, 576, 2304]);  addmm_181 = None
        view_872 = torch.ops.aten.view.default(view_871, [8, 576, 3, 16, 48]);  view_871 = None
        permute_570 = torch.ops.aten.permute.default(view_872, [2, 0, 3, 1, 4]);  view_872 = None
        select_129 = torch.ops.aten.select.int(permute_570, 0, 0)
        mul_442 = torch.ops.aten.mul.Tensor(select_129, 0.14433756729740643);  select_129 = None
        select_130 = torch.ops.aten.select.int(permute_570, 0, 1)
        select_131 = torch.ops.aten.select.int(permute_570, 0, 2);  permute_570 = None
        permute_571 = torch.ops.aten.permute.default(select_130, [0, 1, 3, 2]);  select_130 = None
        expand_169 = torch.ops.aten.expand.default(mul_442, [8, 16, 576, 48]);  mul_442 = None
        clone_598 = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_873 = torch.ops.aten.view.default(clone_598, [128, 576, 48]);  clone_598 = None
        expand_170 = torch.ops.aten.expand.default(permute_571, [8, 16, 48, 576]);  permute_571 = None
        clone_599 = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_874 = torch.ops.aten.view.default(clone_599, [128, 48, 576]);  clone_599 = None
        bmm_84 = torch.ops.aten.bmm.default(view_873, view_874);  view_873 = view_874 = None
        view_875 = torch.ops.aten.view.default(bmm_84, [8, 16, 576, 576]);  bmm_84 = None
        permute_572 = torch.ops.aten.permute.default(view_875, [0, 2, 3, 1]);  view_875 = None
        permute_573 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        clone_600 = torch.ops.aten.clone.default(permute_572, memory_format = torch.contiguous_format);  permute_572 = None
        view_876 = torch.ops.aten.view.default(clone_600, [2654208, 16]);  clone_600 = None
        mm_84 = torch.ops.aten.mm.default(view_876, permute_573);  view_876 = permute_573 = None
        view_877 = torch.ops.aten.view.default(mm_84, [8, 576, 576, 16]);  mm_84 = None
        add_398 = torch.ops.aten.add.Tensor(view_877, arg118_1);  view_877 = arg118_1 = None
        permute_574 = torch.ops.aten.permute.default(add_398, [0, 3, 1, 2]);  add_398 = None
        clone_601 = torch.ops.aten.clone.default(permute_574, memory_format = torch.contiguous_format);  permute_574 = None
        amax_42 = torch.ops.aten.amax.default(clone_601, [-1], True)
        sub_132 = torch.ops.aten.sub.Tensor(clone_601, amax_42);  clone_601 = amax_42 = None
        exp_42 = torch.ops.aten.exp.default(sub_132);  sub_132 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42 = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        permute_575 = torch.ops.aten.permute.default(div_42, [0, 2, 3, 1]);  div_42 = None
        permute_576 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        clone_602 = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
        view_878 = torch.ops.aten.view.default(clone_602, [2654208, 16]);  clone_602 = None
        mm_85 = torch.ops.aten.mm.default(view_878, permute_576);  view_878 = permute_576 = None
        view_879 = torch.ops.aten.view.default(mm_85, [8, 576, 576, 16]);  mm_85 = None
        add_399 = torch.ops.aten.add.Tensor(view_879, arg120_1);  view_879 = arg120_1 = None
        permute_577 = torch.ops.aten.permute.default(add_399, [0, 3, 1, 2]);  add_399 = None
        expand_171 = torch.ops.aten.expand.default(permute_577, [8, 16, 576, 576]);  permute_577 = None
        clone_604 = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_880 = torch.ops.aten.view.default(clone_604, [128, 576, 576]);  clone_604 = None
        expand_172 = torch.ops.aten.expand.default(select_131, [8, 16, 576, 48]);  select_131 = None
        clone_605 = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_881 = torch.ops.aten.view.default(clone_605, [128, 576, 48]);  clone_605 = None
        bmm_85 = torch.ops.aten.bmm.default(view_880, view_881);  view_880 = view_881 = None
        view_882 = torch.ops.aten.view.default(bmm_85, [8, 16, 576, 48]);  bmm_85 = None
        permute_578 = torch.ops.aten.permute.default(view_882, [0, 2, 1, 3]);  view_882 = None
        clone_606 = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
        view_883 = torch.ops.aten.view.default(clone_606, [8, 576, 768]);  clone_606 = None
        view_884 = torch.ops.aten.view.default(view_883, [4608, 768]);  view_883 = None
        permute_579 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg122_1, view_884, permute_579);  arg122_1 = view_884 = permute_579 = None
        view_885 = torch.ops.aten.view.default(addmm_182, [8, 576, 768]);  addmm_182 = None
        mul_443 = torch.ops.aten.mul.Tensor(arg112_1, view_885);  arg112_1 = view_885 = None
        add_400 = torch.ops.aten.add.Tensor(add_395, mul_443);  add_395 = mul_443 = None
        clone_608 = torch.ops.aten.clone.default(add_400, memory_format = torch.contiguous_format)
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_608, [2], correction = 0, keepdim = True)
        getitem_188 = var_mean_90[0]
        getitem_189 = var_mean_90[1];  var_mean_90 = None
        add_401 = torch.ops.aten.add.Tensor(getitem_188, 1e-06);  getitem_188 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
        sub_133 = torch.ops.aten.sub.Tensor(clone_608, getitem_189);  clone_608 = getitem_189 = None
        mul_444 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_90);  sub_133 = rsqrt_90 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, arg124_1);  mul_444 = arg124_1 = None
        add_402 = torch.ops.aten.add.Tensor(mul_445, arg125_1);  mul_445 = arg125_1 = None
        view_886 = torch.ops.aten.view.default(add_402, [4608, 768]);  add_402 = None
        permute_580 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg127_1, view_886, permute_580);  arg127_1 = view_886 = permute_580 = None
        view_887 = torch.ops.aten.view.default(addmm_183, [8, 576, 3072]);  addmm_183 = None
        mul_446 = torch.ops.aten.mul.Tensor(view_887, 0.5)
        mul_447 = torch.ops.aten.mul.Tensor(view_887, 0.7071067811865476);  view_887 = None
        erf_44 = torch.ops.aten.erf.default(mul_447);  mul_447 = None
        add_403 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_446, add_403);  mul_446 = add_403 = None
        view_888 = torch.ops.aten.view.default(mul_448, [4608, 3072]);  mul_448 = None
        permute_581 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_184 = torch.ops.aten.addmm.default(arg129_1, view_888, permute_581);  arg129_1 = view_888 = permute_581 = None
        view_889 = torch.ops.aten.view.default(addmm_184, [8, 576, 768]);  addmm_184 = None
        mul_449 = torch.ops.aten.mul.Tensor(arg123_1, view_889);  arg123_1 = view_889 = None
        add_404 = torch.ops.aten.add.Tensor(add_400, mul_449);  add_400 = mul_449 = None
        clone_611 = torch.ops.aten.clone.default(add_404, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_611, [2], correction = 0, keepdim = True)
        getitem_190 = var_mean_91[0]
        getitem_191 = var_mean_91[1];  var_mean_91 = None
        add_405 = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
        sub_134 = torch.ops.aten.sub.Tensor(clone_611, getitem_191);  clone_611 = getitem_191 = None
        mul_450 = torch.ops.aten.mul.Tensor(sub_134, rsqrt_91);  sub_134 = rsqrt_91 = None
        mul_451 = torch.ops.aten.mul.Tensor(mul_450, arg131_1);  mul_450 = arg131_1 = None
        add_406 = torch.ops.aten.add.Tensor(mul_451, arg132_1);  mul_451 = arg132_1 = None
        view_890 = torch.ops.aten.view.default(add_406, [4608, 768]);  add_406 = None
        permute_582 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg134_1, view_890, permute_582);  arg134_1 = view_890 = permute_582 = None
        view_891 = torch.ops.aten.view.default(addmm_185, [8, 576, 2304]);  addmm_185 = None
        view_892 = torch.ops.aten.view.default(view_891, [8, 576, 3, 16, 48]);  view_891 = None
        permute_583 = torch.ops.aten.permute.default(view_892, [2, 0, 3, 1, 4]);  view_892 = None
        select_132 = torch.ops.aten.select.int(permute_583, 0, 0)
        mul_452 = torch.ops.aten.mul.Tensor(select_132, 0.14433756729740643);  select_132 = None
        select_133 = torch.ops.aten.select.int(permute_583, 0, 1)
        select_134 = torch.ops.aten.select.int(permute_583, 0, 2);  permute_583 = None
        permute_584 = torch.ops.aten.permute.default(select_133, [0, 1, 3, 2]);  select_133 = None
        expand_173 = torch.ops.aten.expand.default(mul_452, [8, 16, 576, 48]);  mul_452 = None
        clone_612 = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_893 = torch.ops.aten.view.default(clone_612, [128, 576, 48]);  clone_612 = None
        expand_174 = torch.ops.aten.expand.default(permute_584, [8, 16, 48, 576]);  permute_584 = None
        clone_613 = torch.ops.aten.clone.default(expand_174, memory_format = torch.contiguous_format);  expand_174 = None
        view_894 = torch.ops.aten.view.default(clone_613, [128, 48, 576]);  clone_613 = None
        bmm_86 = torch.ops.aten.bmm.default(view_893, view_894);  view_893 = view_894 = None
        view_895 = torch.ops.aten.view.default(bmm_86, [8, 16, 576, 576]);  bmm_86 = None
        permute_585 = torch.ops.aten.permute.default(view_895, [0, 2, 3, 1]);  view_895 = None
        permute_586 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        clone_614 = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
        view_896 = torch.ops.aten.view.default(clone_614, [2654208, 16]);  clone_614 = None
        mm_86 = torch.ops.aten.mm.default(view_896, permute_586);  view_896 = permute_586 = None
        view_897 = torch.ops.aten.view.default(mm_86, [8, 576, 576, 16]);  mm_86 = None
        add_407 = torch.ops.aten.add.Tensor(view_897, arg136_1);  view_897 = arg136_1 = None
        permute_587 = torch.ops.aten.permute.default(add_407, [0, 3, 1, 2]);  add_407 = None
        clone_615 = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        amax_43 = torch.ops.aten.amax.default(clone_615, [-1], True)
        sub_135 = torch.ops.aten.sub.Tensor(clone_615, amax_43);  clone_615 = amax_43 = None
        exp_43 = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43 = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        permute_588 = torch.ops.aten.permute.default(div_43, [0, 2, 3, 1]);  div_43 = None
        permute_589 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        clone_616 = torch.ops.aten.clone.default(permute_588, memory_format = torch.contiguous_format);  permute_588 = None
        view_898 = torch.ops.aten.view.default(clone_616, [2654208, 16]);  clone_616 = None
        mm_87 = torch.ops.aten.mm.default(view_898, permute_589);  view_898 = permute_589 = None
        view_899 = torch.ops.aten.view.default(mm_87, [8, 576, 576, 16]);  mm_87 = None
        add_408 = torch.ops.aten.add.Tensor(view_899, arg138_1);  view_899 = arg138_1 = None
        permute_590 = torch.ops.aten.permute.default(add_408, [0, 3, 1, 2]);  add_408 = None
        expand_175 = torch.ops.aten.expand.default(permute_590, [8, 16, 576, 576]);  permute_590 = None
        clone_618 = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_900 = torch.ops.aten.view.default(clone_618, [128, 576, 576]);  clone_618 = None
        expand_176 = torch.ops.aten.expand.default(select_134, [8, 16, 576, 48]);  select_134 = None
        clone_619 = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
        view_901 = torch.ops.aten.view.default(clone_619, [128, 576, 48]);  clone_619 = None
        bmm_87 = torch.ops.aten.bmm.default(view_900, view_901);  view_900 = view_901 = None
        view_902 = torch.ops.aten.view.default(bmm_87, [8, 16, 576, 48]);  bmm_87 = None
        permute_591 = torch.ops.aten.permute.default(view_902, [0, 2, 1, 3]);  view_902 = None
        clone_620 = torch.ops.aten.clone.default(permute_591, memory_format = torch.contiguous_format);  permute_591 = None
        view_903 = torch.ops.aten.view.default(clone_620, [8, 576, 768]);  clone_620 = None
        view_904 = torch.ops.aten.view.default(view_903, [4608, 768]);  view_903 = None
        permute_592 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_186 = torch.ops.aten.addmm.default(arg140_1, view_904, permute_592);  arg140_1 = view_904 = permute_592 = None
        view_905 = torch.ops.aten.view.default(addmm_186, [8, 576, 768]);  addmm_186 = None
        mul_453 = torch.ops.aten.mul.Tensor(arg130_1, view_905);  arg130_1 = view_905 = None
        add_409 = torch.ops.aten.add.Tensor(add_404, mul_453);  add_404 = mul_453 = None
        clone_622 = torch.ops.aten.clone.default(add_409, memory_format = torch.contiguous_format)
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_622, [2], correction = 0, keepdim = True)
        getitem_192 = var_mean_92[0]
        getitem_193 = var_mean_92[1];  var_mean_92 = None
        add_410 = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
        sub_136 = torch.ops.aten.sub.Tensor(clone_622, getitem_193);  clone_622 = getitem_193 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_136, rsqrt_92);  sub_136 = rsqrt_92 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, arg142_1);  mul_454 = arg142_1 = None
        add_411 = torch.ops.aten.add.Tensor(mul_455, arg143_1);  mul_455 = arg143_1 = None
        view_906 = torch.ops.aten.view.default(add_411, [4608, 768]);  add_411 = None
        permute_593 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg145_1, view_906, permute_593);  arg145_1 = view_906 = permute_593 = None
        view_907 = torch.ops.aten.view.default(addmm_187, [8, 576, 3072]);  addmm_187 = None
        mul_456 = torch.ops.aten.mul.Tensor(view_907, 0.5)
        mul_457 = torch.ops.aten.mul.Tensor(view_907, 0.7071067811865476);  view_907 = None
        erf_45 = torch.ops.aten.erf.default(mul_457);  mul_457 = None
        add_412 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_456, add_412);  mul_456 = add_412 = None
        view_908 = torch.ops.aten.view.default(mul_458, [4608, 3072]);  mul_458 = None
        permute_594 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg147_1, view_908, permute_594);  arg147_1 = view_908 = permute_594 = None
        view_909 = torch.ops.aten.view.default(addmm_188, [8, 576, 768]);  addmm_188 = None
        mul_459 = torch.ops.aten.mul.Tensor(arg141_1, view_909);  arg141_1 = view_909 = None
        add_413 = torch.ops.aten.add.Tensor(add_409, mul_459);  add_409 = mul_459 = None
        clone_625 = torch.ops.aten.clone.default(add_413, memory_format = torch.contiguous_format)
        var_mean_93 = torch.ops.aten.var_mean.correction(clone_625, [2], correction = 0, keepdim = True)
        getitem_194 = var_mean_93[0]
        getitem_195 = var_mean_93[1];  var_mean_93 = None
        add_414 = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
        sub_137 = torch.ops.aten.sub.Tensor(clone_625, getitem_195);  clone_625 = getitem_195 = None
        mul_460 = torch.ops.aten.mul.Tensor(sub_137, rsqrt_93);  sub_137 = rsqrt_93 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, arg149_1);  mul_460 = arg149_1 = None
        add_415 = torch.ops.aten.add.Tensor(mul_461, arg150_1);  mul_461 = arg150_1 = None
        view_910 = torch.ops.aten.view.default(add_415, [4608, 768]);  add_415 = None
        permute_595 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg152_1, view_910, permute_595);  arg152_1 = view_910 = permute_595 = None
        view_911 = torch.ops.aten.view.default(addmm_189, [8, 576, 2304]);  addmm_189 = None
        view_912 = torch.ops.aten.view.default(view_911, [8, 576, 3, 16, 48]);  view_911 = None
        permute_596 = torch.ops.aten.permute.default(view_912, [2, 0, 3, 1, 4]);  view_912 = None
        select_135 = torch.ops.aten.select.int(permute_596, 0, 0)
        mul_462 = torch.ops.aten.mul.Tensor(select_135, 0.14433756729740643);  select_135 = None
        select_136 = torch.ops.aten.select.int(permute_596, 0, 1)
        select_137 = torch.ops.aten.select.int(permute_596, 0, 2);  permute_596 = None
        permute_597 = torch.ops.aten.permute.default(select_136, [0, 1, 3, 2]);  select_136 = None
        expand_177 = torch.ops.aten.expand.default(mul_462, [8, 16, 576, 48]);  mul_462 = None
        clone_626 = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_913 = torch.ops.aten.view.default(clone_626, [128, 576, 48]);  clone_626 = None
        expand_178 = torch.ops.aten.expand.default(permute_597, [8, 16, 48, 576]);  permute_597 = None
        clone_627 = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_914 = torch.ops.aten.view.default(clone_627, [128, 48, 576]);  clone_627 = None
        bmm_88 = torch.ops.aten.bmm.default(view_913, view_914);  view_913 = view_914 = None
        view_915 = torch.ops.aten.view.default(bmm_88, [8, 16, 576, 576]);  bmm_88 = None
        permute_598 = torch.ops.aten.permute.default(view_915, [0, 2, 3, 1]);  view_915 = None
        permute_599 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        clone_628 = torch.ops.aten.clone.default(permute_598, memory_format = torch.contiguous_format);  permute_598 = None
        view_916 = torch.ops.aten.view.default(clone_628, [2654208, 16]);  clone_628 = None
        mm_88 = torch.ops.aten.mm.default(view_916, permute_599);  view_916 = permute_599 = None
        view_917 = torch.ops.aten.view.default(mm_88, [8, 576, 576, 16]);  mm_88 = None
        add_416 = torch.ops.aten.add.Tensor(view_917, arg154_1);  view_917 = arg154_1 = None
        permute_600 = torch.ops.aten.permute.default(add_416, [0, 3, 1, 2]);  add_416 = None
        clone_629 = torch.ops.aten.clone.default(permute_600, memory_format = torch.contiguous_format);  permute_600 = None
        amax_44 = torch.ops.aten.amax.default(clone_629, [-1], True)
        sub_138 = torch.ops.aten.sub.Tensor(clone_629, amax_44);  clone_629 = amax_44 = None
        exp_44 = torch.ops.aten.exp.default(sub_138);  sub_138 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44 = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        permute_601 = torch.ops.aten.permute.default(div_44, [0, 2, 3, 1]);  div_44 = None
        permute_602 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        clone_630 = torch.ops.aten.clone.default(permute_601, memory_format = torch.contiguous_format);  permute_601 = None
        view_918 = torch.ops.aten.view.default(clone_630, [2654208, 16]);  clone_630 = None
        mm_89 = torch.ops.aten.mm.default(view_918, permute_602);  view_918 = permute_602 = None
        view_919 = torch.ops.aten.view.default(mm_89, [8, 576, 576, 16]);  mm_89 = None
        add_417 = torch.ops.aten.add.Tensor(view_919, arg156_1);  view_919 = arg156_1 = None
        permute_603 = torch.ops.aten.permute.default(add_417, [0, 3, 1, 2]);  add_417 = None
        expand_179 = torch.ops.aten.expand.default(permute_603, [8, 16, 576, 576]);  permute_603 = None
        clone_632 = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_920 = torch.ops.aten.view.default(clone_632, [128, 576, 576]);  clone_632 = None
        expand_180 = torch.ops.aten.expand.default(select_137, [8, 16, 576, 48]);  select_137 = None
        clone_633 = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_921 = torch.ops.aten.view.default(clone_633, [128, 576, 48]);  clone_633 = None
        bmm_89 = torch.ops.aten.bmm.default(view_920, view_921);  view_920 = view_921 = None
        view_922 = torch.ops.aten.view.default(bmm_89, [8, 16, 576, 48]);  bmm_89 = None
        permute_604 = torch.ops.aten.permute.default(view_922, [0, 2, 1, 3]);  view_922 = None
        clone_634 = torch.ops.aten.clone.default(permute_604, memory_format = torch.contiguous_format);  permute_604 = None
        view_923 = torch.ops.aten.view.default(clone_634, [8, 576, 768]);  clone_634 = None
        view_924 = torch.ops.aten.view.default(view_923, [4608, 768]);  view_923 = None
        permute_605 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg158_1, view_924, permute_605);  arg158_1 = view_924 = permute_605 = None
        view_925 = torch.ops.aten.view.default(addmm_190, [8, 576, 768]);  addmm_190 = None
        mul_463 = torch.ops.aten.mul.Tensor(arg148_1, view_925);  arg148_1 = view_925 = None
        add_418 = torch.ops.aten.add.Tensor(add_413, mul_463);  add_413 = mul_463 = None
        clone_636 = torch.ops.aten.clone.default(add_418, memory_format = torch.contiguous_format)
        var_mean_94 = torch.ops.aten.var_mean.correction(clone_636, [2], correction = 0, keepdim = True)
        getitem_196 = var_mean_94[0]
        getitem_197 = var_mean_94[1];  var_mean_94 = None
        add_419 = torch.ops.aten.add.Tensor(getitem_196, 1e-06);  getitem_196 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
        sub_139 = torch.ops.aten.sub.Tensor(clone_636, getitem_197);  clone_636 = getitem_197 = None
        mul_464 = torch.ops.aten.mul.Tensor(sub_139, rsqrt_94);  sub_139 = rsqrt_94 = None
        mul_465 = torch.ops.aten.mul.Tensor(mul_464, arg160_1);  mul_464 = arg160_1 = None
        add_420 = torch.ops.aten.add.Tensor(mul_465, arg161_1);  mul_465 = arg161_1 = None
        view_926 = torch.ops.aten.view.default(add_420, [4608, 768]);  add_420 = None
        permute_606 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_191 = torch.ops.aten.addmm.default(arg163_1, view_926, permute_606);  arg163_1 = view_926 = permute_606 = None
        view_927 = torch.ops.aten.view.default(addmm_191, [8, 576, 3072]);  addmm_191 = None
        mul_466 = torch.ops.aten.mul.Tensor(view_927, 0.5)
        mul_467 = torch.ops.aten.mul.Tensor(view_927, 0.7071067811865476);  view_927 = None
        erf_46 = torch.ops.aten.erf.default(mul_467);  mul_467 = None
        add_421 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_468 = torch.ops.aten.mul.Tensor(mul_466, add_421);  mul_466 = add_421 = None
        view_928 = torch.ops.aten.view.default(mul_468, [4608, 3072]);  mul_468 = None
        permute_607 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_192 = torch.ops.aten.addmm.default(arg165_1, view_928, permute_607);  arg165_1 = view_928 = permute_607 = None
        view_929 = torch.ops.aten.view.default(addmm_192, [8, 576, 768]);  addmm_192 = None
        mul_469 = torch.ops.aten.mul.Tensor(arg159_1, view_929);  arg159_1 = view_929 = None
        add_422 = torch.ops.aten.add.Tensor(add_418, mul_469);  add_418 = mul_469 = None
        clone_639 = torch.ops.aten.clone.default(add_422, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_639, [2], correction = 0, keepdim = True)
        getitem_198 = var_mean_95[0]
        getitem_199 = var_mean_95[1];  var_mean_95 = None
        add_423 = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_423);  add_423 = None
        sub_140 = torch.ops.aten.sub.Tensor(clone_639, getitem_199);  clone_639 = getitem_199 = None
        mul_470 = torch.ops.aten.mul.Tensor(sub_140, rsqrt_95);  sub_140 = rsqrt_95 = None
        mul_471 = torch.ops.aten.mul.Tensor(mul_470, arg167_1);  mul_470 = arg167_1 = None
        add_424 = torch.ops.aten.add.Tensor(mul_471, arg168_1);  mul_471 = arg168_1 = None
        view_930 = torch.ops.aten.view.default(add_424, [4608, 768]);  add_424 = None
        permute_608 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_193 = torch.ops.aten.addmm.default(arg170_1, view_930, permute_608);  arg170_1 = view_930 = permute_608 = None
        view_931 = torch.ops.aten.view.default(addmm_193, [8, 576, 2304]);  addmm_193 = None
        view_932 = torch.ops.aten.view.default(view_931, [8, 576, 3, 16, 48]);  view_931 = None
        permute_609 = torch.ops.aten.permute.default(view_932, [2, 0, 3, 1, 4]);  view_932 = None
        select_138 = torch.ops.aten.select.int(permute_609, 0, 0)
        mul_472 = torch.ops.aten.mul.Tensor(select_138, 0.14433756729740643);  select_138 = None
        select_139 = torch.ops.aten.select.int(permute_609, 0, 1)
        select_140 = torch.ops.aten.select.int(permute_609, 0, 2);  permute_609 = None
        permute_610 = torch.ops.aten.permute.default(select_139, [0, 1, 3, 2]);  select_139 = None
        expand_181 = torch.ops.aten.expand.default(mul_472, [8, 16, 576, 48]);  mul_472 = None
        clone_640 = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_933 = torch.ops.aten.view.default(clone_640, [128, 576, 48]);  clone_640 = None
        expand_182 = torch.ops.aten.expand.default(permute_610, [8, 16, 48, 576]);  permute_610 = None
        clone_641 = torch.ops.aten.clone.default(expand_182, memory_format = torch.contiguous_format);  expand_182 = None
        view_934 = torch.ops.aten.view.default(clone_641, [128, 48, 576]);  clone_641 = None
        bmm_90 = torch.ops.aten.bmm.default(view_933, view_934);  view_933 = view_934 = None
        view_935 = torch.ops.aten.view.default(bmm_90, [8, 16, 576, 576]);  bmm_90 = None
        permute_611 = torch.ops.aten.permute.default(view_935, [0, 2, 3, 1]);  view_935 = None
        permute_612 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        clone_642 = torch.ops.aten.clone.default(permute_611, memory_format = torch.contiguous_format);  permute_611 = None
        view_936 = torch.ops.aten.view.default(clone_642, [2654208, 16]);  clone_642 = None
        mm_90 = torch.ops.aten.mm.default(view_936, permute_612);  view_936 = permute_612 = None
        view_937 = torch.ops.aten.view.default(mm_90, [8, 576, 576, 16]);  mm_90 = None
        add_425 = torch.ops.aten.add.Tensor(view_937, arg172_1);  view_937 = arg172_1 = None
        permute_613 = torch.ops.aten.permute.default(add_425, [0, 3, 1, 2]);  add_425 = None
        clone_643 = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
        amax_45 = torch.ops.aten.amax.default(clone_643, [-1], True)
        sub_141 = torch.ops.aten.sub.Tensor(clone_643, amax_45);  clone_643 = amax_45 = None
        exp_45 = torch.ops.aten.exp.default(sub_141);  sub_141 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45 = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        permute_614 = torch.ops.aten.permute.default(div_45, [0, 2, 3, 1]);  div_45 = None
        permute_615 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        clone_644 = torch.ops.aten.clone.default(permute_614, memory_format = torch.contiguous_format);  permute_614 = None
        view_938 = torch.ops.aten.view.default(clone_644, [2654208, 16]);  clone_644 = None
        mm_91 = torch.ops.aten.mm.default(view_938, permute_615);  view_938 = permute_615 = None
        view_939 = torch.ops.aten.view.default(mm_91, [8, 576, 576, 16]);  mm_91 = None
        add_426 = torch.ops.aten.add.Tensor(view_939, arg174_1);  view_939 = arg174_1 = None
        permute_616 = torch.ops.aten.permute.default(add_426, [0, 3, 1, 2]);  add_426 = None
        expand_183 = torch.ops.aten.expand.default(permute_616, [8, 16, 576, 576]);  permute_616 = None
        clone_646 = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_940 = torch.ops.aten.view.default(clone_646, [128, 576, 576]);  clone_646 = None
        expand_184 = torch.ops.aten.expand.default(select_140, [8, 16, 576, 48]);  select_140 = None
        clone_647 = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
        view_941 = torch.ops.aten.view.default(clone_647, [128, 576, 48]);  clone_647 = None
        bmm_91 = torch.ops.aten.bmm.default(view_940, view_941);  view_940 = view_941 = None
        view_942 = torch.ops.aten.view.default(bmm_91, [8, 16, 576, 48]);  bmm_91 = None
        permute_617 = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None
        clone_648 = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
        view_943 = torch.ops.aten.view.default(clone_648, [8, 576, 768]);  clone_648 = None
        view_944 = torch.ops.aten.view.default(view_943, [4608, 768]);  view_943 = None
        permute_618 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_194 = torch.ops.aten.addmm.default(arg176_1, view_944, permute_618);  arg176_1 = view_944 = permute_618 = None
        view_945 = torch.ops.aten.view.default(addmm_194, [8, 576, 768]);  addmm_194 = None
        mul_473 = torch.ops.aten.mul.Tensor(arg166_1, view_945);  arg166_1 = view_945 = None
        add_427 = torch.ops.aten.add.Tensor(add_422, mul_473);  add_422 = mul_473 = None
        clone_650 = torch.ops.aten.clone.default(add_427, memory_format = torch.contiguous_format)
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_650, [2], correction = 0, keepdim = True)
        getitem_200 = var_mean_96[0]
        getitem_201 = var_mean_96[1];  var_mean_96 = None
        add_428 = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_428);  add_428 = None
        sub_142 = torch.ops.aten.sub.Tensor(clone_650, getitem_201);  clone_650 = getitem_201 = None
        mul_474 = torch.ops.aten.mul.Tensor(sub_142, rsqrt_96);  sub_142 = rsqrt_96 = None
        mul_475 = torch.ops.aten.mul.Tensor(mul_474, arg178_1);  mul_474 = arg178_1 = None
        add_429 = torch.ops.aten.add.Tensor(mul_475, arg179_1);  mul_475 = arg179_1 = None
        view_946 = torch.ops.aten.view.default(add_429, [4608, 768]);  add_429 = None
        permute_619 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_195 = torch.ops.aten.addmm.default(arg181_1, view_946, permute_619);  arg181_1 = view_946 = permute_619 = None
        view_947 = torch.ops.aten.view.default(addmm_195, [8, 576, 3072]);  addmm_195 = None
        mul_476 = torch.ops.aten.mul.Tensor(view_947, 0.5)
        mul_477 = torch.ops.aten.mul.Tensor(view_947, 0.7071067811865476);  view_947 = None
        erf_47 = torch.ops.aten.erf.default(mul_477);  mul_477 = None
        add_430 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_478 = torch.ops.aten.mul.Tensor(mul_476, add_430);  mul_476 = add_430 = None
        view_948 = torch.ops.aten.view.default(mul_478, [4608, 3072]);  mul_478 = None
        permute_620 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_196 = torch.ops.aten.addmm.default(arg183_1, view_948, permute_620);  arg183_1 = view_948 = permute_620 = None
        view_949 = torch.ops.aten.view.default(addmm_196, [8, 576, 768]);  addmm_196 = None
        mul_479 = torch.ops.aten.mul.Tensor(arg177_1, view_949);  arg177_1 = view_949 = None
        add_431 = torch.ops.aten.add.Tensor(add_427, mul_479);  add_427 = mul_479 = None
        clone_653 = torch.ops.aten.clone.default(add_431, memory_format = torch.contiguous_format)
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_653, [2], correction = 0, keepdim = True)
        getitem_202 = var_mean_97[0]
        getitem_203 = var_mean_97[1];  var_mean_97 = None
        add_432 = torch.ops.aten.add.Tensor(getitem_202, 1e-06);  getitem_202 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_432);  add_432 = None
        sub_143 = torch.ops.aten.sub.Tensor(clone_653, getitem_203);  clone_653 = getitem_203 = None
        mul_480 = torch.ops.aten.mul.Tensor(sub_143, rsqrt_97);  sub_143 = rsqrt_97 = None
        mul_481 = torch.ops.aten.mul.Tensor(mul_480, arg185_1);  mul_480 = arg185_1 = None
        add_433 = torch.ops.aten.add.Tensor(mul_481, arg186_1);  mul_481 = arg186_1 = None
        view_950 = torch.ops.aten.view.default(add_433, [4608, 768]);  add_433 = None
        permute_621 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_197 = torch.ops.aten.addmm.default(arg188_1, view_950, permute_621);  arg188_1 = view_950 = permute_621 = None
        view_951 = torch.ops.aten.view.default(addmm_197, [8, 576, 2304]);  addmm_197 = None
        view_952 = torch.ops.aten.view.default(view_951, [8, 576, 3, 16, 48]);  view_951 = None
        permute_622 = torch.ops.aten.permute.default(view_952, [2, 0, 3, 1, 4]);  view_952 = None
        select_141 = torch.ops.aten.select.int(permute_622, 0, 0)
        mul_482 = torch.ops.aten.mul.Tensor(select_141, 0.14433756729740643);  select_141 = None
        select_142 = torch.ops.aten.select.int(permute_622, 0, 1)
        select_143 = torch.ops.aten.select.int(permute_622, 0, 2);  permute_622 = None
        permute_623 = torch.ops.aten.permute.default(select_142, [0, 1, 3, 2]);  select_142 = None
        expand_185 = torch.ops.aten.expand.default(mul_482, [8, 16, 576, 48]);  mul_482 = None
        clone_654 = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_953 = torch.ops.aten.view.default(clone_654, [128, 576, 48]);  clone_654 = None
        expand_186 = torch.ops.aten.expand.default(permute_623, [8, 16, 48, 576]);  permute_623 = None
        clone_655 = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_954 = torch.ops.aten.view.default(clone_655, [128, 48, 576]);  clone_655 = None
        bmm_92 = torch.ops.aten.bmm.default(view_953, view_954);  view_953 = view_954 = None
        view_955 = torch.ops.aten.view.default(bmm_92, [8, 16, 576, 576]);  bmm_92 = None
        permute_624 = torch.ops.aten.permute.default(view_955, [0, 2, 3, 1]);  view_955 = None
        permute_625 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        clone_656 = torch.ops.aten.clone.default(permute_624, memory_format = torch.contiguous_format);  permute_624 = None
        view_956 = torch.ops.aten.view.default(clone_656, [2654208, 16]);  clone_656 = None
        mm_92 = torch.ops.aten.mm.default(view_956, permute_625);  view_956 = permute_625 = None
        view_957 = torch.ops.aten.view.default(mm_92, [8, 576, 576, 16]);  mm_92 = None
        add_434 = torch.ops.aten.add.Tensor(view_957, arg190_1);  view_957 = arg190_1 = None
        permute_626 = torch.ops.aten.permute.default(add_434, [0, 3, 1, 2]);  add_434 = None
        clone_657 = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
        amax_46 = torch.ops.aten.amax.default(clone_657, [-1], True)
        sub_144 = torch.ops.aten.sub.Tensor(clone_657, amax_46);  clone_657 = amax_46 = None
        exp_46 = torch.ops.aten.exp.default(sub_144);  sub_144 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46 = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        permute_627 = torch.ops.aten.permute.default(div_46, [0, 2, 3, 1]);  div_46 = None
        permute_628 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        clone_658 = torch.ops.aten.clone.default(permute_627, memory_format = torch.contiguous_format);  permute_627 = None
        view_958 = torch.ops.aten.view.default(clone_658, [2654208, 16]);  clone_658 = None
        mm_93 = torch.ops.aten.mm.default(view_958, permute_628);  view_958 = permute_628 = None
        view_959 = torch.ops.aten.view.default(mm_93, [8, 576, 576, 16]);  mm_93 = None
        add_435 = torch.ops.aten.add.Tensor(view_959, arg192_1);  view_959 = arg192_1 = None
        permute_629 = torch.ops.aten.permute.default(add_435, [0, 3, 1, 2]);  add_435 = None
        expand_187 = torch.ops.aten.expand.default(permute_629, [8, 16, 576, 576]);  permute_629 = None
        clone_660 = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_960 = torch.ops.aten.view.default(clone_660, [128, 576, 576]);  clone_660 = None
        expand_188 = torch.ops.aten.expand.default(select_143, [8, 16, 576, 48]);  select_143 = None
        clone_661 = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_961 = torch.ops.aten.view.default(clone_661, [128, 576, 48]);  clone_661 = None
        bmm_93 = torch.ops.aten.bmm.default(view_960, view_961);  view_960 = view_961 = None
        view_962 = torch.ops.aten.view.default(bmm_93, [8, 16, 576, 48]);  bmm_93 = None
        permute_630 = torch.ops.aten.permute.default(view_962, [0, 2, 1, 3]);  view_962 = None
        clone_662 = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        view_963 = torch.ops.aten.view.default(clone_662, [8, 576, 768]);  clone_662 = None
        view_964 = torch.ops.aten.view.default(view_963, [4608, 768]);  view_963 = None
        permute_631 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_198 = torch.ops.aten.addmm.default(arg194_1, view_964, permute_631);  arg194_1 = view_964 = permute_631 = None
        view_965 = torch.ops.aten.view.default(addmm_198, [8, 576, 768]);  addmm_198 = None
        mul_483 = torch.ops.aten.mul.Tensor(arg184_1, view_965);  arg184_1 = view_965 = None
        add_436 = torch.ops.aten.add.Tensor(add_431, mul_483);  add_431 = mul_483 = None
        clone_664 = torch.ops.aten.clone.default(add_436, memory_format = torch.contiguous_format)
        var_mean_98 = torch.ops.aten.var_mean.correction(clone_664, [2], correction = 0, keepdim = True)
        getitem_204 = var_mean_98[0]
        getitem_205 = var_mean_98[1];  var_mean_98 = None
        add_437 = torch.ops.aten.add.Tensor(getitem_204, 1e-06);  getitem_204 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_437);  add_437 = None
        sub_145 = torch.ops.aten.sub.Tensor(clone_664, getitem_205);  clone_664 = getitem_205 = None
        mul_484 = torch.ops.aten.mul.Tensor(sub_145, rsqrt_98);  sub_145 = rsqrt_98 = None
        mul_485 = torch.ops.aten.mul.Tensor(mul_484, arg196_1);  mul_484 = arg196_1 = None
        add_438 = torch.ops.aten.add.Tensor(mul_485, arg197_1);  mul_485 = arg197_1 = None
        view_966 = torch.ops.aten.view.default(add_438, [4608, 768]);  add_438 = None
        permute_632 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_199 = torch.ops.aten.addmm.default(arg199_1, view_966, permute_632);  arg199_1 = view_966 = permute_632 = None
        view_967 = torch.ops.aten.view.default(addmm_199, [8, 576, 3072]);  addmm_199 = None
        mul_486 = torch.ops.aten.mul.Tensor(view_967, 0.5)
        mul_487 = torch.ops.aten.mul.Tensor(view_967, 0.7071067811865476);  view_967 = None
        erf_48 = torch.ops.aten.erf.default(mul_487);  mul_487 = None
        add_439 = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_488 = torch.ops.aten.mul.Tensor(mul_486, add_439);  mul_486 = add_439 = None
        view_968 = torch.ops.aten.view.default(mul_488, [4608, 3072]);  mul_488 = None
        permute_633 = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_200 = torch.ops.aten.addmm.default(arg201_1, view_968, permute_633);  arg201_1 = view_968 = permute_633 = None
        view_969 = torch.ops.aten.view.default(addmm_200, [8, 576, 768]);  addmm_200 = None
        mul_489 = torch.ops.aten.mul.Tensor(arg195_1, view_969);  arg195_1 = view_969 = None
        add_440 = torch.ops.aten.add.Tensor(add_436, mul_489);  add_436 = mul_489 = None
        clone_667 = torch.ops.aten.clone.default(add_440, memory_format = torch.contiguous_format)
        var_mean_99 = torch.ops.aten.var_mean.correction(clone_667, [2], correction = 0, keepdim = True)
        getitem_206 = var_mean_99[0]
        getitem_207 = var_mean_99[1];  var_mean_99 = None
        add_441 = torch.ops.aten.add.Tensor(getitem_206, 1e-06);  getitem_206 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_441);  add_441 = None
        sub_146 = torch.ops.aten.sub.Tensor(clone_667, getitem_207);  clone_667 = getitem_207 = None
        mul_490 = torch.ops.aten.mul.Tensor(sub_146, rsqrt_99);  sub_146 = rsqrt_99 = None
        mul_491 = torch.ops.aten.mul.Tensor(mul_490, arg203_1);  mul_490 = arg203_1 = None
        add_442 = torch.ops.aten.add.Tensor(mul_491, arg204_1);  mul_491 = arg204_1 = None
        view_970 = torch.ops.aten.view.default(add_442, [4608, 768]);  add_442 = None
        permute_634 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_201 = torch.ops.aten.addmm.default(arg206_1, view_970, permute_634);  arg206_1 = view_970 = permute_634 = None
        view_971 = torch.ops.aten.view.default(addmm_201, [8, 576, 2304]);  addmm_201 = None
        view_972 = torch.ops.aten.view.default(view_971, [8, 576, 3, 16, 48]);  view_971 = None
        permute_635 = torch.ops.aten.permute.default(view_972, [2, 0, 3, 1, 4]);  view_972 = None
        select_144 = torch.ops.aten.select.int(permute_635, 0, 0)
        mul_492 = torch.ops.aten.mul.Tensor(select_144, 0.14433756729740643);  select_144 = None
        select_145 = torch.ops.aten.select.int(permute_635, 0, 1)
        select_146 = torch.ops.aten.select.int(permute_635, 0, 2);  permute_635 = None
        permute_636 = torch.ops.aten.permute.default(select_145, [0, 1, 3, 2]);  select_145 = None
        expand_189 = torch.ops.aten.expand.default(mul_492, [8, 16, 576, 48]);  mul_492 = None
        clone_668 = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_973 = torch.ops.aten.view.default(clone_668, [128, 576, 48]);  clone_668 = None
        expand_190 = torch.ops.aten.expand.default(permute_636, [8, 16, 48, 576]);  permute_636 = None
        clone_669 = torch.ops.aten.clone.default(expand_190, memory_format = torch.contiguous_format);  expand_190 = None
        view_974 = torch.ops.aten.view.default(clone_669, [128, 48, 576]);  clone_669 = None
        bmm_94 = torch.ops.aten.bmm.default(view_973, view_974);  view_973 = view_974 = None
        view_975 = torch.ops.aten.view.default(bmm_94, [8, 16, 576, 576]);  bmm_94 = None
        permute_637 = torch.ops.aten.permute.default(view_975, [0, 2, 3, 1]);  view_975 = None
        permute_638 = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        clone_670 = torch.ops.aten.clone.default(permute_637, memory_format = torch.contiguous_format);  permute_637 = None
        view_976 = torch.ops.aten.view.default(clone_670, [2654208, 16]);  clone_670 = None
        mm_94 = torch.ops.aten.mm.default(view_976, permute_638);  view_976 = permute_638 = None
        view_977 = torch.ops.aten.view.default(mm_94, [8, 576, 576, 16]);  mm_94 = None
        add_443 = torch.ops.aten.add.Tensor(view_977, arg208_1);  view_977 = arg208_1 = None
        permute_639 = torch.ops.aten.permute.default(add_443, [0, 3, 1, 2]);  add_443 = None
        clone_671 = torch.ops.aten.clone.default(permute_639, memory_format = torch.contiguous_format);  permute_639 = None
        amax_47 = torch.ops.aten.amax.default(clone_671, [-1], True)
        sub_147 = torch.ops.aten.sub.Tensor(clone_671, amax_47);  clone_671 = amax_47 = None
        exp_47 = torch.ops.aten.exp.default(sub_147);  sub_147 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        permute_640 = torch.ops.aten.permute.default(div_47, [0, 2, 3, 1]);  div_47 = None
        permute_641 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        clone_672 = torch.ops.aten.clone.default(permute_640, memory_format = torch.contiguous_format);  permute_640 = None
        view_978 = torch.ops.aten.view.default(clone_672, [2654208, 16]);  clone_672 = None
        mm_95 = torch.ops.aten.mm.default(view_978, permute_641);  view_978 = permute_641 = None
        view_979 = torch.ops.aten.view.default(mm_95, [8, 576, 576, 16]);  mm_95 = None
        add_444 = torch.ops.aten.add.Tensor(view_979, arg210_1);  view_979 = arg210_1 = None
        permute_642 = torch.ops.aten.permute.default(add_444, [0, 3, 1, 2]);  add_444 = None
        expand_191 = torch.ops.aten.expand.default(permute_642, [8, 16, 576, 576]);  permute_642 = None
        clone_674 = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_980 = torch.ops.aten.view.default(clone_674, [128, 576, 576]);  clone_674 = None
        expand_192 = torch.ops.aten.expand.default(select_146, [8, 16, 576, 48]);  select_146 = None
        clone_675 = torch.ops.aten.clone.default(expand_192, memory_format = torch.contiguous_format);  expand_192 = None
        view_981 = torch.ops.aten.view.default(clone_675, [128, 576, 48]);  clone_675 = None
        bmm_95 = torch.ops.aten.bmm.default(view_980, view_981);  view_980 = view_981 = None
        view_982 = torch.ops.aten.view.default(bmm_95, [8, 16, 576, 48]);  bmm_95 = None
        permute_643 = torch.ops.aten.permute.default(view_982, [0, 2, 1, 3]);  view_982 = None
        clone_676 = torch.ops.aten.clone.default(permute_643, memory_format = torch.contiguous_format);  permute_643 = None
        view_983 = torch.ops.aten.view.default(clone_676, [8, 576, 768]);  clone_676 = None
        view_984 = torch.ops.aten.view.default(view_983, [4608, 768]);  view_983 = None
        permute_644 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_202 = torch.ops.aten.addmm.default(arg212_1, view_984, permute_644);  arg212_1 = view_984 = permute_644 = None
        view_985 = torch.ops.aten.view.default(addmm_202, [8, 576, 768]);  addmm_202 = None
        mul_493 = torch.ops.aten.mul.Tensor(arg202_1, view_985);  arg202_1 = view_985 = None
        add_445 = torch.ops.aten.add.Tensor(add_440, mul_493);  add_440 = mul_493 = None
        clone_678 = torch.ops.aten.clone.default(add_445, memory_format = torch.contiguous_format)
        var_mean_100 = torch.ops.aten.var_mean.correction(clone_678, [2], correction = 0, keepdim = True)
        getitem_208 = var_mean_100[0]
        getitem_209 = var_mean_100[1];  var_mean_100 = None
        add_446 = torch.ops.aten.add.Tensor(getitem_208, 1e-06);  getitem_208 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
        sub_148 = torch.ops.aten.sub.Tensor(clone_678, getitem_209);  clone_678 = getitem_209 = None
        mul_494 = torch.ops.aten.mul.Tensor(sub_148, rsqrt_100);  sub_148 = rsqrt_100 = None
        mul_495 = torch.ops.aten.mul.Tensor(mul_494, arg214_1);  mul_494 = arg214_1 = None
        add_447 = torch.ops.aten.add.Tensor(mul_495, arg215_1);  mul_495 = arg215_1 = None
        view_986 = torch.ops.aten.view.default(add_447, [4608, 768]);  add_447 = None
        permute_645 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_203 = torch.ops.aten.addmm.default(arg217_1, view_986, permute_645);  arg217_1 = view_986 = permute_645 = None
        view_987 = torch.ops.aten.view.default(addmm_203, [8, 576, 3072]);  addmm_203 = None
        mul_496 = torch.ops.aten.mul.Tensor(view_987, 0.5)
        mul_497 = torch.ops.aten.mul.Tensor(view_987, 0.7071067811865476);  view_987 = None
        erf_49 = torch.ops.aten.erf.default(mul_497);  mul_497 = None
        add_448 = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_498 = torch.ops.aten.mul.Tensor(mul_496, add_448);  mul_496 = add_448 = None
        view_988 = torch.ops.aten.view.default(mul_498, [4608, 3072]);  mul_498 = None
        permute_646 = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_204 = torch.ops.aten.addmm.default(arg219_1, view_988, permute_646);  arg219_1 = view_988 = permute_646 = None
        view_989 = torch.ops.aten.view.default(addmm_204, [8, 576, 768]);  addmm_204 = None
        mul_499 = torch.ops.aten.mul.Tensor(arg213_1, view_989);  arg213_1 = view_989 = None
        add_449 = torch.ops.aten.add.Tensor(add_445, mul_499);  add_445 = mul_499 = None
        clone_681 = torch.ops.aten.clone.default(add_449, memory_format = torch.contiguous_format)
        var_mean_101 = torch.ops.aten.var_mean.correction(clone_681, [2], correction = 0, keepdim = True)
        getitem_210 = var_mean_101[0]
        getitem_211 = var_mean_101[1];  var_mean_101 = None
        add_450 = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_450);  add_450 = None
        sub_149 = torch.ops.aten.sub.Tensor(clone_681, getitem_211);  clone_681 = getitem_211 = None
        mul_500 = torch.ops.aten.mul.Tensor(sub_149, rsqrt_101);  sub_149 = rsqrt_101 = None
        mul_501 = torch.ops.aten.mul.Tensor(mul_500, arg221_1);  mul_500 = arg221_1 = None
        add_451 = torch.ops.aten.add.Tensor(mul_501, arg222_1);  mul_501 = arg222_1 = None
        view_990 = torch.ops.aten.view.default(add_451, [4608, 768]);  add_451 = None
        permute_647 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_205 = torch.ops.aten.addmm.default(arg224_1, view_990, permute_647);  arg224_1 = view_990 = permute_647 = None
        view_991 = torch.ops.aten.view.default(addmm_205, [8, 576, 2304]);  addmm_205 = None
        view_992 = torch.ops.aten.view.default(view_991, [8, 576, 3, 16, 48]);  view_991 = None
        permute_648 = torch.ops.aten.permute.default(view_992, [2, 0, 3, 1, 4]);  view_992 = None
        select_147 = torch.ops.aten.select.int(permute_648, 0, 0)
        mul_502 = torch.ops.aten.mul.Tensor(select_147, 0.14433756729740643);  select_147 = None
        select_148 = torch.ops.aten.select.int(permute_648, 0, 1)
        select_149 = torch.ops.aten.select.int(permute_648, 0, 2);  permute_648 = None
        permute_649 = torch.ops.aten.permute.default(select_148, [0, 1, 3, 2]);  select_148 = None
        expand_193 = torch.ops.aten.expand.default(mul_502, [8, 16, 576, 48]);  mul_502 = None
        clone_682 = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_993 = torch.ops.aten.view.default(clone_682, [128, 576, 48]);  clone_682 = None
        expand_194 = torch.ops.aten.expand.default(permute_649, [8, 16, 48, 576]);  permute_649 = None
        clone_683 = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_994 = torch.ops.aten.view.default(clone_683, [128, 48, 576]);  clone_683 = None
        bmm_96 = torch.ops.aten.bmm.default(view_993, view_994);  view_993 = view_994 = None
        view_995 = torch.ops.aten.view.default(bmm_96, [8, 16, 576, 576]);  bmm_96 = None
        permute_650 = torch.ops.aten.permute.default(view_995, [0, 2, 3, 1]);  view_995 = None
        permute_651 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        clone_684 = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
        view_996 = torch.ops.aten.view.default(clone_684, [2654208, 16]);  clone_684 = None
        mm_96 = torch.ops.aten.mm.default(view_996, permute_651);  view_996 = permute_651 = None
        view_997 = torch.ops.aten.view.default(mm_96, [8, 576, 576, 16]);  mm_96 = None
        add_452 = torch.ops.aten.add.Tensor(view_997, arg226_1);  view_997 = arg226_1 = None
        permute_652 = torch.ops.aten.permute.default(add_452, [0, 3, 1, 2]);  add_452 = None
        clone_685 = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
        amax_48 = torch.ops.aten.amax.default(clone_685, [-1], True)
        sub_150 = torch.ops.aten.sub.Tensor(clone_685, amax_48);  clone_685 = amax_48 = None
        exp_48 = torch.ops.aten.exp.default(sub_150);  sub_150 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(exp_48, [-1], True)
        div_48 = torch.ops.aten.div.Tensor(exp_48, sum_49);  exp_48 = sum_49 = None
        permute_653 = torch.ops.aten.permute.default(div_48, [0, 2, 3, 1]);  div_48 = None
        permute_654 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        clone_686 = torch.ops.aten.clone.default(permute_653, memory_format = torch.contiguous_format);  permute_653 = None
        view_998 = torch.ops.aten.view.default(clone_686, [2654208, 16]);  clone_686 = None
        mm_97 = torch.ops.aten.mm.default(view_998, permute_654);  view_998 = permute_654 = None
        view_999 = torch.ops.aten.view.default(mm_97, [8, 576, 576, 16]);  mm_97 = None
        add_453 = torch.ops.aten.add.Tensor(view_999, arg228_1);  view_999 = arg228_1 = None
        permute_655 = torch.ops.aten.permute.default(add_453, [0, 3, 1, 2]);  add_453 = None
        expand_195 = torch.ops.aten.expand.default(permute_655, [8, 16, 576, 576]);  permute_655 = None
        clone_688 = torch.ops.aten.clone.default(expand_195, memory_format = torch.contiguous_format);  expand_195 = None
        view_1000 = torch.ops.aten.view.default(clone_688, [128, 576, 576]);  clone_688 = None
        expand_196 = torch.ops.aten.expand.default(select_149, [8, 16, 576, 48]);  select_149 = None
        clone_689 = torch.ops.aten.clone.default(expand_196, memory_format = torch.contiguous_format);  expand_196 = None
        view_1001 = torch.ops.aten.view.default(clone_689, [128, 576, 48]);  clone_689 = None
        bmm_97 = torch.ops.aten.bmm.default(view_1000, view_1001);  view_1000 = view_1001 = None
        view_1002 = torch.ops.aten.view.default(bmm_97, [8, 16, 576, 48]);  bmm_97 = None
        permute_656 = torch.ops.aten.permute.default(view_1002, [0, 2, 1, 3]);  view_1002 = None
        clone_690 = torch.ops.aten.clone.default(permute_656, memory_format = torch.contiguous_format);  permute_656 = None
        view_1003 = torch.ops.aten.view.default(clone_690, [8, 576, 768]);  clone_690 = None
        view_1004 = torch.ops.aten.view.default(view_1003, [4608, 768]);  view_1003 = None
        permute_657 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_206 = torch.ops.aten.addmm.default(arg230_1, view_1004, permute_657);  arg230_1 = view_1004 = permute_657 = None
        view_1005 = torch.ops.aten.view.default(addmm_206, [8, 576, 768]);  addmm_206 = None
        mul_503 = torch.ops.aten.mul.Tensor(arg220_1, view_1005);  arg220_1 = view_1005 = None
        add_454 = torch.ops.aten.add.Tensor(add_449, mul_503);  add_449 = mul_503 = None
        clone_692 = torch.ops.aten.clone.default(add_454, memory_format = torch.contiguous_format)
        var_mean_102 = torch.ops.aten.var_mean.correction(clone_692, [2], correction = 0, keepdim = True)
        getitem_212 = var_mean_102[0]
        getitem_213 = var_mean_102[1];  var_mean_102 = None
        add_455 = torch.ops.aten.add.Tensor(getitem_212, 1e-06);  getitem_212 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_455);  add_455 = None
        sub_151 = torch.ops.aten.sub.Tensor(clone_692, getitem_213);  clone_692 = getitem_213 = None
        mul_504 = torch.ops.aten.mul.Tensor(sub_151, rsqrt_102);  sub_151 = rsqrt_102 = None
        mul_505 = torch.ops.aten.mul.Tensor(mul_504, arg232_1);  mul_504 = arg232_1 = None
        add_456 = torch.ops.aten.add.Tensor(mul_505, arg233_1);  mul_505 = arg233_1 = None
        view_1006 = torch.ops.aten.view.default(add_456, [4608, 768]);  add_456 = None
        permute_658 = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_207 = torch.ops.aten.addmm.default(arg235_1, view_1006, permute_658);  arg235_1 = view_1006 = permute_658 = None
        view_1007 = torch.ops.aten.view.default(addmm_207, [8, 576, 3072]);  addmm_207 = None
        mul_506 = torch.ops.aten.mul.Tensor(view_1007, 0.5)
        mul_507 = torch.ops.aten.mul.Tensor(view_1007, 0.7071067811865476);  view_1007 = None
        erf_50 = torch.ops.aten.erf.default(mul_507);  mul_507 = None
        add_457 = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_508 = torch.ops.aten.mul.Tensor(mul_506, add_457);  mul_506 = add_457 = None
        view_1008 = torch.ops.aten.view.default(mul_508, [4608, 3072]);  mul_508 = None
        permute_659 = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_208 = torch.ops.aten.addmm.default(arg237_1, view_1008, permute_659);  arg237_1 = view_1008 = permute_659 = None
        view_1009 = torch.ops.aten.view.default(addmm_208, [8, 576, 768]);  addmm_208 = None
        mul_509 = torch.ops.aten.mul.Tensor(arg231_1, view_1009);  arg231_1 = view_1009 = None
        add_458 = torch.ops.aten.add.Tensor(add_454, mul_509);  add_454 = mul_509 = None
        clone_695 = torch.ops.aten.clone.default(add_458, memory_format = torch.contiguous_format)
        var_mean_103 = torch.ops.aten.var_mean.correction(clone_695, [2], correction = 0, keepdim = True)
        getitem_214 = var_mean_103[0]
        getitem_215 = var_mean_103[1];  var_mean_103 = None
        add_459 = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_459);  add_459 = None
        sub_152 = torch.ops.aten.sub.Tensor(clone_695, getitem_215);  clone_695 = getitem_215 = None
        mul_510 = torch.ops.aten.mul.Tensor(sub_152, rsqrt_103);  sub_152 = rsqrt_103 = None
        mul_511 = torch.ops.aten.mul.Tensor(mul_510, arg239_1);  mul_510 = arg239_1 = None
        add_460 = torch.ops.aten.add.Tensor(mul_511, arg240_1);  mul_511 = arg240_1 = None
        view_1010 = torch.ops.aten.view.default(add_460, [4608, 768]);  add_460 = None
        permute_660 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_209 = torch.ops.aten.addmm.default(arg242_1, view_1010, permute_660);  arg242_1 = view_1010 = permute_660 = None
        view_1011 = torch.ops.aten.view.default(addmm_209, [8, 576, 2304]);  addmm_209 = None
        view_1012 = torch.ops.aten.view.default(view_1011, [8, 576, 3, 16, 48]);  view_1011 = None
        permute_661 = torch.ops.aten.permute.default(view_1012, [2, 0, 3, 1, 4]);  view_1012 = None
        select_150 = torch.ops.aten.select.int(permute_661, 0, 0)
        mul_512 = torch.ops.aten.mul.Tensor(select_150, 0.14433756729740643);  select_150 = None
        select_151 = torch.ops.aten.select.int(permute_661, 0, 1)
        select_152 = torch.ops.aten.select.int(permute_661, 0, 2);  permute_661 = None
        permute_662 = torch.ops.aten.permute.default(select_151, [0, 1, 3, 2]);  select_151 = None
        expand_197 = torch.ops.aten.expand.default(mul_512, [8, 16, 576, 48]);  mul_512 = None
        clone_696 = torch.ops.aten.clone.default(expand_197, memory_format = torch.contiguous_format);  expand_197 = None
        view_1013 = torch.ops.aten.view.default(clone_696, [128, 576, 48]);  clone_696 = None
        expand_198 = torch.ops.aten.expand.default(permute_662, [8, 16, 48, 576]);  permute_662 = None
        clone_697 = torch.ops.aten.clone.default(expand_198, memory_format = torch.contiguous_format);  expand_198 = None
        view_1014 = torch.ops.aten.view.default(clone_697, [128, 48, 576]);  clone_697 = None
        bmm_98 = torch.ops.aten.bmm.default(view_1013, view_1014);  view_1013 = view_1014 = None
        view_1015 = torch.ops.aten.view.default(bmm_98, [8, 16, 576, 576]);  bmm_98 = None
        permute_663 = torch.ops.aten.permute.default(view_1015, [0, 2, 3, 1]);  view_1015 = None
        permute_664 = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        clone_698 = torch.ops.aten.clone.default(permute_663, memory_format = torch.contiguous_format);  permute_663 = None
        view_1016 = torch.ops.aten.view.default(clone_698, [2654208, 16]);  clone_698 = None
        mm_98 = torch.ops.aten.mm.default(view_1016, permute_664);  view_1016 = permute_664 = None
        view_1017 = torch.ops.aten.view.default(mm_98, [8, 576, 576, 16]);  mm_98 = None
        add_461 = torch.ops.aten.add.Tensor(view_1017, arg244_1);  view_1017 = arg244_1 = None
        permute_665 = torch.ops.aten.permute.default(add_461, [0, 3, 1, 2]);  add_461 = None
        clone_699 = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
        amax_49 = torch.ops.aten.amax.default(clone_699, [-1], True)
        sub_153 = torch.ops.aten.sub.Tensor(clone_699, amax_49);  clone_699 = amax_49 = None
        exp_49 = torch.ops.aten.exp.default(sub_153);  sub_153 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(exp_49, [-1], True)
        div_49 = torch.ops.aten.div.Tensor(exp_49, sum_50);  exp_49 = sum_50 = None
        permute_666 = torch.ops.aten.permute.default(div_49, [0, 2, 3, 1]);  div_49 = None
        permute_667 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        clone_700 = torch.ops.aten.clone.default(permute_666, memory_format = torch.contiguous_format);  permute_666 = None
        view_1018 = torch.ops.aten.view.default(clone_700, [2654208, 16]);  clone_700 = None
        mm_99 = torch.ops.aten.mm.default(view_1018, permute_667);  view_1018 = permute_667 = None
        view_1019 = torch.ops.aten.view.default(mm_99, [8, 576, 576, 16]);  mm_99 = None
        add_462 = torch.ops.aten.add.Tensor(view_1019, arg246_1);  view_1019 = arg246_1 = None
        permute_668 = torch.ops.aten.permute.default(add_462, [0, 3, 1, 2]);  add_462 = None
        expand_199 = torch.ops.aten.expand.default(permute_668, [8, 16, 576, 576]);  permute_668 = None
        clone_702 = torch.ops.aten.clone.default(expand_199, memory_format = torch.contiguous_format);  expand_199 = None
        view_1020 = torch.ops.aten.view.default(clone_702, [128, 576, 576]);  clone_702 = None
        expand_200 = torch.ops.aten.expand.default(select_152, [8, 16, 576, 48]);  select_152 = None
        clone_703 = torch.ops.aten.clone.default(expand_200, memory_format = torch.contiguous_format);  expand_200 = None
        view_1021 = torch.ops.aten.view.default(clone_703, [128, 576, 48]);  clone_703 = None
        bmm_99 = torch.ops.aten.bmm.default(view_1020, view_1021);  view_1020 = view_1021 = None
        view_1022 = torch.ops.aten.view.default(bmm_99, [8, 16, 576, 48]);  bmm_99 = None
        permute_669 = torch.ops.aten.permute.default(view_1022, [0, 2, 1, 3]);  view_1022 = None
        clone_704 = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
        view_1023 = torch.ops.aten.view.default(clone_704, [8, 576, 768]);  clone_704 = None
        view_1024 = torch.ops.aten.view.default(view_1023, [4608, 768]);  view_1023 = None
        permute_670 = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_210 = torch.ops.aten.addmm.default(arg248_1, view_1024, permute_670);  arg248_1 = view_1024 = permute_670 = None
        view_1025 = torch.ops.aten.view.default(addmm_210, [8, 576, 768]);  addmm_210 = None
        mul_513 = torch.ops.aten.mul.Tensor(arg238_1, view_1025);  arg238_1 = view_1025 = None
        add_463 = torch.ops.aten.add.Tensor(add_458, mul_513);  add_458 = mul_513 = None
        clone_706 = torch.ops.aten.clone.default(add_463, memory_format = torch.contiguous_format)
        var_mean_104 = torch.ops.aten.var_mean.correction(clone_706, [2], correction = 0, keepdim = True)
        getitem_216 = var_mean_104[0]
        getitem_217 = var_mean_104[1];  var_mean_104 = None
        add_464 = torch.ops.aten.add.Tensor(getitem_216, 1e-06);  getitem_216 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_464);  add_464 = None
        sub_154 = torch.ops.aten.sub.Tensor(clone_706, getitem_217);  clone_706 = getitem_217 = None
        mul_514 = torch.ops.aten.mul.Tensor(sub_154, rsqrt_104);  sub_154 = rsqrt_104 = None
        mul_515 = torch.ops.aten.mul.Tensor(mul_514, arg250_1);  mul_514 = arg250_1 = None
        add_465 = torch.ops.aten.add.Tensor(mul_515, arg251_1);  mul_515 = arg251_1 = None
        view_1026 = torch.ops.aten.view.default(add_465, [4608, 768]);  add_465 = None
        permute_671 = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        addmm_211 = torch.ops.aten.addmm.default(arg253_1, view_1026, permute_671);  arg253_1 = view_1026 = permute_671 = None
        view_1027 = torch.ops.aten.view.default(addmm_211, [8, 576, 3072]);  addmm_211 = None
        mul_516 = torch.ops.aten.mul.Tensor(view_1027, 0.5)
        mul_517 = torch.ops.aten.mul.Tensor(view_1027, 0.7071067811865476);  view_1027 = None
        erf_51 = torch.ops.aten.erf.default(mul_517);  mul_517 = None
        add_466 = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_518 = torch.ops.aten.mul.Tensor(mul_516, add_466);  mul_516 = add_466 = None
        view_1028 = torch.ops.aten.view.default(mul_518, [4608, 3072]);  mul_518 = None
        permute_672 = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        addmm_212 = torch.ops.aten.addmm.default(arg255_1, view_1028, permute_672);  arg255_1 = view_1028 = permute_672 = None
        view_1029 = torch.ops.aten.view.default(addmm_212, [8, 576, 768]);  addmm_212 = None
        mul_519 = torch.ops.aten.mul.Tensor(arg249_1, view_1029);  arg249_1 = view_1029 = None
        add_467 = torch.ops.aten.add.Tensor(add_463, mul_519);  add_463 = mul_519 = None
        clone_709 = torch.ops.aten.clone.default(add_467, memory_format = torch.contiguous_format)
        var_mean_105 = torch.ops.aten.var_mean.correction(clone_709, [2], correction = 0, keepdim = True)
        getitem_218 = var_mean_105[0]
        getitem_219 = var_mean_105[1];  var_mean_105 = None
        add_468 = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_468);  add_468 = None
        sub_155 = torch.ops.aten.sub.Tensor(clone_709, getitem_219);  clone_709 = getitem_219 = None
        mul_520 = torch.ops.aten.mul.Tensor(sub_155, rsqrt_105);  sub_155 = rsqrt_105 = None
        mul_521 = torch.ops.aten.mul.Tensor(mul_520, arg257_1);  mul_520 = arg257_1 = None
        add_469 = torch.ops.aten.add.Tensor(mul_521, arg258_1);  mul_521 = arg258_1 = None
        view_1030 = torch.ops.aten.view.default(add_469, [4608, 768]);  add_469 = None
        permute_673 = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_213 = torch.ops.aten.addmm.default(arg260_1, view_1030, permute_673);  arg260_1 = view_1030 = permute_673 = None
        view_1031 = torch.ops.aten.view.default(addmm_213, [8, 576, 2304]);  addmm_213 = None
        view_1032 = torch.ops.aten.view.default(view_1031, [8, 576, 3, 16, 48]);  view_1031 = None
        permute_674 = torch.ops.aten.permute.default(view_1032, [2, 0, 3, 1, 4]);  view_1032 = None
        select_153 = torch.ops.aten.select.int(permute_674, 0, 0)
        mul_522 = torch.ops.aten.mul.Tensor(select_153, 0.14433756729740643);  select_153 = None
        select_154 = torch.ops.aten.select.int(permute_674, 0, 1)
        select_155 = torch.ops.aten.select.int(permute_674, 0, 2);  permute_674 = None
        permute_675 = torch.ops.aten.permute.default(select_154, [0, 1, 3, 2]);  select_154 = None
        expand_201 = torch.ops.aten.expand.default(mul_522, [8, 16, 576, 48]);  mul_522 = None
        clone_710 = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_1033 = torch.ops.aten.view.default(clone_710, [128, 576, 48]);  clone_710 = None
        expand_202 = torch.ops.aten.expand.default(permute_675, [8, 16, 48, 576]);  permute_675 = None
        clone_711 = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_1034 = torch.ops.aten.view.default(clone_711, [128, 48, 576]);  clone_711 = None
        bmm_100 = torch.ops.aten.bmm.default(view_1033, view_1034);  view_1033 = view_1034 = None
        view_1035 = torch.ops.aten.view.default(bmm_100, [8, 16, 576, 576]);  bmm_100 = None
        permute_676 = torch.ops.aten.permute.default(view_1035, [0, 2, 3, 1]);  view_1035 = None
        permute_677 = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        clone_712 = torch.ops.aten.clone.default(permute_676, memory_format = torch.contiguous_format);  permute_676 = None
        view_1036 = torch.ops.aten.view.default(clone_712, [2654208, 16]);  clone_712 = None
        mm_100 = torch.ops.aten.mm.default(view_1036, permute_677);  view_1036 = permute_677 = None
        view_1037 = torch.ops.aten.view.default(mm_100, [8, 576, 576, 16]);  mm_100 = None
        add_470 = torch.ops.aten.add.Tensor(view_1037, arg262_1);  view_1037 = arg262_1 = None
        permute_678 = torch.ops.aten.permute.default(add_470, [0, 3, 1, 2]);  add_470 = None
        clone_713 = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
        amax_50 = torch.ops.aten.amax.default(clone_713, [-1], True)
        sub_156 = torch.ops.aten.sub.Tensor(clone_713, amax_50);  clone_713 = amax_50 = None
        exp_50 = torch.ops.aten.exp.default(sub_156);  sub_156 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(exp_50, [-1], True)
        div_50 = torch.ops.aten.div.Tensor(exp_50, sum_51);  exp_50 = sum_51 = None
        permute_679 = torch.ops.aten.permute.default(div_50, [0, 2, 3, 1]);  div_50 = None
        permute_680 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        clone_714 = torch.ops.aten.clone.default(permute_679, memory_format = torch.contiguous_format);  permute_679 = None
        view_1038 = torch.ops.aten.view.default(clone_714, [2654208, 16]);  clone_714 = None
        mm_101 = torch.ops.aten.mm.default(view_1038, permute_680);  view_1038 = permute_680 = None
        view_1039 = torch.ops.aten.view.default(mm_101, [8, 576, 576, 16]);  mm_101 = None
        add_471 = torch.ops.aten.add.Tensor(view_1039, arg264_1);  view_1039 = arg264_1 = None
        permute_681 = torch.ops.aten.permute.default(add_471, [0, 3, 1, 2]);  add_471 = None
        expand_203 = torch.ops.aten.expand.default(permute_681, [8, 16, 576, 576]);  permute_681 = None
        clone_716 = torch.ops.aten.clone.default(expand_203, memory_format = torch.contiguous_format);  expand_203 = None
        view_1040 = torch.ops.aten.view.default(clone_716, [128, 576, 576]);  clone_716 = None
        expand_204 = torch.ops.aten.expand.default(select_155, [8, 16, 576, 48]);  select_155 = None
        clone_717 = torch.ops.aten.clone.default(expand_204, memory_format = torch.contiguous_format);  expand_204 = None
        view_1041 = torch.ops.aten.view.default(clone_717, [128, 576, 48]);  clone_717 = None
        bmm_101 = torch.ops.aten.bmm.default(view_1040, view_1041);  view_1040 = view_1041 = None
        view_1042 = torch.ops.aten.view.default(bmm_101, [8, 16, 576, 48]);  bmm_101 = None
        permute_682 = torch.ops.aten.permute.default(view_1042, [0, 2, 1, 3]);  view_1042 = None
        clone_718 = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None
        view_1043 = torch.ops.aten.view.default(clone_718, [8, 576, 768]);  clone_718 = None
        view_1044 = torch.ops.aten.view.default(view_1043, [4608, 768]);  view_1043 = None
        permute_683 = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_214 = torch.ops.aten.addmm.default(arg266_1, view_1044, permute_683);  arg266_1 = view_1044 = permute_683 = None
        view_1045 = torch.ops.aten.view.default(addmm_214, [8, 576, 768]);  addmm_214 = None
        mul_523 = torch.ops.aten.mul.Tensor(arg256_1, view_1045);  arg256_1 = view_1045 = None
        add_472 = torch.ops.aten.add.Tensor(add_467, mul_523);  add_467 = mul_523 = None
        clone_720 = torch.ops.aten.clone.default(add_472, memory_format = torch.contiguous_format)
        var_mean_106 = torch.ops.aten.var_mean.correction(clone_720, [2], correction = 0, keepdim = True)
        getitem_220 = var_mean_106[0]
        getitem_221 = var_mean_106[1];  var_mean_106 = None
        add_473 = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_106 = torch.ops.aten.rsqrt.default(add_473);  add_473 = None
        sub_157 = torch.ops.aten.sub.Tensor(clone_720, getitem_221);  clone_720 = getitem_221 = None
        mul_524 = torch.ops.aten.mul.Tensor(sub_157, rsqrt_106);  sub_157 = rsqrt_106 = None
        mul_525 = torch.ops.aten.mul.Tensor(mul_524, arg268_1);  mul_524 = arg268_1 = None
        add_474 = torch.ops.aten.add.Tensor(mul_525, arg269_1);  mul_525 = arg269_1 = None
        view_1046 = torch.ops.aten.view.default(add_474, [4608, 768]);  add_474 = None
        permute_684 = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_215 = torch.ops.aten.addmm.default(arg271_1, view_1046, permute_684);  arg271_1 = view_1046 = permute_684 = None
        view_1047 = torch.ops.aten.view.default(addmm_215, [8, 576, 3072]);  addmm_215 = None
        mul_526 = torch.ops.aten.mul.Tensor(view_1047, 0.5)
        mul_527 = torch.ops.aten.mul.Tensor(view_1047, 0.7071067811865476);  view_1047 = None
        erf_52 = torch.ops.aten.erf.default(mul_527);  mul_527 = None
        add_475 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_528 = torch.ops.aten.mul.Tensor(mul_526, add_475);  mul_526 = add_475 = None
        view_1048 = torch.ops.aten.view.default(mul_528, [4608, 3072]);  mul_528 = None
        permute_685 = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_216 = torch.ops.aten.addmm.default(arg273_1, view_1048, permute_685);  arg273_1 = view_1048 = permute_685 = None
        view_1049 = torch.ops.aten.view.default(addmm_216, [8, 576, 768]);  addmm_216 = None
        mul_529 = torch.ops.aten.mul.Tensor(arg267_1, view_1049);  arg267_1 = view_1049 = None
        add_476 = torch.ops.aten.add.Tensor(add_472, mul_529);  add_472 = mul_529 = None
        clone_723 = torch.ops.aten.clone.default(add_476, memory_format = torch.contiguous_format)
        var_mean_107 = torch.ops.aten.var_mean.correction(clone_723, [2], correction = 0, keepdim = True)
        getitem_222 = var_mean_107[0]
        getitem_223 = var_mean_107[1];  var_mean_107 = None
        add_477 = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_107 = torch.ops.aten.rsqrt.default(add_477);  add_477 = None
        sub_158 = torch.ops.aten.sub.Tensor(clone_723, getitem_223);  clone_723 = getitem_223 = None
        mul_530 = torch.ops.aten.mul.Tensor(sub_158, rsqrt_107);  sub_158 = rsqrt_107 = None
        mul_531 = torch.ops.aten.mul.Tensor(mul_530, arg275_1);  mul_530 = arg275_1 = None
        add_478 = torch.ops.aten.add.Tensor(mul_531, arg276_1);  mul_531 = arg276_1 = None
        view_1050 = torch.ops.aten.view.default(add_478, [4608, 768]);  add_478 = None
        permute_686 = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_217 = torch.ops.aten.addmm.default(arg278_1, view_1050, permute_686);  arg278_1 = view_1050 = permute_686 = None
        view_1051 = torch.ops.aten.view.default(addmm_217, [8, 576, 2304]);  addmm_217 = None
        view_1052 = torch.ops.aten.view.default(view_1051, [8, 576, 3, 16, 48]);  view_1051 = None
        permute_687 = torch.ops.aten.permute.default(view_1052, [2, 0, 3, 1, 4]);  view_1052 = None
        select_156 = torch.ops.aten.select.int(permute_687, 0, 0)
        mul_532 = torch.ops.aten.mul.Tensor(select_156, 0.14433756729740643);  select_156 = None
        select_157 = torch.ops.aten.select.int(permute_687, 0, 1)
        select_158 = torch.ops.aten.select.int(permute_687, 0, 2);  permute_687 = None
        permute_688 = torch.ops.aten.permute.default(select_157, [0, 1, 3, 2]);  select_157 = None
        expand_205 = torch.ops.aten.expand.default(mul_532, [8, 16, 576, 48]);  mul_532 = None
        clone_724 = torch.ops.aten.clone.default(expand_205, memory_format = torch.contiguous_format);  expand_205 = None
        view_1053 = torch.ops.aten.view.default(clone_724, [128, 576, 48]);  clone_724 = None
        expand_206 = torch.ops.aten.expand.default(permute_688, [8, 16, 48, 576]);  permute_688 = None
        clone_725 = torch.ops.aten.clone.default(expand_206, memory_format = torch.contiguous_format);  expand_206 = None
        view_1054 = torch.ops.aten.view.default(clone_725, [128, 48, 576]);  clone_725 = None
        bmm_102 = torch.ops.aten.bmm.default(view_1053, view_1054);  view_1053 = view_1054 = None
        view_1055 = torch.ops.aten.view.default(bmm_102, [8, 16, 576, 576]);  bmm_102 = None
        permute_689 = torch.ops.aten.permute.default(view_1055, [0, 2, 3, 1]);  view_1055 = None
        permute_690 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        clone_726 = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
        view_1056 = torch.ops.aten.view.default(clone_726, [2654208, 16]);  clone_726 = None
        mm_102 = torch.ops.aten.mm.default(view_1056, permute_690);  view_1056 = permute_690 = None
        view_1057 = torch.ops.aten.view.default(mm_102, [8, 576, 576, 16]);  mm_102 = None
        add_479 = torch.ops.aten.add.Tensor(view_1057, arg280_1);  view_1057 = arg280_1 = None
        permute_691 = torch.ops.aten.permute.default(add_479, [0, 3, 1, 2]);  add_479 = None
        clone_727 = torch.ops.aten.clone.default(permute_691, memory_format = torch.contiguous_format);  permute_691 = None
        amax_51 = torch.ops.aten.amax.default(clone_727, [-1], True)
        sub_159 = torch.ops.aten.sub.Tensor(clone_727, amax_51);  clone_727 = amax_51 = None
        exp_51 = torch.ops.aten.exp.default(sub_159);  sub_159 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(exp_51, [-1], True)
        div_51 = torch.ops.aten.div.Tensor(exp_51, sum_52);  exp_51 = sum_52 = None
        permute_692 = torch.ops.aten.permute.default(div_51, [0, 2, 3, 1]);  div_51 = None
        permute_693 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        clone_728 = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
        view_1058 = torch.ops.aten.view.default(clone_728, [2654208, 16]);  clone_728 = None
        mm_103 = torch.ops.aten.mm.default(view_1058, permute_693);  view_1058 = permute_693 = None
        view_1059 = torch.ops.aten.view.default(mm_103, [8, 576, 576, 16]);  mm_103 = None
        add_480 = torch.ops.aten.add.Tensor(view_1059, arg282_1);  view_1059 = arg282_1 = None
        permute_694 = torch.ops.aten.permute.default(add_480, [0, 3, 1, 2]);  add_480 = None
        expand_207 = torch.ops.aten.expand.default(permute_694, [8, 16, 576, 576]);  permute_694 = None
        clone_730 = torch.ops.aten.clone.default(expand_207, memory_format = torch.contiguous_format);  expand_207 = None
        view_1060 = torch.ops.aten.view.default(clone_730, [128, 576, 576]);  clone_730 = None
        expand_208 = torch.ops.aten.expand.default(select_158, [8, 16, 576, 48]);  select_158 = None
        clone_731 = torch.ops.aten.clone.default(expand_208, memory_format = torch.contiguous_format);  expand_208 = None
        view_1061 = torch.ops.aten.view.default(clone_731, [128, 576, 48]);  clone_731 = None
        bmm_103 = torch.ops.aten.bmm.default(view_1060, view_1061);  view_1060 = view_1061 = None
        view_1062 = torch.ops.aten.view.default(bmm_103, [8, 16, 576, 48]);  bmm_103 = None
        permute_695 = torch.ops.aten.permute.default(view_1062, [0, 2, 1, 3]);  view_1062 = None
        clone_732 = torch.ops.aten.clone.default(permute_695, memory_format = torch.contiguous_format);  permute_695 = None
        view_1063 = torch.ops.aten.view.default(clone_732, [8, 576, 768]);  clone_732 = None
        view_1064 = torch.ops.aten.view.default(view_1063, [4608, 768]);  view_1063 = None
        permute_696 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_218 = torch.ops.aten.addmm.default(arg284_1, view_1064, permute_696);  arg284_1 = view_1064 = permute_696 = None
        view_1065 = torch.ops.aten.view.default(addmm_218, [8, 576, 768]);  addmm_218 = None
        mul_533 = torch.ops.aten.mul.Tensor(arg274_1, view_1065);  arg274_1 = view_1065 = None
        add_481 = torch.ops.aten.add.Tensor(add_476, mul_533);  add_476 = mul_533 = None
        clone_734 = torch.ops.aten.clone.default(add_481, memory_format = torch.contiguous_format)
        var_mean_108 = torch.ops.aten.var_mean.correction(clone_734, [2], correction = 0, keepdim = True)
        getitem_224 = var_mean_108[0]
        getitem_225 = var_mean_108[1];  var_mean_108 = None
        add_482 = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
        rsqrt_108 = torch.ops.aten.rsqrt.default(add_482);  add_482 = None
        sub_160 = torch.ops.aten.sub.Tensor(clone_734, getitem_225);  clone_734 = getitem_225 = None
        mul_534 = torch.ops.aten.mul.Tensor(sub_160, rsqrt_108);  sub_160 = rsqrt_108 = None
        mul_535 = torch.ops.aten.mul.Tensor(mul_534, arg286_1);  mul_534 = arg286_1 = None
        add_483 = torch.ops.aten.add.Tensor(mul_535, arg287_1);  mul_535 = arg287_1 = None
        view_1066 = torch.ops.aten.view.default(add_483, [4608, 768]);  add_483 = None
        permute_697 = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_219 = torch.ops.aten.addmm.default(arg289_1, view_1066, permute_697);  arg289_1 = view_1066 = permute_697 = None
        view_1067 = torch.ops.aten.view.default(addmm_219, [8, 576, 3072]);  addmm_219 = None
        mul_536 = torch.ops.aten.mul.Tensor(view_1067, 0.5)
        mul_537 = torch.ops.aten.mul.Tensor(view_1067, 0.7071067811865476);  view_1067 = None
        erf_53 = torch.ops.aten.erf.default(mul_537);  mul_537 = None
        add_484 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_538 = torch.ops.aten.mul.Tensor(mul_536, add_484);  mul_536 = add_484 = None
        view_1068 = torch.ops.aten.view.default(mul_538, [4608, 3072]);  mul_538 = None
        permute_698 = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_220 = torch.ops.aten.addmm.default(arg291_1, view_1068, permute_698);  arg291_1 = view_1068 = permute_698 = None
        view_1069 = torch.ops.aten.view.default(addmm_220, [8, 576, 768]);  addmm_220 = None
        mul_539 = torch.ops.aten.mul.Tensor(arg285_1, view_1069);  arg285_1 = view_1069 = None
        add_485 = torch.ops.aten.add.Tensor(add_481, mul_539);  add_481 = mul_539 = None
        clone_737 = torch.ops.aten.clone.default(add_485, memory_format = torch.contiguous_format)
        var_mean_109 = torch.ops.aten.var_mean.correction(clone_737, [2], correction = 0, keepdim = True)
        getitem_226 = var_mean_109[0]
        getitem_227 = var_mean_109[1];  var_mean_109 = None
        add_486 = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
        rsqrt_109 = torch.ops.aten.rsqrt.default(add_486);  add_486 = None
        sub_161 = torch.ops.aten.sub.Tensor(clone_737, getitem_227);  clone_737 = getitem_227 = None
        mul_540 = torch.ops.aten.mul.Tensor(sub_161, rsqrt_109);  sub_161 = rsqrt_109 = None
        mul_541 = torch.ops.aten.mul.Tensor(mul_540, arg293_1);  mul_540 = arg293_1 = None
        add_487 = torch.ops.aten.add.Tensor(mul_541, arg294_1);  mul_541 = arg294_1 = None
        view_1070 = torch.ops.aten.view.default(add_487, [4608, 768]);  add_487 = None
        permute_699 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_221 = torch.ops.aten.addmm.default(arg296_1, view_1070, permute_699);  arg296_1 = view_1070 = permute_699 = None
        view_1071 = torch.ops.aten.view.default(addmm_221, [8, 576, 2304]);  addmm_221 = None
        view_1072 = torch.ops.aten.view.default(view_1071, [8, 576, 3, 16, 48]);  view_1071 = None
        permute_700 = torch.ops.aten.permute.default(view_1072, [2, 0, 3, 1, 4]);  view_1072 = None
        select_159 = torch.ops.aten.select.int(permute_700, 0, 0)
        mul_542 = torch.ops.aten.mul.Tensor(select_159, 0.14433756729740643);  select_159 = None
        select_160 = torch.ops.aten.select.int(permute_700, 0, 1)
        select_161 = torch.ops.aten.select.int(permute_700, 0, 2);  permute_700 = None
        permute_701 = torch.ops.aten.permute.default(select_160, [0, 1, 3, 2]);  select_160 = None
        expand_209 = torch.ops.aten.expand.default(mul_542, [8, 16, 576, 48]);  mul_542 = None
        clone_738 = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_1073 = torch.ops.aten.view.default(clone_738, [128, 576, 48]);  clone_738 = None
        expand_210 = torch.ops.aten.expand.default(permute_701, [8, 16, 48, 576]);  permute_701 = None
        clone_739 = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_1074 = torch.ops.aten.view.default(clone_739, [128, 48, 576]);  clone_739 = None
        bmm_104 = torch.ops.aten.bmm.default(view_1073, view_1074);  view_1073 = view_1074 = None
        view_1075 = torch.ops.aten.view.default(bmm_104, [8, 16, 576, 576]);  bmm_104 = None
        permute_702 = torch.ops.aten.permute.default(view_1075, [0, 2, 3, 1]);  view_1075 = None
        permute_703 = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        clone_740 = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
        view_1076 = torch.ops.aten.view.default(clone_740, [2654208, 16]);  clone_740 = None
        mm_104 = torch.ops.aten.mm.default(view_1076, permute_703);  view_1076 = permute_703 = None
        view_1077 = torch.ops.aten.view.default(mm_104, [8, 576, 576, 16]);  mm_104 = None
        add_488 = torch.ops.aten.add.Tensor(view_1077, arg298_1);  view_1077 = arg298_1 = None
        permute_704 = torch.ops.aten.permute.default(add_488, [0, 3, 1, 2]);  add_488 = None
        clone_741 = torch.ops.aten.clone.default(permute_704, memory_format = torch.contiguous_format);  permute_704 = None
        amax_52 = torch.ops.aten.amax.default(clone_741, [-1], True)
        sub_162 = torch.ops.aten.sub.Tensor(clone_741, amax_52);  clone_741 = amax_52 = None
        exp_52 = torch.ops.aten.exp.default(sub_162);  sub_162 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(exp_52, [-1], True)
        div_52 = torch.ops.aten.div.Tensor(exp_52, sum_53);  exp_52 = sum_53 = None
        permute_705 = torch.ops.aten.permute.default(div_52, [0, 2, 3, 1]);  div_52 = None
        permute_706 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        clone_742 = torch.ops.aten.clone.default(permute_705, memory_format = torch.contiguous_format);  permute_705 = None
        view_1078 = torch.ops.aten.view.default(clone_742, [2654208, 16]);  clone_742 = None
        mm_105 = torch.ops.aten.mm.default(view_1078, permute_706);  view_1078 = permute_706 = None
        view_1079 = torch.ops.aten.view.default(mm_105, [8, 576, 576, 16]);  mm_105 = None
        add_489 = torch.ops.aten.add.Tensor(view_1079, arg300_1);  view_1079 = arg300_1 = None
        permute_707 = torch.ops.aten.permute.default(add_489, [0, 3, 1, 2]);  add_489 = None
        expand_211 = torch.ops.aten.expand.default(permute_707, [8, 16, 576, 576]);  permute_707 = None
        clone_744 = torch.ops.aten.clone.default(expand_211, memory_format = torch.contiguous_format);  expand_211 = None
        view_1080 = torch.ops.aten.view.default(clone_744, [128, 576, 576]);  clone_744 = None
        expand_212 = torch.ops.aten.expand.default(select_161, [8, 16, 576, 48]);  select_161 = None
        clone_745 = torch.ops.aten.clone.default(expand_212, memory_format = torch.contiguous_format);  expand_212 = None
        view_1081 = torch.ops.aten.view.default(clone_745, [128, 576, 48]);  clone_745 = None
        bmm_105 = torch.ops.aten.bmm.default(view_1080, view_1081);  view_1080 = view_1081 = None
        view_1082 = torch.ops.aten.view.default(bmm_105, [8, 16, 576, 48]);  bmm_105 = None
        permute_708 = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None
        clone_746 = torch.ops.aten.clone.default(permute_708, memory_format = torch.contiguous_format);  permute_708 = None
        view_1083 = torch.ops.aten.view.default(clone_746, [8, 576, 768]);  clone_746 = None
        view_1084 = torch.ops.aten.view.default(view_1083, [4608, 768]);  view_1083 = None
        permute_709 = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_222 = torch.ops.aten.addmm.default(arg302_1, view_1084, permute_709);  arg302_1 = view_1084 = permute_709 = None
        view_1085 = torch.ops.aten.view.default(addmm_222, [8, 576, 768]);  addmm_222 = None
        mul_543 = torch.ops.aten.mul.Tensor(arg292_1, view_1085);  arg292_1 = view_1085 = None
        add_490 = torch.ops.aten.add.Tensor(add_485, mul_543);  add_485 = mul_543 = None
        clone_748 = torch.ops.aten.clone.default(add_490, memory_format = torch.contiguous_format)
        var_mean_110 = torch.ops.aten.var_mean.correction(clone_748, [2], correction = 0, keepdim = True)
        getitem_228 = var_mean_110[0]
        getitem_229 = var_mean_110[1];  var_mean_110 = None
        add_491 = torch.ops.aten.add.Tensor(getitem_228, 1e-06);  getitem_228 = None
        rsqrt_110 = torch.ops.aten.rsqrt.default(add_491);  add_491 = None
        sub_163 = torch.ops.aten.sub.Tensor(clone_748, getitem_229);  clone_748 = getitem_229 = None
        mul_544 = torch.ops.aten.mul.Tensor(sub_163, rsqrt_110);  sub_163 = rsqrt_110 = None
        mul_545 = torch.ops.aten.mul.Tensor(mul_544, arg304_1);  mul_544 = arg304_1 = None
        add_492 = torch.ops.aten.add.Tensor(mul_545, arg305_1);  mul_545 = arg305_1 = None
        view_1086 = torch.ops.aten.view.default(add_492, [4608, 768]);  add_492 = None
        permute_710 = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_223 = torch.ops.aten.addmm.default(arg307_1, view_1086, permute_710);  arg307_1 = view_1086 = permute_710 = None
        view_1087 = torch.ops.aten.view.default(addmm_223, [8, 576, 3072]);  addmm_223 = None
        mul_546 = torch.ops.aten.mul.Tensor(view_1087, 0.5)
        mul_547 = torch.ops.aten.mul.Tensor(view_1087, 0.7071067811865476);  view_1087 = None
        erf_54 = torch.ops.aten.erf.default(mul_547);  mul_547 = None
        add_493 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_548 = torch.ops.aten.mul.Tensor(mul_546, add_493);  mul_546 = add_493 = None
        view_1088 = torch.ops.aten.view.default(mul_548, [4608, 3072]);  mul_548 = None
        permute_711 = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        addmm_224 = torch.ops.aten.addmm.default(arg309_1, view_1088, permute_711);  arg309_1 = view_1088 = permute_711 = None
        view_1089 = torch.ops.aten.view.default(addmm_224, [8, 576, 768]);  addmm_224 = None
        mul_549 = torch.ops.aten.mul.Tensor(arg303_1, view_1089);  arg303_1 = view_1089 = None
        add_494 = torch.ops.aten.add.Tensor(add_490, mul_549);  add_490 = mul_549 = None
        clone_751 = torch.ops.aten.clone.default(add_494, memory_format = torch.contiguous_format)
        var_mean_111 = torch.ops.aten.var_mean.correction(clone_751, [2], correction = 0, keepdim = True)
        getitem_230 = var_mean_111[0]
        getitem_231 = var_mean_111[1];  var_mean_111 = None
        add_495 = torch.ops.aten.add.Tensor(getitem_230, 1e-06);  getitem_230 = None
        rsqrt_111 = torch.ops.aten.rsqrt.default(add_495);  add_495 = None
        sub_164 = torch.ops.aten.sub.Tensor(clone_751, getitem_231);  clone_751 = getitem_231 = None
        mul_550 = torch.ops.aten.mul.Tensor(sub_164, rsqrt_111);  sub_164 = rsqrt_111 = None
        mul_551 = torch.ops.aten.mul.Tensor(mul_550, arg311_1);  mul_550 = arg311_1 = None
        add_496 = torch.ops.aten.add.Tensor(mul_551, arg312_1);  mul_551 = arg312_1 = None
        view_1090 = torch.ops.aten.view.default(add_496, [4608, 768]);  add_496 = None
        permute_712 = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_225 = torch.ops.aten.addmm.default(arg314_1, view_1090, permute_712);  arg314_1 = view_1090 = permute_712 = None
        view_1091 = torch.ops.aten.view.default(addmm_225, [8, 576, 2304]);  addmm_225 = None
        view_1092 = torch.ops.aten.view.default(view_1091, [8, 576, 3, 16, 48]);  view_1091 = None
        permute_713 = torch.ops.aten.permute.default(view_1092, [2, 0, 3, 1, 4]);  view_1092 = None
        select_162 = torch.ops.aten.select.int(permute_713, 0, 0)
        mul_552 = torch.ops.aten.mul.Tensor(select_162, 0.14433756729740643);  select_162 = None
        select_163 = torch.ops.aten.select.int(permute_713, 0, 1)
        select_164 = torch.ops.aten.select.int(permute_713, 0, 2);  permute_713 = None
        permute_714 = torch.ops.aten.permute.default(select_163, [0, 1, 3, 2]);  select_163 = None
        expand_213 = torch.ops.aten.expand.default(mul_552, [8, 16, 576, 48]);  mul_552 = None
        clone_752 = torch.ops.aten.clone.default(expand_213, memory_format = torch.contiguous_format);  expand_213 = None
        view_1093 = torch.ops.aten.view.default(clone_752, [128, 576, 48]);  clone_752 = None
        expand_214 = torch.ops.aten.expand.default(permute_714, [8, 16, 48, 576]);  permute_714 = None
        clone_753 = torch.ops.aten.clone.default(expand_214, memory_format = torch.contiguous_format);  expand_214 = None
        view_1094 = torch.ops.aten.view.default(clone_753, [128, 48, 576]);  clone_753 = None
        bmm_106 = torch.ops.aten.bmm.default(view_1093, view_1094);  view_1093 = view_1094 = None
        view_1095 = torch.ops.aten.view.default(bmm_106, [8, 16, 576, 576]);  bmm_106 = None
        permute_715 = torch.ops.aten.permute.default(view_1095, [0, 2, 3, 1]);  view_1095 = None
        permute_716 = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        clone_754 = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None
        view_1096 = torch.ops.aten.view.default(clone_754, [2654208, 16]);  clone_754 = None
        mm_106 = torch.ops.aten.mm.default(view_1096, permute_716);  view_1096 = permute_716 = None
        view_1097 = torch.ops.aten.view.default(mm_106, [8, 576, 576, 16]);  mm_106 = None
        add_497 = torch.ops.aten.add.Tensor(view_1097, arg316_1);  view_1097 = arg316_1 = None
        permute_717 = torch.ops.aten.permute.default(add_497, [0, 3, 1, 2]);  add_497 = None
        clone_755 = torch.ops.aten.clone.default(permute_717, memory_format = torch.contiguous_format);  permute_717 = None
        amax_53 = torch.ops.aten.amax.default(clone_755, [-1], True)
        sub_165 = torch.ops.aten.sub.Tensor(clone_755, amax_53);  clone_755 = amax_53 = None
        exp_53 = torch.ops.aten.exp.default(sub_165);  sub_165 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(exp_53, [-1], True)
        div_53 = torch.ops.aten.div.Tensor(exp_53, sum_54);  exp_53 = sum_54 = None
        permute_718 = torch.ops.aten.permute.default(div_53, [0, 2, 3, 1]);  div_53 = None
        permute_719 = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        clone_756 = torch.ops.aten.clone.default(permute_718, memory_format = torch.contiguous_format);  permute_718 = None
        view_1098 = torch.ops.aten.view.default(clone_756, [2654208, 16]);  clone_756 = None
        mm_107 = torch.ops.aten.mm.default(view_1098, permute_719);  view_1098 = permute_719 = None
        view_1099 = torch.ops.aten.view.default(mm_107, [8, 576, 576, 16]);  mm_107 = None
        add_498 = torch.ops.aten.add.Tensor(view_1099, arg318_1);  view_1099 = arg318_1 = None
        permute_720 = torch.ops.aten.permute.default(add_498, [0, 3, 1, 2]);  add_498 = None
        expand_215 = torch.ops.aten.expand.default(permute_720, [8, 16, 576, 576]);  permute_720 = None
        clone_758 = torch.ops.aten.clone.default(expand_215, memory_format = torch.contiguous_format);  expand_215 = None
        view_1100 = torch.ops.aten.view.default(clone_758, [128, 576, 576]);  clone_758 = None
        expand_216 = torch.ops.aten.expand.default(select_164, [8, 16, 576, 48]);  select_164 = None
        clone_759 = torch.ops.aten.clone.default(expand_216, memory_format = torch.contiguous_format);  expand_216 = None
        view_1101 = torch.ops.aten.view.default(clone_759, [128, 576, 48]);  clone_759 = None
        bmm_107 = torch.ops.aten.bmm.default(view_1100, view_1101);  view_1100 = view_1101 = None
        view_1102 = torch.ops.aten.view.default(bmm_107, [8, 16, 576, 48]);  bmm_107 = None
        permute_721 = torch.ops.aten.permute.default(view_1102, [0, 2, 1, 3]);  view_1102 = None
        clone_760 = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
        view_1103 = torch.ops.aten.view.default(clone_760, [8, 576, 768]);  clone_760 = None
        view_1104 = torch.ops.aten.view.default(view_1103, [4608, 768]);  view_1103 = None
        permute_722 = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_226 = torch.ops.aten.addmm.default(arg320_1, view_1104, permute_722);  arg320_1 = view_1104 = permute_722 = None
        view_1105 = torch.ops.aten.view.default(addmm_226, [8, 576, 768]);  addmm_226 = None
        mul_553 = torch.ops.aten.mul.Tensor(arg310_1, view_1105);  arg310_1 = view_1105 = None
        add_499 = torch.ops.aten.add.Tensor(add_494, mul_553);  add_494 = mul_553 = None
        clone_762 = torch.ops.aten.clone.default(add_499, memory_format = torch.contiguous_format)
        var_mean_112 = torch.ops.aten.var_mean.correction(clone_762, [2], correction = 0, keepdim = True)
        getitem_232 = var_mean_112[0]
        getitem_233 = var_mean_112[1];  var_mean_112 = None
        add_500 = torch.ops.aten.add.Tensor(getitem_232, 1e-06);  getitem_232 = None
        rsqrt_112 = torch.ops.aten.rsqrt.default(add_500);  add_500 = None
        sub_166 = torch.ops.aten.sub.Tensor(clone_762, getitem_233);  clone_762 = getitem_233 = None
        mul_554 = torch.ops.aten.mul.Tensor(sub_166, rsqrt_112);  sub_166 = rsqrt_112 = None
        mul_555 = torch.ops.aten.mul.Tensor(mul_554, arg322_1);  mul_554 = arg322_1 = None
        add_501 = torch.ops.aten.add.Tensor(mul_555, arg323_1);  mul_555 = arg323_1 = None
        view_1106 = torch.ops.aten.view.default(add_501, [4608, 768]);  add_501 = None
        permute_723 = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        addmm_227 = torch.ops.aten.addmm.default(arg325_1, view_1106, permute_723);  arg325_1 = view_1106 = permute_723 = None
        view_1107 = torch.ops.aten.view.default(addmm_227, [8, 576, 3072]);  addmm_227 = None
        mul_556 = torch.ops.aten.mul.Tensor(view_1107, 0.5)
        mul_557 = torch.ops.aten.mul.Tensor(view_1107, 0.7071067811865476);  view_1107 = None
        erf_55 = torch.ops.aten.erf.default(mul_557);  mul_557 = None
        add_502 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_558 = torch.ops.aten.mul.Tensor(mul_556, add_502);  mul_556 = add_502 = None
        view_1108 = torch.ops.aten.view.default(mul_558, [4608, 3072]);  mul_558 = None
        permute_724 = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_228 = torch.ops.aten.addmm.default(arg327_1, view_1108, permute_724);  arg327_1 = view_1108 = permute_724 = None
        view_1109 = torch.ops.aten.view.default(addmm_228, [8, 576, 768]);  addmm_228 = None
        mul_559 = torch.ops.aten.mul.Tensor(arg321_1, view_1109);  arg321_1 = view_1109 = None
        add_503 = torch.ops.aten.add.Tensor(add_499, mul_559);  add_499 = mul_559 = None
        clone_765 = torch.ops.aten.clone.default(add_503, memory_format = torch.contiguous_format)
        var_mean_113 = torch.ops.aten.var_mean.correction(clone_765, [2], correction = 0, keepdim = True)
        getitem_234 = var_mean_113[0]
        getitem_235 = var_mean_113[1];  var_mean_113 = None
        add_504 = torch.ops.aten.add.Tensor(getitem_234, 1e-06);  getitem_234 = None
        rsqrt_113 = torch.ops.aten.rsqrt.default(add_504);  add_504 = None
        sub_167 = torch.ops.aten.sub.Tensor(clone_765, getitem_235);  clone_765 = getitem_235 = None
        mul_560 = torch.ops.aten.mul.Tensor(sub_167, rsqrt_113);  sub_167 = rsqrt_113 = None
        mul_561 = torch.ops.aten.mul.Tensor(mul_560, arg329_1);  mul_560 = arg329_1 = None
        add_505 = torch.ops.aten.add.Tensor(mul_561, arg330_1);  mul_561 = arg330_1 = None
        view_1110 = torch.ops.aten.view.default(add_505, [4608, 768]);  add_505 = None
        permute_725 = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_229 = torch.ops.aten.addmm.default(arg332_1, view_1110, permute_725);  arg332_1 = view_1110 = permute_725 = None
        view_1111 = torch.ops.aten.view.default(addmm_229, [8, 576, 2304]);  addmm_229 = None
        view_1112 = torch.ops.aten.view.default(view_1111, [8, 576, 3, 16, 48]);  view_1111 = None
        permute_726 = torch.ops.aten.permute.default(view_1112, [2, 0, 3, 1, 4]);  view_1112 = None
        select_165 = torch.ops.aten.select.int(permute_726, 0, 0)
        mul_562 = torch.ops.aten.mul.Tensor(select_165, 0.14433756729740643);  select_165 = None
        select_166 = torch.ops.aten.select.int(permute_726, 0, 1)
        select_167 = torch.ops.aten.select.int(permute_726, 0, 2);  permute_726 = None
        permute_727 = torch.ops.aten.permute.default(select_166, [0, 1, 3, 2]);  select_166 = None
        expand_217 = torch.ops.aten.expand.default(mul_562, [8, 16, 576, 48]);  mul_562 = None
        clone_766 = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_1113 = torch.ops.aten.view.default(clone_766, [128, 576, 48]);  clone_766 = None
        expand_218 = torch.ops.aten.expand.default(permute_727, [8, 16, 48, 576]);  permute_727 = None
        clone_767 = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_1114 = torch.ops.aten.view.default(clone_767, [128, 48, 576]);  clone_767 = None
        bmm_108 = torch.ops.aten.bmm.default(view_1113, view_1114);  view_1113 = view_1114 = None
        view_1115 = torch.ops.aten.view.default(bmm_108, [8, 16, 576, 576]);  bmm_108 = None
        permute_728 = torch.ops.aten.permute.default(view_1115, [0, 2, 3, 1]);  view_1115 = None
        permute_729 = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        clone_768 = torch.ops.aten.clone.default(permute_728, memory_format = torch.contiguous_format);  permute_728 = None
        view_1116 = torch.ops.aten.view.default(clone_768, [2654208, 16]);  clone_768 = None
        mm_108 = torch.ops.aten.mm.default(view_1116, permute_729);  view_1116 = permute_729 = None
        view_1117 = torch.ops.aten.view.default(mm_108, [8, 576, 576, 16]);  mm_108 = None
        add_506 = torch.ops.aten.add.Tensor(view_1117, arg334_1);  view_1117 = arg334_1 = None
        permute_730 = torch.ops.aten.permute.default(add_506, [0, 3, 1, 2]);  add_506 = None
        clone_769 = torch.ops.aten.clone.default(permute_730, memory_format = torch.contiguous_format);  permute_730 = None
        amax_54 = torch.ops.aten.amax.default(clone_769, [-1], True)
        sub_168 = torch.ops.aten.sub.Tensor(clone_769, amax_54);  clone_769 = amax_54 = None
        exp_54 = torch.ops.aten.exp.default(sub_168);  sub_168 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(exp_54, [-1], True)
        div_54 = torch.ops.aten.div.Tensor(exp_54, sum_55);  exp_54 = sum_55 = None
        permute_731 = torch.ops.aten.permute.default(div_54, [0, 2, 3, 1]);  div_54 = None
        permute_732 = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        clone_770 = torch.ops.aten.clone.default(permute_731, memory_format = torch.contiguous_format);  permute_731 = None
        view_1118 = torch.ops.aten.view.default(clone_770, [2654208, 16]);  clone_770 = None
        mm_109 = torch.ops.aten.mm.default(view_1118, permute_732);  view_1118 = permute_732 = None
        view_1119 = torch.ops.aten.view.default(mm_109, [8, 576, 576, 16]);  mm_109 = None
        add_507 = torch.ops.aten.add.Tensor(view_1119, arg336_1);  view_1119 = arg336_1 = None
        permute_733 = torch.ops.aten.permute.default(add_507, [0, 3, 1, 2]);  add_507 = None
        expand_219 = torch.ops.aten.expand.default(permute_733, [8, 16, 576, 576]);  permute_733 = None
        clone_772 = torch.ops.aten.clone.default(expand_219, memory_format = torch.contiguous_format);  expand_219 = None
        view_1120 = torch.ops.aten.view.default(clone_772, [128, 576, 576]);  clone_772 = None
        expand_220 = torch.ops.aten.expand.default(select_167, [8, 16, 576, 48]);  select_167 = None
        clone_773 = torch.ops.aten.clone.default(expand_220, memory_format = torch.contiguous_format);  expand_220 = None
        view_1121 = torch.ops.aten.view.default(clone_773, [128, 576, 48]);  clone_773 = None
        bmm_109 = torch.ops.aten.bmm.default(view_1120, view_1121);  view_1120 = view_1121 = None
        view_1122 = torch.ops.aten.view.default(bmm_109, [8, 16, 576, 48]);  bmm_109 = None
        permute_734 = torch.ops.aten.permute.default(view_1122, [0, 2, 1, 3]);  view_1122 = None
        clone_774 = torch.ops.aten.clone.default(permute_734, memory_format = torch.contiguous_format);  permute_734 = None
        view_1123 = torch.ops.aten.view.default(clone_774, [8, 576, 768]);  clone_774 = None
        view_1124 = torch.ops.aten.view.default(view_1123, [4608, 768]);  view_1123 = None
        permute_735 = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_230 = torch.ops.aten.addmm.default(arg338_1, view_1124, permute_735);  arg338_1 = view_1124 = permute_735 = None
        view_1125 = torch.ops.aten.view.default(addmm_230, [8, 576, 768]);  addmm_230 = None
        mul_563 = torch.ops.aten.mul.Tensor(arg328_1, view_1125);  arg328_1 = view_1125 = None
        add_508 = torch.ops.aten.add.Tensor(add_503, mul_563);  add_503 = mul_563 = None
        clone_776 = torch.ops.aten.clone.default(add_508, memory_format = torch.contiguous_format)
        var_mean_114 = torch.ops.aten.var_mean.correction(clone_776, [2], correction = 0, keepdim = True)
        getitem_236 = var_mean_114[0]
        getitem_237 = var_mean_114[1];  var_mean_114 = None
        add_509 = torch.ops.aten.add.Tensor(getitem_236, 1e-06);  getitem_236 = None
        rsqrt_114 = torch.ops.aten.rsqrt.default(add_509);  add_509 = None
        sub_169 = torch.ops.aten.sub.Tensor(clone_776, getitem_237);  clone_776 = getitem_237 = None
        mul_564 = torch.ops.aten.mul.Tensor(sub_169, rsqrt_114);  sub_169 = rsqrt_114 = None
        mul_565 = torch.ops.aten.mul.Tensor(mul_564, arg340_1);  mul_564 = arg340_1 = None
        add_510 = torch.ops.aten.add.Tensor(mul_565, arg341_1);  mul_565 = arg341_1 = None
        view_1126 = torch.ops.aten.view.default(add_510, [4608, 768]);  add_510 = None
        permute_736 = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
        addmm_231 = torch.ops.aten.addmm.default(arg343_1, view_1126, permute_736);  arg343_1 = view_1126 = permute_736 = None
        view_1127 = torch.ops.aten.view.default(addmm_231, [8, 576, 3072]);  addmm_231 = None
        mul_566 = torch.ops.aten.mul.Tensor(view_1127, 0.5)
        mul_567 = torch.ops.aten.mul.Tensor(view_1127, 0.7071067811865476);  view_1127 = None
        erf_56 = torch.ops.aten.erf.default(mul_567);  mul_567 = None
        add_511 = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_568 = torch.ops.aten.mul.Tensor(mul_566, add_511);  mul_566 = add_511 = None
        view_1128 = torch.ops.aten.view.default(mul_568, [4608, 3072]);  mul_568 = None
        permute_737 = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        addmm_232 = torch.ops.aten.addmm.default(arg345_1, view_1128, permute_737);  arg345_1 = view_1128 = permute_737 = None
        view_1129 = torch.ops.aten.view.default(addmm_232, [8, 576, 768]);  addmm_232 = None
        mul_569 = torch.ops.aten.mul.Tensor(arg339_1, view_1129);  arg339_1 = view_1129 = None
        add_512 = torch.ops.aten.add.Tensor(add_508, mul_569);  add_508 = mul_569 = None
        clone_779 = torch.ops.aten.clone.default(add_512, memory_format = torch.contiguous_format)
        var_mean_115 = torch.ops.aten.var_mean.correction(clone_779, [2], correction = 0, keepdim = True)
        getitem_238 = var_mean_115[0]
        getitem_239 = var_mean_115[1];  var_mean_115 = None
        add_513 = torch.ops.aten.add.Tensor(getitem_238, 1e-06);  getitem_238 = None
        rsqrt_115 = torch.ops.aten.rsqrt.default(add_513);  add_513 = None
        sub_170 = torch.ops.aten.sub.Tensor(clone_779, getitem_239);  clone_779 = getitem_239 = None
        mul_570 = torch.ops.aten.mul.Tensor(sub_170, rsqrt_115);  sub_170 = rsqrt_115 = None
        mul_571 = torch.ops.aten.mul.Tensor(mul_570, arg347_1);  mul_570 = arg347_1 = None
        add_514 = torch.ops.aten.add.Tensor(mul_571, arg348_1);  mul_571 = arg348_1 = None
        view_1130 = torch.ops.aten.view.default(add_514, [4608, 768]);  add_514 = None
        permute_738 = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_233 = torch.ops.aten.addmm.default(arg350_1, view_1130, permute_738);  arg350_1 = view_1130 = permute_738 = None
        view_1131 = torch.ops.aten.view.default(addmm_233, [8, 576, 2304]);  addmm_233 = None
        view_1132 = torch.ops.aten.view.default(view_1131, [8, 576, 3, 16, 48]);  view_1131 = None
        permute_739 = torch.ops.aten.permute.default(view_1132, [2, 0, 3, 1, 4]);  view_1132 = None
        select_168 = torch.ops.aten.select.int(permute_739, 0, 0)
        mul_572 = torch.ops.aten.mul.Tensor(select_168, 0.14433756729740643);  select_168 = None
        select_169 = torch.ops.aten.select.int(permute_739, 0, 1)
        select_170 = torch.ops.aten.select.int(permute_739, 0, 2);  permute_739 = None
        permute_740 = torch.ops.aten.permute.default(select_169, [0, 1, 3, 2]);  select_169 = None
        expand_221 = torch.ops.aten.expand.default(mul_572, [8, 16, 576, 48]);  mul_572 = None
        clone_780 = torch.ops.aten.clone.default(expand_221, memory_format = torch.contiguous_format);  expand_221 = None
        view_1133 = torch.ops.aten.view.default(clone_780, [128, 576, 48]);  clone_780 = None
        expand_222 = torch.ops.aten.expand.default(permute_740, [8, 16, 48, 576]);  permute_740 = None
        clone_781 = torch.ops.aten.clone.default(expand_222, memory_format = torch.contiguous_format);  expand_222 = None
        view_1134 = torch.ops.aten.view.default(clone_781, [128, 48, 576]);  clone_781 = None
        bmm_110 = torch.ops.aten.bmm.default(view_1133, view_1134);  view_1133 = view_1134 = None
        view_1135 = torch.ops.aten.view.default(bmm_110, [8, 16, 576, 576]);  bmm_110 = None
        permute_741 = torch.ops.aten.permute.default(view_1135, [0, 2, 3, 1]);  view_1135 = None
        permute_742 = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        clone_782 = torch.ops.aten.clone.default(permute_741, memory_format = torch.contiguous_format);  permute_741 = None
        view_1136 = torch.ops.aten.view.default(clone_782, [2654208, 16]);  clone_782 = None
        mm_110 = torch.ops.aten.mm.default(view_1136, permute_742);  view_1136 = permute_742 = None
        view_1137 = torch.ops.aten.view.default(mm_110, [8, 576, 576, 16]);  mm_110 = None
        add_515 = torch.ops.aten.add.Tensor(view_1137, arg352_1);  view_1137 = arg352_1 = None
        permute_743 = torch.ops.aten.permute.default(add_515, [0, 3, 1, 2]);  add_515 = None
        clone_783 = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
        amax_55 = torch.ops.aten.amax.default(clone_783, [-1], True)
        sub_171 = torch.ops.aten.sub.Tensor(clone_783, amax_55);  clone_783 = amax_55 = None
        exp_55 = torch.ops.aten.exp.default(sub_171);  sub_171 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(exp_55, [-1], True)
        div_55 = torch.ops.aten.div.Tensor(exp_55, sum_56);  exp_55 = sum_56 = None
        permute_744 = torch.ops.aten.permute.default(div_55, [0, 2, 3, 1]);  div_55 = None
        permute_745 = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        clone_784 = torch.ops.aten.clone.default(permute_744, memory_format = torch.contiguous_format);  permute_744 = None
        view_1138 = torch.ops.aten.view.default(clone_784, [2654208, 16]);  clone_784 = None
        mm_111 = torch.ops.aten.mm.default(view_1138, permute_745);  view_1138 = permute_745 = None
        view_1139 = torch.ops.aten.view.default(mm_111, [8, 576, 576, 16]);  mm_111 = None
        add_516 = torch.ops.aten.add.Tensor(view_1139, arg354_1);  view_1139 = arg354_1 = None
        permute_746 = torch.ops.aten.permute.default(add_516, [0, 3, 1, 2]);  add_516 = None
        expand_223 = torch.ops.aten.expand.default(permute_746, [8, 16, 576, 576]);  permute_746 = None
        clone_786 = torch.ops.aten.clone.default(expand_223, memory_format = torch.contiguous_format);  expand_223 = None
        view_1140 = torch.ops.aten.view.default(clone_786, [128, 576, 576]);  clone_786 = None
        expand_224 = torch.ops.aten.expand.default(select_170, [8, 16, 576, 48]);  select_170 = None
        clone_787 = torch.ops.aten.clone.default(expand_224, memory_format = torch.contiguous_format);  expand_224 = None
        view_1141 = torch.ops.aten.view.default(clone_787, [128, 576, 48]);  clone_787 = None
        bmm_111 = torch.ops.aten.bmm.default(view_1140, view_1141);  view_1140 = view_1141 = None
        view_1142 = torch.ops.aten.view.default(bmm_111, [8, 16, 576, 48]);  bmm_111 = None
        permute_747 = torch.ops.aten.permute.default(view_1142, [0, 2, 1, 3]);  view_1142 = None
        clone_788 = torch.ops.aten.clone.default(permute_747, memory_format = torch.contiguous_format);  permute_747 = None
        view_1143 = torch.ops.aten.view.default(clone_788, [8, 576, 768]);  clone_788 = None
        view_1144 = torch.ops.aten.view.default(view_1143, [4608, 768]);  view_1143 = None
        permute_748 = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        addmm_234 = torch.ops.aten.addmm.default(arg356_1, view_1144, permute_748);  arg356_1 = view_1144 = permute_748 = None
        view_1145 = torch.ops.aten.view.default(addmm_234, [8, 576, 768]);  addmm_234 = None
        mul_573 = torch.ops.aten.mul.Tensor(arg346_1, view_1145);  arg346_1 = view_1145 = None
        add_517 = torch.ops.aten.add.Tensor(add_512, mul_573);  add_512 = mul_573 = None
        clone_790 = torch.ops.aten.clone.default(add_517, memory_format = torch.contiguous_format)
        var_mean_116 = torch.ops.aten.var_mean.correction(clone_790, [2], correction = 0, keepdim = True)
        getitem_240 = var_mean_116[0]
        getitem_241 = var_mean_116[1];  var_mean_116 = None
        add_518 = torch.ops.aten.add.Tensor(getitem_240, 1e-06);  getitem_240 = None
        rsqrt_116 = torch.ops.aten.rsqrt.default(add_518);  add_518 = None
        sub_172 = torch.ops.aten.sub.Tensor(clone_790, getitem_241);  clone_790 = getitem_241 = None
        mul_574 = torch.ops.aten.mul.Tensor(sub_172, rsqrt_116);  sub_172 = rsqrt_116 = None
        mul_575 = torch.ops.aten.mul.Tensor(mul_574, arg358_1);  mul_574 = arg358_1 = None
        add_519 = torch.ops.aten.add.Tensor(mul_575, arg359_1);  mul_575 = arg359_1 = None
        view_1146 = torch.ops.aten.view.default(add_519, [4608, 768]);  add_519 = None
        permute_749 = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_235 = torch.ops.aten.addmm.default(arg361_1, view_1146, permute_749);  arg361_1 = view_1146 = permute_749 = None
        view_1147 = torch.ops.aten.view.default(addmm_235, [8, 576, 3072]);  addmm_235 = None
        mul_576 = torch.ops.aten.mul.Tensor(view_1147, 0.5)
        mul_577 = torch.ops.aten.mul.Tensor(view_1147, 0.7071067811865476);  view_1147 = None
        erf_57 = torch.ops.aten.erf.default(mul_577);  mul_577 = None
        add_520 = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_578 = torch.ops.aten.mul.Tensor(mul_576, add_520);  mul_576 = add_520 = None
        view_1148 = torch.ops.aten.view.default(mul_578, [4608, 3072]);  mul_578 = None
        permute_750 = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        addmm_236 = torch.ops.aten.addmm.default(arg363_1, view_1148, permute_750);  arg363_1 = view_1148 = permute_750 = None
        view_1149 = torch.ops.aten.view.default(addmm_236, [8, 576, 768]);  addmm_236 = None
        mul_579 = torch.ops.aten.mul.Tensor(arg357_1, view_1149);  arg357_1 = view_1149 = None
        add_521 = torch.ops.aten.add.Tensor(add_517, mul_579);  add_517 = mul_579 = None
        clone_793 = torch.ops.aten.clone.default(add_521, memory_format = torch.contiguous_format)
        var_mean_117 = torch.ops.aten.var_mean.correction(clone_793, [2], correction = 0, keepdim = True)
        getitem_242 = var_mean_117[0]
        getitem_243 = var_mean_117[1];  var_mean_117 = None
        add_522 = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_117 = torch.ops.aten.rsqrt.default(add_522);  add_522 = None
        sub_173 = torch.ops.aten.sub.Tensor(clone_793, getitem_243);  clone_793 = getitem_243 = None
        mul_580 = torch.ops.aten.mul.Tensor(sub_173, rsqrt_117);  sub_173 = rsqrt_117 = None
        mul_581 = torch.ops.aten.mul.Tensor(mul_580, arg365_1);  mul_580 = arg365_1 = None
        add_523 = torch.ops.aten.add.Tensor(mul_581, arg366_1);  mul_581 = arg366_1 = None
        view_1150 = torch.ops.aten.view.default(add_523, [4608, 768]);  add_523 = None
        permute_751 = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_237 = torch.ops.aten.addmm.default(arg368_1, view_1150, permute_751);  arg368_1 = view_1150 = permute_751 = None
        view_1151 = torch.ops.aten.view.default(addmm_237, [8, 576, 2304]);  addmm_237 = None
        view_1152 = torch.ops.aten.view.default(view_1151, [8, 576, 3, 16, 48]);  view_1151 = None
        permute_752 = torch.ops.aten.permute.default(view_1152, [2, 0, 3, 1, 4]);  view_1152 = None
        select_171 = torch.ops.aten.select.int(permute_752, 0, 0)
        mul_582 = torch.ops.aten.mul.Tensor(select_171, 0.14433756729740643);  select_171 = None
        select_172 = torch.ops.aten.select.int(permute_752, 0, 1)
        select_173 = torch.ops.aten.select.int(permute_752, 0, 2);  permute_752 = None
        permute_753 = torch.ops.aten.permute.default(select_172, [0, 1, 3, 2]);  select_172 = None
        expand_225 = torch.ops.aten.expand.default(mul_582, [8, 16, 576, 48]);  mul_582 = None
        clone_794 = torch.ops.aten.clone.default(expand_225, memory_format = torch.contiguous_format);  expand_225 = None
        view_1153 = torch.ops.aten.view.default(clone_794, [128, 576, 48]);  clone_794 = None
        expand_226 = torch.ops.aten.expand.default(permute_753, [8, 16, 48, 576]);  permute_753 = None
        clone_795 = torch.ops.aten.clone.default(expand_226, memory_format = torch.contiguous_format);  expand_226 = None
        view_1154 = torch.ops.aten.view.default(clone_795, [128, 48, 576]);  clone_795 = None
        bmm_112 = torch.ops.aten.bmm.default(view_1153, view_1154);  view_1153 = view_1154 = None
        view_1155 = torch.ops.aten.view.default(bmm_112, [8, 16, 576, 576]);  bmm_112 = None
        permute_754 = torch.ops.aten.permute.default(view_1155, [0, 2, 3, 1]);  view_1155 = None
        permute_755 = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        clone_796 = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_1156 = torch.ops.aten.view.default(clone_796, [2654208, 16]);  clone_796 = None
        mm_112 = torch.ops.aten.mm.default(view_1156, permute_755);  view_1156 = permute_755 = None
        view_1157 = torch.ops.aten.view.default(mm_112, [8, 576, 576, 16]);  mm_112 = None
        add_524 = torch.ops.aten.add.Tensor(view_1157, arg370_1);  view_1157 = arg370_1 = None
        permute_756 = torch.ops.aten.permute.default(add_524, [0, 3, 1, 2]);  add_524 = None
        clone_797 = torch.ops.aten.clone.default(permute_756, memory_format = torch.contiguous_format);  permute_756 = None
        amax_56 = torch.ops.aten.amax.default(clone_797, [-1], True)
        sub_174 = torch.ops.aten.sub.Tensor(clone_797, amax_56);  clone_797 = amax_56 = None
        exp_56 = torch.ops.aten.exp.default(sub_174);  sub_174 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(exp_56, [-1], True)
        div_56 = torch.ops.aten.div.Tensor(exp_56, sum_57);  exp_56 = sum_57 = None
        permute_757 = torch.ops.aten.permute.default(div_56, [0, 2, 3, 1]);  div_56 = None
        permute_758 = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        clone_798 = torch.ops.aten.clone.default(permute_757, memory_format = torch.contiguous_format);  permute_757 = None
        view_1158 = torch.ops.aten.view.default(clone_798, [2654208, 16]);  clone_798 = None
        mm_113 = torch.ops.aten.mm.default(view_1158, permute_758);  view_1158 = permute_758 = None
        view_1159 = torch.ops.aten.view.default(mm_113, [8, 576, 576, 16]);  mm_113 = None
        add_525 = torch.ops.aten.add.Tensor(view_1159, arg372_1);  view_1159 = arg372_1 = None
        permute_759 = torch.ops.aten.permute.default(add_525, [0, 3, 1, 2]);  add_525 = None
        expand_227 = torch.ops.aten.expand.default(permute_759, [8, 16, 576, 576]);  permute_759 = None
        clone_800 = torch.ops.aten.clone.default(expand_227, memory_format = torch.contiguous_format);  expand_227 = None
        view_1160 = torch.ops.aten.view.default(clone_800, [128, 576, 576]);  clone_800 = None
        expand_228 = torch.ops.aten.expand.default(select_173, [8, 16, 576, 48]);  select_173 = None
        clone_801 = torch.ops.aten.clone.default(expand_228, memory_format = torch.contiguous_format);  expand_228 = None
        view_1161 = torch.ops.aten.view.default(clone_801, [128, 576, 48]);  clone_801 = None
        bmm_113 = torch.ops.aten.bmm.default(view_1160, view_1161);  view_1160 = view_1161 = None
        view_1162 = torch.ops.aten.view.default(bmm_113, [8, 16, 576, 48]);  bmm_113 = None
        permute_760 = torch.ops.aten.permute.default(view_1162, [0, 2, 1, 3]);  view_1162 = None
        clone_802 = torch.ops.aten.clone.default(permute_760, memory_format = torch.contiguous_format);  permute_760 = None
        view_1163 = torch.ops.aten.view.default(clone_802, [8, 576, 768]);  clone_802 = None
        view_1164 = torch.ops.aten.view.default(view_1163, [4608, 768]);  view_1163 = None
        permute_761 = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_238 = torch.ops.aten.addmm.default(arg374_1, view_1164, permute_761);  arg374_1 = view_1164 = permute_761 = None
        view_1165 = torch.ops.aten.view.default(addmm_238, [8, 576, 768]);  addmm_238 = None
        mul_583 = torch.ops.aten.mul.Tensor(arg364_1, view_1165);  arg364_1 = view_1165 = None
        add_526 = torch.ops.aten.add.Tensor(add_521, mul_583);  add_521 = mul_583 = None
        clone_804 = torch.ops.aten.clone.default(add_526, memory_format = torch.contiguous_format)
        var_mean_118 = torch.ops.aten.var_mean.correction(clone_804, [2], correction = 0, keepdim = True)
        getitem_244 = var_mean_118[0]
        getitem_245 = var_mean_118[1];  var_mean_118 = None
        add_527 = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
        rsqrt_118 = torch.ops.aten.rsqrt.default(add_527);  add_527 = None
        sub_175 = torch.ops.aten.sub.Tensor(clone_804, getitem_245);  clone_804 = getitem_245 = None
        mul_584 = torch.ops.aten.mul.Tensor(sub_175, rsqrt_118);  sub_175 = rsqrt_118 = None
        mul_585 = torch.ops.aten.mul.Tensor(mul_584, arg376_1);  mul_584 = arg376_1 = None
        add_528 = torch.ops.aten.add.Tensor(mul_585, arg377_1);  mul_585 = arg377_1 = None
        view_1166 = torch.ops.aten.view.default(add_528, [4608, 768]);  add_528 = None
        permute_762 = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_239 = torch.ops.aten.addmm.default(arg379_1, view_1166, permute_762);  arg379_1 = view_1166 = permute_762 = None
        view_1167 = torch.ops.aten.view.default(addmm_239, [8, 576, 3072]);  addmm_239 = None
        mul_586 = torch.ops.aten.mul.Tensor(view_1167, 0.5)
        mul_587 = torch.ops.aten.mul.Tensor(view_1167, 0.7071067811865476);  view_1167 = None
        erf_58 = torch.ops.aten.erf.default(mul_587);  mul_587 = None
        add_529 = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_588 = torch.ops.aten.mul.Tensor(mul_586, add_529);  mul_586 = add_529 = None
        view_1168 = torch.ops.aten.view.default(mul_588, [4608, 3072]);  mul_588 = None
        permute_763 = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_240 = torch.ops.aten.addmm.default(arg381_1, view_1168, permute_763);  arg381_1 = view_1168 = permute_763 = None
        view_1169 = torch.ops.aten.view.default(addmm_240, [8, 576, 768]);  addmm_240 = None
        mul_589 = torch.ops.aten.mul.Tensor(arg375_1, view_1169);  arg375_1 = view_1169 = None
        add_530 = torch.ops.aten.add.Tensor(add_526, mul_589);  add_526 = mul_589 = None
        clone_807 = torch.ops.aten.clone.default(add_530, memory_format = torch.contiguous_format)
        var_mean_119 = torch.ops.aten.var_mean.correction(clone_807, [2], correction = 0, keepdim = True)
        getitem_246 = var_mean_119[0]
        getitem_247 = var_mean_119[1];  var_mean_119 = None
        add_531 = torch.ops.aten.add.Tensor(getitem_246, 1e-06);  getitem_246 = None
        rsqrt_119 = torch.ops.aten.rsqrt.default(add_531);  add_531 = None
        sub_176 = torch.ops.aten.sub.Tensor(clone_807, getitem_247);  clone_807 = getitem_247 = None
        mul_590 = torch.ops.aten.mul.Tensor(sub_176, rsqrt_119);  sub_176 = rsqrt_119 = None
        mul_591 = torch.ops.aten.mul.Tensor(mul_590, arg383_1);  mul_590 = arg383_1 = None
        add_532 = torch.ops.aten.add.Tensor(mul_591, arg384_1);  mul_591 = arg384_1 = None
        view_1170 = torch.ops.aten.view.default(add_532, [4608, 768]);  add_532 = None
        permute_764 = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_241 = torch.ops.aten.addmm.default(arg386_1, view_1170, permute_764);  arg386_1 = view_1170 = permute_764 = None
        view_1171 = torch.ops.aten.view.default(addmm_241, [8, 576, 2304]);  addmm_241 = None
        view_1172 = torch.ops.aten.view.default(view_1171, [8, 576, 3, 16, 48]);  view_1171 = None
        permute_765 = torch.ops.aten.permute.default(view_1172, [2, 0, 3, 1, 4]);  view_1172 = None
        select_174 = torch.ops.aten.select.int(permute_765, 0, 0)
        mul_592 = torch.ops.aten.mul.Tensor(select_174, 0.14433756729740643);  select_174 = None
        select_175 = torch.ops.aten.select.int(permute_765, 0, 1)
        select_176 = torch.ops.aten.select.int(permute_765, 0, 2);  permute_765 = None
        permute_766 = torch.ops.aten.permute.default(select_175, [0, 1, 3, 2]);  select_175 = None
        expand_229 = torch.ops.aten.expand.default(mul_592, [8, 16, 576, 48]);  mul_592 = None
        clone_808 = torch.ops.aten.clone.default(expand_229, memory_format = torch.contiguous_format);  expand_229 = None
        view_1173 = torch.ops.aten.view.default(clone_808, [128, 576, 48]);  clone_808 = None
        expand_230 = torch.ops.aten.expand.default(permute_766, [8, 16, 48, 576]);  permute_766 = None
        clone_809 = torch.ops.aten.clone.default(expand_230, memory_format = torch.contiguous_format);  expand_230 = None
        view_1174 = torch.ops.aten.view.default(clone_809, [128, 48, 576]);  clone_809 = None
        bmm_114 = torch.ops.aten.bmm.default(view_1173, view_1174);  view_1173 = view_1174 = None
        view_1175 = torch.ops.aten.view.default(bmm_114, [8, 16, 576, 576]);  bmm_114 = None
        permute_767 = torch.ops.aten.permute.default(view_1175, [0, 2, 3, 1]);  view_1175 = None
        permute_768 = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        clone_810 = torch.ops.aten.clone.default(permute_767, memory_format = torch.contiguous_format);  permute_767 = None
        view_1176 = torch.ops.aten.view.default(clone_810, [2654208, 16]);  clone_810 = None
        mm_114 = torch.ops.aten.mm.default(view_1176, permute_768);  view_1176 = permute_768 = None
        view_1177 = torch.ops.aten.view.default(mm_114, [8, 576, 576, 16]);  mm_114 = None
        add_533 = torch.ops.aten.add.Tensor(view_1177, arg388_1);  view_1177 = arg388_1 = None
        permute_769 = torch.ops.aten.permute.default(add_533, [0, 3, 1, 2]);  add_533 = None
        clone_811 = torch.ops.aten.clone.default(permute_769, memory_format = torch.contiguous_format);  permute_769 = None
        amax_57 = torch.ops.aten.amax.default(clone_811, [-1], True)
        sub_177 = torch.ops.aten.sub.Tensor(clone_811, amax_57);  clone_811 = amax_57 = None
        exp_57 = torch.ops.aten.exp.default(sub_177);  sub_177 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(exp_57, [-1], True)
        div_57 = torch.ops.aten.div.Tensor(exp_57, sum_58);  exp_57 = sum_58 = None
        permute_770 = torch.ops.aten.permute.default(div_57, [0, 2, 3, 1]);  div_57 = None
        permute_771 = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        clone_812 = torch.ops.aten.clone.default(permute_770, memory_format = torch.contiguous_format);  permute_770 = None
        view_1178 = torch.ops.aten.view.default(clone_812, [2654208, 16]);  clone_812 = None
        mm_115 = torch.ops.aten.mm.default(view_1178, permute_771);  view_1178 = permute_771 = None
        view_1179 = torch.ops.aten.view.default(mm_115, [8, 576, 576, 16]);  mm_115 = None
        add_534 = torch.ops.aten.add.Tensor(view_1179, arg390_1);  view_1179 = arg390_1 = None
        permute_772 = torch.ops.aten.permute.default(add_534, [0, 3, 1, 2]);  add_534 = None
        expand_231 = torch.ops.aten.expand.default(permute_772, [8, 16, 576, 576]);  permute_772 = None
        clone_814 = torch.ops.aten.clone.default(expand_231, memory_format = torch.contiguous_format);  expand_231 = None
        view_1180 = torch.ops.aten.view.default(clone_814, [128, 576, 576]);  clone_814 = None
        expand_232 = torch.ops.aten.expand.default(select_176, [8, 16, 576, 48]);  select_176 = None
        clone_815 = torch.ops.aten.clone.default(expand_232, memory_format = torch.contiguous_format);  expand_232 = None
        view_1181 = torch.ops.aten.view.default(clone_815, [128, 576, 48]);  clone_815 = None
        bmm_115 = torch.ops.aten.bmm.default(view_1180, view_1181);  view_1180 = view_1181 = None
        view_1182 = torch.ops.aten.view.default(bmm_115, [8, 16, 576, 48]);  bmm_115 = None
        permute_773 = torch.ops.aten.permute.default(view_1182, [0, 2, 1, 3]);  view_1182 = None
        clone_816 = torch.ops.aten.clone.default(permute_773, memory_format = torch.contiguous_format);  permute_773 = None
        view_1183 = torch.ops.aten.view.default(clone_816, [8, 576, 768]);  clone_816 = None
        view_1184 = torch.ops.aten.view.default(view_1183, [4608, 768]);  view_1183 = None
        permute_774 = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
        addmm_242 = torch.ops.aten.addmm.default(arg392_1, view_1184, permute_774);  arg392_1 = view_1184 = permute_774 = None
        view_1185 = torch.ops.aten.view.default(addmm_242, [8, 576, 768]);  addmm_242 = None
        mul_593 = torch.ops.aten.mul.Tensor(arg382_1, view_1185);  arg382_1 = view_1185 = None
        add_535 = torch.ops.aten.add.Tensor(add_530, mul_593);  add_530 = mul_593 = None
        clone_818 = torch.ops.aten.clone.default(add_535, memory_format = torch.contiguous_format)
        var_mean_120 = torch.ops.aten.var_mean.correction(clone_818, [2], correction = 0, keepdim = True)
        getitem_248 = var_mean_120[0]
        getitem_249 = var_mean_120[1];  var_mean_120 = None
        add_536 = torch.ops.aten.add.Tensor(getitem_248, 1e-06);  getitem_248 = None
        rsqrt_120 = torch.ops.aten.rsqrt.default(add_536);  add_536 = None
        sub_178 = torch.ops.aten.sub.Tensor(clone_818, getitem_249);  clone_818 = getitem_249 = None
        mul_594 = torch.ops.aten.mul.Tensor(sub_178, rsqrt_120);  sub_178 = rsqrt_120 = None
        mul_595 = torch.ops.aten.mul.Tensor(mul_594, arg394_1);  mul_594 = arg394_1 = None
        add_537 = torch.ops.aten.add.Tensor(mul_595, arg395_1);  mul_595 = arg395_1 = None
        view_1186 = torch.ops.aten.view.default(add_537, [4608, 768]);  add_537 = None
        permute_775 = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
        addmm_243 = torch.ops.aten.addmm.default(arg397_1, view_1186, permute_775);  arg397_1 = view_1186 = permute_775 = None
        view_1187 = torch.ops.aten.view.default(addmm_243, [8, 576, 3072]);  addmm_243 = None
        mul_596 = torch.ops.aten.mul.Tensor(view_1187, 0.5)
        mul_597 = torch.ops.aten.mul.Tensor(view_1187, 0.7071067811865476);  view_1187 = None
        erf_59 = torch.ops.aten.erf.default(mul_597);  mul_597 = None
        add_538 = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_598 = torch.ops.aten.mul.Tensor(mul_596, add_538);  mul_596 = add_538 = None
        view_1188 = torch.ops.aten.view.default(mul_598, [4608, 3072]);  mul_598 = None
        permute_776 = torch.ops.aten.permute.default(arg398_1, [1, 0]);  arg398_1 = None
        addmm_244 = torch.ops.aten.addmm.default(arg399_1, view_1188, permute_776);  arg399_1 = view_1188 = permute_776 = None
        view_1189 = torch.ops.aten.view.default(addmm_244, [8, 576, 768]);  addmm_244 = None
        mul_599 = torch.ops.aten.mul.Tensor(arg393_1, view_1189);  arg393_1 = view_1189 = None
        add_539 = torch.ops.aten.add.Tensor(add_535, mul_599);  add_535 = mul_599 = None
        clone_821 = torch.ops.aten.clone.default(add_539, memory_format = torch.contiguous_format)
        var_mean_121 = torch.ops.aten.var_mean.correction(clone_821, [2], correction = 0, keepdim = True)
        getitem_250 = var_mean_121[0]
        getitem_251 = var_mean_121[1];  var_mean_121 = None
        add_540 = torch.ops.aten.add.Tensor(getitem_250, 1e-06);  getitem_250 = None
        rsqrt_121 = torch.ops.aten.rsqrt.default(add_540);  add_540 = None
        sub_179 = torch.ops.aten.sub.Tensor(clone_821, getitem_251);  clone_821 = getitem_251 = None
        mul_600 = torch.ops.aten.mul.Tensor(sub_179, rsqrt_121);  sub_179 = rsqrt_121 = None
        mul_601 = torch.ops.aten.mul.Tensor(mul_600, arg401_1);  mul_600 = arg401_1 = None
        add_541 = torch.ops.aten.add.Tensor(mul_601, arg402_1);  mul_601 = arg402_1 = None
        view_1190 = torch.ops.aten.view.default(add_541, [4608, 768]);  add_541 = None
        permute_777 = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        addmm_245 = torch.ops.aten.addmm.default(arg404_1, view_1190, permute_777);  arg404_1 = view_1190 = permute_777 = None
        view_1191 = torch.ops.aten.view.default(addmm_245, [8, 576, 2304]);  addmm_245 = None
        view_1192 = torch.ops.aten.view.default(view_1191, [8, 576, 3, 16, 48]);  view_1191 = None
        permute_778 = torch.ops.aten.permute.default(view_1192, [2, 0, 3, 1, 4]);  view_1192 = None
        select_177 = torch.ops.aten.select.int(permute_778, 0, 0)
        mul_602 = torch.ops.aten.mul.Tensor(select_177, 0.14433756729740643);  select_177 = None
        select_178 = torch.ops.aten.select.int(permute_778, 0, 1)
        select_179 = torch.ops.aten.select.int(permute_778, 0, 2);  permute_778 = None
        permute_779 = torch.ops.aten.permute.default(select_178, [0, 1, 3, 2]);  select_178 = None
        expand_233 = torch.ops.aten.expand.default(mul_602, [8, 16, 576, 48]);  mul_602 = None
        clone_822 = torch.ops.aten.clone.default(expand_233, memory_format = torch.contiguous_format);  expand_233 = None
        view_1193 = torch.ops.aten.view.default(clone_822, [128, 576, 48]);  clone_822 = None
        expand_234 = torch.ops.aten.expand.default(permute_779, [8, 16, 48, 576]);  permute_779 = None
        clone_823 = torch.ops.aten.clone.default(expand_234, memory_format = torch.contiguous_format);  expand_234 = None
        view_1194 = torch.ops.aten.view.default(clone_823, [128, 48, 576]);  clone_823 = None
        bmm_116 = torch.ops.aten.bmm.default(view_1193, view_1194);  view_1193 = view_1194 = None
        view_1195 = torch.ops.aten.view.default(bmm_116, [8, 16, 576, 576]);  bmm_116 = None
        permute_780 = torch.ops.aten.permute.default(view_1195, [0, 2, 3, 1]);  view_1195 = None
        permute_781 = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        clone_824 = torch.ops.aten.clone.default(permute_780, memory_format = torch.contiguous_format);  permute_780 = None
        view_1196 = torch.ops.aten.view.default(clone_824, [2654208, 16]);  clone_824 = None
        mm_116 = torch.ops.aten.mm.default(view_1196, permute_781);  view_1196 = permute_781 = None
        view_1197 = torch.ops.aten.view.default(mm_116, [8, 576, 576, 16]);  mm_116 = None
        add_542 = torch.ops.aten.add.Tensor(view_1197, arg406_1);  view_1197 = arg406_1 = None
        permute_782 = torch.ops.aten.permute.default(add_542, [0, 3, 1, 2]);  add_542 = None
        clone_825 = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
        amax_58 = torch.ops.aten.amax.default(clone_825, [-1], True)
        sub_180 = torch.ops.aten.sub.Tensor(clone_825, amax_58);  clone_825 = amax_58 = None
        exp_58 = torch.ops.aten.exp.default(sub_180);  sub_180 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(exp_58, [-1], True)
        div_58 = torch.ops.aten.div.Tensor(exp_58, sum_59);  exp_58 = sum_59 = None
        permute_783 = torch.ops.aten.permute.default(div_58, [0, 2, 3, 1]);  div_58 = None
        permute_784 = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        clone_826 = torch.ops.aten.clone.default(permute_783, memory_format = torch.contiguous_format);  permute_783 = None
        view_1198 = torch.ops.aten.view.default(clone_826, [2654208, 16]);  clone_826 = None
        mm_117 = torch.ops.aten.mm.default(view_1198, permute_784);  view_1198 = permute_784 = None
        view_1199 = torch.ops.aten.view.default(mm_117, [8, 576, 576, 16]);  mm_117 = None
        add_543 = torch.ops.aten.add.Tensor(view_1199, arg408_1);  view_1199 = arg408_1 = None
        permute_785 = torch.ops.aten.permute.default(add_543, [0, 3, 1, 2]);  add_543 = None
        expand_235 = torch.ops.aten.expand.default(permute_785, [8, 16, 576, 576]);  permute_785 = None
        clone_828 = torch.ops.aten.clone.default(expand_235, memory_format = torch.contiguous_format);  expand_235 = None
        view_1200 = torch.ops.aten.view.default(clone_828, [128, 576, 576]);  clone_828 = None
        expand_236 = torch.ops.aten.expand.default(select_179, [8, 16, 576, 48]);  select_179 = None
        clone_829 = torch.ops.aten.clone.default(expand_236, memory_format = torch.contiguous_format);  expand_236 = None
        view_1201 = torch.ops.aten.view.default(clone_829, [128, 576, 48]);  clone_829 = None
        bmm_117 = torch.ops.aten.bmm.default(view_1200, view_1201);  view_1200 = view_1201 = None
        view_1202 = torch.ops.aten.view.default(bmm_117, [8, 16, 576, 48]);  bmm_117 = None
        permute_786 = torch.ops.aten.permute.default(view_1202, [0, 2, 1, 3]);  view_1202 = None
        clone_830 = torch.ops.aten.clone.default(permute_786, memory_format = torch.contiguous_format);  permute_786 = None
        view_1203 = torch.ops.aten.view.default(clone_830, [8, 576, 768]);  clone_830 = None
        view_1204 = torch.ops.aten.view.default(view_1203, [4608, 768]);  view_1203 = None
        permute_787 = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_246 = torch.ops.aten.addmm.default(arg410_1, view_1204, permute_787);  arg410_1 = view_1204 = permute_787 = None
        view_1205 = torch.ops.aten.view.default(addmm_246, [8, 576, 768]);  addmm_246 = None
        mul_603 = torch.ops.aten.mul.Tensor(arg400_1, view_1205);  arg400_1 = view_1205 = None
        add_544 = torch.ops.aten.add.Tensor(add_539, mul_603);  add_539 = mul_603 = None
        clone_832 = torch.ops.aten.clone.default(add_544, memory_format = torch.contiguous_format)
        var_mean_122 = torch.ops.aten.var_mean.correction(clone_832, [2], correction = 0, keepdim = True)
        getitem_252 = var_mean_122[0]
        getitem_253 = var_mean_122[1];  var_mean_122 = None
        add_545 = torch.ops.aten.add.Tensor(getitem_252, 1e-06);  getitem_252 = None
        rsqrt_122 = torch.ops.aten.rsqrt.default(add_545);  add_545 = None
        sub_181 = torch.ops.aten.sub.Tensor(clone_832, getitem_253);  clone_832 = getitem_253 = None
        mul_604 = torch.ops.aten.mul.Tensor(sub_181, rsqrt_122);  sub_181 = rsqrt_122 = None
        mul_605 = torch.ops.aten.mul.Tensor(mul_604, arg412_1);  mul_604 = arg412_1 = None
        add_546 = torch.ops.aten.add.Tensor(mul_605, arg413_1);  mul_605 = arg413_1 = None
        view_1206 = torch.ops.aten.view.default(add_546, [4608, 768]);  add_546 = None
        permute_788 = torch.ops.aten.permute.default(arg414_1, [1, 0]);  arg414_1 = None
        addmm_247 = torch.ops.aten.addmm.default(arg415_1, view_1206, permute_788);  arg415_1 = view_1206 = permute_788 = None
        view_1207 = torch.ops.aten.view.default(addmm_247, [8, 576, 3072]);  addmm_247 = None
        mul_606 = torch.ops.aten.mul.Tensor(view_1207, 0.5)
        mul_607 = torch.ops.aten.mul.Tensor(view_1207, 0.7071067811865476);  view_1207 = None
        erf_60 = torch.ops.aten.erf.default(mul_607);  mul_607 = None
        add_547 = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_608 = torch.ops.aten.mul.Tensor(mul_606, add_547);  mul_606 = add_547 = None
        view_1208 = torch.ops.aten.view.default(mul_608, [4608, 3072]);  mul_608 = None
        permute_789 = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
        addmm_248 = torch.ops.aten.addmm.default(arg417_1, view_1208, permute_789);  arg417_1 = view_1208 = permute_789 = None
        view_1209 = torch.ops.aten.view.default(addmm_248, [8, 576, 768]);  addmm_248 = None
        mul_609 = torch.ops.aten.mul.Tensor(arg411_1, view_1209);  arg411_1 = view_1209 = None
        add_548 = torch.ops.aten.add.Tensor(add_544, mul_609);  add_544 = mul_609 = None
        clone_835 = torch.ops.aten.clone.default(add_548, memory_format = torch.contiguous_format)
        var_mean_123 = torch.ops.aten.var_mean.correction(clone_835, [2], correction = 0, keepdim = True)
        getitem_254 = var_mean_123[0]
        getitem_255 = var_mean_123[1];  var_mean_123 = None
        add_549 = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_123 = torch.ops.aten.rsqrt.default(add_549);  add_549 = None
        sub_182 = torch.ops.aten.sub.Tensor(clone_835, getitem_255);  clone_835 = getitem_255 = None
        mul_610 = torch.ops.aten.mul.Tensor(sub_182, rsqrt_123);  sub_182 = rsqrt_123 = None
        mul_611 = torch.ops.aten.mul.Tensor(mul_610, arg419_1);  mul_610 = arg419_1 = None
        add_550 = torch.ops.aten.add.Tensor(mul_611, arg420_1);  mul_611 = arg420_1 = None
        view_1210 = torch.ops.aten.view.default(add_550, [4608, 768]);  add_550 = None
        permute_790 = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        addmm_249 = torch.ops.aten.addmm.default(arg422_1, view_1210, permute_790);  arg422_1 = view_1210 = permute_790 = None
        view_1211 = torch.ops.aten.view.default(addmm_249, [8, 576, 2304]);  addmm_249 = None
        view_1212 = torch.ops.aten.view.default(view_1211, [8, 576, 3, 16, 48]);  view_1211 = None
        permute_791 = torch.ops.aten.permute.default(view_1212, [2, 0, 3, 1, 4]);  view_1212 = None
        select_180 = torch.ops.aten.select.int(permute_791, 0, 0)
        mul_612 = torch.ops.aten.mul.Tensor(select_180, 0.14433756729740643);  select_180 = None
        select_181 = torch.ops.aten.select.int(permute_791, 0, 1)
        select_182 = torch.ops.aten.select.int(permute_791, 0, 2);  permute_791 = None
        permute_792 = torch.ops.aten.permute.default(select_181, [0, 1, 3, 2]);  select_181 = None
        expand_237 = torch.ops.aten.expand.default(mul_612, [8, 16, 576, 48]);  mul_612 = None
        clone_836 = torch.ops.aten.clone.default(expand_237, memory_format = torch.contiguous_format);  expand_237 = None
        view_1213 = torch.ops.aten.view.default(clone_836, [128, 576, 48]);  clone_836 = None
        expand_238 = torch.ops.aten.expand.default(permute_792, [8, 16, 48, 576]);  permute_792 = None
        clone_837 = torch.ops.aten.clone.default(expand_238, memory_format = torch.contiguous_format);  expand_238 = None
        view_1214 = torch.ops.aten.view.default(clone_837, [128, 48, 576]);  clone_837 = None
        bmm_118 = torch.ops.aten.bmm.default(view_1213, view_1214);  view_1213 = view_1214 = None
        view_1215 = torch.ops.aten.view.default(bmm_118, [8, 16, 576, 576]);  bmm_118 = None
        permute_793 = torch.ops.aten.permute.default(view_1215, [0, 2, 3, 1]);  view_1215 = None
        permute_794 = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        clone_838 = torch.ops.aten.clone.default(permute_793, memory_format = torch.contiguous_format);  permute_793 = None
        view_1216 = torch.ops.aten.view.default(clone_838, [2654208, 16]);  clone_838 = None
        mm_118 = torch.ops.aten.mm.default(view_1216, permute_794);  view_1216 = permute_794 = None
        view_1217 = torch.ops.aten.view.default(mm_118, [8, 576, 576, 16]);  mm_118 = None
        add_551 = torch.ops.aten.add.Tensor(view_1217, arg424_1);  view_1217 = arg424_1 = None
        permute_795 = torch.ops.aten.permute.default(add_551, [0, 3, 1, 2]);  add_551 = None
        clone_839 = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        amax_59 = torch.ops.aten.amax.default(clone_839, [-1], True)
        sub_183 = torch.ops.aten.sub.Tensor(clone_839, amax_59);  clone_839 = amax_59 = None
        exp_59 = torch.ops.aten.exp.default(sub_183);  sub_183 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(exp_59, [-1], True)
        div_59 = torch.ops.aten.div.Tensor(exp_59, sum_60);  exp_59 = sum_60 = None
        permute_796 = torch.ops.aten.permute.default(div_59, [0, 2, 3, 1]);  div_59 = None
        permute_797 = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        clone_840 = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
        view_1218 = torch.ops.aten.view.default(clone_840, [2654208, 16]);  clone_840 = None
        mm_119 = torch.ops.aten.mm.default(view_1218, permute_797);  view_1218 = permute_797 = None
        view_1219 = torch.ops.aten.view.default(mm_119, [8, 576, 576, 16]);  mm_119 = None
        add_552 = torch.ops.aten.add.Tensor(view_1219, arg426_1);  view_1219 = arg426_1 = None
        permute_798 = torch.ops.aten.permute.default(add_552, [0, 3, 1, 2]);  add_552 = None
        expand_239 = torch.ops.aten.expand.default(permute_798, [8, 16, 576, 576]);  permute_798 = None
        clone_842 = torch.ops.aten.clone.default(expand_239, memory_format = torch.contiguous_format);  expand_239 = None
        view_1220 = torch.ops.aten.view.default(clone_842, [128, 576, 576]);  clone_842 = None
        expand_240 = torch.ops.aten.expand.default(select_182, [8, 16, 576, 48]);  select_182 = None
        clone_843 = torch.ops.aten.clone.default(expand_240, memory_format = torch.contiguous_format);  expand_240 = None
        view_1221 = torch.ops.aten.view.default(clone_843, [128, 576, 48]);  clone_843 = None
        bmm_119 = torch.ops.aten.bmm.default(view_1220, view_1221);  view_1220 = view_1221 = None
        view_1222 = torch.ops.aten.view.default(bmm_119, [8, 16, 576, 48]);  bmm_119 = None
        permute_799 = torch.ops.aten.permute.default(view_1222, [0, 2, 1, 3]);  view_1222 = None
        clone_844 = torch.ops.aten.clone.default(permute_799, memory_format = torch.contiguous_format);  permute_799 = None
        view_1223 = torch.ops.aten.view.default(clone_844, [8, 576, 768]);  clone_844 = None
        view_1224 = torch.ops.aten.view.default(view_1223, [4608, 768]);  view_1223 = None
        permute_800 = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
        addmm_250 = torch.ops.aten.addmm.default(arg428_1, view_1224, permute_800);  arg428_1 = view_1224 = permute_800 = None
        view_1225 = torch.ops.aten.view.default(addmm_250, [8, 576, 768]);  addmm_250 = None
        mul_613 = torch.ops.aten.mul.Tensor(arg418_1, view_1225);  arg418_1 = view_1225 = None
        add_553 = torch.ops.aten.add.Tensor(add_548, mul_613);  add_548 = mul_613 = None
        clone_846 = torch.ops.aten.clone.default(add_553, memory_format = torch.contiguous_format)
        var_mean_124 = torch.ops.aten.var_mean.correction(clone_846, [2], correction = 0, keepdim = True)
        getitem_256 = var_mean_124[0]
        getitem_257 = var_mean_124[1];  var_mean_124 = None
        add_554 = torch.ops.aten.add.Tensor(getitem_256, 1e-06);  getitem_256 = None
        rsqrt_124 = torch.ops.aten.rsqrt.default(add_554);  add_554 = None
        sub_184 = torch.ops.aten.sub.Tensor(clone_846, getitem_257);  clone_846 = getitem_257 = None
        mul_614 = torch.ops.aten.mul.Tensor(sub_184, rsqrt_124);  sub_184 = rsqrt_124 = None
        mul_615 = torch.ops.aten.mul.Tensor(mul_614, arg430_1);  mul_614 = arg430_1 = None
        add_555 = torch.ops.aten.add.Tensor(mul_615, arg431_1);  mul_615 = arg431_1 = None
        view_1226 = torch.ops.aten.view.default(add_555, [4608, 768]);  add_555 = None
        permute_801 = torch.ops.aten.permute.default(arg432_1, [1, 0]);  arg432_1 = None
        addmm_251 = torch.ops.aten.addmm.default(arg433_1, view_1226, permute_801);  arg433_1 = view_1226 = permute_801 = None
        view_1227 = torch.ops.aten.view.default(addmm_251, [8, 576, 3072]);  addmm_251 = None
        mul_616 = torch.ops.aten.mul.Tensor(view_1227, 0.5)
        mul_617 = torch.ops.aten.mul.Tensor(view_1227, 0.7071067811865476);  view_1227 = None
        erf_61 = torch.ops.aten.erf.default(mul_617);  mul_617 = None
        add_556 = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_618 = torch.ops.aten.mul.Tensor(mul_616, add_556);  mul_616 = add_556 = None
        view_1228 = torch.ops.aten.view.default(mul_618, [4608, 3072]);  mul_618 = None
        permute_802 = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
        addmm_252 = torch.ops.aten.addmm.default(arg435_1, view_1228, permute_802);  arg435_1 = view_1228 = permute_802 = None
        view_1229 = torch.ops.aten.view.default(addmm_252, [8, 576, 768]);  addmm_252 = None
        mul_619 = torch.ops.aten.mul.Tensor(arg429_1, view_1229);  arg429_1 = view_1229 = None
        add_557 = torch.ops.aten.add.Tensor(add_553, mul_619);  add_553 = mul_619 = None
        clone_849 = torch.ops.aten.clone.default(add_557, memory_format = torch.contiguous_format)
        var_mean_125 = torch.ops.aten.var_mean.correction(clone_849, [2], correction = 0, keepdim = True)
        getitem_258 = var_mean_125[0]
        getitem_259 = var_mean_125[1];  var_mean_125 = None
        add_558 = torch.ops.aten.add.Tensor(getitem_258, 1e-06);  getitem_258 = None
        rsqrt_125 = torch.ops.aten.rsqrt.default(add_558);  add_558 = None
        sub_185 = torch.ops.aten.sub.Tensor(clone_849, getitem_259);  clone_849 = getitem_259 = None
        mul_620 = torch.ops.aten.mul.Tensor(sub_185, rsqrt_125);  sub_185 = rsqrt_125 = None
        mul_621 = torch.ops.aten.mul.Tensor(mul_620, arg437_1);  mul_620 = arg437_1 = None
        add_559 = torch.ops.aten.add.Tensor(mul_621, arg438_1);  mul_621 = arg438_1 = None
        view_1230 = torch.ops.aten.view.default(add_559, [4608, 768]);  add_559 = None
        permute_803 = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
        addmm_253 = torch.ops.aten.addmm.default(arg440_1, view_1230, permute_803);  arg440_1 = view_1230 = permute_803 = None
        view_1231 = torch.ops.aten.view.default(addmm_253, [8, 576, 2304]);  addmm_253 = None
        view_1232 = torch.ops.aten.view.default(view_1231, [8, 576, 3, 16, 48]);  view_1231 = None
        permute_804 = torch.ops.aten.permute.default(view_1232, [2, 0, 3, 1, 4]);  view_1232 = None
        select_183 = torch.ops.aten.select.int(permute_804, 0, 0)
        mul_622 = torch.ops.aten.mul.Tensor(select_183, 0.14433756729740643);  select_183 = None
        select_184 = torch.ops.aten.select.int(permute_804, 0, 1)
        select_185 = torch.ops.aten.select.int(permute_804, 0, 2);  permute_804 = None
        permute_805 = torch.ops.aten.permute.default(select_184, [0, 1, 3, 2]);  select_184 = None
        expand_241 = torch.ops.aten.expand.default(mul_622, [8, 16, 576, 48]);  mul_622 = None
        clone_850 = torch.ops.aten.clone.default(expand_241, memory_format = torch.contiguous_format);  expand_241 = None
        view_1233 = torch.ops.aten.view.default(clone_850, [128, 576, 48]);  clone_850 = None
        expand_242 = torch.ops.aten.expand.default(permute_805, [8, 16, 48, 576]);  permute_805 = None
        clone_851 = torch.ops.aten.clone.default(expand_242, memory_format = torch.contiguous_format);  expand_242 = None
        view_1234 = torch.ops.aten.view.default(clone_851, [128, 48, 576]);  clone_851 = None
        bmm_120 = torch.ops.aten.bmm.default(view_1233, view_1234);  view_1233 = view_1234 = None
        view_1235 = torch.ops.aten.view.default(bmm_120, [8, 16, 576, 576]);  bmm_120 = None
        permute_806 = torch.ops.aten.permute.default(view_1235, [0, 2, 3, 1]);  view_1235 = None
        permute_807 = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        clone_852 = torch.ops.aten.clone.default(permute_806, memory_format = torch.contiguous_format);  permute_806 = None
        view_1236 = torch.ops.aten.view.default(clone_852, [2654208, 16]);  clone_852 = None
        mm_120 = torch.ops.aten.mm.default(view_1236, permute_807);  view_1236 = permute_807 = None
        view_1237 = torch.ops.aten.view.default(mm_120, [8, 576, 576, 16]);  mm_120 = None
        add_560 = torch.ops.aten.add.Tensor(view_1237, arg442_1);  view_1237 = arg442_1 = None
        permute_808 = torch.ops.aten.permute.default(add_560, [0, 3, 1, 2]);  add_560 = None
        clone_853 = torch.ops.aten.clone.default(permute_808, memory_format = torch.contiguous_format);  permute_808 = None
        amax_60 = torch.ops.aten.amax.default(clone_853, [-1], True)
        sub_186 = torch.ops.aten.sub.Tensor(clone_853, amax_60);  clone_853 = amax_60 = None
        exp_60 = torch.ops.aten.exp.default(sub_186);  sub_186 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(exp_60, [-1], True)
        div_60 = torch.ops.aten.div.Tensor(exp_60, sum_61);  exp_60 = sum_61 = None
        permute_809 = torch.ops.aten.permute.default(div_60, [0, 2, 3, 1]);  div_60 = None
        permute_810 = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        clone_854 = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
        view_1238 = torch.ops.aten.view.default(clone_854, [2654208, 16]);  clone_854 = None
        mm_121 = torch.ops.aten.mm.default(view_1238, permute_810);  view_1238 = permute_810 = None
        view_1239 = torch.ops.aten.view.default(mm_121, [8, 576, 576, 16]);  mm_121 = None
        add_561 = torch.ops.aten.add.Tensor(view_1239, arg444_1);  view_1239 = arg444_1 = None
        permute_811 = torch.ops.aten.permute.default(add_561, [0, 3, 1, 2]);  add_561 = None
        expand_243 = torch.ops.aten.expand.default(permute_811, [8, 16, 576, 576]);  permute_811 = None
        clone_856 = torch.ops.aten.clone.default(expand_243, memory_format = torch.contiguous_format);  expand_243 = None
        view_1240 = torch.ops.aten.view.default(clone_856, [128, 576, 576]);  clone_856 = None
        expand_244 = torch.ops.aten.expand.default(select_185, [8, 16, 576, 48]);  select_185 = None
        clone_857 = torch.ops.aten.clone.default(expand_244, memory_format = torch.contiguous_format);  expand_244 = None
        view_1241 = torch.ops.aten.view.default(clone_857, [128, 576, 48]);  clone_857 = None
        bmm_121 = torch.ops.aten.bmm.default(view_1240, view_1241);  view_1240 = view_1241 = None
        view_1242 = torch.ops.aten.view.default(bmm_121, [8, 16, 576, 48]);  bmm_121 = None
        permute_812 = torch.ops.aten.permute.default(view_1242, [0, 2, 1, 3]);  view_1242 = None
        clone_858 = torch.ops.aten.clone.default(permute_812, memory_format = torch.contiguous_format);  permute_812 = None
        view_1243 = torch.ops.aten.view.default(clone_858, [8, 576, 768]);  clone_858 = None
        view_1244 = torch.ops.aten.view.default(view_1243, [4608, 768]);  view_1243 = None
        permute_813 = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        addmm_254 = torch.ops.aten.addmm.default(arg446_1, view_1244, permute_813);  arg446_1 = view_1244 = permute_813 = None
        view_1245 = torch.ops.aten.view.default(addmm_254, [8, 576, 768]);  addmm_254 = None
        mul_623 = torch.ops.aten.mul.Tensor(arg436_1, view_1245);  arg436_1 = view_1245 = None
        add_562 = torch.ops.aten.add.Tensor(add_557, mul_623);  add_557 = mul_623 = None
        clone_860 = torch.ops.aten.clone.default(add_562, memory_format = torch.contiguous_format)
        var_mean_126 = torch.ops.aten.var_mean.correction(clone_860, [2], correction = 0, keepdim = True)
        getitem_260 = var_mean_126[0]
        getitem_261 = var_mean_126[1];  var_mean_126 = None
        add_563 = torch.ops.aten.add.Tensor(getitem_260, 1e-06);  getitem_260 = None
        rsqrt_126 = torch.ops.aten.rsqrt.default(add_563);  add_563 = None
        sub_187 = torch.ops.aten.sub.Tensor(clone_860, getitem_261);  clone_860 = getitem_261 = None
        mul_624 = torch.ops.aten.mul.Tensor(sub_187, rsqrt_126);  sub_187 = rsqrt_126 = None
        mul_625 = torch.ops.aten.mul.Tensor(mul_624, arg448_1);  mul_624 = arg448_1 = None
        add_564 = torch.ops.aten.add.Tensor(mul_625, arg449_1);  mul_625 = arg449_1 = None
        view_1246 = torch.ops.aten.view.default(add_564, [4608, 768]);  add_564 = None
        permute_814 = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
        addmm_255 = torch.ops.aten.addmm.default(arg451_1, view_1246, permute_814);  arg451_1 = view_1246 = permute_814 = None
        view_1247 = torch.ops.aten.view.default(addmm_255, [8, 576, 3072]);  addmm_255 = None
        mul_626 = torch.ops.aten.mul.Tensor(view_1247, 0.5)
        mul_627 = torch.ops.aten.mul.Tensor(view_1247, 0.7071067811865476);  view_1247 = None
        erf_62 = torch.ops.aten.erf.default(mul_627);  mul_627 = None
        add_565 = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_628 = torch.ops.aten.mul.Tensor(mul_626, add_565);  mul_626 = add_565 = None
        view_1248 = torch.ops.aten.view.default(mul_628, [4608, 3072]);  mul_628 = None
        permute_815 = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
        addmm_256 = torch.ops.aten.addmm.default(arg453_1, view_1248, permute_815);  arg453_1 = view_1248 = permute_815 = None
        view_1249 = torch.ops.aten.view.default(addmm_256, [8, 576, 768]);  addmm_256 = None
        mul_629 = torch.ops.aten.mul.Tensor(arg447_1, view_1249);  arg447_1 = view_1249 = None
        add_566 = torch.ops.aten.add.Tensor(add_562, mul_629);  add_562 = mul_629 = None
        clone_863 = torch.ops.aten.clone.default(add_566, memory_format = torch.contiguous_format)
        var_mean_127 = torch.ops.aten.var_mean.correction(clone_863, [2], correction = 0, keepdim = True)
        getitem_262 = var_mean_127[0]
        getitem_263 = var_mean_127[1];  var_mean_127 = None
        add_567 = torch.ops.aten.add.Tensor(getitem_262, 1e-06);  getitem_262 = None
        rsqrt_127 = torch.ops.aten.rsqrt.default(add_567);  add_567 = None
        sub_188 = torch.ops.aten.sub.Tensor(clone_863, getitem_263);  clone_863 = getitem_263 = None
        mul_630 = torch.ops.aten.mul.Tensor(sub_188, rsqrt_127);  sub_188 = rsqrt_127 = None
        mul_631 = torch.ops.aten.mul.Tensor(mul_630, arg455_1);  mul_630 = arg455_1 = None
        add_568 = torch.ops.aten.add.Tensor(mul_631, arg456_1);  mul_631 = arg456_1 = None
        view_1250 = torch.ops.aten.view.default(add_568, [4608, 768]);  add_568 = None
        permute_816 = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        addmm_257 = torch.ops.aten.addmm.default(arg458_1, view_1250, permute_816);  arg458_1 = view_1250 = permute_816 = None
        view_1251 = torch.ops.aten.view.default(addmm_257, [8, 576, 2304]);  addmm_257 = None
        view_1252 = torch.ops.aten.view.default(view_1251, [8, 576, 3, 16, 48]);  view_1251 = None
        permute_817 = torch.ops.aten.permute.default(view_1252, [2, 0, 3, 1, 4]);  view_1252 = None
        select_186 = torch.ops.aten.select.int(permute_817, 0, 0)
        mul_632 = torch.ops.aten.mul.Tensor(select_186, 0.14433756729740643);  select_186 = None
        select_187 = torch.ops.aten.select.int(permute_817, 0, 1)
        select_188 = torch.ops.aten.select.int(permute_817, 0, 2);  permute_817 = None
        permute_818 = torch.ops.aten.permute.default(select_187, [0, 1, 3, 2]);  select_187 = None
        expand_245 = torch.ops.aten.expand.default(mul_632, [8, 16, 576, 48]);  mul_632 = None
        clone_864 = torch.ops.aten.clone.default(expand_245, memory_format = torch.contiguous_format);  expand_245 = None
        view_1253 = torch.ops.aten.view.default(clone_864, [128, 576, 48]);  clone_864 = None
        expand_246 = torch.ops.aten.expand.default(permute_818, [8, 16, 48, 576]);  permute_818 = None
        clone_865 = torch.ops.aten.clone.default(expand_246, memory_format = torch.contiguous_format);  expand_246 = None
        view_1254 = torch.ops.aten.view.default(clone_865, [128, 48, 576]);  clone_865 = None
        bmm_122 = torch.ops.aten.bmm.default(view_1253, view_1254);  view_1253 = view_1254 = None
        view_1255 = torch.ops.aten.view.default(bmm_122, [8, 16, 576, 576]);  bmm_122 = None
        permute_819 = torch.ops.aten.permute.default(view_1255, [0, 2, 3, 1]);  view_1255 = None
        permute_820 = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        clone_866 = torch.ops.aten.clone.default(permute_819, memory_format = torch.contiguous_format);  permute_819 = None
        view_1256 = torch.ops.aten.view.default(clone_866, [2654208, 16]);  clone_866 = None
        mm_122 = torch.ops.aten.mm.default(view_1256, permute_820);  view_1256 = permute_820 = None
        view_1257 = torch.ops.aten.view.default(mm_122, [8, 576, 576, 16]);  mm_122 = None
        add_569 = torch.ops.aten.add.Tensor(view_1257, arg460_1);  view_1257 = arg460_1 = None
        permute_821 = torch.ops.aten.permute.default(add_569, [0, 3, 1, 2]);  add_569 = None
        clone_867 = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
        amax_61 = torch.ops.aten.amax.default(clone_867, [-1], True)
        sub_189 = torch.ops.aten.sub.Tensor(clone_867, amax_61);  clone_867 = amax_61 = None
        exp_61 = torch.ops.aten.exp.default(sub_189);  sub_189 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(exp_61, [-1], True)
        div_61 = torch.ops.aten.div.Tensor(exp_61, sum_62);  exp_61 = sum_62 = None
        permute_822 = torch.ops.aten.permute.default(div_61, [0, 2, 3, 1]);  div_61 = None
        permute_823 = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        clone_868 = torch.ops.aten.clone.default(permute_822, memory_format = torch.contiguous_format);  permute_822 = None
        view_1258 = torch.ops.aten.view.default(clone_868, [2654208, 16]);  clone_868 = None
        mm_123 = torch.ops.aten.mm.default(view_1258, permute_823);  view_1258 = permute_823 = None
        view_1259 = torch.ops.aten.view.default(mm_123, [8, 576, 576, 16]);  mm_123 = None
        add_570 = torch.ops.aten.add.Tensor(view_1259, arg462_1);  view_1259 = arg462_1 = None
        permute_824 = torch.ops.aten.permute.default(add_570, [0, 3, 1, 2]);  add_570 = None
        expand_247 = torch.ops.aten.expand.default(permute_824, [8, 16, 576, 576]);  permute_824 = None
        clone_870 = torch.ops.aten.clone.default(expand_247, memory_format = torch.contiguous_format);  expand_247 = None
        view_1260 = torch.ops.aten.view.default(clone_870, [128, 576, 576]);  clone_870 = None
        expand_248 = torch.ops.aten.expand.default(select_188, [8, 16, 576, 48]);  select_188 = None
        clone_871 = torch.ops.aten.clone.default(expand_248, memory_format = torch.contiguous_format);  expand_248 = None
        view_1261 = torch.ops.aten.view.default(clone_871, [128, 576, 48]);  clone_871 = None
        bmm_123 = torch.ops.aten.bmm.default(view_1260, view_1261);  view_1260 = view_1261 = None
        view_1262 = torch.ops.aten.view.default(bmm_123, [8, 16, 576, 48]);  bmm_123 = None
        permute_825 = torch.ops.aten.permute.default(view_1262, [0, 2, 1, 3]);  view_1262 = None
        clone_872 = torch.ops.aten.clone.default(permute_825, memory_format = torch.contiguous_format);  permute_825 = None
        view_1263 = torch.ops.aten.view.default(clone_872, [8, 576, 768]);  clone_872 = None
        view_1264 = torch.ops.aten.view.default(view_1263, [4608, 768]);  view_1263 = None
        permute_826 = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_258 = torch.ops.aten.addmm.default(arg464_1, view_1264, permute_826);  arg464_1 = view_1264 = permute_826 = None
        view_1265 = torch.ops.aten.view.default(addmm_258, [8, 576, 768]);  addmm_258 = None
        mul_633 = torch.ops.aten.mul.Tensor(arg454_1, view_1265);  arg454_1 = view_1265 = None
        add_571 = torch.ops.aten.add.Tensor(add_566, mul_633);  add_566 = mul_633 = None
        clone_874 = torch.ops.aten.clone.default(add_571, memory_format = torch.contiguous_format)
        var_mean_128 = torch.ops.aten.var_mean.correction(clone_874, [2], correction = 0, keepdim = True)
        getitem_264 = var_mean_128[0]
        getitem_265 = var_mean_128[1];  var_mean_128 = None
        add_572 = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
        rsqrt_128 = torch.ops.aten.rsqrt.default(add_572);  add_572 = None
        sub_190 = torch.ops.aten.sub.Tensor(clone_874, getitem_265);  clone_874 = getitem_265 = None
        mul_634 = torch.ops.aten.mul.Tensor(sub_190, rsqrt_128);  sub_190 = rsqrt_128 = None
        mul_635 = torch.ops.aten.mul.Tensor(mul_634, arg466_1);  mul_634 = arg466_1 = None
        add_573 = torch.ops.aten.add.Tensor(mul_635, arg467_1);  mul_635 = arg467_1 = None
        view_1266 = torch.ops.aten.view.default(add_573, [4608, 768]);  add_573 = None
        permute_827 = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
        addmm_259 = torch.ops.aten.addmm.default(arg469_1, view_1266, permute_827);  arg469_1 = view_1266 = permute_827 = None
        view_1267 = torch.ops.aten.view.default(addmm_259, [8, 576, 3072]);  addmm_259 = None
        mul_636 = torch.ops.aten.mul.Tensor(view_1267, 0.5)
        mul_637 = torch.ops.aten.mul.Tensor(view_1267, 0.7071067811865476);  view_1267 = None
        erf_63 = torch.ops.aten.erf.default(mul_637);  mul_637 = None
        add_574 = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_638 = torch.ops.aten.mul.Tensor(mul_636, add_574);  mul_636 = add_574 = None
        view_1268 = torch.ops.aten.view.default(mul_638, [4608, 3072]);  mul_638 = None
        permute_828 = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
        addmm_260 = torch.ops.aten.addmm.default(arg471_1, view_1268, permute_828);  arg471_1 = view_1268 = permute_828 = None
        view_1269 = torch.ops.aten.view.default(addmm_260, [8, 576, 768]);  addmm_260 = None
        mul_639 = torch.ops.aten.mul.Tensor(arg465_1, view_1269);  arg465_1 = view_1269 = None
        add_575 = torch.ops.aten.add.Tensor(add_571, mul_639);  add_571 = mul_639 = None
        clone_877 = torch.ops.aten.clone.default(add_575, memory_format = torch.contiguous_format)
        var_mean_129 = torch.ops.aten.var_mean.correction(clone_877, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_129[0]
        getitem_267 = var_mean_129[1];  var_mean_129 = None
        add_576 = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_129 = torch.ops.aten.rsqrt.default(add_576);  add_576 = None
        sub_191 = torch.ops.aten.sub.Tensor(clone_877, getitem_267);  clone_877 = getitem_267 = None
        mul_640 = torch.ops.aten.mul.Tensor(sub_191, rsqrt_129);  sub_191 = rsqrt_129 = None
        mul_641 = torch.ops.aten.mul.Tensor(mul_640, arg473_1);  mul_640 = arg473_1 = None
        add_577 = torch.ops.aten.add.Tensor(mul_641, arg474_1);  mul_641 = arg474_1 = None
        view_1270 = torch.ops.aten.view.default(add_577, [4608, 768]);  add_577 = None
        permute_829 = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_261 = torch.ops.aten.addmm.default(arg476_1, view_1270, permute_829);  arg476_1 = view_1270 = permute_829 = None
        view_1271 = torch.ops.aten.view.default(addmm_261, [8, 576, 2304]);  addmm_261 = None
        view_1272 = torch.ops.aten.view.default(view_1271, [8, 576, 3, 16, 48]);  view_1271 = None
        permute_830 = torch.ops.aten.permute.default(view_1272, [2, 0, 3, 1, 4]);  view_1272 = None
        select_189 = torch.ops.aten.select.int(permute_830, 0, 0)
        mul_642 = torch.ops.aten.mul.Tensor(select_189, 0.14433756729740643);  select_189 = None
        select_190 = torch.ops.aten.select.int(permute_830, 0, 1)
        select_191 = torch.ops.aten.select.int(permute_830, 0, 2);  permute_830 = None
        permute_831 = torch.ops.aten.permute.default(select_190, [0, 1, 3, 2]);  select_190 = None
        expand_249 = torch.ops.aten.expand.default(mul_642, [8, 16, 576, 48]);  mul_642 = None
        clone_878 = torch.ops.aten.clone.default(expand_249, memory_format = torch.contiguous_format);  expand_249 = None
        view_1273 = torch.ops.aten.view.default(clone_878, [128, 576, 48]);  clone_878 = None
        expand_250 = torch.ops.aten.expand.default(permute_831, [8, 16, 48, 576]);  permute_831 = None
        clone_879 = torch.ops.aten.clone.default(expand_250, memory_format = torch.contiguous_format);  expand_250 = None
        view_1274 = torch.ops.aten.view.default(clone_879, [128, 48, 576]);  clone_879 = None
        bmm_124 = torch.ops.aten.bmm.default(view_1273, view_1274);  view_1273 = view_1274 = None
        view_1275 = torch.ops.aten.view.default(bmm_124, [8, 16, 576, 576]);  bmm_124 = None
        permute_832 = torch.ops.aten.permute.default(view_1275, [0, 2, 3, 1]);  view_1275 = None
        permute_833 = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        clone_880 = torch.ops.aten.clone.default(permute_832, memory_format = torch.contiguous_format);  permute_832 = None
        view_1276 = torch.ops.aten.view.default(clone_880, [2654208, 16]);  clone_880 = None
        mm_124 = torch.ops.aten.mm.default(view_1276, permute_833);  view_1276 = permute_833 = None
        view_1277 = torch.ops.aten.view.default(mm_124, [8, 576, 576, 16]);  mm_124 = None
        add_578 = torch.ops.aten.add.Tensor(view_1277, arg478_1);  view_1277 = arg478_1 = None
        permute_834 = torch.ops.aten.permute.default(add_578, [0, 3, 1, 2]);  add_578 = None
        clone_881 = torch.ops.aten.clone.default(permute_834, memory_format = torch.contiguous_format);  permute_834 = None
        amax_62 = torch.ops.aten.amax.default(clone_881, [-1], True)
        sub_192 = torch.ops.aten.sub.Tensor(clone_881, amax_62);  clone_881 = amax_62 = None
        exp_62 = torch.ops.aten.exp.default(sub_192);  sub_192 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(exp_62, [-1], True)
        div_62 = torch.ops.aten.div.Tensor(exp_62, sum_63);  exp_62 = sum_63 = None
        permute_835 = torch.ops.aten.permute.default(div_62, [0, 2, 3, 1]);  div_62 = None
        permute_836 = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        clone_882 = torch.ops.aten.clone.default(permute_835, memory_format = torch.contiguous_format);  permute_835 = None
        view_1278 = torch.ops.aten.view.default(clone_882, [2654208, 16]);  clone_882 = None
        mm_125 = torch.ops.aten.mm.default(view_1278, permute_836);  view_1278 = permute_836 = None
        view_1279 = torch.ops.aten.view.default(mm_125, [8, 576, 576, 16]);  mm_125 = None
        add_579 = torch.ops.aten.add.Tensor(view_1279, arg480_1);  view_1279 = arg480_1 = None
        permute_837 = torch.ops.aten.permute.default(add_579, [0, 3, 1, 2]);  add_579 = None
        expand_251 = torch.ops.aten.expand.default(permute_837, [8, 16, 576, 576]);  permute_837 = None
        clone_884 = torch.ops.aten.clone.default(expand_251, memory_format = torch.contiguous_format);  expand_251 = None
        view_1280 = torch.ops.aten.view.default(clone_884, [128, 576, 576]);  clone_884 = None
        expand_252 = torch.ops.aten.expand.default(select_191, [8, 16, 576, 48]);  select_191 = None
        clone_885 = torch.ops.aten.clone.default(expand_252, memory_format = torch.contiguous_format);  expand_252 = None
        view_1281 = torch.ops.aten.view.default(clone_885, [128, 576, 48]);  clone_885 = None
        bmm_125 = torch.ops.aten.bmm.default(view_1280, view_1281);  view_1280 = view_1281 = None
        view_1282 = torch.ops.aten.view.default(bmm_125, [8, 16, 576, 48]);  bmm_125 = None
        permute_838 = torch.ops.aten.permute.default(view_1282, [0, 2, 1, 3]);  view_1282 = None
        clone_886 = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_1283 = torch.ops.aten.view.default(clone_886, [8, 576, 768]);  clone_886 = None
        view_1284 = torch.ops.aten.view.default(view_1283, [4608, 768]);  view_1283 = None
        permute_839 = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        addmm_262 = torch.ops.aten.addmm.default(arg482_1, view_1284, permute_839);  arg482_1 = view_1284 = permute_839 = None
        view_1285 = torch.ops.aten.view.default(addmm_262, [8, 576, 768]);  addmm_262 = None
        mul_643 = torch.ops.aten.mul.Tensor(arg472_1, view_1285);  arg472_1 = view_1285 = None
        add_580 = torch.ops.aten.add.Tensor(add_575, mul_643);  add_575 = mul_643 = None
        clone_888 = torch.ops.aten.clone.default(add_580, memory_format = torch.contiguous_format)
        var_mean_130 = torch.ops.aten.var_mean.correction(clone_888, [2], correction = 0, keepdim = True)
        getitem_268 = var_mean_130[0]
        getitem_269 = var_mean_130[1];  var_mean_130 = None
        add_581 = torch.ops.aten.add.Tensor(getitem_268, 1e-06);  getitem_268 = None
        rsqrt_130 = torch.ops.aten.rsqrt.default(add_581);  add_581 = None
        sub_193 = torch.ops.aten.sub.Tensor(clone_888, getitem_269);  clone_888 = getitem_269 = None
        mul_644 = torch.ops.aten.mul.Tensor(sub_193, rsqrt_130);  sub_193 = rsqrt_130 = None
        mul_645 = torch.ops.aten.mul.Tensor(mul_644, arg484_1);  mul_644 = arg484_1 = None
        add_582 = torch.ops.aten.add.Tensor(mul_645, arg485_1);  mul_645 = arg485_1 = None
        view_1286 = torch.ops.aten.view.default(add_582, [4608, 768]);  add_582 = None
        permute_840 = torch.ops.aten.permute.default(arg486_1, [1, 0]);  arg486_1 = None
        addmm_263 = torch.ops.aten.addmm.default(arg487_1, view_1286, permute_840);  arg487_1 = view_1286 = permute_840 = None
        view_1287 = torch.ops.aten.view.default(addmm_263, [8, 576, 3072]);  addmm_263 = None
        mul_646 = torch.ops.aten.mul.Tensor(view_1287, 0.5)
        mul_647 = torch.ops.aten.mul.Tensor(view_1287, 0.7071067811865476);  view_1287 = None
        erf_64 = torch.ops.aten.erf.default(mul_647);  mul_647 = None
        add_583 = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_648 = torch.ops.aten.mul.Tensor(mul_646, add_583);  mul_646 = add_583 = None
        view_1288 = torch.ops.aten.view.default(mul_648, [4608, 3072]);  mul_648 = None
        permute_841 = torch.ops.aten.permute.default(arg488_1, [1, 0]);  arg488_1 = None
        addmm_264 = torch.ops.aten.addmm.default(arg489_1, view_1288, permute_841);  arg489_1 = view_1288 = permute_841 = None
        view_1289 = torch.ops.aten.view.default(addmm_264, [8, 576, 768]);  addmm_264 = None
        mul_649 = torch.ops.aten.mul.Tensor(arg483_1, view_1289);  arg483_1 = view_1289 = None
        add_584 = torch.ops.aten.add.Tensor(add_580, mul_649);  add_580 = mul_649 = None
        clone_891 = torch.ops.aten.clone.default(add_584, memory_format = torch.contiguous_format)
        var_mean_131 = torch.ops.aten.var_mean.correction(clone_891, [2], correction = 0, keepdim = True)
        getitem_270 = var_mean_131[0]
        getitem_271 = var_mean_131[1];  var_mean_131 = None
        add_585 = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_131 = torch.ops.aten.rsqrt.default(add_585);  add_585 = None
        sub_194 = torch.ops.aten.sub.Tensor(clone_891, getitem_271);  clone_891 = getitem_271 = None
        mul_650 = torch.ops.aten.mul.Tensor(sub_194, rsqrt_131);  sub_194 = rsqrt_131 = None
        mul_651 = torch.ops.aten.mul.Tensor(mul_650, arg491_1);  mul_650 = arg491_1 = None
        add_586 = torch.ops.aten.add.Tensor(mul_651, arg492_1);  mul_651 = arg492_1 = None
        view_1290 = torch.ops.aten.view.default(add_586, [4608, 768]);  add_586 = None
        permute_842 = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_265 = torch.ops.aten.addmm.default(arg494_1, view_1290, permute_842);  arg494_1 = view_1290 = permute_842 = None
        view_1291 = torch.ops.aten.view.default(addmm_265, [8, 576, 2304]);  addmm_265 = None
        view_1292 = torch.ops.aten.view.default(view_1291, [8, 576, 3, 16, 48]);  view_1291 = None
        permute_843 = torch.ops.aten.permute.default(view_1292, [2, 0, 3, 1, 4]);  view_1292 = None
        select_192 = torch.ops.aten.select.int(permute_843, 0, 0)
        mul_652 = torch.ops.aten.mul.Tensor(select_192, 0.14433756729740643);  select_192 = None
        select_193 = torch.ops.aten.select.int(permute_843, 0, 1)
        select_194 = torch.ops.aten.select.int(permute_843, 0, 2);  permute_843 = None
        permute_844 = torch.ops.aten.permute.default(select_193, [0, 1, 3, 2]);  select_193 = None
        expand_253 = torch.ops.aten.expand.default(mul_652, [8, 16, 576, 48]);  mul_652 = None
        clone_892 = torch.ops.aten.clone.default(expand_253, memory_format = torch.contiguous_format);  expand_253 = None
        view_1293 = torch.ops.aten.view.default(clone_892, [128, 576, 48]);  clone_892 = None
        expand_254 = torch.ops.aten.expand.default(permute_844, [8, 16, 48, 576]);  permute_844 = None
        clone_893 = torch.ops.aten.clone.default(expand_254, memory_format = torch.contiguous_format);  expand_254 = None
        view_1294 = torch.ops.aten.view.default(clone_893, [128, 48, 576]);  clone_893 = None
        bmm_126 = torch.ops.aten.bmm.default(view_1293, view_1294);  view_1293 = view_1294 = None
        view_1295 = torch.ops.aten.view.default(bmm_126, [8, 16, 576, 576]);  bmm_126 = None
        permute_845 = torch.ops.aten.permute.default(view_1295, [0, 2, 3, 1]);  view_1295 = None
        permute_846 = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        clone_894 = torch.ops.aten.clone.default(permute_845, memory_format = torch.contiguous_format);  permute_845 = None
        view_1296 = torch.ops.aten.view.default(clone_894, [2654208, 16]);  clone_894 = None
        mm_126 = torch.ops.aten.mm.default(view_1296, permute_846);  view_1296 = permute_846 = None
        view_1297 = torch.ops.aten.view.default(mm_126, [8, 576, 576, 16]);  mm_126 = None
        add_587 = torch.ops.aten.add.Tensor(view_1297, arg496_1);  view_1297 = arg496_1 = None
        permute_847 = torch.ops.aten.permute.default(add_587, [0, 3, 1, 2]);  add_587 = None
        clone_895 = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
        amax_63 = torch.ops.aten.amax.default(clone_895, [-1], True)
        sub_195 = torch.ops.aten.sub.Tensor(clone_895, amax_63);  clone_895 = amax_63 = None
        exp_63 = torch.ops.aten.exp.default(sub_195);  sub_195 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(exp_63, [-1], True)
        div_63 = torch.ops.aten.div.Tensor(exp_63, sum_64);  exp_63 = sum_64 = None
        permute_848 = torch.ops.aten.permute.default(div_63, [0, 2, 3, 1]);  div_63 = None
        permute_849 = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
        clone_896 = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
        view_1298 = torch.ops.aten.view.default(clone_896, [2654208, 16]);  clone_896 = None
        mm_127 = torch.ops.aten.mm.default(view_1298, permute_849);  view_1298 = permute_849 = None
        view_1299 = torch.ops.aten.view.default(mm_127, [8, 576, 576, 16]);  mm_127 = None
        add_588 = torch.ops.aten.add.Tensor(view_1299, arg498_1);  view_1299 = arg498_1 = None
        permute_850 = torch.ops.aten.permute.default(add_588, [0, 3, 1, 2]);  add_588 = None
        expand_255 = torch.ops.aten.expand.default(permute_850, [8, 16, 576, 576]);  permute_850 = None
        clone_898 = torch.ops.aten.clone.default(expand_255, memory_format = torch.contiguous_format);  expand_255 = None
        view_1300 = torch.ops.aten.view.default(clone_898, [128, 576, 576]);  clone_898 = None
        expand_256 = torch.ops.aten.expand.default(select_194, [8, 16, 576, 48]);  select_194 = None
        clone_899 = torch.ops.aten.clone.default(expand_256, memory_format = torch.contiguous_format);  expand_256 = None
        view_1301 = torch.ops.aten.view.default(clone_899, [128, 576, 48]);  clone_899 = None
        bmm_127 = torch.ops.aten.bmm.default(view_1300, view_1301);  view_1300 = view_1301 = None
        view_1302 = torch.ops.aten.view.default(bmm_127, [8, 16, 576, 48]);  bmm_127 = None
        permute_851 = torch.ops.aten.permute.default(view_1302, [0, 2, 1, 3]);  view_1302 = None
        clone_900 = torch.ops.aten.clone.default(permute_851, memory_format = torch.contiguous_format);  permute_851 = None
        view_1303 = torch.ops.aten.view.default(clone_900, [8, 576, 768]);  clone_900 = None
        view_1304 = torch.ops.aten.view.default(view_1303, [4608, 768]);  view_1303 = None
        permute_852 = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_266 = torch.ops.aten.addmm.default(arg500_1, view_1304, permute_852);  arg500_1 = view_1304 = permute_852 = None
        view_1305 = torch.ops.aten.view.default(addmm_266, [8, 576, 768]);  addmm_266 = None
        mul_653 = torch.ops.aten.mul.Tensor(arg490_1, view_1305);  arg490_1 = view_1305 = None
        add_589 = torch.ops.aten.add.Tensor(add_584, mul_653);  add_584 = mul_653 = None
        clone_902 = torch.ops.aten.clone.default(add_589, memory_format = torch.contiguous_format)
        var_mean_132 = torch.ops.aten.var_mean.correction(clone_902, [2], correction = 0, keepdim = True)
        getitem_272 = var_mean_132[0]
        getitem_273 = var_mean_132[1];  var_mean_132 = None
        add_590 = torch.ops.aten.add.Tensor(getitem_272, 1e-06);  getitem_272 = None
        rsqrt_132 = torch.ops.aten.rsqrt.default(add_590);  add_590 = None
        sub_196 = torch.ops.aten.sub.Tensor(clone_902, getitem_273);  clone_902 = getitem_273 = None
        mul_654 = torch.ops.aten.mul.Tensor(sub_196, rsqrt_132);  sub_196 = rsqrt_132 = None
        mul_655 = torch.ops.aten.mul.Tensor(mul_654, arg502_1);  mul_654 = arg502_1 = None
        add_591 = torch.ops.aten.add.Tensor(mul_655, arg503_1);  mul_655 = arg503_1 = None
        view_1306 = torch.ops.aten.view.default(add_591, [4608, 768]);  add_591 = None
        permute_853 = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
        addmm_267 = torch.ops.aten.addmm.default(arg505_1, view_1306, permute_853);  arg505_1 = view_1306 = permute_853 = None
        view_1307 = torch.ops.aten.view.default(addmm_267, [8, 576, 3072]);  addmm_267 = None
        mul_656 = torch.ops.aten.mul.Tensor(view_1307, 0.5)
        mul_657 = torch.ops.aten.mul.Tensor(view_1307, 0.7071067811865476);  view_1307 = None
        erf_65 = torch.ops.aten.erf.default(mul_657);  mul_657 = None
        add_592 = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_658 = torch.ops.aten.mul.Tensor(mul_656, add_592);  mul_656 = add_592 = None
        view_1308 = torch.ops.aten.view.default(mul_658, [4608, 3072]);  mul_658 = None
        permute_854 = torch.ops.aten.permute.default(arg506_1, [1, 0]);  arg506_1 = None
        addmm_268 = torch.ops.aten.addmm.default(arg507_1, view_1308, permute_854);  arg507_1 = view_1308 = permute_854 = None
        view_1309 = torch.ops.aten.view.default(addmm_268, [8, 576, 768]);  addmm_268 = None
        mul_659 = torch.ops.aten.mul.Tensor(arg501_1, view_1309);  arg501_1 = view_1309 = None
        add_593 = torch.ops.aten.add.Tensor(add_589, mul_659);  add_589 = mul_659 = None
        clone_905 = torch.ops.aten.clone.default(add_593, memory_format = torch.contiguous_format)
        var_mean_133 = torch.ops.aten.var_mean.correction(clone_905, [2], correction = 0, keepdim = True)
        getitem_274 = var_mean_133[0]
        getitem_275 = var_mean_133[1];  var_mean_133 = None
        add_594 = torch.ops.aten.add.Tensor(getitem_274, 1e-06);  getitem_274 = None
        rsqrt_133 = torch.ops.aten.rsqrt.default(add_594);  add_594 = None
        sub_197 = torch.ops.aten.sub.Tensor(clone_905, getitem_275);  clone_905 = getitem_275 = None
        mul_660 = torch.ops.aten.mul.Tensor(sub_197, rsqrt_133);  sub_197 = rsqrt_133 = None
        mul_661 = torch.ops.aten.mul.Tensor(mul_660, arg509_1);  mul_660 = arg509_1 = None
        add_595 = torch.ops.aten.add.Tensor(mul_661, arg510_1);  mul_661 = arg510_1 = None
        view_1310 = torch.ops.aten.view.default(add_595, [4608, 768]);  add_595 = None
        permute_855 = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
        addmm_269 = torch.ops.aten.addmm.default(arg512_1, view_1310, permute_855);  arg512_1 = view_1310 = permute_855 = None
        view_1311 = torch.ops.aten.view.default(addmm_269, [8, 576, 2304]);  addmm_269 = None
        view_1312 = torch.ops.aten.view.default(view_1311, [8, 576, 3, 16, 48]);  view_1311 = None
        permute_856 = torch.ops.aten.permute.default(view_1312, [2, 0, 3, 1, 4]);  view_1312 = None
        select_195 = torch.ops.aten.select.int(permute_856, 0, 0)
        mul_662 = torch.ops.aten.mul.Tensor(select_195, 0.14433756729740643);  select_195 = None
        select_196 = torch.ops.aten.select.int(permute_856, 0, 1)
        select_197 = torch.ops.aten.select.int(permute_856, 0, 2);  permute_856 = None
        permute_857 = torch.ops.aten.permute.default(select_196, [0, 1, 3, 2]);  select_196 = None
        expand_257 = torch.ops.aten.expand.default(mul_662, [8, 16, 576, 48]);  mul_662 = None
        clone_906 = torch.ops.aten.clone.default(expand_257, memory_format = torch.contiguous_format);  expand_257 = None
        view_1313 = torch.ops.aten.view.default(clone_906, [128, 576, 48]);  clone_906 = None
        expand_258 = torch.ops.aten.expand.default(permute_857, [8, 16, 48, 576]);  permute_857 = None
        clone_907 = torch.ops.aten.clone.default(expand_258, memory_format = torch.contiguous_format);  expand_258 = None
        view_1314 = torch.ops.aten.view.default(clone_907, [128, 48, 576]);  clone_907 = None
        bmm_128 = torch.ops.aten.bmm.default(view_1313, view_1314);  view_1313 = view_1314 = None
        view_1315 = torch.ops.aten.view.default(bmm_128, [8, 16, 576, 576]);  bmm_128 = None
        permute_858 = torch.ops.aten.permute.default(view_1315, [0, 2, 3, 1]);  view_1315 = None
        permute_859 = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
        clone_908 = torch.ops.aten.clone.default(permute_858, memory_format = torch.contiguous_format);  permute_858 = None
        view_1316 = torch.ops.aten.view.default(clone_908, [2654208, 16]);  clone_908 = None
        mm_128 = torch.ops.aten.mm.default(view_1316, permute_859);  view_1316 = permute_859 = None
        view_1317 = torch.ops.aten.view.default(mm_128, [8, 576, 576, 16]);  mm_128 = None
        add_596 = torch.ops.aten.add.Tensor(view_1317, arg514_1);  view_1317 = arg514_1 = None
        permute_860 = torch.ops.aten.permute.default(add_596, [0, 3, 1, 2]);  add_596 = None
        clone_909 = torch.ops.aten.clone.default(permute_860, memory_format = torch.contiguous_format);  permute_860 = None
        amax_64 = torch.ops.aten.amax.default(clone_909, [-1], True)
        sub_198 = torch.ops.aten.sub.Tensor(clone_909, amax_64);  clone_909 = amax_64 = None
        exp_64 = torch.ops.aten.exp.default(sub_198);  sub_198 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(exp_64, [-1], True)
        div_64 = torch.ops.aten.div.Tensor(exp_64, sum_65);  exp_64 = sum_65 = None
        permute_861 = torch.ops.aten.permute.default(div_64, [0, 2, 3, 1]);  div_64 = None
        permute_862 = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        clone_910 = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
        view_1318 = torch.ops.aten.view.default(clone_910, [2654208, 16]);  clone_910 = None
        mm_129 = torch.ops.aten.mm.default(view_1318, permute_862);  view_1318 = permute_862 = None
        view_1319 = torch.ops.aten.view.default(mm_129, [8, 576, 576, 16]);  mm_129 = None
        add_597 = torch.ops.aten.add.Tensor(view_1319, arg516_1);  view_1319 = arg516_1 = None
        permute_863 = torch.ops.aten.permute.default(add_597, [0, 3, 1, 2]);  add_597 = None
        expand_259 = torch.ops.aten.expand.default(permute_863, [8, 16, 576, 576]);  permute_863 = None
        clone_912 = torch.ops.aten.clone.default(expand_259, memory_format = torch.contiguous_format);  expand_259 = None
        view_1320 = torch.ops.aten.view.default(clone_912, [128, 576, 576]);  clone_912 = None
        expand_260 = torch.ops.aten.expand.default(select_197, [8, 16, 576, 48]);  select_197 = None
        clone_913 = torch.ops.aten.clone.default(expand_260, memory_format = torch.contiguous_format);  expand_260 = None
        view_1321 = torch.ops.aten.view.default(clone_913, [128, 576, 48]);  clone_913 = None
        bmm_129 = torch.ops.aten.bmm.default(view_1320, view_1321);  view_1320 = view_1321 = None
        view_1322 = torch.ops.aten.view.default(bmm_129, [8, 16, 576, 48]);  bmm_129 = None
        permute_864 = torch.ops.aten.permute.default(view_1322, [0, 2, 1, 3]);  view_1322 = None
        clone_914 = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
        view_1323 = torch.ops.aten.view.default(clone_914, [8, 576, 768]);  clone_914 = None
        view_1324 = torch.ops.aten.view.default(view_1323, [4608, 768]);  view_1323 = None
        permute_865 = torch.ops.aten.permute.default(arg517_1, [1, 0]);  arg517_1 = None
        addmm_270 = torch.ops.aten.addmm.default(arg518_1, view_1324, permute_865);  arg518_1 = view_1324 = permute_865 = None
        view_1325 = torch.ops.aten.view.default(addmm_270, [8, 576, 768]);  addmm_270 = None
        mul_663 = torch.ops.aten.mul.Tensor(arg508_1, view_1325);  arg508_1 = view_1325 = None
        add_598 = torch.ops.aten.add.Tensor(add_593, mul_663);  add_593 = mul_663 = None
        clone_916 = torch.ops.aten.clone.default(add_598, memory_format = torch.contiguous_format)
        var_mean_134 = torch.ops.aten.var_mean.correction(clone_916, [2], correction = 0, keepdim = True)
        getitem_276 = var_mean_134[0]
        getitem_277 = var_mean_134[1];  var_mean_134 = None
        add_599 = torch.ops.aten.add.Tensor(getitem_276, 1e-06);  getitem_276 = None
        rsqrt_134 = torch.ops.aten.rsqrt.default(add_599);  add_599 = None
        sub_199 = torch.ops.aten.sub.Tensor(clone_916, getitem_277);  clone_916 = getitem_277 = None
        mul_664 = torch.ops.aten.mul.Tensor(sub_199, rsqrt_134);  sub_199 = rsqrt_134 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, arg520_1);  mul_664 = arg520_1 = None
        add_600 = torch.ops.aten.add.Tensor(mul_665, arg521_1);  mul_665 = arg521_1 = None
        view_1326 = torch.ops.aten.view.default(add_600, [4608, 768]);  add_600 = None
        permute_866 = torch.ops.aten.permute.default(arg522_1, [1, 0]);  arg522_1 = None
        addmm_271 = torch.ops.aten.addmm.default(arg523_1, view_1326, permute_866);  arg523_1 = view_1326 = permute_866 = None
        view_1327 = torch.ops.aten.view.default(addmm_271, [8, 576, 3072]);  addmm_271 = None
        mul_666 = torch.ops.aten.mul.Tensor(view_1327, 0.5)
        mul_667 = torch.ops.aten.mul.Tensor(view_1327, 0.7071067811865476);  view_1327 = None
        erf_66 = torch.ops.aten.erf.default(mul_667);  mul_667 = None
        add_601 = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_668 = torch.ops.aten.mul.Tensor(mul_666, add_601);  mul_666 = add_601 = None
        view_1328 = torch.ops.aten.view.default(mul_668, [4608, 3072]);  mul_668 = None
        permute_867 = torch.ops.aten.permute.default(arg524_1, [1, 0]);  arg524_1 = None
        addmm_272 = torch.ops.aten.addmm.default(arg525_1, view_1328, permute_867);  arg525_1 = view_1328 = permute_867 = None
        view_1329 = torch.ops.aten.view.default(addmm_272, [8, 576, 768]);  addmm_272 = None
        mul_669 = torch.ops.aten.mul.Tensor(arg519_1, view_1329);  arg519_1 = view_1329 = None
        add_602 = torch.ops.aten.add.Tensor(add_598, mul_669);  add_598 = mul_669 = None
        clone_919 = torch.ops.aten.clone.default(add_602, memory_format = torch.contiguous_format)
        var_mean_135 = torch.ops.aten.var_mean.correction(clone_919, [2], correction = 0, keepdim = True)
        getitem_278 = var_mean_135[0]
        getitem_279 = var_mean_135[1];  var_mean_135 = None
        add_603 = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
        rsqrt_135 = torch.ops.aten.rsqrt.default(add_603);  add_603 = None
        sub_200 = torch.ops.aten.sub.Tensor(clone_919, getitem_279);  clone_919 = getitem_279 = None
        mul_670 = torch.ops.aten.mul.Tensor(sub_200, rsqrt_135);  sub_200 = rsqrt_135 = None
        mul_671 = torch.ops.aten.mul.Tensor(mul_670, arg527_1);  mul_670 = arg527_1 = None
        add_604 = torch.ops.aten.add.Tensor(mul_671, arg528_1);  mul_671 = arg528_1 = None
        view_1330 = torch.ops.aten.view.default(add_604, [4608, 768]);  add_604 = None
        permute_868 = torch.ops.aten.permute.default(arg529_1, [1, 0]);  arg529_1 = None
        addmm_273 = torch.ops.aten.addmm.default(arg530_1, view_1330, permute_868);  arg530_1 = view_1330 = permute_868 = None
        view_1331 = torch.ops.aten.view.default(addmm_273, [8, 576, 2304]);  addmm_273 = None
        view_1332 = torch.ops.aten.view.default(view_1331, [8, 576, 3, 16, 48]);  view_1331 = None
        permute_869 = torch.ops.aten.permute.default(view_1332, [2, 0, 3, 1, 4]);  view_1332 = None
        select_198 = torch.ops.aten.select.int(permute_869, 0, 0)
        mul_672 = torch.ops.aten.mul.Tensor(select_198, 0.14433756729740643);  select_198 = None
        select_199 = torch.ops.aten.select.int(permute_869, 0, 1)
        select_200 = torch.ops.aten.select.int(permute_869, 0, 2);  permute_869 = None
        permute_870 = torch.ops.aten.permute.default(select_199, [0, 1, 3, 2]);  select_199 = None
        expand_261 = torch.ops.aten.expand.default(mul_672, [8, 16, 576, 48]);  mul_672 = None
        clone_920 = torch.ops.aten.clone.default(expand_261, memory_format = torch.contiguous_format);  expand_261 = None
        view_1333 = torch.ops.aten.view.default(clone_920, [128, 576, 48]);  clone_920 = None
        expand_262 = torch.ops.aten.expand.default(permute_870, [8, 16, 48, 576]);  permute_870 = None
        clone_921 = torch.ops.aten.clone.default(expand_262, memory_format = torch.contiguous_format);  expand_262 = None
        view_1334 = torch.ops.aten.view.default(clone_921, [128, 48, 576]);  clone_921 = None
        bmm_130 = torch.ops.aten.bmm.default(view_1333, view_1334);  view_1333 = view_1334 = None
        view_1335 = torch.ops.aten.view.default(bmm_130, [8, 16, 576, 576]);  bmm_130 = None
        permute_871 = torch.ops.aten.permute.default(view_1335, [0, 2, 3, 1]);  view_1335 = None
        permute_872 = torch.ops.aten.permute.default(arg531_1, [1, 0]);  arg531_1 = None
        clone_922 = torch.ops.aten.clone.default(permute_871, memory_format = torch.contiguous_format);  permute_871 = None
        view_1336 = torch.ops.aten.view.default(clone_922, [2654208, 16]);  clone_922 = None
        mm_130 = torch.ops.aten.mm.default(view_1336, permute_872);  view_1336 = permute_872 = None
        view_1337 = torch.ops.aten.view.default(mm_130, [8, 576, 576, 16]);  mm_130 = None
        add_605 = torch.ops.aten.add.Tensor(view_1337, arg532_1);  view_1337 = arg532_1 = None
        permute_873 = torch.ops.aten.permute.default(add_605, [0, 3, 1, 2]);  add_605 = None
        clone_923 = torch.ops.aten.clone.default(permute_873, memory_format = torch.contiguous_format);  permute_873 = None
        amax_65 = torch.ops.aten.amax.default(clone_923, [-1], True)
        sub_201 = torch.ops.aten.sub.Tensor(clone_923, amax_65);  clone_923 = amax_65 = None
        exp_65 = torch.ops.aten.exp.default(sub_201);  sub_201 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(exp_65, [-1], True)
        div_65 = torch.ops.aten.div.Tensor(exp_65, sum_66);  exp_65 = sum_66 = None
        permute_874 = torch.ops.aten.permute.default(div_65, [0, 2, 3, 1]);  div_65 = None
        permute_875 = torch.ops.aten.permute.default(arg533_1, [1, 0]);  arg533_1 = None
        clone_924 = torch.ops.aten.clone.default(permute_874, memory_format = torch.contiguous_format);  permute_874 = None
        view_1338 = torch.ops.aten.view.default(clone_924, [2654208, 16]);  clone_924 = None
        mm_131 = torch.ops.aten.mm.default(view_1338, permute_875);  view_1338 = permute_875 = None
        view_1339 = torch.ops.aten.view.default(mm_131, [8, 576, 576, 16]);  mm_131 = None
        add_606 = torch.ops.aten.add.Tensor(view_1339, arg534_1);  view_1339 = arg534_1 = None
        permute_876 = torch.ops.aten.permute.default(add_606, [0, 3, 1, 2]);  add_606 = None
        expand_263 = torch.ops.aten.expand.default(permute_876, [8, 16, 576, 576]);  permute_876 = None
        clone_926 = torch.ops.aten.clone.default(expand_263, memory_format = torch.contiguous_format);  expand_263 = None
        view_1340 = torch.ops.aten.view.default(clone_926, [128, 576, 576]);  clone_926 = None
        expand_264 = torch.ops.aten.expand.default(select_200, [8, 16, 576, 48]);  select_200 = None
        clone_927 = torch.ops.aten.clone.default(expand_264, memory_format = torch.contiguous_format);  expand_264 = None
        view_1341 = torch.ops.aten.view.default(clone_927, [128, 576, 48]);  clone_927 = None
        bmm_131 = torch.ops.aten.bmm.default(view_1340, view_1341);  view_1340 = view_1341 = None
        view_1342 = torch.ops.aten.view.default(bmm_131, [8, 16, 576, 48]);  bmm_131 = None
        permute_877 = torch.ops.aten.permute.default(view_1342, [0, 2, 1, 3]);  view_1342 = None
        clone_928 = torch.ops.aten.clone.default(permute_877, memory_format = torch.contiguous_format);  permute_877 = None
        view_1343 = torch.ops.aten.view.default(clone_928, [8, 576, 768]);  clone_928 = None
        view_1344 = torch.ops.aten.view.default(view_1343, [4608, 768]);  view_1343 = None
        permute_878 = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
        addmm_274 = torch.ops.aten.addmm.default(arg536_1, view_1344, permute_878);  arg536_1 = view_1344 = permute_878 = None
        view_1345 = torch.ops.aten.view.default(addmm_274, [8, 576, 768]);  addmm_274 = None
        mul_673 = torch.ops.aten.mul.Tensor(arg526_1, view_1345);  arg526_1 = view_1345 = None
        add_607 = torch.ops.aten.add.Tensor(add_602, mul_673);  add_602 = mul_673 = None
        clone_930 = torch.ops.aten.clone.default(add_607, memory_format = torch.contiguous_format)
        var_mean_136 = torch.ops.aten.var_mean.correction(clone_930, [2], correction = 0, keepdim = True)
        getitem_280 = var_mean_136[0]
        getitem_281 = var_mean_136[1];  var_mean_136 = None
        add_608 = torch.ops.aten.add.Tensor(getitem_280, 1e-06);  getitem_280 = None
        rsqrt_136 = torch.ops.aten.rsqrt.default(add_608);  add_608 = None
        sub_202 = torch.ops.aten.sub.Tensor(clone_930, getitem_281);  clone_930 = getitem_281 = None
        mul_674 = torch.ops.aten.mul.Tensor(sub_202, rsqrt_136);  sub_202 = rsqrt_136 = None
        mul_675 = torch.ops.aten.mul.Tensor(mul_674, arg538_1);  mul_674 = arg538_1 = None
        add_609 = torch.ops.aten.add.Tensor(mul_675, arg539_1);  mul_675 = arg539_1 = None
        view_1346 = torch.ops.aten.view.default(add_609, [4608, 768]);  add_609 = None
        permute_879 = torch.ops.aten.permute.default(arg540_1, [1, 0]);  arg540_1 = None
        addmm_275 = torch.ops.aten.addmm.default(arg541_1, view_1346, permute_879);  arg541_1 = view_1346 = permute_879 = None
        view_1347 = torch.ops.aten.view.default(addmm_275, [8, 576, 3072]);  addmm_275 = None
        mul_676 = torch.ops.aten.mul.Tensor(view_1347, 0.5)
        mul_677 = torch.ops.aten.mul.Tensor(view_1347, 0.7071067811865476);  view_1347 = None
        erf_67 = torch.ops.aten.erf.default(mul_677);  mul_677 = None
        add_610 = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_678 = torch.ops.aten.mul.Tensor(mul_676, add_610);  mul_676 = add_610 = None
        view_1348 = torch.ops.aten.view.default(mul_678, [4608, 3072]);  mul_678 = None
        permute_880 = torch.ops.aten.permute.default(arg542_1, [1, 0]);  arg542_1 = None
        addmm_276 = torch.ops.aten.addmm.default(arg543_1, view_1348, permute_880);  arg543_1 = view_1348 = permute_880 = None
        view_1349 = torch.ops.aten.view.default(addmm_276, [8, 576, 768]);  addmm_276 = None
        mul_679 = torch.ops.aten.mul.Tensor(arg537_1, view_1349);  arg537_1 = view_1349 = None
        add_611 = torch.ops.aten.add.Tensor(add_607, mul_679);  add_607 = mul_679 = None
        clone_933 = torch.ops.aten.clone.default(add_611, memory_format = torch.contiguous_format)
        var_mean_137 = torch.ops.aten.var_mean.correction(clone_933, [2], correction = 0, keepdim = True)
        getitem_282 = var_mean_137[0]
        getitem_283 = var_mean_137[1];  var_mean_137 = None
        add_612 = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
        rsqrt_137 = torch.ops.aten.rsqrt.default(add_612);  add_612 = None
        sub_203 = torch.ops.aten.sub.Tensor(clone_933, getitem_283);  clone_933 = getitem_283 = None
        mul_680 = torch.ops.aten.mul.Tensor(sub_203, rsqrt_137);  sub_203 = rsqrt_137 = None
        mul_681 = torch.ops.aten.mul.Tensor(mul_680, arg545_1);  mul_680 = arg545_1 = None
        add_613 = torch.ops.aten.add.Tensor(mul_681, arg546_1);  mul_681 = arg546_1 = None
        view_1350 = torch.ops.aten.view.default(add_613, [4608, 768]);  add_613 = None
        permute_881 = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
        addmm_277 = torch.ops.aten.addmm.default(arg548_1, view_1350, permute_881);  arg548_1 = view_1350 = permute_881 = None
        view_1351 = torch.ops.aten.view.default(addmm_277, [8, 576, 2304]);  addmm_277 = None
        view_1352 = torch.ops.aten.view.default(view_1351, [8, 576, 3, 16, 48]);  view_1351 = None
        permute_882 = torch.ops.aten.permute.default(view_1352, [2, 0, 3, 1, 4]);  view_1352 = None
        select_201 = torch.ops.aten.select.int(permute_882, 0, 0)
        mul_682 = torch.ops.aten.mul.Tensor(select_201, 0.14433756729740643);  select_201 = None
        select_202 = torch.ops.aten.select.int(permute_882, 0, 1)
        select_203 = torch.ops.aten.select.int(permute_882, 0, 2);  permute_882 = None
        permute_883 = torch.ops.aten.permute.default(select_202, [0, 1, 3, 2]);  select_202 = None
        expand_265 = torch.ops.aten.expand.default(mul_682, [8, 16, 576, 48]);  mul_682 = None
        clone_934 = torch.ops.aten.clone.default(expand_265, memory_format = torch.contiguous_format);  expand_265 = None
        view_1353 = torch.ops.aten.view.default(clone_934, [128, 576, 48]);  clone_934 = None
        expand_266 = torch.ops.aten.expand.default(permute_883, [8, 16, 48, 576]);  permute_883 = None
        clone_935 = torch.ops.aten.clone.default(expand_266, memory_format = torch.contiguous_format);  expand_266 = None
        view_1354 = torch.ops.aten.view.default(clone_935, [128, 48, 576]);  clone_935 = None
        bmm_132 = torch.ops.aten.bmm.default(view_1353, view_1354);  view_1353 = view_1354 = None
        view_1355 = torch.ops.aten.view.default(bmm_132, [8, 16, 576, 576]);  bmm_132 = None
        permute_884 = torch.ops.aten.permute.default(view_1355, [0, 2, 3, 1]);  view_1355 = None
        permute_885 = torch.ops.aten.permute.default(arg549_1, [1, 0]);  arg549_1 = None
        clone_936 = torch.ops.aten.clone.default(permute_884, memory_format = torch.contiguous_format);  permute_884 = None
        view_1356 = torch.ops.aten.view.default(clone_936, [2654208, 16]);  clone_936 = None
        mm_132 = torch.ops.aten.mm.default(view_1356, permute_885);  view_1356 = permute_885 = None
        view_1357 = torch.ops.aten.view.default(mm_132, [8, 576, 576, 16]);  mm_132 = None
        add_614 = torch.ops.aten.add.Tensor(view_1357, arg550_1);  view_1357 = arg550_1 = None
        permute_886 = torch.ops.aten.permute.default(add_614, [0, 3, 1, 2]);  add_614 = None
        clone_937 = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
        amax_66 = torch.ops.aten.amax.default(clone_937, [-1], True)
        sub_204 = torch.ops.aten.sub.Tensor(clone_937, amax_66);  clone_937 = amax_66 = None
        exp_66 = torch.ops.aten.exp.default(sub_204);  sub_204 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(exp_66, [-1], True)
        div_66 = torch.ops.aten.div.Tensor(exp_66, sum_67);  exp_66 = sum_67 = None
        permute_887 = torch.ops.aten.permute.default(div_66, [0, 2, 3, 1]);  div_66 = None
        permute_888 = torch.ops.aten.permute.default(arg551_1, [1, 0]);  arg551_1 = None
        clone_938 = torch.ops.aten.clone.default(permute_887, memory_format = torch.contiguous_format);  permute_887 = None
        view_1358 = torch.ops.aten.view.default(clone_938, [2654208, 16]);  clone_938 = None
        mm_133 = torch.ops.aten.mm.default(view_1358, permute_888);  view_1358 = permute_888 = None
        view_1359 = torch.ops.aten.view.default(mm_133, [8, 576, 576, 16]);  mm_133 = None
        add_615 = torch.ops.aten.add.Tensor(view_1359, arg552_1);  view_1359 = arg552_1 = None
        permute_889 = torch.ops.aten.permute.default(add_615, [0, 3, 1, 2]);  add_615 = None
        expand_267 = torch.ops.aten.expand.default(permute_889, [8, 16, 576, 576]);  permute_889 = None
        clone_940 = torch.ops.aten.clone.default(expand_267, memory_format = torch.contiguous_format);  expand_267 = None
        view_1360 = torch.ops.aten.view.default(clone_940, [128, 576, 576]);  clone_940 = None
        expand_268 = torch.ops.aten.expand.default(select_203, [8, 16, 576, 48]);  select_203 = None
        clone_941 = torch.ops.aten.clone.default(expand_268, memory_format = torch.contiguous_format);  expand_268 = None
        view_1361 = torch.ops.aten.view.default(clone_941, [128, 576, 48]);  clone_941 = None
        bmm_133 = torch.ops.aten.bmm.default(view_1360, view_1361);  view_1360 = view_1361 = None
        view_1362 = torch.ops.aten.view.default(bmm_133, [8, 16, 576, 48]);  bmm_133 = None
        permute_890 = torch.ops.aten.permute.default(view_1362, [0, 2, 1, 3]);  view_1362 = None
        clone_942 = torch.ops.aten.clone.default(permute_890, memory_format = torch.contiguous_format);  permute_890 = None
        view_1363 = torch.ops.aten.view.default(clone_942, [8, 576, 768]);  clone_942 = None
        view_1364 = torch.ops.aten.view.default(view_1363, [4608, 768]);  view_1363 = None
        permute_891 = torch.ops.aten.permute.default(arg553_1, [1, 0]);  arg553_1 = None
        addmm_278 = torch.ops.aten.addmm.default(arg554_1, view_1364, permute_891);  arg554_1 = view_1364 = permute_891 = None
        view_1365 = torch.ops.aten.view.default(addmm_278, [8, 576, 768]);  addmm_278 = None
        mul_683 = torch.ops.aten.mul.Tensor(arg544_1, view_1365);  arg544_1 = view_1365 = None
        add_616 = torch.ops.aten.add.Tensor(add_611, mul_683);  add_611 = mul_683 = None
        clone_944 = torch.ops.aten.clone.default(add_616, memory_format = torch.contiguous_format)
        var_mean_138 = torch.ops.aten.var_mean.correction(clone_944, [2], correction = 0, keepdim = True)
        getitem_284 = var_mean_138[0]
        getitem_285 = var_mean_138[1];  var_mean_138 = None
        add_617 = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_138 = torch.ops.aten.rsqrt.default(add_617);  add_617 = None
        sub_205 = torch.ops.aten.sub.Tensor(clone_944, getitem_285);  clone_944 = getitem_285 = None
        mul_684 = torch.ops.aten.mul.Tensor(sub_205, rsqrt_138);  sub_205 = rsqrt_138 = None
        mul_685 = torch.ops.aten.mul.Tensor(mul_684, arg556_1);  mul_684 = arg556_1 = None
        add_618 = torch.ops.aten.add.Tensor(mul_685, arg557_1);  mul_685 = arg557_1 = None
        view_1366 = torch.ops.aten.view.default(add_618, [4608, 768]);  add_618 = None
        permute_892 = torch.ops.aten.permute.default(arg558_1, [1, 0]);  arg558_1 = None
        addmm_279 = torch.ops.aten.addmm.default(arg559_1, view_1366, permute_892);  arg559_1 = view_1366 = permute_892 = None
        view_1367 = torch.ops.aten.view.default(addmm_279, [8, 576, 3072]);  addmm_279 = None
        mul_686 = torch.ops.aten.mul.Tensor(view_1367, 0.5)
        mul_687 = torch.ops.aten.mul.Tensor(view_1367, 0.7071067811865476);  view_1367 = None
        erf_68 = torch.ops.aten.erf.default(mul_687);  mul_687 = None
        add_619 = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_688 = torch.ops.aten.mul.Tensor(mul_686, add_619);  mul_686 = add_619 = None
        view_1368 = torch.ops.aten.view.default(mul_688, [4608, 3072]);  mul_688 = None
        permute_893 = torch.ops.aten.permute.default(arg560_1, [1, 0]);  arg560_1 = None
        addmm_280 = torch.ops.aten.addmm.default(arg561_1, view_1368, permute_893);  arg561_1 = view_1368 = permute_893 = None
        view_1369 = torch.ops.aten.view.default(addmm_280, [8, 576, 768]);  addmm_280 = None
        mul_689 = torch.ops.aten.mul.Tensor(arg555_1, view_1369);  arg555_1 = view_1369 = None
        add_620 = torch.ops.aten.add.Tensor(add_616, mul_689);  add_616 = mul_689 = None
        clone_947 = torch.ops.aten.clone.default(add_620, memory_format = torch.contiguous_format)
        var_mean_139 = torch.ops.aten.var_mean.correction(clone_947, [2], correction = 0, keepdim = True)
        getitem_286 = var_mean_139[0]
        getitem_287 = var_mean_139[1];  var_mean_139 = None
        add_621 = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_139 = torch.ops.aten.rsqrt.default(add_621);  add_621 = None
        sub_206 = torch.ops.aten.sub.Tensor(clone_947, getitem_287);  clone_947 = getitem_287 = None
        mul_690 = torch.ops.aten.mul.Tensor(sub_206, rsqrt_139);  sub_206 = rsqrt_139 = None
        mul_691 = torch.ops.aten.mul.Tensor(mul_690, arg563_1);  mul_690 = arg563_1 = None
        add_622 = torch.ops.aten.add.Tensor(mul_691, arg564_1);  mul_691 = arg564_1 = None
        view_1370 = torch.ops.aten.view.default(add_622, [4608, 768]);  add_622 = None
        permute_894 = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
        addmm_281 = torch.ops.aten.addmm.default(arg566_1, view_1370, permute_894);  arg566_1 = view_1370 = permute_894 = None
        view_1371 = torch.ops.aten.view.default(addmm_281, [8, 576, 2304]);  addmm_281 = None
        view_1372 = torch.ops.aten.view.default(view_1371, [8, 576, 3, 16, 48]);  view_1371 = None
        permute_895 = torch.ops.aten.permute.default(view_1372, [2, 0, 3, 1, 4]);  view_1372 = None
        select_204 = torch.ops.aten.select.int(permute_895, 0, 0)
        mul_692 = torch.ops.aten.mul.Tensor(select_204, 0.14433756729740643);  select_204 = None
        select_205 = torch.ops.aten.select.int(permute_895, 0, 1)
        select_206 = torch.ops.aten.select.int(permute_895, 0, 2);  permute_895 = None
        permute_896 = torch.ops.aten.permute.default(select_205, [0, 1, 3, 2]);  select_205 = None
        expand_269 = torch.ops.aten.expand.default(mul_692, [8, 16, 576, 48]);  mul_692 = None
        clone_948 = torch.ops.aten.clone.default(expand_269, memory_format = torch.contiguous_format);  expand_269 = None
        view_1373 = torch.ops.aten.view.default(clone_948, [128, 576, 48]);  clone_948 = None
        expand_270 = torch.ops.aten.expand.default(permute_896, [8, 16, 48, 576]);  permute_896 = None
        clone_949 = torch.ops.aten.clone.default(expand_270, memory_format = torch.contiguous_format);  expand_270 = None
        view_1374 = torch.ops.aten.view.default(clone_949, [128, 48, 576]);  clone_949 = None
        bmm_134 = torch.ops.aten.bmm.default(view_1373, view_1374);  view_1373 = view_1374 = None
        view_1375 = torch.ops.aten.view.default(bmm_134, [8, 16, 576, 576]);  bmm_134 = None
        permute_897 = torch.ops.aten.permute.default(view_1375, [0, 2, 3, 1]);  view_1375 = None
        permute_898 = torch.ops.aten.permute.default(arg567_1, [1, 0]);  arg567_1 = None
        clone_950 = torch.ops.aten.clone.default(permute_897, memory_format = torch.contiguous_format);  permute_897 = None
        view_1376 = torch.ops.aten.view.default(clone_950, [2654208, 16]);  clone_950 = None
        mm_134 = torch.ops.aten.mm.default(view_1376, permute_898);  view_1376 = permute_898 = None
        view_1377 = torch.ops.aten.view.default(mm_134, [8, 576, 576, 16]);  mm_134 = None
        add_623 = torch.ops.aten.add.Tensor(view_1377, arg568_1);  view_1377 = arg568_1 = None
        permute_899 = torch.ops.aten.permute.default(add_623, [0, 3, 1, 2]);  add_623 = None
        clone_951 = torch.ops.aten.clone.default(permute_899, memory_format = torch.contiguous_format);  permute_899 = None
        amax_67 = torch.ops.aten.amax.default(clone_951, [-1], True)
        sub_207 = torch.ops.aten.sub.Tensor(clone_951, amax_67);  clone_951 = amax_67 = None
        exp_67 = torch.ops.aten.exp.default(sub_207);  sub_207 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(exp_67, [-1], True)
        div_67 = torch.ops.aten.div.Tensor(exp_67, sum_68);  exp_67 = sum_68 = None
        permute_900 = torch.ops.aten.permute.default(div_67, [0, 2, 3, 1]);  div_67 = None
        permute_901 = torch.ops.aten.permute.default(arg569_1, [1, 0]);  arg569_1 = None
        clone_952 = torch.ops.aten.clone.default(permute_900, memory_format = torch.contiguous_format);  permute_900 = None
        view_1378 = torch.ops.aten.view.default(clone_952, [2654208, 16]);  clone_952 = None
        mm_135 = torch.ops.aten.mm.default(view_1378, permute_901);  view_1378 = permute_901 = None
        view_1379 = torch.ops.aten.view.default(mm_135, [8, 576, 576, 16]);  mm_135 = None
        add_624 = torch.ops.aten.add.Tensor(view_1379, arg570_1);  view_1379 = arg570_1 = None
        permute_902 = torch.ops.aten.permute.default(add_624, [0, 3, 1, 2]);  add_624 = None
        expand_271 = torch.ops.aten.expand.default(permute_902, [8, 16, 576, 576]);  permute_902 = None
        clone_954 = torch.ops.aten.clone.default(expand_271, memory_format = torch.contiguous_format);  expand_271 = None
        view_1380 = torch.ops.aten.view.default(clone_954, [128, 576, 576]);  clone_954 = None
        expand_272 = torch.ops.aten.expand.default(select_206, [8, 16, 576, 48]);  select_206 = None
        clone_955 = torch.ops.aten.clone.default(expand_272, memory_format = torch.contiguous_format);  expand_272 = None
        view_1381 = torch.ops.aten.view.default(clone_955, [128, 576, 48]);  clone_955 = None
        bmm_135 = torch.ops.aten.bmm.default(view_1380, view_1381);  view_1380 = view_1381 = None
        view_1382 = torch.ops.aten.view.default(bmm_135, [8, 16, 576, 48]);  bmm_135 = None
        permute_903 = torch.ops.aten.permute.default(view_1382, [0, 2, 1, 3]);  view_1382 = None
        clone_956 = torch.ops.aten.clone.default(permute_903, memory_format = torch.contiguous_format);  permute_903 = None
        view_1383 = torch.ops.aten.view.default(clone_956, [8, 576, 768]);  clone_956 = None
        view_1384 = torch.ops.aten.view.default(view_1383, [4608, 768]);  view_1383 = None
        permute_904 = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
        addmm_282 = torch.ops.aten.addmm.default(arg572_1, view_1384, permute_904);  arg572_1 = view_1384 = permute_904 = None
        view_1385 = torch.ops.aten.view.default(addmm_282, [8, 576, 768]);  addmm_282 = None
        mul_693 = torch.ops.aten.mul.Tensor(arg562_1, view_1385);  arg562_1 = view_1385 = None
        add_625 = torch.ops.aten.add.Tensor(add_620, mul_693);  add_620 = mul_693 = None
        clone_958 = torch.ops.aten.clone.default(add_625, memory_format = torch.contiguous_format)
        var_mean_140 = torch.ops.aten.var_mean.correction(clone_958, [2], correction = 0, keepdim = True)
        getitem_288 = var_mean_140[0]
        getitem_289 = var_mean_140[1];  var_mean_140 = None
        add_626 = torch.ops.aten.add.Tensor(getitem_288, 1e-06);  getitem_288 = None
        rsqrt_140 = torch.ops.aten.rsqrt.default(add_626);  add_626 = None
        sub_208 = torch.ops.aten.sub.Tensor(clone_958, getitem_289);  clone_958 = getitem_289 = None
        mul_694 = torch.ops.aten.mul.Tensor(sub_208, rsqrt_140);  sub_208 = rsqrt_140 = None
        mul_695 = torch.ops.aten.mul.Tensor(mul_694, arg574_1);  mul_694 = arg574_1 = None
        add_627 = torch.ops.aten.add.Tensor(mul_695, arg575_1);  mul_695 = arg575_1 = None
        view_1386 = torch.ops.aten.view.default(add_627, [4608, 768]);  add_627 = None
        permute_905 = torch.ops.aten.permute.default(arg576_1, [1, 0]);  arg576_1 = None
        addmm_283 = torch.ops.aten.addmm.default(arg577_1, view_1386, permute_905);  arg577_1 = view_1386 = permute_905 = None
        view_1387 = torch.ops.aten.view.default(addmm_283, [8, 576, 3072]);  addmm_283 = None
        mul_696 = torch.ops.aten.mul.Tensor(view_1387, 0.5)
        mul_697 = torch.ops.aten.mul.Tensor(view_1387, 0.7071067811865476);  view_1387 = None
        erf_69 = torch.ops.aten.erf.default(mul_697);  mul_697 = None
        add_628 = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_698 = torch.ops.aten.mul.Tensor(mul_696, add_628);  mul_696 = add_628 = None
        view_1388 = torch.ops.aten.view.default(mul_698, [4608, 3072]);  mul_698 = None
        permute_906 = torch.ops.aten.permute.default(arg578_1, [1, 0]);  arg578_1 = None
        addmm_284 = torch.ops.aten.addmm.default(arg579_1, view_1388, permute_906);  arg579_1 = view_1388 = permute_906 = None
        view_1389 = torch.ops.aten.view.default(addmm_284, [8, 576, 768]);  addmm_284 = None
        mul_699 = torch.ops.aten.mul.Tensor(arg573_1, view_1389);  arg573_1 = view_1389 = None
        add_629 = torch.ops.aten.add.Tensor(add_625, mul_699);  add_625 = mul_699 = None
        clone_961 = torch.ops.aten.clone.default(add_629, memory_format = torch.contiguous_format)
        var_mean_141 = torch.ops.aten.var_mean.correction(clone_961, [2], correction = 0, keepdim = True)
        getitem_290 = var_mean_141[0]
        getitem_291 = var_mean_141[1];  var_mean_141 = None
        add_630 = torch.ops.aten.add.Tensor(getitem_290, 1e-06);  getitem_290 = None
        rsqrt_141 = torch.ops.aten.rsqrt.default(add_630);  add_630 = None
        sub_209 = torch.ops.aten.sub.Tensor(clone_961, getitem_291);  clone_961 = getitem_291 = None
        mul_700 = torch.ops.aten.mul.Tensor(sub_209, rsqrt_141);  sub_209 = rsqrt_141 = None
        mul_701 = torch.ops.aten.mul.Tensor(mul_700, arg581_1);  mul_700 = arg581_1 = None
        add_631 = torch.ops.aten.add.Tensor(mul_701, arg582_1);  mul_701 = arg582_1 = None
        view_1390 = torch.ops.aten.view.default(add_631, [4608, 768]);  add_631 = None
        permute_907 = torch.ops.aten.permute.default(arg583_1, [1, 0]);  arg583_1 = None
        addmm_285 = torch.ops.aten.addmm.default(arg584_1, view_1390, permute_907);  arg584_1 = view_1390 = permute_907 = None
        view_1391 = torch.ops.aten.view.default(addmm_285, [8, 576, 2304]);  addmm_285 = None
        view_1392 = torch.ops.aten.view.default(view_1391, [8, 576, 3, 16, 48]);  view_1391 = None
        permute_908 = torch.ops.aten.permute.default(view_1392, [2, 0, 3, 1, 4]);  view_1392 = None
        select_207 = torch.ops.aten.select.int(permute_908, 0, 0)
        mul_702 = torch.ops.aten.mul.Tensor(select_207, 0.14433756729740643);  select_207 = None
        select_208 = torch.ops.aten.select.int(permute_908, 0, 1)
        select_209 = torch.ops.aten.select.int(permute_908, 0, 2);  permute_908 = None
        permute_909 = torch.ops.aten.permute.default(select_208, [0, 1, 3, 2]);  select_208 = None
        expand_273 = torch.ops.aten.expand.default(mul_702, [8, 16, 576, 48]);  mul_702 = None
        clone_962 = torch.ops.aten.clone.default(expand_273, memory_format = torch.contiguous_format);  expand_273 = None
        view_1393 = torch.ops.aten.view.default(clone_962, [128, 576, 48]);  clone_962 = None
        expand_274 = torch.ops.aten.expand.default(permute_909, [8, 16, 48, 576]);  permute_909 = None
        clone_963 = torch.ops.aten.clone.default(expand_274, memory_format = torch.contiguous_format);  expand_274 = None
        view_1394 = torch.ops.aten.view.default(clone_963, [128, 48, 576]);  clone_963 = None
        bmm_136 = torch.ops.aten.bmm.default(view_1393, view_1394);  view_1393 = view_1394 = None
        view_1395 = torch.ops.aten.view.default(bmm_136, [8, 16, 576, 576]);  bmm_136 = None
        permute_910 = torch.ops.aten.permute.default(view_1395, [0, 2, 3, 1]);  view_1395 = None
        permute_911 = torch.ops.aten.permute.default(arg585_1, [1, 0]);  arg585_1 = None
        clone_964 = torch.ops.aten.clone.default(permute_910, memory_format = torch.contiguous_format);  permute_910 = None
        view_1396 = torch.ops.aten.view.default(clone_964, [2654208, 16]);  clone_964 = None
        mm_136 = torch.ops.aten.mm.default(view_1396, permute_911);  view_1396 = permute_911 = None
        view_1397 = torch.ops.aten.view.default(mm_136, [8, 576, 576, 16]);  mm_136 = None
        add_632 = torch.ops.aten.add.Tensor(view_1397, arg586_1);  view_1397 = arg586_1 = None
        permute_912 = torch.ops.aten.permute.default(add_632, [0, 3, 1, 2]);  add_632 = None
        clone_965 = torch.ops.aten.clone.default(permute_912, memory_format = torch.contiguous_format);  permute_912 = None
        amax_68 = torch.ops.aten.amax.default(clone_965, [-1], True)
        sub_210 = torch.ops.aten.sub.Tensor(clone_965, amax_68);  clone_965 = amax_68 = None
        exp_68 = torch.ops.aten.exp.default(sub_210);  sub_210 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(exp_68, [-1], True)
        div_68 = torch.ops.aten.div.Tensor(exp_68, sum_69);  exp_68 = sum_69 = None
        permute_913 = torch.ops.aten.permute.default(div_68, [0, 2, 3, 1]);  div_68 = None
        permute_914 = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
        clone_966 = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None
        view_1398 = torch.ops.aten.view.default(clone_966, [2654208, 16]);  clone_966 = None
        mm_137 = torch.ops.aten.mm.default(view_1398, permute_914);  view_1398 = permute_914 = None
        view_1399 = torch.ops.aten.view.default(mm_137, [8, 576, 576, 16]);  mm_137 = None
        add_633 = torch.ops.aten.add.Tensor(view_1399, arg588_1);  view_1399 = arg588_1 = None
        permute_915 = torch.ops.aten.permute.default(add_633, [0, 3, 1, 2]);  add_633 = None
        expand_275 = torch.ops.aten.expand.default(permute_915, [8, 16, 576, 576]);  permute_915 = None
        clone_968 = torch.ops.aten.clone.default(expand_275, memory_format = torch.contiguous_format);  expand_275 = None
        view_1400 = torch.ops.aten.view.default(clone_968, [128, 576, 576]);  clone_968 = None
        expand_276 = torch.ops.aten.expand.default(select_209, [8, 16, 576, 48]);  select_209 = None
        clone_969 = torch.ops.aten.clone.default(expand_276, memory_format = torch.contiguous_format);  expand_276 = None
        view_1401 = torch.ops.aten.view.default(clone_969, [128, 576, 48]);  clone_969 = None
        bmm_137 = torch.ops.aten.bmm.default(view_1400, view_1401);  view_1400 = view_1401 = None
        view_1402 = torch.ops.aten.view.default(bmm_137, [8, 16, 576, 48]);  bmm_137 = None
        permute_916 = torch.ops.aten.permute.default(view_1402, [0, 2, 1, 3]);  view_1402 = None
        clone_970 = torch.ops.aten.clone.default(permute_916, memory_format = torch.contiguous_format);  permute_916 = None
        view_1403 = torch.ops.aten.view.default(clone_970, [8, 576, 768]);  clone_970 = None
        view_1404 = torch.ops.aten.view.default(view_1403, [4608, 768]);  view_1403 = None
        permute_917 = torch.ops.aten.permute.default(arg589_1, [1, 0]);  arg589_1 = None
        addmm_286 = torch.ops.aten.addmm.default(arg590_1, view_1404, permute_917);  arg590_1 = view_1404 = permute_917 = None
        view_1405 = torch.ops.aten.view.default(addmm_286, [8, 576, 768]);  addmm_286 = None
        mul_703 = torch.ops.aten.mul.Tensor(arg580_1, view_1405);  arg580_1 = view_1405 = None
        add_634 = torch.ops.aten.add.Tensor(add_629, mul_703);  add_629 = mul_703 = None
        clone_972 = torch.ops.aten.clone.default(add_634, memory_format = torch.contiguous_format)
        var_mean_142 = torch.ops.aten.var_mean.correction(clone_972, [2], correction = 0, keepdim = True)
        getitem_292 = var_mean_142[0]
        getitem_293 = var_mean_142[1];  var_mean_142 = None
        add_635 = torch.ops.aten.add.Tensor(getitem_292, 1e-06);  getitem_292 = None
        rsqrt_142 = torch.ops.aten.rsqrt.default(add_635);  add_635 = None
        sub_211 = torch.ops.aten.sub.Tensor(clone_972, getitem_293);  clone_972 = getitem_293 = None
        mul_704 = torch.ops.aten.mul.Tensor(sub_211, rsqrt_142);  sub_211 = rsqrt_142 = None
        mul_705 = torch.ops.aten.mul.Tensor(mul_704, arg592_1);  mul_704 = arg592_1 = None
        add_636 = torch.ops.aten.add.Tensor(mul_705, arg593_1);  mul_705 = arg593_1 = None
        view_1406 = torch.ops.aten.view.default(add_636, [4608, 768]);  add_636 = None
        permute_918 = torch.ops.aten.permute.default(arg594_1, [1, 0]);  arg594_1 = None
        addmm_287 = torch.ops.aten.addmm.default(arg595_1, view_1406, permute_918);  arg595_1 = view_1406 = permute_918 = None
        view_1407 = torch.ops.aten.view.default(addmm_287, [8, 576, 3072]);  addmm_287 = None
        mul_706 = torch.ops.aten.mul.Tensor(view_1407, 0.5)
        mul_707 = torch.ops.aten.mul.Tensor(view_1407, 0.7071067811865476);  view_1407 = None
        erf_70 = torch.ops.aten.erf.default(mul_707);  mul_707 = None
        add_637 = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_708 = torch.ops.aten.mul.Tensor(mul_706, add_637);  mul_706 = add_637 = None
        view_1408 = torch.ops.aten.view.default(mul_708, [4608, 3072]);  mul_708 = None
        permute_919 = torch.ops.aten.permute.default(arg596_1, [1, 0]);  arg596_1 = None
        addmm_288 = torch.ops.aten.addmm.default(arg597_1, view_1408, permute_919);  arg597_1 = view_1408 = permute_919 = None
        view_1409 = torch.ops.aten.view.default(addmm_288, [8, 576, 768]);  addmm_288 = None
        mul_709 = torch.ops.aten.mul.Tensor(arg591_1, view_1409);  arg591_1 = view_1409 = None
        add_638 = torch.ops.aten.add.Tensor(add_634, mul_709);  add_634 = mul_709 = None
        clone_975 = torch.ops.aten.clone.default(add_638, memory_format = torch.contiguous_format)
        var_mean_143 = torch.ops.aten.var_mean.correction(clone_975, [2], correction = 0, keepdim = True)
        getitem_294 = var_mean_143[0]
        getitem_295 = var_mean_143[1];  var_mean_143 = None
        add_639 = torch.ops.aten.add.Tensor(getitem_294, 1e-06);  getitem_294 = None
        rsqrt_143 = torch.ops.aten.rsqrt.default(add_639);  add_639 = None
        sub_212 = torch.ops.aten.sub.Tensor(clone_975, getitem_295);  clone_975 = getitem_295 = None
        mul_710 = torch.ops.aten.mul.Tensor(sub_212, rsqrt_143);  sub_212 = rsqrt_143 = None
        mul_711 = torch.ops.aten.mul.Tensor(mul_710, arg599_1);  mul_710 = arg599_1 = None
        add_640 = torch.ops.aten.add.Tensor(mul_711, arg600_1);  mul_711 = arg600_1 = None
        view_1410 = torch.ops.aten.view.default(add_640, [4608, 768]);  add_640 = None
        permute_920 = torch.ops.aten.permute.default(arg601_1, [1, 0]);  arg601_1 = None
        addmm_289 = torch.ops.aten.addmm.default(arg602_1, view_1410, permute_920);  arg602_1 = view_1410 = permute_920 = None
        view_1411 = torch.ops.aten.view.default(addmm_289, [8, 576, 2304]);  addmm_289 = None
        view_1412 = torch.ops.aten.view.default(view_1411, [8, 576, 3, 16, 48]);  view_1411 = None
        permute_921 = torch.ops.aten.permute.default(view_1412, [2, 0, 3, 1, 4]);  view_1412 = None
        select_210 = torch.ops.aten.select.int(permute_921, 0, 0)
        mul_712 = torch.ops.aten.mul.Tensor(select_210, 0.14433756729740643);  select_210 = None
        select_211 = torch.ops.aten.select.int(permute_921, 0, 1)
        select_212 = torch.ops.aten.select.int(permute_921, 0, 2);  permute_921 = None
        permute_922 = torch.ops.aten.permute.default(select_211, [0, 1, 3, 2]);  select_211 = None
        expand_277 = torch.ops.aten.expand.default(mul_712, [8, 16, 576, 48]);  mul_712 = None
        clone_976 = torch.ops.aten.clone.default(expand_277, memory_format = torch.contiguous_format);  expand_277 = None
        view_1413 = torch.ops.aten.view.default(clone_976, [128, 576, 48]);  clone_976 = None
        expand_278 = torch.ops.aten.expand.default(permute_922, [8, 16, 48, 576]);  permute_922 = None
        clone_977 = torch.ops.aten.clone.default(expand_278, memory_format = torch.contiguous_format);  expand_278 = None
        view_1414 = torch.ops.aten.view.default(clone_977, [128, 48, 576]);  clone_977 = None
        bmm_138 = torch.ops.aten.bmm.default(view_1413, view_1414);  view_1413 = view_1414 = None
        view_1415 = torch.ops.aten.view.default(bmm_138, [8, 16, 576, 576]);  bmm_138 = None
        permute_923 = torch.ops.aten.permute.default(view_1415, [0, 2, 3, 1]);  view_1415 = None
        permute_924 = torch.ops.aten.permute.default(arg603_1, [1, 0]);  arg603_1 = None
        clone_978 = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
        view_1416 = torch.ops.aten.view.default(clone_978, [2654208, 16]);  clone_978 = None
        mm_138 = torch.ops.aten.mm.default(view_1416, permute_924);  view_1416 = permute_924 = None
        view_1417 = torch.ops.aten.view.default(mm_138, [8, 576, 576, 16]);  mm_138 = None
        add_641 = torch.ops.aten.add.Tensor(view_1417, arg604_1);  view_1417 = arg604_1 = None
        permute_925 = torch.ops.aten.permute.default(add_641, [0, 3, 1, 2]);  add_641 = None
        clone_979 = torch.ops.aten.clone.default(permute_925, memory_format = torch.contiguous_format);  permute_925 = None
        amax_69 = torch.ops.aten.amax.default(clone_979, [-1], True)
        sub_213 = torch.ops.aten.sub.Tensor(clone_979, amax_69);  clone_979 = amax_69 = None
        exp_69 = torch.ops.aten.exp.default(sub_213);  sub_213 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(exp_69, [-1], True)
        div_69 = torch.ops.aten.div.Tensor(exp_69, sum_70);  exp_69 = sum_70 = None
        permute_926 = torch.ops.aten.permute.default(div_69, [0, 2, 3, 1]);  div_69 = None
        permute_927 = torch.ops.aten.permute.default(arg605_1, [1, 0]);  arg605_1 = None
        clone_980 = torch.ops.aten.clone.default(permute_926, memory_format = torch.contiguous_format);  permute_926 = None
        view_1418 = torch.ops.aten.view.default(clone_980, [2654208, 16]);  clone_980 = None
        mm_139 = torch.ops.aten.mm.default(view_1418, permute_927);  view_1418 = permute_927 = None
        view_1419 = torch.ops.aten.view.default(mm_139, [8, 576, 576, 16]);  mm_139 = None
        add_642 = torch.ops.aten.add.Tensor(view_1419, arg606_1);  view_1419 = arg606_1 = None
        permute_928 = torch.ops.aten.permute.default(add_642, [0, 3, 1, 2]);  add_642 = None
        expand_279 = torch.ops.aten.expand.default(permute_928, [8, 16, 576, 576]);  permute_928 = None
        clone_982 = torch.ops.aten.clone.default(expand_279, memory_format = torch.contiguous_format);  expand_279 = None
        view_1420 = torch.ops.aten.view.default(clone_982, [128, 576, 576]);  clone_982 = None
        expand_280 = torch.ops.aten.expand.default(select_212, [8, 16, 576, 48]);  select_212 = None
        clone_983 = torch.ops.aten.clone.default(expand_280, memory_format = torch.contiguous_format);  expand_280 = None
        view_1421 = torch.ops.aten.view.default(clone_983, [128, 576, 48]);  clone_983 = None
        bmm_139 = torch.ops.aten.bmm.default(view_1420, view_1421);  view_1420 = view_1421 = None
        view_1422 = torch.ops.aten.view.default(bmm_139, [8, 16, 576, 48]);  bmm_139 = None
        permute_929 = torch.ops.aten.permute.default(view_1422, [0, 2, 1, 3]);  view_1422 = None
        clone_984 = torch.ops.aten.clone.default(permute_929, memory_format = torch.contiguous_format);  permute_929 = None
        view_1423 = torch.ops.aten.view.default(clone_984, [8, 576, 768]);  clone_984 = None
        view_1424 = torch.ops.aten.view.default(view_1423, [4608, 768]);  view_1423 = None
        permute_930 = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
        addmm_290 = torch.ops.aten.addmm.default(arg608_1, view_1424, permute_930);  arg608_1 = view_1424 = permute_930 = None
        view_1425 = torch.ops.aten.view.default(addmm_290, [8, 576, 768]);  addmm_290 = None
        mul_713 = torch.ops.aten.mul.Tensor(arg598_1, view_1425);  arg598_1 = view_1425 = None
        add_643 = torch.ops.aten.add.Tensor(add_638, mul_713);  add_638 = mul_713 = None
        clone_986 = torch.ops.aten.clone.default(add_643, memory_format = torch.contiguous_format)
        var_mean_144 = torch.ops.aten.var_mean.correction(clone_986, [2], correction = 0, keepdim = True)
        getitem_296 = var_mean_144[0]
        getitem_297 = var_mean_144[1];  var_mean_144 = None
        add_644 = torch.ops.aten.add.Tensor(getitem_296, 1e-06);  getitem_296 = None
        rsqrt_144 = torch.ops.aten.rsqrt.default(add_644);  add_644 = None
        sub_214 = torch.ops.aten.sub.Tensor(clone_986, getitem_297);  clone_986 = getitem_297 = None
        mul_714 = torch.ops.aten.mul.Tensor(sub_214, rsqrt_144);  sub_214 = rsqrt_144 = None
        mul_715 = torch.ops.aten.mul.Tensor(mul_714, arg610_1);  mul_714 = arg610_1 = None
        add_645 = torch.ops.aten.add.Tensor(mul_715, arg611_1);  mul_715 = arg611_1 = None
        view_1426 = torch.ops.aten.view.default(add_645, [4608, 768]);  add_645 = None
        permute_931 = torch.ops.aten.permute.default(arg612_1, [1, 0]);  arg612_1 = None
        addmm_291 = torch.ops.aten.addmm.default(arg613_1, view_1426, permute_931);  arg613_1 = view_1426 = permute_931 = None
        view_1427 = torch.ops.aten.view.default(addmm_291, [8, 576, 3072]);  addmm_291 = None
        mul_716 = torch.ops.aten.mul.Tensor(view_1427, 0.5)
        mul_717 = torch.ops.aten.mul.Tensor(view_1427, 0.7071067811865476);  view_1427 = None
        erf_71 = torch.ops.aten.erf.default(mul_717);  mul_717 = None
        add_646 = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_718 = torch.ops.aten.mul.Tensor(mul_716, add_646);  mul_716 = add_646 = None
        view_1428 = torch.ops.aten.view.default(mul_718, [4608, 3072]);  mul_718 = None
        permute_932 = torch.ops.aten.permute.default(arg614_1, [1, 0]);  arg614_1 = None
        addmm_292 = torch.ops.aten.addmm.default(arg615_1, view_1428, permute_932);  arg615_1 = view_1428 = permute_932 = None
        view_1429 = torch.ops.aten.view.default(addmm_292, [8, 576, 768]);  addmm_292 = None
        mul_719 = torch.ops.aten.mul.Tensor(arg609_1, view_1429);  arg609_1 = view_1429 = None
        add_647 = torch.ops.aten.add.Tensor(add_643, mul_719);  add_643 = mul_719 = None
        clone_989 = torch.ops.aten.clone.default(add_647, memory_format = torch.contiguous_format)
        var_mean_145 = torch.ops.aten.var_mean.correction(clone_989, [2], correction = 0, keepdim = True)
        getitem_298 = var_mean_145[0]
        getitem_299 = var_mean_145[1];  var_mean_145 = None
        add_648 = torch.ops.aten.add.Tensor(getitem_298, 1e-06);  getitem_298 = None
        rsqrt_145 = torch.ops.aten.rsqrt.default(add_648);  add_648 = None
        sub_215 = torch.ops.aten.sub.Tensor(clone_989, getitem_299);  clone_989 = getitem_299 = None
        mul_720 = torch.ops.aten.mul.Tensor(sub_215, rsqrt_145);  sub_215 = rsqrt_145 = None
        mul_721 = torch.ops.aten.mul.Tensor(mul_720, arg617_1);  mul_720 = arg617_1 = None
        add_649 = torch.ops.aten.add.Tensor(mul_721, arg618_1);  mul_721 = arg618_1 = None
        view_1430 = torch.ops.aten.view.default(add_649, [4608, 768]);  add_649 = None
        permute_933 = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
        addmm_293 = torch.ops.aten.addmm.default(arg620_1, view_1430, permute_933);  arg620_1 = view_1430 = permute_933 = None
        view_1431 = torch.ops.aten.view.default(addmm_293, [8, 576, 2304]);  addmm_293 = None
        view_1432 = torch.ops.aten.view.default(view_1431, [8, 576, 3, 16, 48]);  view_1431 = None
        permute_934 = torch.ops.aten.permute.default(view_1432, [2, 0, 3, 1, 4]);  view_1432 = None
        select_213 = torch.ops.aten.select.int(permute_934, 0, 0)
        mul_722 = torch.ops.aten.mul.Tensor(select_213, 0.14433756729740643);  select_213 = None
        select_214 = torch.ops.aten.select.int(permute_934, 0, 1)
        select_215 = torch.ops.aten.select.int(permute_934, 0, 2);  permute_934 = None
        permute_935 = torch.ops.aten.permute.default(select_214, [0, 1, 3, 2]);  select_214 = None
        expand_281 = torch.ops.aten.expand.default(mul_722, [8, 16, 576, 48]);  mul_722 = None
        clone_990 = torch.ops.aten.clone.default(expand_281, memory_format = torch.contiguous_format);  expand_281 = None
        view_1433 = torch.ops.aten.view.default(clone_990, [128, 576, 48]);  clone_990 = None
        expand_282 = torch.ops.aten.expand.default(permute_935, [8, 16, 48, 576]);  permute_935 = None
        clone_991 = torch.ops.aten.clone.default(expand_282, memory_format = torch.contiguous_format);  expand_282 = None
        view_1434 = torch.ops.aten.view.default(clone_991, [128, 48, 576]);  clone_991 = None
        bmm_140 = torch.ops.aten.bmm.default(view_1433, view_1434);  view_1433 = view_1434 = None
        view_1435 = torch.ops.aten.view.default(bmm_140, [8, 16, 576, 576]);  bmm_140 = None
        permute_936 = torch.ops.aten.permute.default(view_1435, [0, 2, 3, 1]);  view_1435 = None
        permute_937 = torch.ops.aten.permute.default(arg621_1, [1, 0]);  arg621_1 = None
        clone_992 = torch.ops.aten.clone.default(permute_936, memory_format = torch.contiguous_format);  permute_936 = None
        view_1436 = torch.ops.aten.view.default(clone_992, [2654208, 16]);  clone_992 = None
        mm_140 = torch.ops.aten.mm.default(view_1436, permute_937);  view_1436 = permute_937 = None
        view_1437 = torch.ops.aten.view.default(mm_140, [8, 576, 576, 16]);  mm_140 = None
        add_650 = torch.ops.aten.add.Tensor(view_1437, arg622_1);  view_1437 = arg622_1 = None
        permute_938 = torch.ops.aten.permute.default(add_650, [0, 3, 1, 2]);  add_650 = None
        clone_993 = torch.ops.aten.clone.default(permute_938, memory_format = torch.contiguous_format);  permute_938 = None
        amax_70 = torch.ops.aten.amax.default(clone_993, [-1], True)
        sub_216 = torch.ops.aten.sub.Tensor(clone_993, amax_70);  clone_993 = amax_70 = None
        exp_70 = torch.ops.aten.exp.default(sub_216);  sub_216 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(exp_70, [-1], True)
        div_70 = torch.ops.aten.div.Tensor(exp_70, sum_71);  exp_70 = sum_71 = None
        permute_939 = torch.ops.aten.permute.default(div_70, [0, 2, 3, 1]);  div_70 = None
        permute_940 = torch.ops.aten.permute.default(arg623_1, [1, 0]);  arg623_1 = None
        clone_994 = torch.ops.aten.clone.default(permute_939, memory_format = torch.contiguous_format);  permute_939 = None
        view_1438 = torch.ops.aten.view.default(clone_994, [2654208, 16]);  clone_994 = None
        mm_141 = torch.ops.aten.mm.default(view_1438, permute_940);  view_1438 = permute_940 = None
        view_1439 = torch.ops.aten.view.default(mm_141, [8, 576, 576, 16]);  mm_141 = None
        add_651 = torch.ops.aten.add.Tensor(view_1439, arg624_1);  view_1439 = arg624_1 = None
        permute_941 = torch.ops.aten.permute.default(add_651, [0, 3, 1, 2]);  add_651 = None
        expand_283 = torch.ops.aten.expand.default(permute_941, [8, 16, 576, 576]);  permute_941 = None
        clone_996 = torch.ops.aten.clone.default(expand_283, memory_format = torch.contiguous_format);  expand_283 = None
        view_1440 = torch.ops.aten.view.default(clone_996, [128, 576, 576]);  clone_996 = None
        expand_284 = torch.ops.aten.expand.default(select_215, [8, 16, 576, 48]);  select_215 = None
        clone_997 = torch.ops.aten.clone.default(expand_284, memory_format = torch.contiguous_format);  expand_284 = None
        view_1441 = torch.ops.aten.view.default(clone_997, [128, 576, 48]);  clone_997 = None
        bmm_141 = torch.ops.aten.bmm.default(view_1440, view_1441);  view_1440 = view_1441 = None
        view_1442 = torch.ops.aten.view.default(bmm_141, [8, 16, 576, 48]);  bmm_141 = None
        permute_942 = torch.ops.aten.permute.default(view_1442, [0, 2, 1, 3]);  view_1442 = None
        clone_998 = torch.ops.aten.clone.default(permute_942, memory_format = torch.contiguous_format);  permute_942 = None
        view_1443 = torch.ops.aten.view.default(clone_998, [8, 576, 768]);  clone_998 = None
        view_1444 = torch.ops.aten.view.default(view_1443, [4608, 768]);  view_1443 = None
        permute_943 = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
        addmm_294 = torch.ops.aten.addmm.default(arg626_1, view_1444, permute_943);  arg626_1 = view_1444 = permute_943 = None
        view_1445 = torch.ops.aten.view.default(addmm_294, [8, 576, 768]);  addmm_294 = None
        mul_723 = torch.ops.aten.mul.Tensor(arg616_1, view_1445);  arg616_1 = view_1445 = None
        add_652 = torch.ops.aten.add.Tensor(add_647, mul_723);  add_647 = mul_723 = None
        clone_1000 = torch.ops.aten.clone.default(add_652, memory_format = torch.contiguous_format)
        var_mean_146 = torch.ops.aten.var_mean.correction(clone_1000, [2], correction = 0, keepdim = True)
        getitem_300 = var_mean_146[0]
        getitem_301 = var_mean_146[1];  var_mean_146 = None
        add_653 = torch.ops.aten.add.Tensor(getitem_300, 1e-06);  getitem_300 = None
        rsqrt_146 = torch.ops.aten.rsqrt.default(add_653);  add_653 = None
        sub_217 = torch.ops.aten.sub.Tensor(clone_1000, getitem_301);  clone_1000 = getitem_301 = None
        mul_724 = torch.ops.aten.mul.Tensor(sub_217, rsqrt_146);  sub_217 = rsqrt_146 = None
        mul_725 = torch.ops.aten.mul.Tensor(mul_724, arg628_1);  mul_724 = arg628_1 = None
        add_654 = torch.ops.aten.add.Tensor(mul_725, arg629_1);  mul_725 = arg629_1 = None
        view_1446 = torch.ops.aten.view.default(add_654, [4608, 768]);  add_654 = None
        permute_944 = torch.ops.aten.permute.default(arg630_1, [1, 0]);  arg630_1 = None
        addmm_295 = torch.ops.aten.addmm.default(arg631_1, view_1446, permute_944);  arg631_1 = view_1446 = permute_944 = None
        view_1447 = torch.ops.aten.view.default(addmm_295, [8, 576, 3072]);  addmm_295 = None
        mul_726 = torch.ops.aten.mul.Tensor(view_1447, 0.5)
        mul_727 = torch.ops.aten.mul.Tensor(view_1447, 0.7071067811865476);  view_1447 = None
        erf_72 = torch.ops.aten.erf.default(mul_727);  mul_727 = None
        add_655 = torch.ops.aten.add.Tensor(erf_72, 1);  erf_72 = None
        mul_728 = torch.ops.aten.mul.Tensor(mul_726, add_655);  mul_726 = add_655 = None
        view_1448 = torch.ops.aten.view.default(mul_728, [4608, 3072]);  mul_728 = None
        permute_945 = torch.ops.aten.permute.default(arg632_1, [1, 0]);  arg632_1 = None
        addmm_296 = torch.ops.aten.addmm.default(arg633_1, view_1448, permute_945);  arg633_1 = view_1448 = permute_945 = None
        view_1449 = torch.ops.aten.view.default(addmm_296, [8, 576, 768]);  addmm_296 = None
        mul_729 = torch.ops.aten.mul.Tensor(arg627_1, view_1449);  arg627_1 = view_1449 = None
        add_656 = torch.ops.aten.add.Tensor(add_652, mul_729);  add_652 = mul_729 = None
        clone_1003 = torch.ops.aten.clone.default(add_656, memory_format = torch.contiguous_format)
        var_mean_147 = torch.ops.aten.var_mean.correction(clone_1003, [2], correction = 0, keepdim = True)
        getitem_302 = var_mean_147[0]
        getitem_303 = var_mean_147[1];  var_mean_147 = None
        add_657 = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_147 = torch.ops.aten.rsqrt.default(add_657);  add_657 = None
        sub_218 = torch.ops.aten.sub.Tensor(clone_1003, getitem_303);  clone_1003 = getitem_303 = None
        mul_730 = torch.ops.aten.mul.Tensor(sub_218, rsqrt_147);  sub_218 = rsqrt_147 = None
        mul_731 = torch.ops.aten.mul.Tensor(mul_730, arg635_1);  mul_730 = arg635_1 = None
        add_658 = torch.ops.aten.add.Tensor(mul_731, arg636_1);  mul_731 = arg636_1 = None
        view_1450 = torch.ops.aten.view.default(add_658, [4608, 768]);  add_658 = None
        permute_946 = torch.ops.aten.permute.default(arg637_1, [1, 0]);  arg637_1 = None
        addmm_297 = torch.ops.aten.addmm.default(arg638_1, view_1450, permute_946);  arg638_1 = view_1450 = permute_946 = None
        view_1451 = torch.ops.aten.view.default(addmm_297, [8, 576, 2304]);  addmm_297 = None
        view_1452 = torch.ops.aten.view.default(view_1451, [8, 576, 3, 16, 48]);  view_1451 = None
        permute_947 = torch.ops.aten.permute.default(view_1452, [2, 0, 3, 1, 4]);  view_1452 = None
        select_216 = torch.ops.aten.select.int(permute_947, 0, 0)
        mul_732 = torch.ops.aten.mul.Tensor(select_216, 0.14433756729740643);  select_216 = None
        select_217 = torch.ops.aten.select.int(permute_947, 0, 1)
        select_218 = torch.ops.aten.select.int(permute_947, 0, 2);  permute_947 = None
        permute_948 = torch.ops.aten.permute.default(select_217, [0, 1, 3, 2]);  select_217 = None
        expand_285 = torch.ops.aten.expand.default(mul_732, [8, 16, 576, 48]);  mul_732 = None
        clone_1004 = torch.ops.aten.clone.default(expand_285, memory_format = torch.contiguous_format);  expand_285 = None
        view_1453 = torch.ops.aten.view.default(clone_1004, [128, 576, 48]);  clone_1004 = None
        expand_286 = torch.ops.aten.expand.default(permute_948, [8, 16, 48, 576]);  permute_948 = None
        clone_1005 = torch.ops.aten.clone.default(expand_286, memory_format = torch.contiguous_format);  expand_286 = None
        view_1454 = torch.ops.aten.view.default(clone_1005, [128, 48, 576]);  clone_1005 = None
        bmm_142 = torch.ops.aten.bmm.default(view_1453, view_1454);  view_1453 = view_1454 = None
        view_1455 = torch.ops.aten.view.default(bmm_142, [8, 16, 576, 576]);  bmm_142 = None
        permute_949 = torch.ops.aten.permute.default(view_1455, [0, 2, 3, 1]);  view_1455 = None
        permute_950 = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
        clone_1006 = torch.ops.aten.clone.default(permute_949, memory_format = torch.contiguous_format);  permute_949 = None
        view_1456 = torch.ops.aten.view.default(clone_1006, [2654208, 16]);  clone_1006 = None
        mm_142 = torch.ops.aten.mm.default(view_1456, permute_950);  view_1456 = permute_950 = None
        view_1457 = torch.ops.aten.view.default(mm_142, [8, 576, 576, 16]);  mm_142 = None
        add_659 = torch.ops.aten.add.Tensor(view_1457, arg640_1);  view_1457 = arg640_1 = None
        permute_951 = torch.ops.aten.permute.default(add_659, [0, 3, 1, 2]);  add_659 = None
        clone_1007 = torch.ops.aten.clone.default(permute_951, memory_format = torch.contiguous_format);  permute_951 = None
        amax_71 = torch.ops.aten.amax.default(clone_1007, [-1], True)
        sub_219 = torch.ops.aten.sub.Tensor(clone_1007, amax_71);  clone_1007 = amax_71 = None
        exp_71 = torch.ops.aten.exp.default(sub_219);  sub_219 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(exp_71, [-1], True)
        div_71 = torch.ops.aten.div.Tensor(exp_71, sum_72);  exp_71 = sum_72 = None
        permute_952 = torch.ops.aten.permute.default(div_71, [0, 2, 3, 1]);  div_71 = None
        permute_953 = torch.ops.aten.permute.default(arg641_1, [1, 0]);  arg641_1 = None
        clone_1008 = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
        view_1458 = torch.ops.aten.view.default(clone_1008, [2654208, 16]);  clone_1008 = None
        mm_143 = torch.ops.aten.mm.default(view_1458, permute_953);  view_1458 = permute_953 = None
        view_1459 = torch.ops.aten.view.default(mm_143, [8, 576, 576, 16]);  mm_143 = None
        add_660 = torch.ops.aten.add.Tensor(view_1459, arg642_1);  view_1459 = arg642_1 = None
        permute_954 = torch.ops.aten.permute.default(add_660, [0, 3, 1, 2]);  add_660 = None
        expand_287 = torch.ops.aten.expand.default(permute_954, [8, 16, 576, 576]);  permute_954 = None
        clone_1010 = torch.ops.aten.clone.default(expand_287, memory_format = torch.contiguous_format);  expand_287 = None
        view_1460 = torch.ops.aten.view.default(clone_1010, [128, 576, 576]);  clone_1010 = None
        expand_288 = torch.ops.aten.expand.default(select_218, [8, 16, 576, 48]);  select_218 = None
        clone_1011 = torch.ops.aten.clone.default(expand_288, memory_format = torch.contiguous_format);  expand_288 = None
        view_1461 = torch.ops.aten.view.default(clone_1011, [128, 576, 48]);  clone_1011 = None
        bmm_143 = torch.ops.aten.bmm.default(view_1460, view_1461);  view_1460 = view_1461 = None
        view_1462 = torch.ops.aten.view.default(bmm_143, [8, 16, 576, 48]);  bmm_143 = None
        permute_955 = torch.ops.aten.permute.default(view_1462, [0, 2, 1, 3]);  view_1462 = None
        clone_1012 = torch.ops.aten.clone.default(permute_955, memory_format = torch.contiguous_format);  permute_955 = None
        view_1463 = torch.ops.aten.view.default(clone_1012, [8, 576, 768]);  clone_1012 = None
        view_1464 = torch.ops.aten.view.default(view_1463, [4608, 768]);  view_1463 = None
        permute_956 = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
        addmm_298 = torch.ops.aten.addmm.default(arg644_1, view_1464, permute_956);  arg644_1 = view_1464 = permute_956 = None
        view_1465 = torch.ops.aten.view.default(addmm_298, [8, 576, 768]);  addmm_298 = None
        mul_733 = torch.ops.aten.mul.Tensor(arg634_1, view_1465);  arg634_1 = view_1465 = None
        add_661 = torch.ops.aten.add.Tensor(add_656, mul_733);  add_656 = mul_733 = None
        clone_1014 = torch.ops.aten.clone.default(add_661, memory_format = torch.contiguous_format)
        var_mean_148 = torch.ops.aten.var_mean.correction(clone_1014, [2], correction = 0, keepdim = True)
        getitem_304 = var_mean_148[0]
        getitem_305 = var_mean_148[1];  var_mean_148 = None
        add_662 = torch.ops.aten.add.Tensor(getitem_304, 1e-06);  getitem_304 = None
        rsqrt_148 = torch.ops.aten.rsqrt.default(add_662);  add_662 = None
        sub_220 = torch.ops.aten.sub.Tensor(clone_1014, getitem_305);  clone_1014 = getitem_305 = None
        mul_734 = torch.ops.aten.mul.Tensor(sub_220, rsqrt_148);  sub_220 = rsqrt_148 = None
        mul_735 = torch.ops.aten.mul.Tensor(mul_734, arg646_1);  mul_734 = arg646_1 = None
        add_663 = torch.ops.aten.add.Tensor(mul_735, arg647_1);  mul_735 = arg647_1 = None
        view_1466 = torch.ops.aten.view.default(add_663, [4608, 768]);  add_663 = None
        permute_957 = torch.ops.aten.permute.default(arg648_1, [1, 0]);  arg648_1 = None
        addmm_299 = torch.ops.aten.addmm.default(arg649_1, view_1466, permute_957);  arg649_1 = view_1466 = permute_957 = None
        view_1467 = torch.ops.aten.view.default(addmm_299, [8, 576, 3072]);  addmm_299 = None
        mul_736 = torch.ops.aten.mul.Tensor(view_1467, 0.5)
        mul_737 = torch.ops.aten.mul.Tensor(view_1467, 0.7071067811865476);  view_1467 = None
        erf_73 = torch.ops.aten.erf.default(mul_737);  mul_737 = None
        add_664 = torch.ops.aten.add.Tensor(erf_73, 1);  erf_73 = None
        mul_738 = torch.ops.aten.mul.Tensor(mul_736, add_664);  mul_736 = add_664 = None
        view_1468 = torch.ops.aten.view.default(mul_738, [4608, 3072]);  mul_738 = None
        permute_958 = torch.ops.aten.permute.default(arg650_1, [1, 0]);  arg650_1 = None
        addmm_300 = torch.ops.aten.addmm.default(arg651_1, view_1468, permute_958);  arg651_1 = view_1468 = permute_958 = None
        view_1469 = torch.ops.aten.view.default(addmm_300, [8, 576, 768]);  addmm_300 = None
        mul_739 = torch.ops.aten.mul.Tensor(arg645_1, view_1469);  arg645_1 = view_1469 = None
        add_665 = torch.ops.aten.add.Tensor(add_661, mul_739);  add_661 = mul_739 = None
        expand_289 = torch.ops.aten.expand.default(arg652_1, [8, -1, -1]);  arg652_1 = None
        cat_3 = torch.ops.aten.cat.default([expand_289, add_665], 1)
        var_mean_149 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
        getitem_306 = var_mean_149[0]
        getitem_307 = var_mean_149[1];  var_mean_149 = None
        add_666 = torch.ops.aten.add.Tensor(getitem_306, 1e-06);  getitem_306 = None
        rsqrt_149 = torch.ops.aten.rsqrt.default(add_666);  add_666 = None
        sub_221 = torch.ops.aten.sub.Tensor(cat_3, getitem_307);  cat_3 = getitem_307 = None
        mul_740 = torch.ops.aten.mul.Tensor(sub_221, rsqrt_149);  sub_221 = rsqrt_149 = None
        mul_741 = torch.ops.aten.mul.Tensor(mul_740, arg654_1);  mul_740 = arg654_1 = None
        add_667 = torch.ops.aten.add.Tensor(mul_741, arg655_1);  mul_741 = arg655_1 = None
        select_219 = torch.ops.aten.select.int(add_667, 1, 0)
        permute_959 = torch.ops.aten.permute.default(arg656_1, [1, 0]);  arg656_1 = None
        addmm_301 = torch.ops.aten.addmm.default(arg657_1, select_219, permute_959);  arg657_1 = select_219 = permute_959 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(addmm_301, 1);  addmm_301 = None
        view_1470 = torch.ops.aten.view.default(unsqueeze_2, [8, 1, 16, 48]);  unsqueeze_2 = None
        permute_960 = torch.ops.aten.permute.default(view_1470, [0, 2, 1, 3]);  view_1470 = None
        view_1471 = torch.ops.aten.view.default(add_667, [4616, 768])
        permute_961 = torch.ops.aten.permute.default(arg658_1, [1, 0]);  arg658_1 = None
        addmm_302 = torch.ops.aten.addmm.default(arg659_1, view_1471, permute_961);  arg659_1 = view_1471 = permute_961 = None
        view_1472 = torch.ops.aten.view.default(addmm_302, [8, 577, 768]);  addmm_302 = None
        view_1473 = torch.ops.aten.view.default(view_1472, [8, 577, 16, 48]);  view_1472 = None
        permute_962 = torch.ops.aten.permute.default(view_1473, [0, 2, 1, 3]);  view_1473 = None
        view_1474 = torch.ops.aten.view.default(add_667, [4616, 768]);  add_667 = None
        permute_963 = torch.ops.aten.permute.default(arg660_1, [1, 0]);  arg660_1 = None
        addmm_303 = torch.ops.aten.addmm.default(arg661_1, view_1474, permute_963);  arg661_1 = view_1474 = permute_963 = None
        view_1475 = torch.ops.aten.view.default(addmm_303, [8, 577, 768]);  addmm_303 = None
        view_1476 = torch.ops.aten.view.default(view_1475, [8, 577, 16, 48]);  view_1475 = None
        permute_964 = torch.ops.aten.permute.default(view_1476, [0, 2, 1, 3]);  view_1476 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_960, permute_962, permute_964, None, False);  permute_960 = permute_962 = permute_964 = None
        getitem_308 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_965 = torch.ops.aten.permute.default(getitem_308, [0, 2, 1, 3]);  getitem_308 = None
        view_1477 = torch.ops.aten.view.default(permute_965, [8, 1, 768]);  permute_965 = None
        view_1478 = torch.ops.aten.view.default(view_1477, [8, 768]);  view_1477 = None
        permute_966 = torch.ops.aten.permute.default(arg662_1, [1, 0]);  arg662_1 = None
        addmm_304 = torch.ops.aten.addmm.default(arg663_1, view_1478, permute_966);  arg663_1 = view_1478 = permute_966 = None
        view_1479 = torch.ops.aten.view.default(addmm_304, [8, 1, 768]);  addmm_304 = None
        mul_742 = torch.ops.aten.mul.Tensor(arg653_1, view_1479);  arg653_1 = view_1479 = None
        add_668 = torch.ops.aten.add.Tensor(expand_289, mul_742);  expand_289 = mul_742 = None
        var_mean_150 = torch.ops.aten.var_mean.correction(add_668, [2], correction = 0, keepdim = True)
        getitem_312 = var_mean_150[0]
        getitem_313 = var_mean_150[1];  var_mean_150 = None
        add_669 = torch.ops.aten.add.Tensor(getitem_312, 1e-06);  getitem_312 = None
        rsqrt_150 = torch.ops.aten.rsqrt.default(add_669);  add_669 = None
        sub_222 = torch.ops.aten.sub.Tensor(add_668, getitem_313);  getitem_313 = None
        mul_743 = torch.ops.aten.mul.Tensor(sub_222, rsqrt_150);  sub_222 = rsqrt_150 = None
        mul_744 = torch.ops.aten.mul.Tensor(mul_743, arg665_1);  mul_743 = arg665_1 = None
        add_670 = torch.ops.aten.add.Tensor(mul_744, arg666_1);  mul_744 = arg666_1 = None
        view_1480 = torch.ops.aten.view.default(add_670, [8, 768]);  add_670 = None
        permute_967 = torch.ops.aten.permute.default(arg667_1, [1, 0]);  arg667_1 = None
        addmm_305 = torch.ops.aten.addmm.default(arg668_1, view_1480, permute_967);  arg668_1 = view_1480 = permute_967 = None
        view_1481 = torch.ops.aten.view.default(addmm_305, [8, 1, 3072]);  addmm_305 = None
        mul_745 = torch.ops.aten.mul.Tensor(view_1481, 0.5)
        mul_746 = torch.ops.aten.mul.Tensor(view_1481, 0.7071067811865476);  view_1481 = None
        erf_74 = torch.ops.aten.erf.default(mul_746);  mul_746 = None
        add_671 = torch.ops.aten.add.Tensor(erf_74, 1);  erf_74 = None
        mul_747 = torch.ops.aten.mul.Tensor(mul_745, add_671);  mul_745 = add_671 = None
        view_1482 = torch.ops.aten.view.default(mul_747, [8, 3072]);  mul_747 = None
        permute_968 = torch.ops.aten.permute.default(arg669_1, [1, 0]);  arg669_1 = None
        addmm_306 = torch.ops.aten.addmm.default(arg670_1, view_1482, permute_968);  arg670_1 = view_1482 = permute_968 = None
        view_1483 = torch.ops.aten.view.default(addmm_306, [8, 1, 768]);  addmm_306 = None
        mul_748 = torch.ops.aten.mul.Tensor(arg664_1, view_1483);  arg664_1 = view_1483 = None
        add_672 = torch.ops.aten.add.Tensor(add_668, mul_748);  add_668 = mul_748 = None
        cat_4 = torch.ops.aten.cat.default([add_672, add_665], 1)
        var_mean_151 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
        getitem_314 = var_mean_151[0]
        getitem_315 = var_mean_151[1];  var_mean_151 = None
        add_673 = torch.ops.aten.add.Tensor(getitem_314, 1e-06);  getitem_314 = None
        rsqrt_151 = torch.ops.aten.rsqrt.default(add_673);  add_673 = None
        sub_223 = torch.ops.aten.sub.Tensor(cat_4, getitem_315);  cat_4 = getitem_315 = None
        mul_749 = torch.ops.aten.mul.Tensor(sub_223, rsqrt_151);  sub_223 = rsqrt_151 = None
        mul_750 = torch.ops.aten.mul.Tensor(mul_749, arg672_1);  mul_749 = arg672_1 = None
        add_674 = torch.ops.aten.add.Tensor(mul_750, arg673_1);  mul_750 = arg673_1 = None
        select_220 = torch.ops.aten.select.int(add_674, 1, 0)
        permute_969 = torch.ops.aten.permute.default(arg674_1, [1, 0]);  arg674_1 = None
        addmm_307 = torch.ops.aten.addmm.default(arg675_1, select_220, permute_969);  arg675_1 = select_220 = permute_969 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(addmm_307, 1);  addmm_307 = None
        view_1484 = torch.ops.aten.view.default(unsqueeze_3, [8, 1, 16, 48]);  unsqueeze_3 = None
        permute_970 = torch.ops.aten.permute.default(view_1484, [0, 2, 1, 3]);  view_1484 = None
        view_1485 = torch.ops.aten.view.default(add_674, [4616, 768])
        permute_971 = torch.ops.aten.permute.default(arg676_1, [1, 0]);  arg676_1 = None
        addmm_308 = torch.ops.aten.addmm.default(arg677_1, view_1485, permute_971);  arg677_1 = view_1485 = permute_971 = None
        view_1486 = torch.ops.aten.view.default(addmm_308, [8, 577, 768]);  addmm_308 = None
        view_1487 = torch.ops.aten.view.default(view_1486, [8, 577, 16, 48]);  view_1486 = None
        permute_972 = torch.ops.aten.permute.default(view_1487, [0, 2, 1, 3]);  view_1487 = None
        view_1488 = torch.ops.aten.view.default(add_674, [4616, 768]);  add_674 = None
        permute_973 = torch.ops.aten.permute.default(arg678_1, [1, 0]);  arg678_1 = None
        addmm_309 = torch.ops.aten.addmm.default(arg679_1, view_1488, permute_973);  arg679_1 = view_1488 = permute_973 = None
        view_1489 = torch.ops.aten.view.default(addmm_309, [8, 577, 768]);  addmm_309 = None
        view_1490 = torch.ops.aten.view.default(view_1489, [8, 577, 16, 48]);  view_1489 = None
        permute_974 = torch.ops.aten.permute.default(view_1490, [0, 2, 1, 3]);  view_1490 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_970, permute_972, permute_974, None, False);  permute_970 = permute_972 = permute_974 = None
        getitem_316 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_975 = torch.ops.aten.permute.default(getitem_316, [0, 2, 1, 3]);  getitem_316 = None
        view_1491 = torch.ops.aten.view.default(permute_975, [8, 1, 768]);  permute_975 = None
        view_1492 = torch.ops.aten.view.default(view_1491, [8, 768]);  view_1491 = None
        permute_976 = torch.ops.aten.permute.default(arg680_1, [1, 0]);  arg680_1 = None
        addmm_310 = torch.ops.aten.addmm.default(arg681_1, view_1492, permute_976);  arg681_1 = view_1492 = permute_976 = None
        view_1493 = torch.ops.aten.view.default(addmm_310, [8, 1, 768]);  addmm_310 = None
        mul_751 = torch.ops.aten.mul.Tensor(arg671_1, view_1493);  arg671_1 = view_1493 = None
        add_675 = torch.ops.aten.add.Tensor(add_672, mul_751);  add_672 = mul_751 = None
        var_mean_152 = torch.ops.aten.var_mean.correction(add_675, [2], correction = 0, keepdim = True)
        getitem_320 = var_mean_152[0]
        getitem_321 = var_mean_152[1];  var_mean_152 = None
        add_676 = torch.ops.aten.add.Tensor(getitem_320, 1e-06);  getitem_320 = None
        rsqrt_152 = torch.ops.aten.rsqrt.default(add_676);  add_676 = None
        sub_224 = torch.ops.aten.sub.Tensor(add_675, getitem_321);  getitem_321 = None
        mul_752 = torch.ops.aten.mul.Tensor(sub_224, rsqrt_152);  sub_224 = rsqrt_152 = None
        mul_753 = torch.ops.aten.mul.Tensor(mul_752, arg683_1);  mul_752 = arg683_1 = None
        add_677 = torch.ops.aten.add.Tensor(mul_753, arg684_1);  mul_753 = arg684_1 = None
        view_1494 = torch.ops.aten.view.default(add_677, [8, 768]);  add_677 = None
        permute_977 = torch.ops.aten.permute.default(arg685_1, [1, 0]);  arg685_1 = None
        addmm_311 = torch.ops.aten.addmm.default(arg686_1, view_1494, permute_977);  arg686_1 = view_1494 = permute_977 = None
        view_1495 = torch.ops.aten.view.default(addmm_311, [8, 1, 3072]);  addmm_311 = None
        mul_754 = torch.ops.aten.mul.Tensor(view_1495, 0.5)
        mul_755 = torch.ops.aten.mul.Tensor(view_1495, 0.7071067811865476);  view_1495 = None
        erf_75 = torch.ops.aten.erf.default(mul_755);  mul_755 = None
        add_678 = torch.ops.aten.add.Tensor(erf_75, 1);  erf_75 = None
        mul_756 = torch.ops.aten.mul.Tensor(mul_754, add_678);  mul_754 = add_678 = None
        view_1496 = torch.ops.aten.view.default(mul_756, [8, 3072]);  mul_756 = None
        permute_978 = torch.ops.aten.permute.default(arg687_1, [1, 0]);  arg687_1 = None
        addmm_312 = torch.ops.aten.addmm.default(arg688_1, view_1496, permute_978);  arg688_1 = view_1496 = permute_978 = None
        view_1497 = torch.ops.aten.view.default(addmm_312, [8, 1, 768]);  addmm_312 = None
        mul_757 = torch.ops.aten.mul.Tensor(arg682_1, view_1497);  arg682_1 = view_1497 = None
        add_679 = torch.ops.aten.add.Tensor(add_675, mul_757);  add_675 = mul_757 = None
        cat_5 = torch.ops.aten.cat.default([add_679, add_665], 1);  add_679 = add_665 = None
        var_mean_153 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
        getitem_322 = var_mean_153[0]
        getitem_323 = var_mean_153[1];  var_mean_153 = None
        add_680 = torch.ops.aten.add.Tensor(getitem_322, 1e-06);  getitem_322 = None
        rsqrt_153 = torch.ops.aten.rsqrt.default(add_680);  add_680 = None
        sub_225 = torch.ops.aten.sub.Tensor(cat_5, getitem_323);  cat_5 = getitem_323 = None
        mul_758 = torch.ops.aten.mul.Tensor(sub_225, rsqrt_153);  sub_225 = rsqrt_153 = None
        mul_759 = torch.ops.aten.mul.Tensor(mul_758, arg689_1);  mul_758 = arg689_1 = None
        add_681 = torch.ops.aten.add.Tensor(mul_759, arg690_1);  mul_759 = arg690_1 = None
        select_221 = torch.ops.aten.select.int(add_681, 1, 0);  add_681 = None
        clone_1023 = torch.ops.aten.clone.default(select_221);  select_221 = None
        permute_979 = torch.ops.aten.permute.default(arg691_1, [1, 0]);  arg691_1 = None
        addmm_313 = torch.ops.aten.addmm.default(arg692_1, clone_1023, permute_979);  arg692_1 = clone_1023 = permute_979 = None
        return (addmm_313,)
        
def load_args(reader):
    buf0 = reader.storage(None, 14155776, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 384, 384), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768, 3, 16, 16), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 576, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2304, 768), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2304,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf9, (16, 16), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf10, (16,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf11, (16, 16), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768, 768), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (3072, 768), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf19, (3072,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768, 3072), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf25, (2304, 768), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf26, (2304,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf27, (16, 16), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf28, (16,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (16, 16), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf30, (16,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768, 768), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (3072, 768), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf37, (3072,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768, 3072), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf43, (2304, 768), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf44, (2304,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf45, (16, 16), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf46, (16,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf47, (16, 16), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf48, (16,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768, 768), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf54, (3072, 768), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf55, (3072,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 3072), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf61, (2304, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf62, (2304,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf63, (16, 16), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf64, (16,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf65, (16, 16), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf66, (16,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768, 768), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf72, (3072, 768), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf73, (3072,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 3072), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf79, (2304, 768), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf80, (2304,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf81, (16, 16), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf82, (16,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf83, (16, 16), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf84, (16,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 768), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf90, (3072, 768), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf91, (3072,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768, 3072), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf97, (2304, 768), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf98, (2304,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf99, (16, 16), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf100, (16,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf101, (16, 16), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf102, (16,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768, 768), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf108, (3072, 768), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf109, (3072,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768, 3072), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf115, (2304, 768), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf116, (2304,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf117, (16, 16), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf118, (16,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf119, (16, 16), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf120, (16,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf126, (3072, 768), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf127, (3072,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768, 3072), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf133, (2304, 768), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf134, (2304,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf135, (16, 16), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf136, (16,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf137, (16, 16), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf138, (16,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768, 768), is_leaf=True)  # arg139_1
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
    buf153 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf153, (16, 16), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf154, (16,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf155, (16, 16), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf156, (16,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768, 768), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf162, (3072, 768), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf163, (3072,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768, 3072), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf169, (2304, 768), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf170, (2304,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf171, (16, 16), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf172, (16,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf173, (16, 16), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf174, (16,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768, 768), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf179, (768,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf180, (3072, 768), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf181, (3072,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768, 3072), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf187, (2304, 768), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf188, (2304,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf189, (16, 16), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf190, (16,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf191, (16, 16), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf192, (16,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768, 768), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf195, (768,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf196, (768,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf198, (3072, 768), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf199, (3072,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf200, (768, 3072), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf201, (768,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf202, (768,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf203, (768,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf204, (768,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf205, (2304, 768), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf206, (2304,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf207, (16, 16), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf208, (16,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf209, (16, 16), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf210, (16,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf211, (768, 768), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf212, (768,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf213, (768,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf214, (768,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf215, (768,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf216, (3072, 768), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf217, (3072,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf218, (768, 3072), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf219, (768,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf220, (768,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf221, (768,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf222, (768,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf223, (2304, 768), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf224, (2304,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf225, (16, 16), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf226, (16,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf227, (16, 16), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf228, (16,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf229, (768, 768), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf230, (768,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf231, (768,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf232, (768,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf233, (768,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf234, (3072, 768), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf235, (3072,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf236, (768, 3072), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf237, (768,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf238, (768,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf239, (768,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf240, (768,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf241, (2304, 768), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf242, (2304,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf243, (16, 16), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf244, (16,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf245, (16, 16), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf246, (16,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf247, (768, 768), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf248, (768,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf249, (768,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf250, (768,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf251, (768,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf252, (3072, 768), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf253, (3072,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf254, (768, 3072), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf255, (768,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf256, (768,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf257, (768,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf258, (768,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf259, (2304, 768), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf260, (2304,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf261, (16, 16), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf262, (16,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf263, (16, 16), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf264, (16,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf265, (768, 768), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf266, (768,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf267, (768,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf268, (768,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf269, (768,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf270, (3072, 768), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf271, (3072,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf272, (768, 3072), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf273, (768,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf274, (768,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf275, (768,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf276, (768,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf277, (2304, 768), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf278, (2304,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf279, (16, 16), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf280, (16,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf281, (16, 16), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf282, (16,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf283, (768, 768), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf284, (768,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf285, (768,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf286, (768,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf287, (768,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf288, (3072, 768), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf289, (3072,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf290, (768, 3072), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf291, (768,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf292, (768,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf293, (768,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf294, (768,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf295, (2304, 768), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf296, (2304,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf297, (16, 16), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf298, (16,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf299, (16, 16), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf300, (16,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf301, (768, 768), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf302, (768,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf303, (768,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf304, (768,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf305, (768,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf306, (3072, 768), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf307, (3072,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf308, (768, 3072), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf309, (768,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf310, (768,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf311, (768,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf312, (768,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf313, (2304, 768), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf314, (2304,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf315, (16, 16), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf316, (16,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf317, (16, 16), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf318, (16,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf319, (768, 768), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf320, (768,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf321, (768,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf322, (768,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf323, (768,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf324, (3072, 768), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf325, (3072,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf326, (768, 3072), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf327, (768,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf328, (768,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf329, (768,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf330, (768,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf331, (2304, 768), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf332, (2304,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf333, (16, 16), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf334, (16,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf335, (16, 16), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf336, (16,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf337, (768, 768), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf338, (768,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf339, (768,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf340, (768,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf341, (768,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf342, (3072, 768), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf343, (3072,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf344, (768, 3072), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf345, (768,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf346, (768,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf347, (768,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf348, (768,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf349, (2304, 768), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf350, (2304,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf351, (16, 16), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf352, (16,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf353, (16, 16), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf354, (16,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf355, (768, 768), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf356, (768,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf357, (768,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf358, (768,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf359, (768,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf360, (3072, 768), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf361, (3072,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf362, (768, 3072), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf363, (768,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf364, (768,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf365, (768,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf366, (768,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf367, (2304, 768), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf368, (2304,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf369, (16, 16), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf370, (16,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf371, (16, 16), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf372, (16,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf373, (768, 768), is_leaf=True)  # arg373_1
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
    buf387 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf387, (16, 16), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf388, (16,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf389, (16, 16), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf390, (16,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf391, (768, 768), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf392, (768,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf393, (768,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf394, (768,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf395, (768,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf396, (3072, 768), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf397, (3072,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf398, (768, 3072), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf399, (768,), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf400, (768,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf401, (768,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf402, (768,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf403, (2304, 768), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf404, (2304,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf405, (16, 16), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf406, (16,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf407, (16, 16), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf408, (16,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf409, (768, 768), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf410, (768,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf411, (768,), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf412, (768,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf413, (768,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf414, (3072, 768), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf415, (3072,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf416, (768, 3072), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf417, (768,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf418, (768,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf419, (768,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf420, (768,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf421, (2304, 768), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf422, (2304,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf423, (16, 16), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf424, (16,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf425, (16, 16), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf426, (16,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf427, (768, 768), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf428, (768,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf429, (768,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf430, (768,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf431, (768,), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf432, (3072, 768), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf433, (3072,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf434, (768, 3072), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf435, (768,), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf436, (768,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf437, (768,), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf438, (768,), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf439, (2304, 768), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf440, (2304,), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf441, (16, 16), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf442, (16,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf443, (16, 16), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf444, (16,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf445, (768, 768), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf446, (768,), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf447, (768,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf448, (768,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf449, (768,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf450, (3072, 768), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf451, (3072,), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf452, (768, 3072), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf453, (768,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf454, (768,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf455, (768,), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf456, (768,), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf457, (2304, 768), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf458, (2304,), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf459, (16, 16), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf460, (16,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf461, (16, 16), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf462, (16,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf463, (768, 768), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf464, (768,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf465, (768,), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf466, (768,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf467, (768,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf468, (3072, 768), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf469, (3072,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf470, (768, 3072), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf471, (768,), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf472, (768,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf473, (768,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf474, (768,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf475, (2304, 768), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf476, (2304,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf477, (16, 16), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf478, (16,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf479, (16, 16), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf480, (16,), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf481, (768, 768), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf482, (768,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf483, (768,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf484, (768,), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf485, (768,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf486, (3072, 768), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf487, (3072,), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf488, (768, 3072), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf489, (768,), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf490, (768,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf491, (768,), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf492, (768,), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf493, (2304, 768), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf494, (2304,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf495, (16, 16), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf496, (16,), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf497, (16, 16), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf498, (16,), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf499, (768, 768), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf500, (768,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf501, (768,), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf502, (768,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf503, (768,), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf504, (3072, 768), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf505, (3072,), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf506, (768, 3072), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf507, (768,), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf508, (768,), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf509, (768,), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf510, (768,), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf511, (2304, 768), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf512, (2304,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf513, (16, 16), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf514, (16,), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf515, (16, 16), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf516, (16,), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf517, (768, 768), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf518, (768,), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf519, (768,), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf520, (768,), is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf521, (768,), is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf522, (3072, 768), is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf523, (3072,), is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf524, (768, 3072), is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf525, (768,), is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf526, (768,), is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf527, (768,), is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf528, (768,), is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf529, (2304, 768), is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf530, (2304,), is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf531, (16, 16), is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf532, (16,), is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf533, (16, 16), is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf534, (16,), is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf535, (768, 768), is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf536, (768,), is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf537, (768,), is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf538, (768,), is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf539, (768,), is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf540, (3072, 768), is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf541, (3072,), is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf542, (768, 3072), is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf543, (768,), is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf544, (768,), is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf545, (768,), is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf546, (768,), is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf547, (2304, 768), is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf548, (2304,), is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf549, (16, 16), is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf550, (16,), is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf551, (16, 16), is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf552, (16,), is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf553, (768, 768), is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf554, (768,), is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf555, (768,), is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf556, (768,), is_leaf=True)  # arg556_1
    buf557 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf557, (768,), is_leaf=True)  # arg557_1
    buf558 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf558, (3072, 768), is_leaf=True)  # arg558_1
    buf559 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf559, (3072,), is_leaf=True)  # arg559_1
    buf560 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf560, (768, 3072), is_leaf=True)  # arg560_1
    buf561 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf561, (768,), is_leaf=True)  # arg561_1
    buf562 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf562, (768,), is_leaf=True)  # arg562_1
    buf563 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf563, (768,), is_leaf=True)  # arg563_1
    buf564 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf564, (768,), is_leaf=True)  # arg564_1
    buf565 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf565, (2304, 768), is_leaf=True)  # arg565_1
    buf566 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf566, (2304,), is_leaf=True)  # arg566_1
    buf567 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf567, (16, 16), is_leaf=True)  # arg567_1
    buf568 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf568, (16,), is_leaf=True)  # arg568_1
    buf569 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf569, (16, 16), is_leaf=True)  # arg569_1
    buf570 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf570, (16,), is_leaf=True)  # arg570_1
    buf571 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf571, (768, 768), is_leaf=True)  # arg571_1
    buf572 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf572, (768,), is_leaf=True)  # arg572_1
    buf573 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf573, (768,), is_leaf=True)  # arg573_1
    buf574 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf574, (768,), is_leaf=True)  # arg574_1
    buf575 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf575, (768,), is_leaf=True)  # arg575_1
    buf576 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf576, (3072, 768), is_leaf=True)  # arg576_1
    buf577 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf577, (3072,), is_leaf=True)  # arg577_1
    buf578 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf578, (768, 3072), is_leaf=True)  # arg578_1
    buf579 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf579, (768,), is_leaf=True)  # arg579_1
    buf580 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf580, (768,), is_leaf=True)  # arg580_1
    buf581 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf581, (768,), is_leaf=True)  # arg581_1
    buf582 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf582, (768,), is_leaf=True)  # arg582_1
    buf583 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf583, (2304, 768), is_leaf=True)  # arg583_1
    buf584 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf584, (2304,), is_leaf=True)  # arg584_1
    buf585 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf585, (16, 16), is_leaf=True)  # arg585_1
    buf586 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf586, (16,), is_leaf=True)  # arg586_1
    buf587 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf587, (16, 16), is_leaf=True)  # arg587_1
    buf588 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf588, (16,), is_leaf=True)  # arg588_1
    buf589 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf589, (768, 768), is_leaf=True)  # arg589_1
    buf590 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf590, (768,), is_leaf=True)  # arg590_1
    buf591 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf591, (768,), is_leaf=True)  # arg591_1
    buf592 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf592, (768,), is_leaf=True)  # arg592_1
    buf593 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf593, (768,), is_leaf=True)  # arg593_1
    buf594 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf594, (3072, 768), is_leaf=True)  # arg594_1
    buf595 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf595, (3072,), is_leaf=True)  # arg595_1
    buf596 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf596, (768, 3072), is_leaf=True)  # arg596_1
    buf597 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf597, (768,), is_leaf=True)  # arg597_1
    buf598 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf598, (768,), is_leaf=True)  # arg598_1
    buf599 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf599, (768,), is_leaf=True)  # arg599_1
    buf600 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf600, (768,), is_leaf=True)  # arg600_1
    buf601 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf601, (2304, 768), is_leaf=True)  # arg601_1
    buf602 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf602, (2304,), is_leaf=True)  # arg602_1
    buf603 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf603, (16, 16), is_leaf=True)  # arg603_1
    buf604 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf604, (16,), is_leaf=True)  # arg604_1
    buf605 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf605, (16, 16), is_leaf=True)  # arg605_1
    buf606 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf606, (16,), is_leaf=True)  # arg606_1
    buf607 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf607, (768, 768), is_leaf=True)  # arg607_1
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
    buf621 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf621, (16, 16), is_leaf=True)  # arg621_1
    buf622 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf622, (16,), is_leaf=True)  # arg622_1
    buf623 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf623, (16, 16), is_leaf=True)  # arg623_1
    buf624 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf624, (16,), is_leaf=True)  # arg624_1
    buf625 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf625, (768, 768), is_leaf=True)  # arg625_1
    buf626 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf626, (768,), is_leaf=True)  # arg626_1
    buf627 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf627, (768,), is_leaf=True)  # arg627_1
    buf628 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf628, (768,), is_leaf=True)  # arg628_1
    buf629 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf629, (768,), is_leaf=True)  # arg629_1
    buf630 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf630, (3072, 768), is_leaf=True)  # arg630_1
    buf631 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf631, (3072,), is_leaf=True)  # arg631_1
    buf632 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf632, (768, 3072), is_leaf=True)  # arg632_1
    buf633 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf633, (768,), is_leaf=True)  # arg633_1
    buf634 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf634, (768,), is_leaf=True)  # arg634_1
    buf635 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf635, (768,), is_leaf=True)  # arg635_1
    buf636 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf636, (768,), is_leaf=True)  # arg636_1
    buf637 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf637, (2304, 768), is_leaf=True)  # arg637_1
    buf638 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf638, (2304,), is_leaf=True)  # arg638_1
    buf639 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf639, (16, 16), is_leaf=True)  # arg639_1
    buf640 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf640, (16,), is_leaf=True)  # arg640_1
    buf641 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf641, (16, 16), is_leaf=True)  # arg641_1
    buf642 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf642, (16,), is_leaf=True)  # arg642_1
    buf643 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf643, (768, 768), is_leaf=True)  # arg643_1
    buf644 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf644, (768,), is_leaf=True)  # arg644_1
    buf645 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf645, (768,), is_leaf=True)  # arg645_1
    buf646 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf646, (768,), is_leaf=True)  # arg646_1
    buf647 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf647, (768,), is_leaf=True)  # arg647_1
    buf648 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf648, (3072, 768), is_leaf=True)  # arg648_1
    buf649 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf649, (3072,), is_leaf=True)  # arg649_1
    buf650 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf650, (768, 3072), is_leaf=True)  # arg650_1
    buf651 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf651, (768,), is_leaf=True)  # arg651_1
    buf652 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf652, (1, 1, 768), is_leaf=True)  # arg652_1
    buf653 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf653, (768,), is_leaf=True)  # arg653_1
    buf654 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf654, (768,), is_leaf=True)  # arg654_1
    buf655 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf655, (768,), is_leaf=True)  # arg655_1
    buf656 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf656, (768, 768), is_leaf=True)  # arg656_1
    buf657 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf657, (768,), is_leaf=True)  # arg657_1
    buf658 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf658, (768, 768), is_leaf=True)  # arg658_1
    buf659 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf659, (768,), is_leaf=True)  # arg659_1
    buf660 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf660, (768, 768), is_leaf=True)  # arg660_1
    buf661 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf661, (768,), is_leaf=True)  # arg661_1
    buf662 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf662, (768, 768), is_leaf=True)  # arg662_1
    buf663 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf663, (768,), is_leaf=True)  # arg663_1
    buf664 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf664, (768,), is_leaf=True)  # arg664_1
    buf665 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf665, (768,), is_leaf=True)  # arg665_1
    buf666 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf666, (768,), is_leaf=True)  # arg666_1
    buf667 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf667, (3072, 768), is_leaf=True)  # arg667_1
    buf668 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf668, (3072,), is_leaf=True)  # arg668_1
    buf669 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf669, (768, 3072), is_leaf=True)  # arg669_1
    buf670 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf670, (768,), is_leaf=True)  # arg670_1
    buf671 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf671, (768,), is_leaf=True)  # arg671_1
    buf672 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf672, (768,), is_leaf=True)  # arg672_1
    buf673 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf673, (768,), is_leaf=True)  # arg673_1
    buf674 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf674, (768, 768), is_leaf=True)  # arg674_1
    buf675 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf675, (768,), is_leaf=True)  # arg675_1
    buf676 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf676, (768, 768), is_leaf=True)  # arg676_1
    buf677 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf677, (768,), is_leaf=True)  # arg677_1
    buf678 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf678, (768, 768), is_leaf=True)  # arg678_1
    buf679 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf679, (768,), is_leaf=True)  # arg679_1
    buf680 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf680, (768, 768), is_leaf=True)  # arg680_1
    buf681 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf681, (768,), is_leaf=True)  # arg681_1
    buf682 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf682, (768,), is_leaf=True)  # arg682_1
    buf683 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf683, (768,), is_leaf=True)  # arg683_1
    buf684 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf684, (768,), is_leaf=True)  # arg684_1
    buf685 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf685, (3072, 768), is_leaf=True)  # arg685_1
    buf686 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf686, (3072,), is_leaf=True)  # arg686_1
    buf687 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf687, (768, 3072), is_leaf=True)  # arg687_1
    buf688 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf688, (768,), is_leaf=True)  # arg688_1
    buf689 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf689, (768,), is_leaf=True)  # arg689_1
    buf690 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf690, (768,), is_leaf=True)  # arg690_1
    buf691 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf691, (1000, 768), is_leaf=True)  # arg691_1
    buf692 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf692, (1000,), is_leaf=True)  # arg692_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)