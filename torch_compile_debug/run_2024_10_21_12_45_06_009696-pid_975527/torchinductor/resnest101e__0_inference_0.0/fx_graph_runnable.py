
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1):
        convolution_172 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_311 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_139 = torch.ops.aten.sqrt.default(add_311);  add_311 = None
        reciprocal_139 = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_450 = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1112 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1113 = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        unsqueeze_1114 = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1115 = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        sub_172 = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1113);  convolution_172 = unsqueeze_1113 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1115);  sub_172 = unsqueeze_1115 = None
        unsqueeze_1116 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_1117 = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1117);  mul_451 = unsqueeze_1117 = None
        unsqueeze_1118 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_1119 = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_312 = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1119);  mul_452 = unsqueeze_1119 = None
        relu_135 = torch.ops.aten.relu.default(add_312);  add_312 = None
        convolution_173 = torch.ops.aten.convolution.default(relu_135, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_135 = arg6_1 = None
        add_313 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_140 = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_140 = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_453 = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1120 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_1121 = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        unsqueeze_1122 = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1123 = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        sub_173 = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1121);  convolution_173 = unsqueeze_1121 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1123);  sub_173 = unsqueeze_1123 = None
        unsqueeze_1124 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_1125 = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1125);  mul_454 = unsqueeze_1125 = None
        unsqueeze_1126 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_1127 = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_314 = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1127);  mul_455 = unsqueeze_1127 = None
        relu_136 = torch.ops.aten.relu.default(add_314);  add_314 = None
        convolution_174 = torch.ops.aten.convolution.default(relu_136, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_136 = arg11_1 = None
        add_315 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_141 = torch.ops.aten.sqrt.default(add_315);  add_315 = None
        reciprocal_141 = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_456 = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1128 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_1129 = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        unsqueeze_1130 = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1131 = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        sub_174 = torch.ops.aten.sub.Tensor(convolution_174, unsqueeze_1129);  convolution_174 = unsqueeze_1129 = None
        mul_457 = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_1131);  sub_174 = unsqueeze_1131 = None
        unsqueeze_1132 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_1133 = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1133);  mul_457 = unsqueeze_1133 = None
        unsqueeze_1134 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_1135 = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_316 = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1135);  mul_458 = unsqueeze_1135 = None
        relu_137 = torch.ops.aten.relu.default(add_316);  add_316 = None
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_137, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_137 = None
        getitem_2 = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        convolution_175 = torch.ops.aten.convolution.default(getitem_2, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        add_317 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_142 = torch.ops.aten.sqrt.default(add_317);  add_317 = None
        reciprocal_142 = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_459 = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1136 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_1137 = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        unsqueeze_1138 = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_1139 = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        sub_175 = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1137);  convolution_175 = unsqueeze_1137 = None
        mul_460 = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_1139);  sub_175 = unsqueeze_1139 = None
        unsqueeze_1140 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_1141 = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1141);  mul_460 = unsqueeze_1141 = None
        unsqueeze_1142 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_1143 = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_318 = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1143);  mul_461 = unsqueeze_1143 = None
        relu_138 = torch.ops.aten.relu.default(add_318);  add_318 = None
        convolution_176 = torch.ops.aten.convolution.default(relu_138, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_138 = arg21_1 = None
        add_319 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_143 = torch.ops.aten.sqrt.default(add_319);  add_319 = None
        reciprocal_143 = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_462 = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1144 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_1145 = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        unsqueeze_1146 = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
        unsqueeze_1147 = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        sub_176 = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1145);  convolution_176 = unsqueeze_1145 = None
        mul_463 = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_1147);  sub_176 = unsqueeze_1147 = None
        unsqueeze_1148 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_1149 = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_464 = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1149);  mul_463 = unsqueeze_1149 = None
        unsqueeze_1150 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_1151 = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_320 = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1151);  mul_464 = unsqueeze_1151 = None
        relu_139 = torch.ops.aten.relu.default(add_320);  add_320 = None
        view_200 = torch.ops.aten.view.default(relu_139, [8, 2, 64, 64, 64]);  relu_139 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(view_200, [1])
        mean_34 = torch.ops.aten.mean.dim(sum_100, [2, 3], True);  sum_100 = None
        convolution_177 = torch.ops.aten.convolution.default(mean_34, arg26_1, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_34 = arg26_1 = arg27_1 = None
        add_321 = torch.ops.aten.add.Tensor(arg29_1, 1e-05);  arg29_1 = None
        sqrt_144 = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_144 = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_465 = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1152 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_1153 = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        unsqueeze_1154 = torch.ops.aten.unsqueeze.default(mul_465, -1);  mul_465 = None
        unsqueeze_1155 = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        sub_177 = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1153);  convolution_177 = unsqueeze_1153 = None
        mul_466 = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_1155);  sub_177 = unsqueeze_1155 = None
        unsqueeze_1156 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_1157 = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_467 = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_1157);  mul_466 = unsqueeze_1157 = None
        unsqueeze_1158 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_1159 = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_322 = torch.ops.aten.add.Tensor(mul_467, unsqueeze_1159);  mul_467 = unsqueeze_1159 = None
        relu_140 = torch.ops.aten.relu.default(add_322);  add_322 = None
        convolution_178 = torch.ops.aten.convolution.default(relu_140, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_140 = arg32_1 = arg33_1 = None
        view_201 = torch.ops.aten.view.default(convolution_178, [8, 1, 2, -1]);  convolution_178 = None
        permute_34 = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        amax_33 = torch.ops.aten.amax.default(permute_34, [1], True)
        sub_178 = torch.ops.aten.sub.Tensor(permute_34, amax_33);  permute_34 = amax_33 = None
        exp_33 = torch.ops.aten.exp.default(sub_178);  sub_178 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(exp_33, [1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_33, sum_101);  exp_33 = sum_101 = None
        view_202 = torch.ops.aten.view.default(div_33, [8, -1]);  div_33 = None
        view_203 = torch.ops.aten.view.default(view_202, [8, -1, 1, 1]);  view_202 = None
        view_204 = torch.ops.aten.view.default(view_203, [8, 2, 64, 1, 1]);  view_203 = None
        mul_468 = torch.ops.aten.mul.Tensor(view_200, view_204);  view_200 = view_204 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(mul_468, [1]);  mul_468 = None
        convolution_179 = torch.ops.aten.convolution.default(sum_102, arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_102 = arg34_1 = None
        add_323 = torch.ops.aten.add.Tensor(arg36_1, 1e-05);  arg36_1 = None
        sqrt_145 = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_145 = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_469 = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1160 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_1161 = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        unsqueeze_1162 = torch.ops.aten.unsqueeze.default(mul_469, -1);  mul_469 = None
        unsqueeze_1163 = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        sub_179 = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1161);  convolution_179 = unsqueeze_1161 = None
        mul_470 = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_1163);  sub_179 = unsqueeze_1163 = None
        unsqueeze_1164 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_1165 = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_471 = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_1165);  mul_470 = unsqueeze_1165 = None
        unsqueeze_1166 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_1167 = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_324 = torch.ops.aten.add.Tensor(mul_471, unsqueeze_1167);  mul_471 = unsqueeze_1167 = None
        convolution_180 = torch.ops.aten.convolution.default(getitem_2, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_2 = arg39_1 = None
        add_325 = torch.ops.aten.add.Tensor(arg41_1, 1e-05);  arg41_1 = None
        sqrt_146 = torch.ops.aten.sqrt.default(add_325);  add_325 = None
        reciprocal_146 = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_472 = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1168 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_1169 = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        unsqueeze_1170 = torch.ops.aten.unsqueeze.default(mul_472, -1);  mul_472 = None
        unsqueeze_1171 = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        sub_180 = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1169);  convolution_180 = unsqueeze_1169 = None
        mul_473 = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_1171);  sub_180 = unsqueeze_1171 = None
        unsqueeze_1172 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_1173 = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_474 = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_1173);  mul_473 = unsqueeze_1173 = None
        unsqueeze_1174 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_1175 = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_326 = torch.ops.aten.add.Tensor(mul_474, unsqueeze_1175);  mul_474 = unsqueeze_1175 = None
        add_327 = torch.ops.aten.add.Tensor(add_324, add_326);  add_324 = add_326 = None
        relu_141 = torch.ops.aten.relu.default(add_327);  add_327 = None
        convolution_181 = torch.ops.aten.convolution.default(relu_141, arg44_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg44_1 = None
        add_328 = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_147 = torch.ops.aten.sqrt.default(add_328);  add_328 = None
        reciprocal_147 = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_475 = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1176 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_1177 = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        unsqueeze_1178 = torch.ops.aten.unsqueeze.default(mul_475, -1);  mul_475 = None
        unsqueeze_1179 = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        sub_181 = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1177);  convolution_181 = unsqueeze_1177 = None
        mul_476 = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_1179);  sub_181 = unsqueeze_1179 = None
        unsqueeze_1180 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_1181 = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_477 = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_1181);  mul_476 = unsqueeze_1181 = None
        unsqueeze_1182 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_1183 = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_329 = torch.ops.aten.add.Tensor(mul_477, unsqueeze_1183);  mul_477 = unsqueeze_1183 = None
        relu_142 = torch.ops.aten.relu.default(add_329);  add_329 = None
        convolution_182 = torch.ops.aten.convolution.default(relu_142, arg49_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_142 = arg49_1 = None
        add_330 = torch.ops.aten.add.Tensor(arg51_1, 1e-05);  arg51_1 = None
        sqrt_148 = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_148 = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_478 = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1184 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_1185 = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        unsqueeze_1186 = torch.ops.aten.unsqueeze.default(mul_478, -1);  mul_478 = None
        unsqueeze_1187 = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        sub_182 = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1185);  convolution_182 = unsqueeze_1185 = None
        mul_479 = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_1187);  sub_182 = unsqueeze_1187 = None
        unsqueeze_1188 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_1189 = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_480 = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_1189);  mul_479 = unsqueeze_1189 = None
        unsqueeze_1190 = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_1191 = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_331 = torch.ops.aten.add.Tensor(mul_480, unsqueeze_1191);  mul_480 = unsqueeze_1191 = None
        relu_143 = torch.ops.aten.relu.default(add_331);  add_331 = None
        view_206 = torch.ops.aten.view.default(relu_143, [8, 2, 64, 64, 64]);  relu_143 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(view_206, [1])
        mean_35 = torch.ops.aten.mean.dim(sum_103, [2, 3], True);  sum_103 = None
        convolution_183 = torch.ops.aten.convolution.default(mean_35, arg54_1, arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_35 = arg54_1 = arg55_1 = None
        add_332 = torch.ops.aten.add.Tensor(arg57_1, 1e-05);  arg57_1 = None
        sqrt_149 = torch.ops.aten.sqrt.default(add_332);  add_332 = None
        reciprocal_149 = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_481 = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_1193 = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194 = torch.ops.aten.unsqueeze.default(mul_481, -1);  mul_481 = None
        unsqueeze_1195 = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_183 = torch.ops.aten.sub.Tensor(convolution_183, unsqueeze_1193);  convolution_183 = unsqueeze_1193 = None
        mul_482 = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_1195);  sub_183 = unsqueeze_1195 = None
        unsqueeze_1196 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_1197 = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_483 = torch.ops.aten.mul.Tensor(mul_482, unsqueeze_1197);  mul_482 = unsqueeze_1197 = None
        unsqueeze_1198 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_1199 = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_333 = torch.ops.aten.add.Tensor(mul_483, unsqueeze_1199);  mul_483 = unsqueeze_1199 = None
        relu_144 = torch.ops.aten.relu.default(add_333);  add_333 = None
        convolution_184 = torch.ops.aten.convolution.default(relu_144, arg60_1, arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_144 = arg60_1 = arg61_1 = None
        view_207 = torch.ops.aten.view.default(convolution_184, [8, 1, 2, -1]);  convolution_184 = None
        permute_35 = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        amax_34 = torch.ops.aten.amax.default(permute_35, [1], True)
        sub_184 = torch.ops.aten.sub.Tensor(permute_35, amax_34);  permute_35 = amax_34 = None
        exp_34 = torch.ops.aten.exp.default(sub_184);  sub_184 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(exp_34, [1], True)
        div_34 = torch.ops.aten.div.Tensor(exp_34, sum_104);  exp_34 = sum_104 = None
        view_208 = torch.ops.aten.view.default(div_34, [8, -1]);  div_34 = None
        view_209 = torch.ops.aten.view.default(view_208, [8, -1, 1, 1]);  view_208 = None
        view_210 = torch.ops.aten.view.default(view_209, [8, 2, 64, 1, 1]);  view_209 = None
        mul_484 = torch.ops.aten.mul.Tensor(view_206, view_210);  view_206 = view_210 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(mul_484, [1]);  mul_484 = None
        convolution_185 = torch.ops.aten.convolution.default(sum_105, arg62_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_105 = arg62_1 = None
        add_334 = torch.ops.aten.add.Tensor(arg64_1, 1e-05);  arg64_1 = None
        sqrt_150 = torch.ops.aten.sqrt.default(add_334);  add_334 = None
        reciprocal_150 = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_485 = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200 = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_1201 = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202 = torch.ops.aten.unsqueeze.default(mul_485, -1);  mul_485 = None
        unsqueeze_1203 = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_185 = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1201);  convolution_185 = unsqueeze_1201 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_1203);  sub_185 = unsqueeze_1203 = None
        unsqueeze_1204 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_1205 = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_487 = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_1205);  mul_486 = unsqueeze_1205 = None
        unsqueeze_1206 = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_1207 = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_335 = torch.ops.aten.add.Tensor(mul_487, unsqueeze_1207);  mul_487 = unsqueeze_1207 = None
        add_336 = torch.ops.aten.add.Tensor(add_335, relu_141);  add_335 = relu_141 = None
        relu_145 = torch.ops.aten.relu.default(add_336);  add_336 = None
        convolution_186 = torch.ops.aten.convolution.default(relu_145, arg67_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg67_1 = None
        add_337 = torch.ops.aten.add.Tensor(arg69_1, 1e-05);  arg69_1 = None
        sqrt_151 = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_151 = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_488 = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_1209 = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210 = torch.ops.aten.unsqueeze.default(mul_488, -1);  mul_488 = None
        unsqueeze_1211 = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_186 = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1209);  convolution_186 = unsqueeze_1209 = None
        mul_489 = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_1211);  sub_186 = unsqueeze_1211 = None
        unsqueeze_1212 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_1213 = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_490 = torch.ops.aten.mul.Tensor(mul_489, unsqueeze_1213);  mul_489 = unsqueeze_1213 = None
        unsqueeze_1214 = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_1215 = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_338 = torch.ops.aten.add.Tensor(mul_490, unsqueeze_1215);  mul_490 = unsqueeze_1215 = None
        relu_146 = torch.ops.aten.relu.default(add_338);  add_338 = None
        convolution_187 = torch.ops.aten.convolution.default(relu_146, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_146 = arg72_1 = None
        add_339 = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_152 = torch.ops.aten.sqrt.default(add_339);  add_339 = None
        reciprocal_152 = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_491 = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_1217 = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218 = torch.ops.aten.unsqueeze.default(mul_491, -1);  mul_491 = None
        unsqueeze_1219 = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_187 = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1217);  convolution_187 = unsqueeze_1217 = None
        mul_492 = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_1219);  sub_187 = unsqueeze_1219 = None
        unsqueeze_1220 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_1221 = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_493 = torch.ops.aten.mul.Tensor(mul_492, unsqueeze_1221);  mul_492 = unsqueeze_1221 = None
        unsqueeze_1222 = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_1223 = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_340 = torch.ops.aten.add.Tensor(mul_493, unsqueeze_1223);  mul_493 = unsqueeze_1223 = None
        relu_147 = torch.ops.aten.relu.default(add_340);  add_340 = None
        view_212 = torch.ops.aten.view.default(relu_147, [8, 2, 64, 64, 64]);  relu_147 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(view_212, [1])
        mean_36 = torch.ops.aten.mean.dim(sum_106, [2, 3], True);  sum_106 = None
        convolution_188 = torch.ops.aten.convolution.default(mean_36, arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_36 = arg77_1 = arg78_1 = None
        add_341 = torch.ops.aten.add.Tensor(arg80_1, 1e-05);  arg80_1 = None
        sqrt_153 = torch.ops.aten.sqrt.default(add_341);  add_341 = None
        reciprocal_153 = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_494 = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_1225 = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226 = torch.ops.aten.unsqueeze.default(mul_494, -1);  mul_494 = None
        unsqueeze_1227 = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_188 = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1225);  convolution_188 = unsqueeze_1225 = None
        mul_495 = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_1227);  sub_188 = unsqueeze_1227 = None
        unsqueeze_1228 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_1229 = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_496 = torch.ops.aten.mul.Tensor(mul_495, unsqueeze_1229);  mul_495 = unsqueeze_1229 = None
        unsqueeze_1230 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_1231 = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_342 = torch.ops.aten.add.Tensor(mul_496, unsqueeze_1231);  mul_496 = unsqueeze_1231 = None
        relu_148 = torch.ops.aten.relu.default(add_342);  add_342 = None
        convolution_189 = torch.ops.aten.convolution.default(relu_148, arg83_1, arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_148 = arg83_1 = arg84_1 = None
        view_213 = torch.ops.aten.view.default(convolution_189, [8, 1, 2, -1]);  convolution_189 = None
        permute_36 = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        amax_35 = torch.ops.aten.amax.default(permute_36, [1], True)
        sub_189 = torch.ops.aten.sub.Tensor(permute_36, amax_35);  permute_36 = amax_35 = None
        exp_35 = torch.ops.aten.exp.default(sub_189);  sub_189 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(exp_35, [1], True)
        div_35 = torch.ops.aten.div.Tensor(exp_35, sum_107);  exp_35 = sum_107 = None
        view_214 = torch.ops.aten.view.default(div_35, [8, -1]);  div_35 = None
        view_215 = torch.ops.aten.view.default(view_214, [8, -1, 1, 1]);  view_214 = None
        view_216 = torch.ops.aten.view.default(view_215, [8, 2, 64, 1, 1]);  view_215 = None
        mul_497 = torch.ops.aten.mul.Tensor(view_212, view_216);  view_212 = view_216 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(mul_497, [1]);  mul_497 = None
        convolution_190 = torch.ops.aten.convolution.default(sum_108, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_108 = arg85_1 = None
        add_343 = torch.ops.aten.add.Tensor(arg87_1, 1e-05);  arg87_1 = None
        sqrt_154 = torch.ops.aten.sqrt.default(add_343);  add_343 = None
        reciprocal_154 = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_498 = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_1233 = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234 = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_1235 = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_190 = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1233);  convolution_190 = unsqueeze_1233 = None
        mul_499 = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_1235);  sub_190 = unsqueeze_1235 = None
        unsqueeze_1236 = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_1237 = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_1237);  mul_499 = unsqueeze_1237 = None
        unsqueeze_1238 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_1239 = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_344 = torch.ops.aten.add.Tensor(mul_500, unsqueeze_1239);  mul_500 = unsqueeze_1239 = None
        add_345 = torch.ops.aten.add.Tensor(add_344, relu_145);  add_344 = relu_145 = None
        relu_149 = torch.ops.aten.relu.default(add_345);  add_345 = None
        convolution_191 = torch.ops.aten.convolution.default(relu_149, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg90_1 = None
        add_346 = torch.ops.aten.add.Tensor(arg92_1, 1e-05);  arg92_1 = None
        sqrt_155 = torch.ops.aten.sqrt.default(add_346);  add_346 = None
        reciprocal_155 = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_501 = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_1241 = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242 = torch.ops.aten.unsqueeze.default(mul_501, -1);  mul_501 = None
        unsqueeze_1243 = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_191 = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1241);  convolution_191 = unsqueeze_1241 = None
        mul_502 = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_1243);  sub_191 = unsqueeze_1243 = None
        unsqueeze_1244 = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_1245 = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_503 = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_1245);  mul_502 = unsqueeze_1245 = None
        unsqueeze_1246 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_1247 = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_347 = torch.ops.aten.add.Tensor(mul_503, unsqueeze_1247);  mul_503 = unsqueeze_1247 = None
        relu_150 = torch.ops.aten.relu.default(add_347);  add_347 = None
        convolution_192 = torch.ops.aten.convolution.default(relu_150, arg95_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_150 = arg95_1 = None
        add_348 = torch.ops.aten.add.Tensor(arg97_1, 1e-05);  arg97_1 = None
        sqrt_156 = torch.ops.aten.sqrt.default(add_348);  add_348 = None
        reciprocal_156 = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_504 = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248 = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_1249 = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250 = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
        unsqueeze_1251 = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_192 = torch.ops.aten.sub.Tensor(convolution_192, unsqueeze_1249);  convolution_192 = unsqueeze_1249 = None
        mul_505 = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_1251);  sub_192 = unsqueeze_1251 = None
        unsqueeze_1252 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_1253 = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_506 = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1253);  mul_505 = unsqueeze_1253 = None
        unsqueeze_1254 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_1255 = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_349 = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1255);  mul_506 = unsqueeze_1255 = None
        relu_151 = torch.ops.aten.relu.default(add_349);  add_349 = None
        view_218 = torch.ops.aten.view.default(relu_151, [8, 2, 128, 64, 64]);  relu_151 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(view_218, [1])
        mean_37 = torch.ops.aten.mean.dim(sum_109, [2, 3], True);  sum_109 = None
        convolution_193 = torch.ops.aten.convolution.default(mean_37, arg100_1, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_37 = arg100_1 = arg101_1 = None
        add_350 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_157 = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_157 = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_507 = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_1257 = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258 = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
        unsqueeze_1259 = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_193 = torch.ops.aten.sub.Tensor(convolution_193, unsqueeze_1257);  convolution_193 = unsqueeze_1257 = None
        mul_508 = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_1259);  sub_193 = unsqueeze_1259 = None
        unsqueeze_1260 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_1261 = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_509 = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_1261);  mul_508 = unsqueeze_1261 = None
        unsqueeze_1262 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_1263 = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_351 = torch.ops.aten.add.Tensor(mul_509, unsqueeze_1263);  mul_509 = unsqueeze_1263 = None
        relu_152 = torch.ops.aten.relu.default(add_351);  add_351 = None
        convolution_194 = torch.ops.aten.convolution.default(relu_152, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_152 = arg106_1 = arg107_1 = None
        view_219 = torch.ops.aten.view.default(convolution_194, [8, 1, 2, -1]);  convolution_194 = None
        permute_37 = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None
        amax_36 = torch.ops.aten.amax.default(permute_37, [1], True)
        sub_194 = torch.ops.aten.sub.Tensor(permute_37, amax_36);  permute_37 = amax_36 = None
        exp_36 = torch.ops.aten.exp.default(sub_194);  sub_194 = None
        sum_110 = torch.ops.aten.sum.dim_IntList(exp_36, [1], True)
        div_36 = torch.ops.aten.div.Tensor(exp_36, sum_110);  exp_36 = sum_110 = None
        view_220 = torch.ops.aten.view.default(div_36, [8, -1]);  div_36 = None
        view_221 = torch.ops.aten.view.default(view_220, [8, -1, 1, 1]);  view_220 = None
        view_222 = torch.ops.aten.view.default(view_221, [8, 2, 128, 1, 1]);  view_221 = None
        mul_510 = torch.ops.aten.mul.Tensor(view_218, view_222);  view_218 = view_222 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(mul_510, [1]);  mul_510 = None
        avg_pool2d_6 = torch.ops.aten.avg_pool2d.default(sum_111, [3, 3], [2, 2], [1, 1]);  sum_111 = None
        convolution_195 = torch.ops.aten.convolution.default(avg_pool2d_6, arg108_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_6 = arg108_1 = None
        add_352 = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
        sqrt_158 = torch.ops.aten.sqrt.default(add_352);  add_352 = None
        reciprocal_158 = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_511 = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_1265 = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266 = torch.ops.aten.unsqueeze.default(mul_511, -1);  mul_511 = None
        unsqueeze_1267 = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_195 = torch.ops.aten.sub.Tensor(convolution_195, unsqueeze_1265);  convolution_195 = unsqueeze_1265 = None
        mul_512 = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_1267);  sub_195 = unsqueeze_1267 = None
        unsqueeze_1268 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_1269 = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_513 = torch.ops.aten.mul.Tensor(mul_512, unsqueeze_1269);  mul_512 = unsqueeze_1269 = None
        unsqueeze_1270 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_1271 = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_353 = torch.ops.aten.add.Tensor(mul_513, unsqueeze_1271);  mul_513 = unsqueeze_1271 = None
        avg_pool2d_7 = torch.ops.aten.avg_pool2d.default(relu_149, [2, 2], [2, 2], [0, 0], True, False);  relu_149 = None
        convolution_196 = torch.ops.aten.convolution.default(avg_pool2d_7, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_7 = arg113_1 = None
        add_354 = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_159 = torch.ops.aten.sqrt.default(add_354);  add_354 = None
        reciprocal_159 = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_514 = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_1273 = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274 = torch.ops.aten.unsqueeze.default(mul_514, -1);  mul_514 = None
        unsqueeze_1275 = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_196 = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1273);  convolution_196 = unsqueeze_1273 = None
        mul_515 = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_1275);  sub_196 = unsqueeze_1275 = None
        unsqueeze_1276 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_1277 = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_516 = torch.ops.aten.mul.Tensor(mul_515, unsqueeze_1277);  mul_515 = unsqueeze_1277 = None
        unsqueeze_1278 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_1279 = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_355 = torch.ops.aten.add.Tensor(mul_516, unsqueeze_1279);  mul_516 = unsqueeze_1279 = None
        add_356 = torch.ops.aten.add.Tensor(add_353, add_355);  add_353 = add_355 = None
        relu_153 = torch.ops.aten.relu.default(add_356);  add_356 = None
        convolution_197 = torch.ops.aten.convolution.default(relu_153, arg118_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg118_1 = None
        add_357 = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_160 = torch.ops.aten.sqrt.default(add_357);  add_357 = None
        reciprocal_160 = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_517 = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1280 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_1281 = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        unsqueeze_1282 = torch.ops.aten.unsqueeze.default(mul_517, -1);  mul_517 = None
        unsqueeze_1283 = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        sub_197 = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1281);  convolution_197 = unsqueeze_1281 = None
        mul_518 = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_1283);  sub_197 = unsqueeze_1283 = None
        unsqueeze_1284 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_1285 = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_519 = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_1285);  mul_518 = unsqueeze_1285 = None
        unsqueeze_1286 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_1287 = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_358 = torch.ops.aten.add.Tensor(mul_519, unsqueeze_1287);  mul_519 = unsqueeze_1287 = None
        relu_154 = torch.ops.aten.relu.default(add_358);  add_358 = None
        convolution_198 = torch.ops.aten.convolution.default(relu_154, arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_154 = arg123_1 = None
        add_359 = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_161 = torch.ops.aten.sqrt.default(add_359);  add_359 = None
        reciprocal_161 = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_520 = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1288 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_1289 = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        unsqueeze_1290 = torch.ops.aten.unsqueeze.default(mul_520, -1);  mul_520 = None
        unsqueeze_1291 = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        sub_198 = torch.ops.aten.sub.Tensor(convolution_198, unsqueeze_1289);  convolution_198 = unsqueeze_1289 = None
        mul_521 = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_1291);  sub_198 = unsqueeze_1291 = None
        unsqueeze_1292 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_1293 = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_522 = torch.ops.aten.mul.Tensor(mul_521, unsqueeze_1293);  mul_521 = unsqueeze_1293 = None
        unsqueeze_1294 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_1295 = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_360 = torch.ops.aten.add.Tensor(mul_522, unsqueeze_1295);  mul_522 = unsqueeze_1295 = None
        relu_155 = torch.ops.aten.relu.default(add_360);  add_360 = None
        view_224 = torch.ops.aten.view.default(relu_155, [8, 2, 128, 32, 32]);  relu_155 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(view_224, [1])
        mean_38 = torch.ops.aten.mean.dim(sum_112, [2, 3], True);  sum_112 = None
        convolution_199 = torch.ops.aten.convolution.default(mean_38, arg128_1, arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_38 = arg128_1 = arg129_1 = None
        add_361 = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_162 = torch.ops.aten.sqrt.default(add_361);  add_361 = None
        reciprocal_162 = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_523 = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1296 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_1297 = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        unsqueeze_1298 = torch.ops.aten.unsqueeze.default(mul_523, -1);  mul_523 = None
        unsqueeze_1299 = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        sub_199 = torch.ops.aten.sub.Tensor(convolution_199, unsqueeze_1297);  convolution_199 = unsqueeze_1297 = None
        mul_524 = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_1299);  sub_199 = unsqueeze_1299 = None
        unsqueeze_1300 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_1301 = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_525 = torch.ops.aten.mul.Tensor(mul_524, unsqueeze_1301);  mul_524 = unsqueeze_1301 = None
        unsqueeze_1302 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_1303 = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_362 = torch.ops.aten.add.Tensor(mul_525, unsqueeze_1303);  mul_525 = unsqueeze_1303 = None
        relu_156 = torch.ops.aten.relu.default(add_362);  add_362 = None
        convolution_200 = torch.ops.aten.convolution.default(relu_156, arg134_1, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_156 = arg134_1 = arg135_1 = None
        view_225 = torch.ops.aten.view.default(convolution_200, [8, 1, 2, -1]);  convolution_200 = None
        permute_38 = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        amax_37 = torch.ops.aten.amax.default(permute_38, [1], True)
        sub_200 = torch.ops.aten.sub.Tensor(permute_38, amax_37);  permute_38 = amax_37 = None
        exp_37 = torch.ops.aten.exp.default(sub_200);  sub_200 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(exp_37, [1], True)
        div_37 = torch.ops.aten.div.Tensor(exp_37, sum_113);  exp_37 = sum_113 = None
        view_226 = torch.ops.aten.view.default(div_37, [8, -1]);  div_37 = None
        view_227 = torch.ops.aten.view.default(view_226, [8, -1, 1, 1]);  view_226 = None
        view_228 = torch.ops.aten.view.default(view_227, [8, 2, 128, 1, 1]);  view_227 = None
        mul_526 = torch.ops.aten.mul.Tensor(view_224, view_228);  view_224 = view_228 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_526, [1]);  mul_526 = None
        convolution_201 = torch.ops.aten.convolution.default(sum_114, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_114 = arg136_1 = None
        add_363 = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_163 = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_163 = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_527 = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1304 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_1305 = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        unsqueeze_1306 = torch.ops.aten.unsqueeze.default(mul_527, -1);  mul_527 = None
        unsqueeze_1307 = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        sub_201 = torch.ops.aten.sub.Tensor(convolution_201, unsqueeze_1305);  convolution_201 = unsqueeze_1305 = None
        mul_528 = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_1307);  sub_201 = unsqueeze_1307 = None
        unsqueeze_1308 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_1309 = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_529 = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_1309);  mul_528 = unsqueeze_1309 = None
        unsqueeze_1310 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_1311 = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_364 = torch.ops.aten.add.Tensor(mul_529, unsqueeze_1311);  mul_529 = unsqueeze_1311 = None
        add_365 = torch.ops.aten.add.Tensor(add_364, relu_153);  add_364 = relu_153 = None
        relu_157 = torch.ops.aten.relu.default(add_365);  add_365 = None
        convolution_202 = torch.ops.aten.convolution.default(relu_157, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg141_1 = None
        add_366 = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_164 = torch.ops.aten.sqrt.default(add_366);  add_366 = None
        reciprocal_164 = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_530 = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1312 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_1313 = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        unsqueeze_1314 = torch.ops.aten.unsqueeze.default(mul_530, -1);  mul_530 = None
        unsqueeze_1315 = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        sub_202 = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1313);  convolution_202 = unsqueeze_1313 = None
        mul_531 = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_1315);  sub_202 = unsqueeze_1315 = None
        unsqueeze_1316 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_1317 = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_532 = torch.ops.aten.mul.Tensor(mul_531, unsqueeze_1317);  mul_531 = unsqueeze_1317 = None
        unsqueeze_1318 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_1319 = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_367 = torch.ops.aten.add.Tensor(mul_532, unsqueeze_1319);  mul_532 = unsqueeze_1319 = None
        relu_158 = torch.ops.aten.relu.default(add_367);  add_367 = None
        convolution_203 = torch.ops.aten.convolution.default(relu_158, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_158 = arg146_1 = None
        add_368 = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_165 = torch.ops.aten.sqrt.default(add_368);  add_368 = None
        reciprocal_165 = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_533 = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1320 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_1321 = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        unsqueeze_1322 = torch.ops.aten.unsqueeze.default(mul_533, -1);  mul_533 = None
        unsqueeze_1323 = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        sub_203 = torch.ops.aten.sub.Tensor(convolution_203, unsqueeze_1321);  convolution_203 = unsqueeze_1321 = None
        mul_534 = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_1323);  sub_203 = unsqueeze_1323 = None
        unsqueeze_1324 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_1325 = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_535 = torch.ops.aten.mul.Tensor(mul_534, unsqueeze_1325);  mul_534 = unsqueeze_1325 = None
        unsqueeze_1326 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_1327 = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_369 = torch.ops.aten.add.Tensor(mul_535, unsqueeze_1327);  mul_535 = unsqueeze_1327 = None
        relu_159 = torch.ops.aten.relu.default(add_369);  add_369 = None
        view_230 = torch.ops.aten.view.default(relu_159, [8, 2, 128, 32, 32]);  relu_159 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(view_230, [1])
        mean_39 = torch.ops.aten.mean.dim(sum_115, [2, 3], True);  sum_115 = None
        convolution_204 = torch.ops.aten.convolution.default(mean_39, arg151_1, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_39 = arg151_1 = arg152_1 = None
        add_370 = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_166 = torch.ops.aten.sqrt.default(add_370);  add_370 = None
        reciprocal_166 = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_536 = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1328 = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_1329 = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        unsqueeze_1330 = torch.ops.aten.unsqueeze.default(mul_536, -1);  mul_536 = None
        unsqueeze_1331 = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        sub_204 = torch.ops.aten.sub.Tensor(convolution_204, unsqueeze_1329);  convolution_204 = unsqueeze_1329 = None
        mul_537 = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_1331);  sub_204 = unsqueeze_1331 = None
        unsqueeze_1332 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_1333 = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_538 = torch.ops.aten.mul.Tensor(mul_537, unsqueeze_1333);  mul_537 = unsqueeze_1333 = None
        unsqueeze_1334 = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_1335 = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_371 = torch.ops.aten.add.Tensor(mul_538, unsqueeze_1335);  mul_538 = unsqueeze_1335 = None
        relu_160 = torch.ops.aten.relu.default(add_371);  add_371 = None
        convolution_205 = torch.ops.aten.convolution.default(relu_160, arg157_1, arg158_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_160 = arg157_1 = arg158_1 = None
        view_231 = torch.ops.aten.view.default(convolution_205, [8, 1, 2, -1]);  convolution_205 = None
        permute_39 = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        amax_38 = torch.ops.aten.amax.default(permute_39, [1], True)
        sub_205 = torch.ops.aten.sub.Tensor(permute_39, amax_38);  permute_39 = amax_38 = None
        exp_38 = torch.ops.aten.exp.default(sub_205);  sub_205 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(exp_38, [1], True)
        div_38 = torch.ops.aten.div.Tensor(exp_38, sum_116);  exp_38 = sum_116 = None
        view_232 = torch.ops.aten.view.default(div_38, [8, -1]);  div_38 = None
        view_233 = torch.ops.aten.view.default(view_232, [8, -1, 1, 1]);  view_232 = None
        view_234 = torch.ops.aten.view.default(view_233, [8, 2, 128, 1, 1]);  view_233 = None
        mul_539 = torch.ops.aten.mul.Tensor(view_230, view_234);  view_230 = view_234 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(mul_539, [1]);  mul_539 = None
        convolution_206 = torch.ops.aten.convolution.default(sum_117, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_117 = arg159_1 = None
        add_372 = torch.ops.aten.add.Tensor(arg161_1, 1e-05);  arg161_1 = None
        sqrt_167 = torch.ops.aten.sqrt.default(add_372);  add_372 = None
        reciprocal_167 = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_540 = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1336 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_1337 = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        unsqueeze_1338 = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
        unsqueeze_1339 = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        sub_206 = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1337);  convolution_206 = unsqueeze_1337 = None
        mul_541 = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_1339);  sub_206 = unsqueeze_1339 = None
        unsqueeze_1340 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_1341 = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_542 = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_1341);  mul_541 = unsqueeze_1341 = None
        unsqueeze_1342 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_1343 = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_373 = torch.ops.aten.add.Tensor(mul_542, unsqueeze_1343);  mul_542 = unsqueeze_1343 = None
        add_374 = torch.ops.aten.add.Tensor(add_373, relu_157);  add_373 = relu_157 = None
        relu_161 = torch.ops.aten.relu.default(add_374);  add_374 = None
        convolution_207 = torch.ops.aten.convolution.default(relu_161, arg164_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg164_1 = None
        add_375 = torch.ops.aten.add.Tensor(arg166_1, 1e-05);  arg166_1 = None
        sqrt_168 = torch.ops.aten.sqrt.default(add_375);  add_375 = None
        reciprocal_168 = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_543 = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1344 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_1345 = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        unsqueeze_1346 = torch.ops.aten.unsqueeze.default(mul_543, -1);  mul_543 = None
        unsqueeze_1347 = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        sub_207 = torch.ops.aten.sub.Tensor(convolution_207, unsqueeze_1345);  convolution_207 = unsqueeze_1345 = None
        mul_544 = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_1347);  sub_207 = unsqueeze_1347 = None
        unsqueeze_1348 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_1349 = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_545 = torch.ops.aten.mul.Tensor(mul_544, unsqueeze_1349);  mul_544 = unsqueeze_1349 = None
        unsqueeze_1350 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_1351 = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_376 = torch.ops.aten.add.Tensor(mul_545, unsqueeze_1351);  mul_545 = unsqueeze_1351 = None
        relu_162 = torch.ops.aten.relu.default(add_376);  add_376 = None
        convolution_208 = torch.ops.aten.convolution.default(relu_162, arg169_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_162 = arg169_1 = None
        add_377 = torch.ops.aten.add.Tensor(arg171_1, 1e-05);  arg171_1 = None
        sqrt_169 = torch.ops.aten.sqrt.default(add_377);  add_377 = None
        reciprocal_169 = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_546 = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1352 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_1353 = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        unsqueeze_1354 = torch.ops.aten.unsqueeze.default(mul_546, -1);  mul_546 = None
        unsqueeze_1355 = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        sub_208 = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_1353);  convolution_208 = unsqueeze_1353 = None
        mul_547 = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_1355);  sub_208 = unsqueeze_1355 = None
        unsqueeze_1356 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_1357 = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_548 = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_1357);  mul_547 = unsqueeze_1357 = None
        unsqueeze_1358 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_1359 = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_378 = torch.ops.aten.add.Tensor(mul_548, unsqueeze_1359);  mul_548 = unsqueeze_1359 = None
        relu_163 = torch.ops.aten.relu.default(add_378);  add_378 = None
        view_236 = torch.ops.aten.view.default(relu_163, [8, 2, 128, 32, 32]);  relu_163 = None
        sum_118 = torch.ops.aten.sum.dim_IntList(view_236, [1])
        mean_40 = torch.ops.aten.mean.dim(sum_118, [2, 3], True);  sum_118 = None
        convolution_209 = torch.ops.aten.convolution.default(mean_40, arg174_1, arg175_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_40 = arg174_1 = arg175_1 = None
        add_379 = torch.ops.aten.add.Tensor(arg177_1, 1e-05);  arg177_1 = None
        sqrt_170 = torch.ops.aten.sqrt.default(add_379);  add_379 = None
        reciprocal_170 = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_549 = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1360 = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_1361 = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        unsqueeze_1362 = torch.ops.aten.unsqueeze.default(mul_549, -1);  mul_549 = None
        unsqueeze_1363 = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        sub_209 = torch.ops.aten.sub.Tensor(convolution_209, unsqueeze_1361);  convolution_209 = unsqueeze_1361 = None
        mul_550 = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_1363);  sub_209 = unsqueeze_1363 = None
        unsqueeze_1364 = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_1365 = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_551 = torch.ops.aten.mul.Tensor(mul_550, unsqueeze_1365);  mul_550 = unsqueeze_1365 = None
        unsqueeze_1366 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_1367 = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_380 = torch.ops.aten.add.Tensor(mul_551, unsqueeze_1367);  mul_551 = unsqueeze_1367 = None
        relu_164 = torch.ops.aten.relu.default(add_380);  add_380 = None
        convolution_210 = torch.ops.aten.convolution.default(relu_164, arg180_1, arg181_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_164 = arg180_1 = arg181_1 = None
        view_237 = torch.ops.aten.view.default(convolution_210, [8, 1, 2, -1]);  convolution_210 = None
        permute_40 = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        amax_39 = torch.ops.aten.amax.default(permute_40, [1], True)
        sub_210 = torch.ops.aten.sub.Tensor(permute_40, amax_39);  permute_40 = amax_39 = None
        exp_39 = torch.ops.aten.exp.default(sub_210);  sub_210 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(exp_39, [1], True)
        div_39 = torch.ops.aten.div.Tensor(exp_39, sum_119);  exp_39 = sum_119 = None
        view_238 = torch.ops.aten.view.default(div_39, [8, -1]);  div_39 = None
        view_239 = torch.ops.aten.view.default(view_238, [8, -1, 1, 1]);  view_238 = None
        view_240 = torch.ops.aten.view.default(view_239, [8, 2, 128, 1, 1]);  view_239 = None
        mul_552 = torch.ops.aten.mul.Tensor(view_236, view_240);  view_236 = view_240 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_552, [1]);  mul_552 = None
        convolution_211 = torch.ops.aten.convolution.default(sum_120, arg182_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_120 = arg182_1 = None
        add_381 = torch.ops.aten.add.Tensor(arg184_1, 1e-05);  arg184_1 = None
        sqrt_171 = torch.ops.aten.sqrt.default(add_381);  add_381 = None
        reciprocal_171 = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_553 = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1368 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_1369 = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        unsqueeze_1370 = torch.ops.aten.unsqueeze.default(mul_553, -1);  mul_553 = None
        unsqueeze_1371 = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        sub_211 = torch.ops.aten.sub.Tensor(convolution_211, unsqueeze_1369);  convolution_211 = unsqueeze_1369 = None
        mul_554 = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_1371);  sub_211 = unsqueeze_1371 = None
        unsqueeze_1372 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_1373 = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_555 = torch.ops.aten.mul.Tensor(mul_554, unsqueeze_1373);  mul_554 = unsqueeze_1373 = None
        unsqueeze_1374 = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_1375 = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_382 = torch.ops.aten.add.Tensor(mul_555, unsqueeze_1375);  mul_555 = unsqueeze_1375 = None
        add_383 = torch.ops.aten.add.Tensor(add_382, relu_161);  add_382 = relu_161 = None
        relu_165 = torch.ops.aten.relu.default(add_383);  add_383 = None
        convolution_212 = torch.ops.aten.convolution.default(relu_165, arg187_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg187_1 = None
        add_384 = torch.ops.aten.add.Tensor(arg189_1, 1e-05);  arg189_1 = None
        sqrt_172 = torch.ops.aten.sqrt.default(add_384);  add_384 = None
        reciprocal_172 = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_556 = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1376 = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_1377 = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        unsqueeze_1378 = torch.ops.aten.unsqueeze.default(mul_556, -1);  mul_556 = None
        unsqueeze_1379 = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        sub_212 = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1377);  convolution_212 = unsqueeze_1377 = None
        mul_557 = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_1379);  sub_212 = unsqueeze_1379 = None
        unsqueeze_1380 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_1381 = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_558 = torch.ops.aten.mul.Tensor(mul_557, unsqueeze_1381);  mul_557 = unsqueeze_1381 = None
        unsqueeze_1382 = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_1383 = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_385 = torch.ops.aten.add.Tensor(mul_558, unsqueeze_1383);  mul_558 = unsqueeze_1383 = None
        relu_166 = torch.ops.aten.relu.default(add_385);  add_385 = None
        convolution_213 = torch.ops.aten.convolution.default(relu_166, arg192_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_166 = arg192_1 = None
        add_386 = torch.ops.aten.add.Tensor(arg194_1, 1e-05);  arg194_1 = None
        sqrt_173 = torch.ops.aten.sqrt.default(add_386);  add_386 = None
        reciprocal_173 = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_559 = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1384 = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_1385 = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        unsqueeze_1386 = torch.ops.aten.unsqueeze.default(mul_559, -1);  mul_559 = None
        unsqueeze_1387 = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        sub_213 = torch.ops.aten.sub.Tensor(convolution_213, unsqueeze_1385);  convolution_213 = unsqueeze_1385 = None
        mul_560 = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_1387);  sub_213 = unsqueeze_1387 = None
        unsqueeze_1388 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_1389 = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_561 = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_1389);  mul_560 = unsqueeze_1389 = None
        unsqueeze_1390 = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_1391 = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_387 = torch.ops.aten.add.Tensor(mul_561, unsqueeze_1391);  mul_561 = unsqueeze_1391 = None
        relu_167 = torch.ops.aten.relu.default(add_387);  add_387 = None
        view_242 = torch.ops.aten.view.default(relu_167, [8, 2, 256, 32, 32]);  relu_167 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(view_242, [1])
        mean_41 = torch.ops.aten.mean.dim(sum_121, [2, 3], True);  sum_121 = None
        convolution_214 = torch.ops.aten.convolution.default(mean_41, arg197_1, arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_41 = arg197_1 = arg198_1 = None
        add_388 = torch.ops.aten.add.Tensor(arg200_1, 1e-05);  arg200_1 = None
        sqrt_174 = torch.ops.aten.sqrt.default(add_388);  add_388 = None
        reciprocal_174 = torch.ops.aten.reciprocal.default(sqrt_174);  sqrt_174 = None
        mul_562 = torch.ops.aten.mul.Tensor(reciprocal_174, 1);  reciprocal_174 = None
        unsqueeze_1392 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_1393 = torch.ops.aten.unsqueeze.default(unsqueeze_1392, -1);  unsqueeze_1392 = None
        unsqueeze_1394 = torch.ops.aten.unsqueeze.default(mul_562, -1);  mul_562 = None
        unsqueeze_1395 = torch.ops.aten.unsqueeze.default(unsqueeze_1394, -1);  unsqueeze_1394 = None
        sub_214 = torch.ops.aten.sub.Tensor(convolution_214, unsqueeze_1393);  convolution_214 = unsqueeze_1393 = None
        mul_563 = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_1395);  sub_214 = unsqueeze_1395 = None
        unsqueeze_1396 = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_1397 = torch.ops.aten.unsqueeze.default(unsqueeze_1396, -1);  unsqueeze_1396 = None
        mul_564 = torch.ops.aten.mul.Tensor(mul_563, unsqueeze_1397);  mul_563 = unsqueeze_1397 = None
        unsqueeze_1398 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_1399 = torch.ops.aten.unsqueeze.default(unsqueeze_1398, -1);  unsqueeze_1398 = None
        add_389 = torch.ops.aten.add.Tensor(mul_564, unsqueeze_1399);  mul_564 = unsqueeze_1399 = None
        relu_168 = torch.ops.aten.relu.default(add_389);  add_389 = None
        convolution_215 = torch.ops.aten.convolution.default(relu_168, arg203_1, arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_168 = arg203_1 = arg204_1 = None
        view_243 = torch.ops.aten.view.default(convolution_215, [8, 1, 2, -1]);  convolution_215 = None
        permute_41 = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        amax_40 = torch.ops.aten.amax.default(permute_41, [1], True)
        sub_215 = torch.ops.aten.sub.Tensor(permute_41, amax_40);  permute_41 = amax_40 = None
        exp_40 = torch.ops.aten.exp.default(sub_215);  sub_215 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(exp_40, [1], True)
        div_40 = torch.ops.aten.div.Tensor(exp_40, sum_122);  exp_40 = sum_122 = None
        view_244 = torch.ops.aten.view.default(div_40, [8, -1]);  div_40 = None
        view_245 = torch.ops.aten.view.default(view_244, [8, -1, 1, 1]);  view_244 = None
        view_246 = torch.ops.aten.view.default(view_245, [8, 2, 256, 1, 1]);  view_245 = None
        mul_565 = torch.ops.aten.mul.Tensor(view_242, view_246);  view_242 = view_246 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(mul_565, [1]);  mul_565 = None
        avg_pool2d_8 = torch.ops.aten.avg_pool2d.default(sum_123, [3, 3], [2, 2], [1, 1]);  sum_123 = None
        convolution_216 = torch.ops.aten.convolution.default(avg_pool2d_8, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_8 = arg205_1 = None
        add_390 = torch.ops.aten.add.Tensor(arg207_1, 1e-05);  arg207_1 = None
        sqrt_175 = torch.ops.aten.sqrt.default(add_390);  add_390 = None
        reciprocal_175 = torch.ops.aten.reciprocal.default(sqrt_175);  sqrt_175 = None
        mul_566 = torch.ops.aten.mul.Tensor(reciprocal_175, 1);  reciprocal_175 = None
        unsqueeze_1400 = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_1401 = torch.ops.aten.unsqueeze.default(unsqueeze_1400, -1);  unsqueeze_1400 = None
        unsqueeze_1402 = torch.ops.aten.unsqueeze.default(mul_566, -1);  mul_566 = None
        unsqueeze_1403 = torch.ops.aten.unsqueeze.default(unsqueeze_1402, -1);  unsqueeze_1402 = None
        sub_216 = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1401);  convolution_216 = unsqueeze_1401 = None
        mul_567 = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_1403);  sub_216 = unsqueeze_1403 = None
        unsqueeze_1404 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_1405 = torch.ops.aten.unsqueeze.default(unsqueeze_1404, -1);  unsqueeze_1404 = None
        mul_568 = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_1405);  mul_567 = unsqueeze_1405 = None
        unsqueeze_1406 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1407 = torch.ops.aten.unsqueeze.default(unsqueeze_1406, -1);  unsqueeze_1406 = None
        add_391 = torch.ops.aten.add.Tensor(mul_568, unsqueeze_1407);  mul_568 = unsqueeze_1407 = None
        avg_pool2d_9 = torch.ops.aten.avg_pool2d.default(relu_165, [2, 2], [2, 2], [0, 0], True, False);  relu_165 = None
        convolution_217 = torch.ops.aten.convolution.default(avg_pool2d_9, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_9 = arg210_1 = None
        add_392 = torch.ops.aten.add.Tensor(arg212_1, 1e-05);  arg212_1 = None
        sqrt_176 = torch.ops.aten.sqrt.default(add_392);  add_392 = None
        reciprocal_176 = torch.ops.aten.reciprocal.default(sqrt_176);  sqrt_176 = None
        mul_569 = torch.ops.aten.mul.Tensor(reciprocal_176, 1);  reciprocal_176 = None
        unsqueeze_1408 = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_1409 = torch.ops.aten.unsqueeze.default(unsqueeze_1408, -1);  unsqueeze_1408 = None
        unsqueeze_1410 = torch.ops.aten.unsqueeze.default(mul_569, -1);  mul_569 = None
        unsqueeze_1411 = torch.ops.aten.unsqueeze.default(unsqueeze_1410, -1);  unsqueeze_1410 = None
        sub_217 = torch.ops.aten.sub.Tensor(convolution_217, unsqueeze_1409);  convolution_217 = unsqueeze_1409 = None
        mul_570 = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_1411);  sub_217 = unsqueeze_1411 = None
        unsqueeze_1412 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_1413 = torch.ops.aten.unsqueeze.default(unsqueeze_1412, -1);  unsqueeze_1412 = None
        mul_571 = torch.ops.aten.mul.Tensor(mul_570, unsqueeze_1413);  mul_570 = unsqueeze_1413 = None
        unsqueeze_1414 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1415 = torch.ops.aten.unsqueeze.default(unsqueeze_1414, -1);  unsqueeze_1414 = None
        add_393 = torch.ops.aten.add.Tensor(mul_571, unsqueeze_1415);  mul_571 = unsqueeze_1415 = None
        add_394 = torch.ops.aten.add.Tensor(add_391, add_393);  add_391 = add_393 = None
        relu_169 = torch.ops.aten.relu.default(add_394);  add_394 = None
        convolution_218 = torch.ops.aten.convolution.default(relu_169, arg215_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg215_1 = None
        add_395 = torch.ops.aten.add.Tensor(arg217_1, 1e-05);  arg217_1 = None
        sqrt_177 = torch.ops.aten.sqrt.default(add_395);  add_395 = None
        reciprocal_177 = torch.ops.aten.reciprocal.default(sqrt_177);  sqrt_177 = None
        mul_572 = torch.ops.aten.mul.Tensor(reciprocal_177, 1);  reciprocal_177 = None
        unsqueeze_1416 = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_1417 = torch.ops.aten.unsqueeze.default(unsqueeze_1416, -1);  unsqueeze_1416 = None
        unsqueeze_1418 = torch.ops.aten.unsqueeze.default(mul_572, -1);  mul_572 = None
        unsqueeze_1419 = torch.ops.aten.unsqueeze.default(unsqueeze_1418, -1);  unsqueeze_1418 = None
        sub_218 = torch.ops.aten.sub.Tensor(convolution_218, unsqueeze_1417);  convolution_218 = unsqueeze_1417 = None
        mul_573 = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_1419);  sub_218 = unsqueeze_1419 = None
        unsqueeze_1420 = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_1421 = torch.ops.aten.unsqueeze.default(unsqueeze_1420, -1);  unsqueeze_1420 = None
        mul_574 = torch.ops.aten.mul.Tensor(mul_573, unsqueeze_1421);  mul_573 = unsqueeze_1421 = None
        unsqueeze_1422 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1423 = torch.ops.aten.unsqueeze.default(unsqueeze_1422, -1);  unsqueeze_1422 = None
        add_396 = torch.ops.aten.add.Tensor(mul_574, unsqueeze_1423);  mul_574 = unsqueeze_1423 = None
        relu_170 = torch.ops.aten.relu.default(add_396);  add_396 = None
        convolution_219 = torch.ops.aten.convolution.default(relu_170, arg220_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_170 = arg220_1 = None
        add_397 = torch.ops.aten.add.Tensor(arg222_1, 1e-05);  arg222_1 = None
        sqrt_178 = torch.ops.aten.sqrt.default(add_397);  add_397 = None
        reciprocal_178 = torch.ops.aten.reciprocal.default(sqrt_178);  sqrt_178 = None
        mul_575 = torch.ops.aten.mul.Tensor(reciprocal_178, 1);  reciprocal_178 = None
        unsqueeze_1424 = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_1425 = torch.ops.aten.unsqueeze.default(unsqueeze_1424, -1);  unsqueeze_1424 = None
        unsqueeze_1426 = torch.ops.aten.unsqueeze.default(mul_575, -1);  mul_575 = None
        unsqueeze_1427 = torch.ops.aten.unsqueeze.default(unsqueeze_1426, -1);  unsqueeze_1426 = None
        sub_219 = torch.ops.aten.sub.Tensor(convolution_219, unsqueeze_1425);  convolution_219 = unsqueeze_1425 = None
        mul_576 = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_1427);  sub_219 = unsqueeze_1427 = None
        unsqueeze_1428 = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_1429 = torch.ops.aten.unsqueeze.default(unsqueeze_1428, -1);  unsqueeze_1428 = None
        mul_577 = torch.ops.aten.mul.Tensor(mul_576, unsqueeze_1429);  mul_576 = unsqueeze_1429 = None
        unsqueeze_1430 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1431 = torch.ops.aten.unsqueeze.default(unsqueeze_1430, -1);  unsqueeze_1430 = None
        add_398 = torch.ops.aten.add.Tensor(mul_577, unsqueeze_1431);  mul_577 = unsqueeze_1431 = None
        relu_171 = torch.ops.aten.relu.default(add_398);  add_398 = None
        view_248 = torch.ops.aten.view.default(relu_171, [8, 2, 256, 16, 16]);  relu_171 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(view_248, [1])
        mean_42 = torch.ops.aten.mean.dim(sum_124, [2, 3], True);  sum_124 = None
        convolution_220 = torch.ops.aten.convolution.default(mean_42, arg225_1, arg226_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_42 = arg225_1 = arg226_1 = None
        add_399 = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_179 = torch.ops.aten.sqrt.default(add_399);  add_399 = None
        reciprocal_179 = torch.ops.aten.reciprocal.default(sqrt_179);  sqrt_179 = None
        mul_578 = torch.ops.aten.mul.Tensor(reciprocal_179, 1);  reciprocal_179 = None
        unsqueeze_1432 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1433 = torch.ops.aten.unsqueeze.default(unsqueeze_1432, -1);  unsqueeze_1432 = None
        unsqueeze_1434 = torch.ops.aten.unsqueeze.default(mul_578, -1);  mul_578 = None
        unsqueeze_1435 = torch.ops.aten.unsqueeze.default(unsqueeze_1434, -1);  unsqueeze_1434 = None
        sub_220 = torch.ops.aten.sub.Tensor(convolution_220, unsqueeze_1433);  convolution_220 = unsqueeze_1433 = None
        mul_579 = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_1435);  sub_220 = unsqueeze_1435 = None
        unsqueeze_1436 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1437 = torch.ops.aten.unsqueeze.default(unsqueeze_1436, -1);  unsqueeze_1436 = None
        mul_580 = torch.ops.aten.mul.Tensor(mul_579, unsqueeze_1437);  mul_579 = unsqueeze_1437 = None
        unsqueeze_1438 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1439 = torch.ops.aten.unsqueeze.default(unsqueeze_1438, -1);  unsqueeze_1438 = None
        add_400 = torch.ops.aten.add.Tensor(mul_580, unsqueeze_1439);  mul_580 = unsqueeze_1439 = None
        relu_172 = torch.ops.aten.relu.default(add_400);  add_400 = None
        convolution_221 = torch.ops.aten.convolution.default(relu_172, arg231_1, arg232_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_172 = arg231_1 = arg232_1 = None
        view_249 = torch.ops.aten.view.default(convolution_221, [8, 1, 2, -1]);  convolution_221 = None
        permute_42 = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        amax_41 = torch.ops.aten.amax.default(permute_42, [1], True)
        sub_221 = torch.ops.aten.sub.Tensor(permute_42, amax_41);  permute_42 = amax_41 = None
        exp_41 = torch.ops.aten.exp.default(sub_221);  sub_221 = None
        sum_125 = torch.ops.aten.sum.dim_IntList(exp_41, [1], True)
        div_41 = torch.ops.aten.div.Tensor(exp_41, sum_125);  exp_41 = sum_125 = None
        view_250 = torch.ops.aten.view.default(div_41, [8, -1]);  div_41 = None
        view_251 = torch.ops.aten.view.default(view_250, [8, -1, 1, 1]);  view_250 = None
        view_252 = torch.ops.aten.view.default(view_251, [8, 2, 256, 1, 1]);  view_251 = None
        mul_581 = torch.ops.aten.mul.Tensor(view_248, view_252);  view_248 = view_252 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(mul_581, [1]);  mul_581 = None
        convolution_222 = torch.ops.aten.convolution.default(sum_126, arg233_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_126 = arg233_1 = None
        add_401 = torch.ops.aten.add.Tensor(arg235_1, 1e-05);  arg235_1 = None
        sqrt_180 = torch.ops.aten.sqrt.default(add_401);  add_401 = None
        reciprocal_180 = torch.ops.aten.reciprocal.default(sqrt_180);  sqrt_180 = None
        mul_582 = torch.ops.aten.mul.Tensor(reciprocal_180, 1);  reciprocal_180 = None
        unsqueeze_1440 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1441 = torch.ops.aten.unsqueeze.default(unsqueeze_1440, -1);  unsqueeze_1440 = None
        unsqueeze_1442 = torch.ops.aten.unsqueeze.default(mul_582, -1);  mul_582 = None
        unsqueeze_1443 = torch.ops.aten.unsqueeze.default(unsqueeze_1442, -1);  unsqueeze_1442 = None
        sub_222 = torch.ops.aten.sub.Tensor(convolution_222, unsqueeze_1441);  convolution_222 = unsqueeze_1441 = None
        mul_583 = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_1443);  sub_222 = unsqueeze_1443 = None
        unsqueeze_1444 = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_1445 = torch.ops.aten.unsqueeze.default(unsqueeze_1444, -1);  unsqueeze_1444 = None
        mul_584 = torch.ops.aten.mul.Tensor(mul_583, unsqueeze_1445);  mul_583 = unsqueeze_1445 = None
        unsqueeze_1446 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1447 = torch.ops.aten.unsqueeze.default(unsqueeze_1446, -1);  unsqueeze_1446 = None
        add_402 = torch.ops.aten.add.Tensor(mul_584, unsqueeze_1447);  mul_584 = unsqueeze_1447 = None
        add_403 = torch.ops.aten.add.Tensor(add_402, relu_169);  add_402 = relu_169 = None
        relu_173 = torch.ops.aten.relu.default(add_403);  add_403 = None
        convolution_223 = torch.ops.aten.convolution.default(relu_173, arg238_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg238_1 = None
        add_404 = torch.ops.aten.add.Tensor(arg240_1, 1e-05);  arg240_1 = None
        sqrt_181 = torch.ops.aten.sqrt.default(add_404);  add_404 = None
        reciprocal_181 = torch.ops.aten.reciprocal.default(sqrt_181);  sqrt_181 = None
        mul_585 = torch.ops.aten.mul.Tensor(reciprocal_181, 1);  reciprocal_181 = None
        unsqueeze_1448 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1449 = torch.ops.aten.unsqueeze.default(unsqueeze_1448, -1);  unsqueeze_1448 = None
        unsqueeze_1450 = torch.ops.aten.unsqueeze.default(mul_585, -1);  mul_585 = None
        unsqueeze_1451 = torch.ops.aten.unsqueeze.default(unsqueeze_1450, -1);  unsqueeze_1450 = None
        sub_223 = torch.ops.aten.sub.Tensor(convolution_223, unsqueeze_1449);  convolution_223 = unsqueeze_1449 = None
        mul_586 = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_1451);  sub_223 = unsqueeze_1451 = None
        unsqueeze_1452 = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_1453 = torch.ops.aten.unsqueeze.default(unsqueeze_1452, -1);  unsqueeze_1452 = None
        mul_587 = torch.ops.aten.mul.Tensor(mul_586, unsqueeze_1453);  mul_586 = unsqueeze_1453 = None
        unsqueeze_1454 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1455 = torch.ops.aten.unsqueeze.default(unsqueeze_1454, -1);  unsqueeze_1454 = None
        add_405 = torch.ops.aten.add.Tensor(mul_587, unsqueeze_1455);  mul_587 = unsqueeze_1455 = None
        relu_174 = torch.ops.aten.relu.default(add_405);  add_405 = None
        convolution_224 = torch.ops.aten.convolution.default(relu_174, arg243_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_174 = arg243_1 = None
        add_406 = torch.ops.aten.add.Tensor(arg245_1, 1e-05);  arg245_1 = None
        sqrt_182 = torch.ops.aten.sqrt.default(add_406);  add_406 = None
        reciprocal_182 = torch.ops.aten.reciprocal.default(sqrt_182);  sqrt_182 = None
        mul_588 = torch.ops.aten.mul.Tensor(reciprocal_182, 1);  reciprocal_182 = None
        unsqueeze_1456 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1457 = torch.ops.aten.unsqueeze.default(unsqueeze_1456, -1);  unsqueeze_1456 = None
        unsqueeze_1458 = torch.ops.aten.unsqueeze.default(mul_588, -1);  mul_588 = None
        unsqueeze_1459 = torch.ops.aten.unsqueeze.default(unsqueeze_1458, -1);  unsqueeze_1458 = None
        sub_224 = torch.ops.aten.sub.Tensor(convolution_224, unsqueeze_1457);  convolution_224 = unsqueeze_1457 = None
        mul_589 = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_1459);  sub_224 = unsqueeze_1459 = None
        unsqueeze_1460 = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_1461 = torch.ops.aten.unsqueeze.default(unsqueeze_1460, -1);  unsqueeze_1460 = None
        mul_590 = torch.ops.aten.mul.Tensor(mul_589, unsqueeze_1461);  mul_589 = unsqueeze_1461 = None
        unsqueeze_1462 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1463 = torch.ops.aten.unsqueeze.default(unsqueeze_1462, -1);  unsqueeze_1462 = None
        add_407 = torch.ops.aten.add.Tensor(mul_590, unsqueeze_1463);  mul_590 = unsqueeze_1463 = None
        relu_175 = torch.ops.aten.relu.default(add_407);  add_407 = None
        view_254 = torch.ops.aten.view.default(relu_175, [8, 2, 256, 16, 16]);  relu_175 = None
        sum_127 = torch.ops.aten.sum.dim_IntList(view_254, [1])
        mean_43 = torch.ops.aten.mean.dim(sum_127, [2, 3], True);  sum_127 = None
        convolution_225 = torch.ops.aten.convolution.default(mean_43, arg248_1, arg249_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_43 = arg248_1 = arg249_1 = None
        add_408 = torch.ops.aten.add.Tensor(arg251_1, 1e-05);  arg251_1 = None
        sqrt_183 = torch.ops.aten.sqrt.default(add_408);  add_408 = None
        reciprocal_183 = torch.ops.aten.reciprocal.default(sqrt_183);  sqrt_183 = None
        mul_591 = torch.ops.aten.mul.Tensor(reciprocal_183, 1);  reciprocal_183 = None
        unsqueeze_1464 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1465 = torch.ops.aten.unsqueeze.default(unsqueeze_1464, -1);  unsqueeze_1464 = None
        unsqueeze_1466 = torch.ops.aten.unsqueeze.default(mul_591, -1);  mul_591 = None
        unsqueeze_1467 = torch.ops.aten.unsqueeze.default(unsqueeze_1466, -1);  unsqueeze_1466 = None
        sub_225 = torch.ops.aten.sub.Tensor(convolution_225, unsqueeze_1465);  convolution_225 = unsqueeze_1465 = None
        mul_592 = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_1467);  sub_225 = unsqueeze_1467 = None
        unsqueeze_1468 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_1469 = torch.ops.aten.unsqueeze.default(unsqueeze_1468, -1);  unsqueeze_1468 = None
        mul_593 = torch.ops.aten.mul.Tensor(mul_592, unsqueeze_1469);  mul_592 = unsqueeze_1469 = None
        unsqueeze_1470 = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_1471 = torch.ops.aten.unsqueeze.default(unsqueeze_1470, -1);  unsqueeze_1470 = None
        add_409 = torch.ops.aten.add.Tensor(mul_593, unsqueeze_1471);  mul_593 = unsqueeze_1471 = None
        relu_176 = torch.ops.aten.relu.default(add_409);  add_409 = None
        convolution_226 = torch.ops.aten.convolution.default(relu_176, arg254_1, arg255_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_176 = arg254_1 = arg255_1 = None
        view_255 = torch.ops.aten.view.default(convolution_226, [8, 1, 2, -1]);  convolution_226 = None
        permute_43 = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        amax_42 = torch.ops.aten.amax.default(permute_43, [1], True)
        sub_226 = torch.ops.aten.sub.Tensor(permute_43, amax_42);  permute_43 = amax_42 = None
        exp_42 = torch.ops.aten.exp.default(sub_226);  sub_226 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(exp_42, [1], True)
        div_42 = torch.ops.aten.div.Tensor(exp_42, sum_128);  exp_42 = sum_128 = None
        view_256 = torch.ops.aten.view.default(div_42, [8, -1]);  div_42 = None
        view_257 = torch.ops.aten.view.default(view_256, [8, -1, 1, 1]);  view_256 = None
        view_258 = torch.ops.aten.view.default(view_257, [8, 2, 256, 1, 1]);  view_257 = None
        mul_594 = torch.ops.aten.mul.Tensor(view_254, view_258);  view_254 = view_258 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(mul_594, [1]);  mul_594 = None
        convolution_227 = torch.ops.aten.convolution.default(sum_129, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_129 = arg256_1 = None
        add_410 = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_184 = torch.ops.aten.sqrt.default(add_410);  add_410 = None
        reciprocal_184 = torch.ops.aten.reciprocal.default(sqrt_184);  sqrt_184 = None
        mul_595 = torch.ops.aten.mul.Tensor(reciprocal_184, 1);  reciprocal_184 = None
        unsqueeze_1472 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1473 = torch.ops.aten.unsqueeze.default(unsqueeze_1472, -1);  unsqueeze_1472 = None
        unsqueeze_1474 = torch.ops.aten.unsqueeze.default(mul_595, -1);  mul_595 = None
        unsqueeze_1475 = torch.ops.aten.unsqueeze.default(unsqueeze_1474, -1);  unsqueeze_1474 = None
        sub_227 = torch.ops.aten.sub.Tensor(convolution_227, unsqueeze_1473);  convolution_227 = unsqueeze_1473 = None
        mul_596 = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_1475);  sub_227 = unsqueeze_1475 = None
        unsqueeze_1476 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1477 = torch.ops.aten.unsqueeze.default(unsqueeze_1476, -1);  unsqueeze_1476 = None
        mul_597 = torch.ops.aten.mul.Tensor(mul_596, unsqueeze_1477);  mul_596 = unsqueeze_1477 = None
        unsqueeze_1478 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1479 = torch.ops.aten.unsqueeze.default(unsqueeze_1478, -1);  unsqueeze_1478 = None
        add_411 = torch.ops.aten.add.Tensor(mul_597, unsqueeze_1479);  mul_597 = unsqueeze_1479 = None
        add_412 = torch.ops.aten.add.Tensor(add_411, relu_173);  add_411 = relu_173 = None
        relu_177 = torch.ops.aten.relu.default(add_412);  add_412 = None
        convolution_228 = torch.ops.aten.convolution.default(relu_177, arg261_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg261_1 = None
        add_413 = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_185 = torch.ops.aten.sqrt.default(add_413);  add_413 = None
        reciprocal_185 = torch.ops.aten.reciprocal.default(sqrt_185);  sqrt_185 = None
        mul_598 = torch.ops.aten.mul.Tensor(reciprocal_185, 1);  reciprocal_185 = None
        unsqueeze_1480 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1481 = torch.ops.aten.unsqueeze.default(unsqueeze_1480, -1);  unsqueeze_1480 = None
        unsqueeze_1482 = torch.ops.aten.unsqueeze.default(mul_598, -1);  mul_598 = None
        unsqueeze_1483 = torch.ops.aten.unsqueeze.default(unsqueeze_1482, -1);  unsqueeze_1482 = None
        sub_228 = torch.ops.aten.sub.Tensor(convolution_228, unsqueeze_1481);  convolution_228 = unsqueeze_1481 = None
        mul_599 = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_1483);  sub_228 = unsqueeze_1483 = None
        unsqueeze_1484 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1485 = torch.ops.aten.unsqueeze.default(unsqueeze_1484, -1);  unsqueeze_1484 = None
        mul_600 = torch.ops.aten.mul.Tensor(mul_599, unsqueeze_1485);  mul_599 = unsqueeze_1485 = None
        unsqueeze_1486 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1487 = torch.ops.aten.unsqueeze.default(unsqueeze_1486, -1);  unsqueeze_1486 = None
        add_414 = torch.ops.aten.add.Tensor(mul_600, unsqueeze_1487);  mul_600 = unsqueeze_1487 = None
        relu_178 = torch.ops.aten.relu.default(add_414);  add_414 = None
        convolution_229 = torch.ops.aten.convolution.default(relu_178, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_178 = arg266_1 = None
        add_415 = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_186 = torch.ops.aten.sqrt.default(add_415);  add_415 = None
        reciprocal_186 = torch.ops.aten.reciprocal.default(sqrt_186);  sqrt_186 = None
        mul_601 = torch.ops.aten.mul.Tensor(reciprocal_186, 1);  reciprocal_186 = None
        unsqueeze_1488 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1489 = torch.ops.aten.unsqueeze.default(unsqueeze_1488, -1);  unsqueeze_1488 = None
        unsqueeze_1490 = torch.ops.aten.unsqueeze.default(mul_601, -1);  mul_601 = None
        unsqueeze_1491 = torch.ops.aten.unsqueeze.default(unsqueeze_1490, -1);  unsqueeze_1490 = None
        sub_229 = torch.ops.aten.sub.Tensor(convolution_229, unsqueeze_1489);  convolution_229 = unsqueeze_1489 = None
        mul_602 = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_1491);  sub_229 = unsqueeze_1491 = None
        unsqueeze_1492 = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1493 = torch.ops.aten.unsqueeze.default(unsqueeze_1492, -1);  unsqueeze_1492 = None
        mul_603 = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_1493);  mul_602 = unsqueeze_1493 = None
        unsqueeze_1494 = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_1495 = torch.ops.aten.unsqueeze.default(unsqueeze_1494, -1);  unsqueeze_1494 = None
        add_416 = torch.ops.aten.add.Tensor(mul_603, unsqueeze_1495);  mul_603 = unsqueeze_1495 = None
        relu_179 = torch.ops.aten.relu.default(add_416);  add_416 = None
        view_260 = torch.ops.aten.view.default(relu_179, [8, 2, 256, 16, 16]);  relu_179 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(view_260, [1])
        mean_44 = torch.ops.aten.mean.dim(sum_130, [2, 3], True);  sum_130 = None
        convolution_230 = torch.ops.aten.convolution.default(mean_44, arg271_1, arg272_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_44 = arg271_1 = arg272_1 = None
        add_417 = torch.ops.aten.add.Tensor(arg274_1, 1e-05);  arg274_1 = None
        sqrt_187 = torch.ops.aten.sqrt.default(add_417);  add_417 = None
        reciprocal_187 = torch.ops.aten.reciprocal.default(sqrt_187);  sqrt_187 = None
        mul_604 = torch.ops.aten.mul.Tensor(reciprocal_187, 1);  reciprocal_187 = None
        unsqueeze_1496 = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_1497 = torch.ops.aten.unsqueeze.default(unsqueeze_1496, -1);  unsqueeze_1496 = None
        unsqueeze_1498 = torch.ops.aten.unsqueeze.default(mul_604, -1);  mul_604 = None
        unsqueeze_1499 = torch.ops.aten.unsqueeze.default(unsqueeze_1498, -1);  unsqueeze_1498 = None
        sub_230 = torch.ops.aten.sub.Tensor(convolution_230, unsqueeze_1497);  convolution_230 = unsqueeze_1497 = None
        mul_605 = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_1499);  sub_230 = unsqueeze_1499 = None
        unsqueeze_1500 = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_1501 = torch.ops.aten.unsqueeze.default(unsqueeze_1500, -1);  unsqueeze_1500 = None
        mul_606 = torch.ops.aten.mul.Tensor(mul_605, unsqueeze_1501);  mul_605 = unsqueeze_1501 = None
        unsqueeze_1502 = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1503 = torch.ops.aten.unsqueeze.default(unsqueeze_1502, -1);  unsqueeze_1502 = None
        add_418 = torch.ops.aten.add.Tensor(mul_606, unsqueeze_1503);  mul_606 = unsqueeze_1503 = None
        relu_180 = torch.ops.aten.relu.default(add_418);  add_418 = None
        convolution_231 = torch.ops.aten.convolution.default(relu_180, arg277_1, arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_180 = arg277_1 = arg278_1 = None
        view_261 = torch.ops.aten.view.default(convolution_231, [8, 1, 2, -1]);  convolution_231 = None
        permute_44 = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        amax_43 = torch.ops.aten.amax.default(permute_44, [1], True)
        sub_231 = torch.ops.aten.sub.Tensor(permute_44, amax_43);  permute_44 = amax_43 = None
        exp_43 = torch.ops.aten.exp.default(sub_231);  sub_231 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(exp_43, [1], True)
        div_43 = torch.ops.aten.div.Tensor(exp_43, sum_131);  exp_43 = sum_131 = None
        view_262 = torch.ops.aten.view.default(div_43, [8, -1]);  div_43 = None
        view_263 = torch.ops.aten.view.default(view_262, [8, -1, 1, 1]);  view_262 = None
        view_264 = torch.ops.aten.view.default(view_263, [8, 2, 256, 1, 1]);  view_263 = None
        mul_607 = torch.ops.aten.mul.Tensor(view_260, view_264);  view_260 = view_264 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(mul_607, [1]);  mul_607 = None
        convolution_232 = torch.ops.aten.convolution.default(sum_132, arg279_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_132 = arg279_1 = None
        add_419 = torch.ops.aten.add.Tensor(arg281_1, 1e-05);  arg281_1 = None
        sqrt_188 = torch.ops.aten.sqrt.default(add_419);  add_419 = None
        reciprocal_188 = torch.ops.aten.reciprocal.default(sqrt_188);  sqrt_188 = None
        mul_608 = torch.ops.aten.mul.Tensor(reciprocal_188, 1);  reciprocal_188 = None
        unsqueeze_1504 = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_1505 = torch.ops.aten.unsqueeze.default(unsqueeze_1504, -1);  unsqueeze_1504 = None
        unsqueeze_1506 = torch.ops.aten.unsqueeze.default(mul_608, -1);  mul_608 = None
        unsqueeze_1507 = torch.ops.aten.unsqueeze.default(unsqueeze_1506, -1);  unsqueeze_1506 = None
        sub_232 = torch.ops.aten.sub.Tensor(convolution_232, unsqueeze_1505);  convolution_232 = unsqueeze_1505 = None
        mul_609 = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_1507);  sub_232 = unsqueeze_1507 = None
        unsqueeze_1508 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1509 = torch.ops.aten.unsqueeze.default(unsqueeze_1508, -1);  unsqueeze_1508 = None
        mul_610 = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_1509);  mul_609 = unsqueeze_1509 = None
        unsqueeze_1510 = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_1511 = torch.ops.aten.unsqueeze.default(unsqueeze_1510, -1);  unsqueeze_1510 = None
        add_420 = torch.ops.aten.add.Tensor(mul_610, unsqueeze_1511);  mul_610 = unsqueeze_1511 = None
        add_421 = torch.ops.aten.add.Tensor(add_420, relu_177);  add_420 = relu_177 = None
        relu_181 = torch.ops.aten.relu.default(add_421);  add_421 = None
        convolution_233 = torch.ops.aten.convolution.default(relu_181, arg284_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg284_1 = None
        add_422 = torch.ops.aten.add.Tensor(arg286_1, 1e-05);  arg286_1 = None
        sqrt_189 = torch.ops.aten.sqrt.default(add_422);  add_422 = None
        reciprocal_189 = torch.ops.aten.reciprocal.default(sqrt_189);  sqrt_189 = None
        mul_611 = torch.ops.aten.mul.Tensor(reciprocal_189, 1);  reciprocal_189 = None
        unsqueeze_1512 = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_1513 = torch.ops.aten.unsqueeze.default(unsqueeze_1512, -1);  unsqueeze_1512 = None
        unsqueeze_1514 = torch.ops.aten.unsqueeze.default(mul_611, -1);  mul_611 = None
        unsqueeze_1515 = torch.ops.aten.unsqueeze.default(unsqueeze_1514, -1);  unsqueeze_1514 = None
        sub_233 = torch.ops.aten.sub.Tensor(convolution_233, unsqueeze_1513);  convolution_233 = unsqueeze_1513 = None
        mul_612 = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_1515);  sub_233 = unsqueeze_1515 = None
        unsqueeze_1516 = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_1517 = torch.ops.aten.unsqueeze.default(unsqueeze_1516, -1);  unsqueeze_1516 = None
        mul_613 = torch.ops.aten.mul.Tensor(mul_612, unsqueeze_1517);  mul_612 = unsqueeze_1517 = None
        unsqueeze_1518 = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_1519 = torch.ops.aten.unsqueeze.default(unsqueeze_1518, -1);  unsqueeze_1518 = None
        add_423 = torch.ops.aten.add.Tensor(mul_613, unsqueeze_1519);  mul_613 = unsqueeze_1519 = None
        relu_182 = torch.ops.aten.relu.default(add_423);  add_423 = None
        convolution_234 = torch.ops.aten.convolution.default(relu_182, arg289_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_182 = arg289_1 = None
        add_424 = torch.ops.aten.add.Tensor(arg291_1, 1e-05);  arg291_1 = None
        sqrt_190 = torch.ops.aten.sqrt.default(add_424);  add_424 = None
        reciprocal_190 = torch.ops.aten.reciprocal.default(sqrt_190);  sqrt_190 = None
        mul_614 = torch.ops.aten.mul.Tensor(reciprocal_190, 1);  reciprocal_190 = None
        unsqueeze_1520 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1521 = torch.ops.aten.unsqueeze.default(unsqueeze_1520, -1);  unsqueeze_1520 = None
        unsqueeze_1522 = torch.ops.aten.unsqueeze.default(mul_614, -1);  mul_614 = None
        unsqueeze_1523 = torch.ops.aten.unsqueeze.default(unsqueeze_1522, -1);  unsqueeze_1522 = None
        sub_234 = torch.ops.aten.sub.Tensor(convolution_234, unsqueeze_1521);  convolution_234 = unsqueeze_1521 = None
        mul_615 = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_1523);  sub_234 = unsqueeze_1523 = None
        unsqueeze_1524 = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1525 = torch.ops.aten.unsqueeze.default(unsqueeze_1524, -1);  unsqueeze_1524 = None
        mul_616 = torch.ops.aten.mul.Tensor(mul_615, unsqueeze_1525);  mul_615 = unsqueeze_1525 = None
        unsqueeze_1526 = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1527 = torch.ops.aten.unsqueeze.default(unsqueeze_1526, -1);  unsqueeze_1526 = None
        add_425 = torch.ops.aten.add.Tensor(mul_616, unsqueeze_1527);  mul_616 = unsqueeze_1527 = None
        relu_183 = torch.ops.aten.relu.default(add_425);  add_425 = None
        view_266 = torch.ops.aten.view.default(relu_183, [8, 2, 256, 16, 16]);  relu_183 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(view_266, [1])
        mean_45 = torch.ops.aten.mean.dim(sum_133, [2, 3], True);  sum_133 = None
        convolution_235 = torch.ops.aten.convolution.default(mean_45, arg294_1, arg295_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_45 = arg294_1 = arg295_1 = None
        add_426 = torch.ops.aten.add.Tensor(arg297_1, 1e-05);  arg297_1 = None
        sqrt_191 = torch.ops.aten.sqrt.default(add_426);  add_426 = None
        reciprocal_191 = torch.ops.aten.reciprocal.default(sqrt_191);  sqrt_191 = None
        mul_617 = torch.ops.aten.mul.Tensor(reciprocal_191, 1);  reciprocal_191 = None
        unsqueeze_1528 = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_1529 = torch.ops.aten.unsqueeze.default(unsqueeze_1528, -1);  unsqueeze_1528 = None
        unsqueeze_1530 = torch.ops.aten.unsqueeze.default(mul_617, -1);  mul_617 = None
        unsqueeze_1531 = torch.ops.aten.unsqueeze.default(unsqueeze_1530, -1);  unsqueeze_1530 = None
        sub_235 = torch.ops.aten.sub.Tensor(convolution_235, unsqueeze_1529);  convolution_235 = unsqueeze_1529 = None
        mul_618 = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_1531);  sub_235 = unsqueeze_1531 = None
        unsqueeze_1532 = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1533 = torch.ops.aten.unsqueeze.default(unsqueeze_1532, -1);  unsqueeze_1532 = None
        mul_619 = torch.ops.aten.mul.Tensor(mul_618, unsqueeze_1533);  mul_618 = unsqueeze_1533 = None
        unsqueeze_1534 = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1535 = torch.ops.aten.unsqueeze.default(unsqueeze_1534, -1);  unsqueeze_1534 = None
        add_427 = torch.ops.aten.add.Tensor(mul_619, unsqueeze_1535);  mul_619 = unsqueeze_1535 = None
        relu_184 = torch.ops.aten.relu.default(add_427);  add_427 = None
        convolution_236 = torch.ops.aten.convolution.default(relu_184, arg300_1, arg301_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_184 = arg300_1 = arg301_1 = None
        view_267 = torch.ops.aten.view.default(convolution_236, [8, 1, 2, -1]);  convolution_236 = None
        permute_45 = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        amax_44 = torch.ops.aten.amax.default(permute_45, [1], True)
        sub_236 = torch.ops.aten.sub.Tensor(permute_45, amax_44);  permute_45 = amax_44 = None
        exp_44 = torch.ops.aten.exp.default(sub_236);  sub_236 = None
        sum_134 = torch.ops.aten.sum.dim_IntList(exp_44, [1], True)
        div_44 = torch.ops.aten.div.Tensor(exp_44, sum_134);  exp_44 = sum_134 = None
        view_268 = torch.ops.aten.view.default(div_44, [8, -1]);  div_44 = None
        view_269 = torch.ops.aten.view.default(view_268, [8, -1, 1, 1]);  view_268 = None
        view_270 = torch.ops.aten.view.default(view_269, [8, 2, 256, 1, 1]);  view_269 = None
        mul_620 = torch.ops.aten.mul.Tensor(view_266, view_270);  view_266 = view_270 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(mul_620, [1]);  mul_620 = None
        convolution_237 = torch.ops.aten.convolution.default(sum_135, arg302_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_135 = arg302_1 = None
        add_428 = torch.ops.aten.add.Tensor(arg304_1, 1e-05);  arg304_1 = None
        sqrt_192 = torch.ops.aten.sqrt.default(add_428);  add_428 = None
        reciprocal_192 = torch.ops.aten.reciprocal.default(sqrt_192);  sqrt_192 = None
        mul_621 = torch.ops.aten.mul.Tensor(reciprocal_192, 1);  reciprocal_192 = None
        unsqueeze_1536 = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1537 = torch.ops.aten.unsqueeze.default(unsqueeze_1536, -1);  unsqueeze_1536 = None
        unsqueeze_1538 = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
        unsqueeze_1539 = torch.ops.aten.unsqueeze.default(unsqueeze_1538, -1);  unsqueeze_1538 = None
        sub_237 = torch.ops.aten.sub.Tensor(convolution_237, unsqueeze_1537);  convolution_237 = unsqueeze_1537 = None
        mul_622 = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_1539);  sub_237 = unsqueeze_1539 = None
        unsqueeze_1540 = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1541 = torch.ops.aten.unsqueeze.default(unsqueeze_1540, -1);  unsqueeze_1540 = None
        mul_623 = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_1541);  mul_622 = unsqueeze_1541 = None
        unsqueeze_1542 = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_1543 = torch.ops.aten.unsqueeze.default(unsqueeze_1542, -1);  unsqueeze_1542 = None
        add_429 = torch.ops.aten.add.Tensor(mul_623, unsqueeze_1543);  mul_623 = unsqueeze_1543 = None
        add_430 = torch.ops.aten.add.Tensor(add_429, relu_181);  add_429 = relu_181 = None
        relu_185 = torch.ops.aten.relu.default(add_430);  add_430 = None
        convolution_238 = torch.ops.aten.convolution.default(relu_185, arg307_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg307_1 = None
        add_431 = torch.ops.aten.add.Tensor(arg309_1, 1e-05);  arg309_1 = None
        sqrt_193 = torch.ops.aten.sqrt.default(add_431);  add_431 = None
        reciprocal_193 = torch.ops.aten.reciprocal.default(sqrt_193);  sqrt_193 = None
        mul_624 = torch.ops.aten.mul.Tensor(reciprocal_193, 1);  reciprocal_193 = None
        unsqueeze_1544 = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_1545 = torch.ops.aten.unsqueeze.default(unsqueeze_1544, -1);  unsqueeze_1544 = None
        unsqueeze_1546 = torch.ops.aten.unsqueeze.default(mul_624, -1);  mul_624 = None
        unsqueeze_1547 = torch.ops.aten.unsqueeze.default(unsqueeze_1546, -1);  unsqueeze_1546 = None
        sub_238 = torch.ops.aten.sub.Tensor(convolution_238, unsqueeze_1545);  convolution_238 = unsqueeze_1545 = None
        mul_625 = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_1547);  sub_238 = unsqueeze_1547 = None
        unsqueeze_1548 = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1549 = torch.ops.aten.unsqueeze.default(unsqueeze_1548, -1);  unsqueeze_1548 = None
        mul_626 = torch.ops.aten.mul.Tensor(mul_625, unsqueeze_1549);  mul_625 = unsqueeze_1549 = None
        unsqueeze_1550 = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1551 = torch.ops.aten.unsqueeze.default(unsqueeze_1550, -1);  unsqueeze_1550 = None
        add_432 = torch.ops.aten.add.Tensor(mul_626, unsqueeze_1551);  mul_626 = unsqueeze_1551 = None
        relu_186 = torch.ops.aten.relu.default(add_432);  add_432 = None
        convolution_239 = torch.ops.aten.convolution.default(relu_186, arg312_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_186 = arg312_1 = None
        add_433 = torch.ops.aten.add.Tensor(arg314_1, 1e-05);  arg314_1 = None
        sqrt_194 = torch.ops.aten.sqrt.default(add_433);  add_433 = None
        reciprocal_194 = torch.ops.aten.reciprocal.default(sqrt_194);  sqrt_194 = None
        mul_627 = torch.ops.aten.mul.Tensor(reciprocal_194, 1);  reciprocal_194 = None
        unsqueeze_1552 = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_1553 = torch.ops.aten.unsqueeze.default(unsqueeze_1552, -1);  unsqueeze_1552 = None
        unsqueeze_1554 = torch.ops.aten.unsqueeze.default(mul_627, -1);  mul_627 = None
        unsqueeze_1555 = torch.ops.aten.unsqueeze.default(unsqueeze_1554, -1);  unsqueeze_1554 = None
        sub_239 = torch.ops.aten.sub.Tensor(convolution_239, unsqueeze_1553);  convolution_239 = unsqueeze_1553 = None
        mul_628 = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_1555);  sub_239 = unsqueeze_1555 = None
        unsqueeze_1556 = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_1557 = torch.ops.aten.unsqueeze.default(unsqueeze_1556, -1);  unsqueeze_1556 = None
        mul_629 = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_1557);  mul_628 = unsqueeze_1557 = None
        unsqueeze_1558 = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
        unsqueeze_1559 = torch.ops.aten.unsqueeze.default(unsqueeze_1558, -1);  unsqueeze_1558 = None
        add_434 = torch.ops.aten.add.Tensor(mul_629, unsqueeze_1559);  mul_629 = unsqueeze_1559 = None
        relu_187 = torch.ops.aten.relu.default(add_434);  add_434 = None
        view_272 = torch.ops.aten.view.default(relu_187, [8, 2, 256, 16, 16]);  relu_187 = None
        sum_136 = torch.ops.aten.sum.dim_IntList(view_272, [1])
        mean_46 = torch.ops.aten.mean.dim(sum_136, [2, 3], True);  sum_136 = None
        convolution_240 = torch.ops.aten.convolution.default(mean_46, arg317_1, arg318_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_46 = arg317_1 = arg318_1 = None
        add_435 = torch.ops.aten.add.Tensor(arg320_1, 1e-05);  arg320_1 = None
        sqrt_195 = torch.ops.aten.sqrt.default(add_435);  add_435 = None
        reciprocal_195 = torch.ops.aten.reciprocal.default(sqrt_195);  sqrt_195 = None
        mul_630 = torch.ops.aten.mul.Tensor(reciprocal_195, 1);  reciprocal_195 = None
        unsqueeze_1560 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1561 = torch.ops.aten.unsqueeze.default(unsqueeze_1560, -1);  unsqueeze_1560 = None
        unsqueeze_1562 = torch.ops.aten.unsqueeze.default(mul_630, -1);  mul_630 = None
        unsqueeze_1563 = torch.ops.aten.unsqueeze.default(unsqueeze_1562, -1);  unsqueeze_1562 = None
        sub_240 = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_1561);  convolution_240 = unsqueeze_1561 = None
        mul_631 = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_1563);  sub_240 = unsqueeze_1563 = None
        unsqueeze_1564 = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_1565 = torch.ops.aten.unsqueeze.default(unsqueeze_1564, -1);  unsqueeze_1564 = None
        mul_632 = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_1565);  mul_631 = unsqueeze_1565 = None
        unsqueeze_1566 = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1567 = torch.ops.aten.unsqueeze.default(unsqueeze_1566, -1);  unsqueeze_1566 = None
        add_436 = torch.ops.aten.add.Tensor(mul_632, unsqueeze_1567);  mul_632 = unsqueeze_1567 = None
        relu_188 = torch.ops.aten.relu.default(add_436);  add_436 = None
        convolution_241 = torch.ops.aten.convolution.default(relu_188, arg323_1, arg324_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_188 = arg323_1 = arg324_1 = None
        view_273 = torch.ops.aten.view.default(convolution_241, [8, 1, 2, -1]);  convolution_241 = None
        permute_46 = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
        amax_45 = torch.ops.aten.amax.default(permute_46, [1], True)
        sub_241 = torch.ops.aten.sub.Tensor(permute_46, amax_45);  permute_46 = amax_45 = None
        exp_45 = torch.ops.aten.exp.default(sub_241);  sub_241 = None
        sum_137 = torch.ops.aten.sum.dim_IntList(exp_45, [1], True)
        div_45 = torch.ops.aten.div.Tensor(exp_45, sum_137);  exp_45 = sum_137 = None
        view_274 = torch.ops.aten.view.default(div_45, [8, -1]);  div_45 = None
        view_275 = torch.ops.aten.view.default(view_274, [8, -1, 1, 1]);  view_274 = None
        view_276 = torch.ops.aten.view.default(view_275, [8, 2, 256, 1, 1]);  view_275 = None
        mul_633 = torch.ops.aten.mul.Tensor(view_272, view_276);  view_272 = view_276 = None
        sum_138 = torch.ops.aten.sum.dim_IntList(mul_633, [1]);  mul_633 = None
        convolution_242 = torch.ops.aten.convolution.default(sum_138, arg325_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_138 = arg325_1 = None
        add_437 = torch.ops.aten.add.Tensor(arg327_1, 1e-05);  arg327_1 = None
        sqrt_196 = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_196 = torch.ops.aten.reciprocal.default(sqrt_196);  sqrt_196 = None
        mul_634 = torch.ops.aten.mul.Tensor(reciprocal_196, 1);  reciprocal_196 = None
        unsqueeze_1568 = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
        unsqueeze_1569 = torch.ops.aten.unsqueeze.default(unsqueeze_1568, -1);  unsqueeze_1568 = None
        unsqueeze_1570 = torch.ops.aten.unsqueeze.default(mul_634, -1);  mul_634 = None
        unsqueeze_1571 = torch.ops.aten.unsqueeze.default(unsqueeze_1570, -1);  unsqueeze_1570 = None
        sub_242 = torch.ops.aten.sub.Tensor(convolution_242, unsqueeze_1569);  convolution_242 = unsqueeze_1569 = None
        mul_635 = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_1571);  sub_242 = unsqueeze_1571 = None
        unsqueeze_1572 = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_1573 = torch.ops.aten.unsqueeze.default(unsqueeze_1572, -1);  unsqueeze_1572 = None
        mul_636 = torch.ops.aten.mul.Tensor(mul_635, unsqueeze_1573);  mul_635 = unsqueeze_1573 = None
        unsqueeze_1574 = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1575 = torch.ops.aten.unsqueeze.default(unsqueeze_1574, -1);  unsqueeze_1574 = None
        add_438 = torch.ops.aten.add.Tensor(mul_636, unsqueeze_1575);  mul_636 = unsqueeze_1575 = None
        add_439 = torch.ops.aten.add.Tensor(add_438, relu_185);  add_438 = relu_185 = None
        relu_189 = torch.ops.aten.relu.default(add_439);  add_439 = None
        convolution_243 = torch.ops.aten.convolution.default(relu_189, arg330_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg330_1 = None
        add_440 = torch.ops.aten.add.Tensor(arg332_1, 1e-05);  arg332_1 = None
        sqrt_197 = torch.ops.aten.sqrt.default(add_440);  add_440 = None
        reciprocal_197 = torch.ops.aten.reciprocal.default(sqrt_197);  sqrt_197 = None
        mul_637 = torch.ops.aten.mul.Tensor(reciprocal_197, 1);  reciprocal_197 = None
        unsqueeze_1576 = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_1577 = torch.ops.aten.unsqueeze.default(unsqueeze_1576, -1);  unsqueeze_1576 = None
        unsqueeze_1578 = torch.ops.aten.unsqueeze.default(mul_637, -1);  mul_637 = None
        unsqueeze_1579 = torch.ops.aten.unsqueeze.default(unsqueeze_1578, -1);  unsqueeze_1578 = None
        sub_243 = torch.ops.aten.sub.Tensor(convolution_243, unsqueeze_1577);  convolution_243 = unsqueeze_1577 = None
        mul_638 = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_1579);  sub_243 = unsqueeze_1579 = None
        unsqueeze_1580 = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_1581 = torch.ops.aten.unsqueeze.default(unsqueeze_1580, -1);  unsqueeze_1580 = None
        mul_639 = torch.ops.aten.mul.Tensor(mul_638, unsqueeze_1581);  mul_638 = unsqueeze_1581 = None
        unsqueeze_1582 = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1583 = torch.ops.aten.unsqueeze.default(unsqueeze_1582, -1);  unsqueeze_1582 = None
        add_441 = torch.ops.aten.add.Tensor(mul_639, unsqueeze_1583);  mul_639 = unsqueeze_1583 = None
        relu_190 = torch.ops.aten.relu.default(add_441);  add_441 = None
        convolution_244 = torch.ops.aten.convolution.default(relu_190, arg335_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_190 = arg335_1 = None
        add_442 = torch.ops.aten.add.Tensor(arg337_1, 1e-05);  arg337_1 = None
        sqrt_198 = torch.ops.aten.sqrt.default(add_442);  add_442 = None
        reciprocal_198 = torch.ops.aten.reciprocal.default(sqrt_198);  sqrt_198 = None
        mul_640 = torch.ops.aten.mul.Tensor(reciprocal_198, 1);  reciprocal_198 = None
        unsqueeze_1584 = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_1585 = torch.ops.aten.unsqueeze.default(unsqueeze_1584, -1);  unsqueeze_1584 = None
        unsqueeze_1586 = torch.ops.aten.unsqueeze.default(mul_640, -1);  mul_640 = None
        unsqueeze_1587 = torch.ops.aten.unsqueeze.default(unsqueeze_1586, -1);  unsqueeze_1586 = None
        sub_244 = torch.ops.aten.sub.Tensor(convolution_244, unsqueeze_1585);  convolution_244 = unsqueeze_1585 = None
        mul_641 = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_1587);  sub_244 = unsqueeze_1587 = None
        unsqueeze_1588 = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
        unsqueeze_1589 = torch.ops.aten.unsqueeze.default(unsqueeze_1588, -1);  unsqueeze_1588 = None
        mul_642 = torch.ops.aten.mul.Tensor(mul_641, unsqueeze_1589);  mul_641 = unsqueeze_1589 = None
        unsqueeze_1590 = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1591 = torch.ops.aten.unsqueeze.default(unsqueeze_1590, -1);  unsqueeze_1590 = None
        add_443 = torch.ops.aten.add.Tensor(mul_642, unsqueeze_1591);  mul_642 = unsqueeze_1591 = None
        relu_191 = torch.ops.aten.relu.default(add_443);  add_443 = None
        view_278 = torch.ops.aten.view.default(relu_191, [8, 2, 256, 16, 16]);  relu_191 = None
        sum_139 = torch.ops.aten.sum.dim_IntList(view_278, [1])
        mean_47 = torch.ops.aten.mean.dim(sum_139, [2, 3], True);  sum_139 = None
        convolution_245 = torch.ops.aten.convolution.default(mean_47, arg340_1, arg341_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_47 = arg340_1 = arg341_1 = None
        add_444 = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_199 = torch.ops.aten.sqrt.default(add_444);  add_444 = None
        reciprocal_199 = torch.ops.aten.reciprocal.default(sqrt_199);  sqrt_199 = None
        mul_643 = torch.ops.aten.mul.Tensor(reciprocal_199, 1);  reciprocal_199 = None
        unsqueeze_1592 = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_1593 = torch.ops.aten.unsqueeze.default(unsqueeze_1592, -1);  unsqueeze_1592 = None
        unsqueeze_1594 = torch.ops.aten.unsqueeze.default(mul_643, -1);  mul_643 = None
        unsqueeze_1595 = torch.ops.aten.unsqueeze.default(unsqueeze_1594, -1);  unsqueeze_1594 = None
        sub_245 = torch.ops.aten.sub.Tensor(convolution_245, unsqueeze_1593);  convolution_245 = unsqueeze_1593 = None
        mul_644 = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_1595);  sub_245 = unsqueeze_1595 = None
        unsqueeze_1596 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1597 = torch.ops.aten.unsqueeze.default(unsqueeze_1596, -1);  unsqueeze_1596 = None
        mul_645 = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_1597);  mul_644 = unsqueeze_1597 = None
        unsqueeze_1598 = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_1599 = torch.ops.aten.unsqueeze.default(unsqueeze_1598, -1);  unsqueeze_1598 = None
        add_445 = torch.ops.aten.add.Tensor(mul_645, unsqueeze_1599);  mul_645 = unsqueeze_1599 = None
        relu_192 = torch.ops.aten.relu.default(add_445);  add_445 = None
        convolution_246 = torch.ops.aten.convolution.default(relu_192, arg346_1, arg347_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_192 = arg346_1 = arg347_1 = None
        view_279 = torch.ops.aten.view.default(convolution_246, [8, 1, 2, -1]);  convolution_246 = None
        permute_47 = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
        amax_46 = torch.ops.aten.amax.default(permute_47, [1], True)
        sub_246 = torch.ops.aten.sub.Tensor(permute_47, amax_46);  permute_47 = amax_46 = None
        exp_46 = torch.ops.aten.exp.default(sub_246);  sub_246 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(exp_46, [1], True)
        div_46 = torch.ops.aten.div.Tensor(exp_46, sum_140);  exp_46 = sum_140 = None
        view_280 = torch.ops.aten.view.default(div_46, [8, -1]);  div_46 = None
        view_281 = torch.ops.aten.view.default(view_280, [8, -1, 1, 1]);  view_280 = None
        view_282 = torch.ops.aten.view.default(view_281, [8, 2, 256, 1, 1]);  view_281 = None
        mul_646 = torch.ops.aten.mul.Tensor(view_278, view_282);  view_278 = view_282 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(mul_646, [1]);  mul_646 = None
        convolution_247 = torch.ops.aten.convolution.default(sum_141, arg348_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_141 = arg348_1 = None
        add_446 = torch.ops.aten.add.Tensor(arg350_1, 1e-05);  arg350_1 = None
        sqrt_200 = torch.ops.aten.sqrt.default(add_446);  add_446 = None
        reciprocal_200 = torch.ops.aten.reciprocal.default(sqrt_200);  sqrt_200 = None
        mul_647 = torch.ops.aten.mul.Tensor(reciprocal_200, 1);  reciprocal_200 = None
        unsqueeze_1600 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1601 = torch.ops.aten.unsqueeze.default(unsqueeze_1600, -1);  unsqueeze_1600 = None
        unsqueeze_1602 = torch.ops.aten.unsqueeze.default(mul_647, -1);  mul_647 = None
        unsqueeze_1603 = torch.ops.aten.unsqueeze.default(unsqueeze_1602, -1);  unsqueeze_1602 = None
        sub_247 = torch.ops.aten.sub.Tensor(convolution_247, unsqueeze_1601);  convolution_247 = unsqueeze_1601 = None
        mul_648 = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_1603);  sub_247 = unsqueeze_1603 = None
        unsqueeze_1604 = torch.ops.aten.unsqueeze.default(arg351_1, -1);  arg351_1 = None
        unsqueeze_1605 = torch.ops.aten.unsqueeze.default(unsqueeze_1604, -1);  unsqueeze_1604 = None
        mul_649 = torch.ops.aten.mul.Tensor(mul_648, unsqueeze_1605);  mul_648 = unsqueeze_1605 = None
        unsqueeze_1606 = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1607 = torch.ops.aten.unsqueeze.default(unsqueeze_1606, -1);  unsqueeze_1606 = None
        add_447 = torch.ops.aten.add.Tensor(mul_649, unsqueeze_1607);  mul_649 = unsqueeze_1607 = None
        add_448 = torch.ops.aten.add.Tensor(add_447, relu_189);  add_447 = relu_189 = None
        relu_193 = torch.ops.aten.relu.default(add_448);  add_448 = None
        convolution_248 = torch.ops.aten.convolution.default(relu_193, arg353_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg353_1 = None
        add_449 = torch.ops.aten.add.Tensor(arg355_1, 1e-05);  arg355_1 = None
        sqrt_201 = torch.ops.aten.sqrt.default(add_449);  add_449 = None
        reciprocal_201 = torch.ops.aten.reciprocal.default(sqrt_201);  sqrt_201 = None
        mul_650 = torch.ops.aten.mul.Tensor(reciprocal_201, 1);  reciprocal_201 = None
        unsqueeze_1608 = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1609 = torch.ops.aten.unsqueeze.default(unsqueeze_1608, -1);  unsqueeze_1608 = None
        unsqueeze_1610 = torch.ops.aten.unsqueeze.default(mul_650, -1);  mul_650 = None
        unsqueeze_1611 = torch.ops.aten.unsqueeze.default(unsqueeze_1610, -1);  unsqueeze_1610 = None
        sub_248 = torch.ops.aten.sub.Tensor(convolution_248, unsqueeze_1609);  convolution_248 = unsqueeze_1609 = None
        mul_651 = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_1611);  sub_248 = unsqueeze_1611 = None
        unsqueeze_1612 = torch.ops.aten.unsqueeze.default(arg356_1, -1);  arg356_1 = None
        unsqueeze_1613 = torch.ops.aten.unsqueeze.default(unsqueeze_1612, -1);  unsqueeze_1612 = None
        mul_652 = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_1613);  mul_651 = unsqueeze_1613 = None
        unsqueeze_1614 = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1615 = torch.ops.aten.unsqueeze.default(unsqueeze_1614, -1);  unsqueeze_1614 = None
        add_450 = torch.ops.aten.add.Tensor(mul_652, unsqueeze_1615);  mul_652 = unsqueeze_1615 = None
        relu_194 = torch.ops.aten.relu.default(add_450);  add_450 = None
        convolution_249 = torch.ops.aten.convolution.default(relu_194, arg358_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_194 = arg358_1 = None
        add_451 = torch.ops.aten.add.Tensor(arg360_1, 1e-05);  arg360_1 = None
        sqrt_202 = torch.ops.aten.sqrt.default(add_451);  add_451 = None
        reciprocal_202 = torch.ops.aten.reciprocal.default(sqrt_202);  sqrt_202 = None
        mul_653 = torch.ops.aten.mul.Tensor(reciprocal_202, 1);  reciprocal_202 = None
        unsqueeze_1616 = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1617 = torch.ops.aten.unsqueeze.default(unsqueeze_1616, -1);  unsqueeze_1616 = None
        unsqueeze_1618 = torch.ops.aten.unsqueeze.default(mul_653, -1);  mul_653 = None
        unsqueeze_1619 = torch.ops.aten.unsqueeze.default(unsqueeze_1618, -1);  unsqueeze_1618 = None
        sub_249 = torch.ops.aten.sub.Tensor(convolution_249, unsqueeze_1617);  convolution_249 = unsqueeze_1617 = None
        mul_654 = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_1619);  sub_249 = unsqueeze_1619 = None
        unsqueeze_1620 = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
        unsqueeze_1621 = torch.ops.aten.unsqueeze.default(unsqueeze_1620, -1);  unsqueeze_1620 = None
        mul_655 = torch.ops.aten.mul.Tensor(mul_654, unsqueeze_1621);  mul_654 = unsqueeze_1621 = None
        unsqueeze_1622 = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_1623 = torch.ops.aten.unsqueeze.default(unsqueeze_1622, -1);  unsqueeze_1622 = None
        add_452 = torch.ops.aten.add.Tensor(mul_655, unsqueeze_1623);  mul_655 = unsqueeze_1623 = None
        relu_195 = torch.ops.aten.relu.default(add_452);  add_452 = None
        view_284 = torch.ops.aten.view.default(relu_195, [8, 2, 256, 16, 16]);  relu_195 = None
        sum_142 = torch.ops.aten.sum.dim_IntList(view_284, [1])
        mean_48 = torch.ops.aten.mean.dim(sum_142, [2, 3], True);  sum_142 = None
        convolution_250 = torch.ops.aten.convolution.default(mean_48, arg363_1, arg364_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_48 = arg363_1 = arg364_1 = None
        add_453 = torch.ops.aten.add.Tensor(arg366_1, 1e-05);  arg366_1 = None
        sqrt_203 = torch.ops.aten.sqrt.default(add_453);  add_453 = None
        reciprocal_203 = torch.ops.aten.reciprocal.default(sqrt_203);  sqrt_203 = None
        mul_656 = torch.ops.aten.mul.Tensor(reciprocal_203, 1);  reciprocal_203 = None
        unsqueeze_1624 = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_1625 = torch.ops.aten.unsqueeze.default(unsqueeze_1624, -1);  unsqueeze_1624 = None
        unsqueeze_1626 = torch.ops.aten.unsqueeze.default(mul_656, -1);  mul_656 = None
        unsqueeze_1627 = torch.ops.aten.unsqueeze.default(unsqueeze_1626, -1);  unsqueeze_1626 = None
        sub_250 = torch.ops.aten.sub.Tensor(convolution_250, unsqueeze_1625);  convolution_250 = unsqueeze_1625 = None
        mul_657 = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_1627);  sub_250 = unsqueeze_1627 = None
        unsqueeze_1628 = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_1629 = torch.ops.aten.unsqueeze.default(unsqueeze_1628, -1);  unsqueeze_1628 = None
        mul_658 = torch.ops.aten.mul.Tensor(mul_657, unsqueeze_1629);  mul_657 = unsqueeze_1629 = None
        unsqueeze_1630 = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_1631 = torch.ops.aten.unsqueeze.default(unsqueeze_1630, -1);  unsqueeze_1630 = None
        add_454 = torch.ops.aten.add.Tensor(mul_658, unsqueeze_1631);  mul_658 = unsqueeze_1631 = None
        relu_196 = torch.ops.aten.relu.default(add_454);  add_454 = None
        convolution_251 = torch.ops.aten.convolution.default(relu_196, arg369_1, arg370_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_196 = arg369_1 = arg370_1 = None
        view_285 = torch.ops.aten.view.default(convolution_251, [8, 1, 2, -1]);  convolution_251 = None
        permute_48 = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        amax_47 = torch.ops.aten.amax.default(permute_48, [1], True)
        sub_251 = torch.ops.aten.sub.Tensor(permute_48, amax_47);  permute_48 = amax_47 = None
        exp_47 = torch.ops.aten.exp.default(sub_251);  sub_251 = None
        sum_143 = torch.ops.aten.sum.dim_IntList(exp_47, [1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_47, sum_143);  exp_47 = sum_143 = None
        view_286 = torch.ops.aten.view.default(div_47, [8, -1]);  div_47 = None
        view_287 = torch.ops.aten.view.default(view_286, [8, -1, 1, 1]);  view_286 = None
        view_288 = torch.ops.aten.view.default(view_287, [8, 2, 256, 1, 1]);  view_287 = None
        mul_659 = torch.ops.aten.mul.Tensor(view_284, view_288);  view_284 = view_288 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(mul_659, [1]);  mul_659 = None
        convolution_252 = torch.ops.aten.convolution.default(sum_144, arg371_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_144 = arg371_1 = None
        add_455 = torch.ops.aten.add.Tensor(arg373_1, 1e-05);  arg373_1 = None
        sqrt_204 = torch.ops.aten.sqrt.default(add_455);  add_455 = None
        reciprocal_204 = torch.ops.aten.reciprocal.default(sqrt_204);  sqrt_204 = None
        mul_660 = torch.ops.aten.mul.Tensor(reciprocal_204, 1);  reciprocal_204 = None
        unsqueeze_1632 = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_1633 = torch.ops.aten.unsqueeze.default(unsqueeze_1632, -1);  unsqueeze_1632 = None
        unsqueeze_1634 = torch.ops.aten.unsqueeze.default(mul_660, -1);  mul_660 = None
        unsqueeze_1635 = torch.ops.aten.unsqueeze.default(unsqueeze_1634, -1);  unsqueeze_1634 = None
        sub_252 = torch.ops.aten.sub.Tensor(convolution_252, unsqueeze_1633);  convolution_252 = unsqueeze_1633 = None
        mul_661 = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_1635);  sub_252 = unsqueeze_1635 = None
        unsqueeze_1636 = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1637 = torch.ops.aten.unsqueeze.default(unsqueeze_1636, -1);  unsqueeze_1636 = None
        mul_662 = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_1637);  mul_661 = unsqueeze_1637 = None
        unsqueeze_1638 = torch.ops.aten.unsqueeze.default(arg375_1, -1);  arg375_1 = None
        unsqueeze_1639 = torch.ops.aten.unsqueeze.default(unsqueeze_1638, -1);  unsqueeze_1638 = None
        add_456 = torch.ops.aten.add.Tensor(mul_662, unsqueeze_1639);  mul_662 = unsqueeze_1639 = None
        add_457 = torch.ops.aten.add.Tensor(add_456, relu_193);  add_456 = relu_193 = None
        relu_197 = torch.ops.aten.relu.default(add_457);  add_457 = None
        convolution_253 = torch.ops.aten.convolution.default(relu_197, arg376_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg376_1 = None
        add_458 = torch.ops.aten.add.Tensor(arg378_1, 1e-05);  arg378_1 = None
        sqrt_205 = torch.ops.aten.sqrt.default(add_458);  add_458 = None
        reciprocal_205 = torch.ops.aten.reciprocal.default(sqrt_205);  sqrt_205 = None
        mul_663 = torch.ops.aten.mul.Tensor(reciprocal_205, 1);  reciprocal_205 = None
        unsqueeze_1640 = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
        unsqueeze_1641 = torch.ops.aten.unsqueeze.default(unsqueeze_1640, -1);  unsqueeze_1640 = None
        unsqueeze_1642 = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_1643 = torch.ops.aten.unsqueeze.default(unsqueeze_1642, -1);  unsqueeze_1642 = None
        sub_253 = torch.ops.aten.sub.Tensor(convolution_253, unsqueeze_1641);  convolution_253 = unsqueeze_1641 = None
        mul_664 = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_1643);  sub_253 = unsqueeze_1643 = None
        unsqueeze_1644 = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1645 = torch.ops.aten.unsqueeze.default(unsqueeze_1644, -1);  unsqueeze_1644 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_1645);  mul_664 = unsqueeze_1645 = None
        unsqueeze_1646 = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_1647 = torch.ops.aten.unsqueeze.default(unsqueeze_1646, -1);  unsqueeze_1646 = None
        add_459 = torch.ops.aten.add.Tensor(mul_665, unsqueeze_1647);  mul_665 = unsqueeze_1647 = None
        relu_198 = torch.ops.aten.relu.default(add_459);  add_459 = None
        convolution_254 = torch.ops.aten.convolution.default(relu_198, arg381_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_198 = arg381_1 = None
        add_460 = torch.ops.aten.add.Tensor(arg383_1, 1e-05);  arg383_1 = None
        sqrt_206 = torch.ops.aten.sqrt.default(add_460);  add_460 = None
        reciprocal_206 = torch.ops.aten.reciprocal.default(sqrt_206);  sqrt_206 = None
        mul_666 = torch.ops.aten.mul.Tensor(reciprocal_206, 1);  reciprocal_206 = None
        unsqueeze_1648 = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
        unsqueeze_1649 = torch.ops.aten.unsqueeze.default(unsqueeze_1648, -1);  unsqueeze_1648 = None
        unsqueeze_1650 = torch.ops.aten.unsqueeze.default(mul_666, -1);  mul_666 = None
        unsqueeze_1651 = torch.ops.aten.unsqueeze.default(unsqueeze_1650, -1);  unsqueeze_1650 = None
        sub_254 = torch.ops.aten.sub.Tensor(convolution_254, unsqueeze_1649);  convolution_254 = unsqueeze_1649 = None
        mul_667 = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_1651);  sub_254 = unsqueeze_1651 = None
        unsqueeze_1652 = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1653 = torch.ops.aten.unsqueeze.default(unsqueeze_1652, -1);  unsqueeze_1652 = None
        mul_668 = torch.ops.aten.mul.Tensor(mul_667, unsqueeze_1653);  mul_667 = unsqueeze_1653 = None
        unsqueeze_1654 = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_1655 = torch.ops.aten.unsqueeze.default(unsqueeze_1654, -1);  unsqueeze_1654 = None
        add_461 = torch.ops.aten.add.Tensor(mul_668, unsqueeze_1655);  mul_668 = unsqueeze_1655 = None
        relu_199 = torch.ops.aten.relu.default(add_461);  add_461 = None
        view_290 = torch.ops.aten.view.default(relu_199, [8, 2, 256, 16, 16]);  relu_199 = None
        sum_145 = torch.ops.aten.sum.dim_IntList(view_290, [1])
        mean_49 = torch.ops.aten.mean.dim(sum_145, [2, 3], True);  sum_145 = None
        convolution_255 = torch.ops.aten.convolution.default(mean_49, arg386_1, arg387_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_49 = arg386_1 = arg387_1 = None
        add_462 = torch.ops.aten.add.Tensor(arg389_1, 1e-05);  arg389_1 = None
        sqrt_207 = torch.ops.aten.sqrt.default(add_462);  add_462 = None
        reciprocal_207 = torch.ops.aten.reciprocal.default(sqrt_207);  sqrt_207 = None
        mul_669 = torch.ops.aten.mul.Tensor(reciprocal_207, 1);  reciprocal_207 = None
        unsqueeze_1656 = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1657 = torch.ops.aten.unsqueeze.default(unsqueeze_1656, -1);  unsqueeze_1656 = None
        unsqueeze_1658 = torch.ops.aten.unsqueeze.default(mul_669, -1);  mul_669 = None
        unsqueeze_1659 = torch.ops.aten.unsqueeze.default(unsqueeze_1658, -1);  unsqueeze_1658 = None
        sub_255 = torch.ops.aten.sub.Tensor(convolution_255, unsqueeze_1657);  convolution_255 = unsqueeze_1657 = None
        mul_670 = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_1659);  sub_255 = unsqueeze_1659 = None
        unsqueeze_1660 = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_1661 = torch.ops.aten.unsqueeze.default(unsqueeze_1660, -1);  unsqueeze_1660 = None
        mul_671 = torch.ops.aten.mul.Tensor(mul_670, unsqueeze_1661);  mul_670 = unsqueeze_1661 = None
        unsqueeze_1662 = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_1663 = torch.ops.aten.unsqueeze.default(unsqueeze_1662, -1);  unsqueeze_1662 = None
        add_463 = torch.ops.aten.add.Tensor(mul_671, unsqueeze_1663);  mul_671 = unsqueeze_1663 = None
        relu_200 = torch.ops.aten.relu.default(add_463);  add_463 = None
        convolution_256 = torch.ops.aten.convolution.default(relu_200, arg392_1, arg393_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_200 = arg392_1 = arg393_1 = None
        view_291 = torch.ops.aten.view.default(convolution_256, [8, 1, 2, -1]);  convolution_256 = None
        permute_49 = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        amax_48 = torch.ops.aten.amax.default(permute_49, [1], True)
        sub_256 = torch.ops.aten.sub.Tensor(permute_49, amax_48);  permute_49 = amax_48 = None
        exp_48 = torch.ops.aten.exp.default(sub_256);  sub_256 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(exp_48, [1], True)
        div_48 = torch.ops.aten.div.Tensor(exp_48, sum_146);  exp_48 = sum_146 = None
        view_292 = torch.ops.aten.view.default(div_48, [8, -1]);  div_48 = None
        view_293 = torch.ops.aten.view.default(view_292, [8, -1, 1, 1]);  view_292 = None
        view_294 = torch.ops.aten.view.default(view_293, [8, 2, 256, 1, 1]);  view_293 = None
        mul_672 = torch.ops.aten.mul.Tensor(view_290, view_294);  view_290 = view_294 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(mul_672, [1]);  mul_672 = None
        convolution_257 = torch.ops.aten.convolution.default(sum_147, arg394_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_147 = arg394_1 = None
        add_464 = torch.ops.aten.add.Tensor(arg396_1, 1e-05);  arg396_1 = None
        sqrt_208 = torch.ops.aten.sqrt.default(add_464);  add_464 = None
        reciprocal_208 = torch.ops.aten.reciprocal.default(sqrt_208);  sqrt_208 = None
        mul_673 = torch.ops.aten.mul.Tensor(reciprocal_208, 1);  reciprocal_208 = None
        unsqueeze_1664 = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_1665 = torch.ops.aten.unsqueeze.default(unsqueeze_1664, -1);  unsqueeze_1664 = None
        unsqueeze_1666 = torch.ops.aten.unsqueeze.default(mul_673, -1);  mul_673 = None
        unsqueeze_1667 = torch.ops.aten.unsqueeze.default(unsqueeze_1666, -1);  unsqueeze_1666 = None
        sub_257 = torch.ops.aten.sub.Tensor(convolution_257, unsqueeze_1665);  convolution_257 = unsqueeze_1665 = None
        mul_674 = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_1667);  sub_257 = unsqueeze_1667 = None
        unsqueeze_1668 = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_1669 = torch.ops.aten.unsqueeze.default(unsqueeze_1668, -1);  unsqueeze_1668 = None
        mul_675 = torch.ops.aten.mul.Tensor(mul_674, unsqueeze_1669);  mul_674 = unsqueeze_1669 = None
        unsqueeze_1670 = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1671 = torch.ops.aten.unsqueeze.default(unsqueeze_1670, -1);  unsqueeze_1670 = None
        add_465 = torch.ops.aten.add.Tensor(mul_675, unsqueeze_1671);  mul_675 = unsqueeze_1671 = None
        add_466 = torch.ops.aten.add.Tensor(add_465, relu_197);  add_465 = relu_197 = None
        relu_201 = torch.ops.aten.relu.default(add_466);  add_466 = None
        convolution_258 = torch.ops.aten.convolution.default(relu_201, arg399_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg399_1 = None
        add_467 = torch.ops.aten.add.Tensor(arg401_1, 1e-05);  arg401_1 = None
        sqrt_209 = torch.ops.aten.sqrt.default(add_467);  add_467 = None
        reciprocal_209 = torch.ops.aten.reciprocal.default(sqrt_209);  sqrt_209 = None
        mul_676 = torch.ops.aten.mul.Tensor(reciprocal_209, 1);  reciprocal_209 = None
        unsqueeze_1672 = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_1673 = torch.ops.aten.unsqueeze.default(unsqueeze_1672, -1);  unsqueeze_1672 = None
        unsqueeze_1674 = torch.ops.aten.unsqueeze.default(mul_676, -1);  mul_676 = None
        unsqueeze_1675 = torch.ops.aten.unsqueeze.default(unsqueeze_1674, -1);  unsqueeze_1674 = None
        sub_258 = torch.ops.aten.sub.Tensor(convolution_258, unsqueeze_1673);  convolution_258 = unsqueeze_1673 = None
        mul_677 = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_1675);  sub_258 = unsqueeze_1675 = None
        unsqueeze_1676 = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
        unsqueeze_1677 = torch.ops.aten.unsqueeze.default(unsqueeze_1676, -1);  unsqueeze_1676 = None
        mul_678 = torch.ops.aten.mul.Tensor(mul_677, unsqueeze_1677);  mul_677 = unsqueeze_1677 = None
        unsqueeze_1678 = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_1679 = torch.ops.aten.unsqueeze.default(unsqueeze_1678, -1);  unsqueeze_1678 = None
        add_468 = torch.ops.aten.add.Tensor(mul_678, unsqueeze_1679);  mul_678 = unsqueeze_1679 = None
        relu_202 = torch.ops.aten.relu.default(add_468);  add_468 = None
        convolution_259 = torch.ops.aten.convolution.default(relu_202, arg404_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_202 = arg404_1 = None
        add_469 = torch.ops.aten.add.Tensor(arg406_1, 1e-05);  arg406_1 = None
        sqrt_210 = torch.ops.aten.sqrt.default(add_469);  add_469 = None
        reciprocal_210 = torch.ops.aten.reciprocal.default(sqrt_210);  sqrt_210 = None
        mul_679 = torch.ops.aten.mul.Tensor(reciprocal_210, 1);  reciprocal_210 = None
        unsqueeze_1680 = torch.ops.aten.unsqueeze.default(arg405_1, -1);  arg405_1 = None
        unsqueeze_1681 = torch.ops.aten.unsqueeze.default(unsqueeze_1680, -1);  unsqueeze_1680 = None
        unsqueeze_1682 = torch.ops.aten.unsqueeze.default(mul_679, -1);  mul_679 = None
        unsqueeze_1683 = torch.ops.aten.unsqueeze.default(unsqueeze_1682, -1);  unsqueeze_1682 = None
        sub_259 = torch.ops.aten.sub.Tensor(convolution_259, unsqueeze_1681);  convolution_259 = unsqueeze_1681 = None
        mul_680 = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_1683);  sub_259 = unsqueeze_1683 = None
        unsqueeze_1684 = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_1685 = torch.ops.aten.unsqueeze.default(unsqueeze_1684, -1);  unsqueeze_1684 = None
        mul_681 = torch.ops.aten.mul.Tensor(mul_680, unsqueeze_1685);  mul_680 = unsqueeze_1685 = None
        unsqueeze_1686 = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
        unsqueeze_1687 = torch.ops.aten.unsqueeze.default(unsqueeze_1686, -1);  unsqueeze_1686 = None
        add_470 = torch.ops.aten.add.Tensor(mul_681, unsqueeze_1687);  mul_681 = unsqueeze_1687 = None
        relu_203 = torch.ops.aten.relu.default(add_470);  add_470 = None
        view_296 = torch.ops.aten.view.default(relu_203, [8, 2, 256, 16, 16]);  relu_203 = None
        sum_148 = torch.ops.aten.sum.dim_IntList(view_296, [1])
        mean_50 = torch.ops.aten.mean.dim(sum_148, [2, 3], True);  sum_148 = None
        convolution_260 = torch.ops.aten.convolution.default(mean_50, arg409_1, arg410_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_50 = arg409_1 = arg410_1 = None
        add_471 = torch.ops.aten.add.Tensor(arg412_1, 1e-05);  arg412_1 = None
        sqrt_211 = torch.ops.aten.sqrt.default(add_471);  add_471 = None
        reciprocal_211 = torch.ops.aten.reciprocal.default(sqrt_211);  sqrt_211 = None
        mul_682 = torch.ops.aten.mul.Tensor(reciprocal_211, 1);  reciprocal_211 = None
        unsqueeze_1688 = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_1689 = torch.ops.aten.unsqueeze.default(unsqueeze_1688, -1);  unsqueeze_1688 = None
        unsqueeze_1690 = torch.ops.aten.unsqueeze.default(mul_682, -1);  mul_682 = None
        unsqueeze_1691 = torch.ops.aten.unsqueeze.default(unsqueeze_1690, -1);  unsqueeze_1690 = None
        sub_260 = torch.ops.aten.sub.Tensor(convolution_260, unsqueeze_1689);  convolution_260 = unsqueeze_1689 = None
        mul_683 = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_1691);  sub_260 = unsqueeze_1691 = None
        unsqueeze_1692 = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
        unsqueeze_1693 = torch.ops.aten.unsqueeze.default(unsqueeze_1692, -1);  unsqueeze_1692 = None
        mul_684 = torch.ops.aten.mul.Tensor(mul_683, unsqueeze_1693);  mul_683 = unsqueeze_1693 = None
        unsqueeze_1694 = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1695 = torch.ops.aten.unsqueeze.default(unsqueeze_1694, -1);  unsqueeze_1694 = None
        add_472 = torch.ops.aten.add.Tensor(mul_684, unsqueeze_1695);  mul_684 = unsqueeze_1695 = None
        relu_204 = torch.ops.aten.relu.default(add_472);  add_472 = None
        convolution_261 = torch.ops.aten.convolution.default(relu_204, arg415_1, arg416_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_204 = arg415_1 = arg416_1 = None
        view_297 = torch.ops.aten.view.default(convolution_261, [8, 1, 2, -1]);  convolution_261 = None
        permute_50 = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
        amax_49 = torch.ops.aten.amax.default(permute_50, [1], True)
        sub_261 = torch.ops.aten.sub.Tensor(permute_50, amax_49);  permute_50 = amax_49 = None
        exp_49 = torch.ops.aten.exp.default(sub_261);  sub_261 = None
        sum_149 = torch.ops.aten.sum.dim_IntList(exp_49, [1], True)
        div_49 = torch.ops.aten.div.Tensor(exp_49, sum_149);  exp_49 = sum_149 = None
        view_298 = torch.ops.aten.view.default(div_49, [8, -1]);  div_49 = None
        view_299 = torch.ops.aten.view.default(view_298, [8, -1, 1, 1]);  view_298 = None
        view_300 = torch.ops.aten.view.default(view_299, [8, 2, 256, 1, 1]);  view_299 = None
        mul_685 = torch.ops.aten.mul.Tensor(view_296, view_300);  view_296 = view_300 = None
        sum_150 = torch.ops.aten.sum.dim_IntList(mul_685, [1]);  mul_685 = None
        convolution_262 = torch.ops.aten.convolution.default(sum_150, arg417_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_150 = arg417_1 = None
        add_473 = torch.ops.aten.add.Tensor(arg419_1, 1e-05);  arg419_1 = None
        sqrt_212 = torch.ops.aten.sqrt.default(add_473);  add_473 = None
        reciprocal_212 = torch.ops.aten.reciprocal.default(sqrt_212);  sqrt_212 = None
        mul_686 = torch.ops.aten.mul.Tensor(reciprocal_212, 1);  reciprocal_212 = None
        unsqueeze_1696 = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_1697 = torch.ops.aten.unsqueeze.default(unsqueeze_1696, -1);  unsqueeze_1696 = None
        unsqueeze_1698 = torch.ops.aten.unsqueeze.default(mul_686, -1);  mul_686 = None
        unsqueeze_1699 = torch.ops.aten.unsqueeze.default(unsqueeze_1698, -1);  unsqueeze_1698 = None
        sub_262 = torch.ops.aten.sub.Tensor(convolution_262, unsqueeze_1697);  convolution_262 = unsqueeze_1697 = None
        mul_687 = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_1699);  sub_262 = unsqueeze_1699 = None
        unsqueeze_1700 = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_1701 = torch.ops.aten.unsqueeze.default(unsqueeze_1700, -1);  unsqueeze_1700 = None
        mul_688 = torch.ops.aten.mul.Tensor(mul_687, unsqueeze_1701);  mul_687 = unsqueeze_1701 = None
        unsqueeze_1702 = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
        unsqueeze_1703 = torch.ops.aten.unsqueeze.default(unsqueeze_1702, -1);  unsqueeze_1702 = None
        add_474 = torch.ops.aten.add.Tensor(mul_688, unsqueeze_1703);  mul_688 = unsqueeze_1703 = None
        add_475 = torch.ops.aten.add.Tensor(add_474, relu_201);  add_474 = relu_201 = None
        relu_205 = torch.ops.aten.relu.default(add_475);  add_475 = None
        convolution_263 = torch.ops.aten.convolution.default(relu_205, arg422_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg422_1 = None
        add_476 = torch.ops.aten.add.Tensor(arg424_1, 1e-05);  arg424_1 = None
        sqrt_213 = torch.ops.aten.sqrt.default(add_476);  add_476 = None
        reciprocal_213 = torch.ops.aten.reciprocal.default(sqrt_213);  sqrt_213 = None
        mul_689 = torch.ops.aten.mul.Tensor(reciprocal_213, 1);  reciprocal_213 = None
        unsqueeze_1704 = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1705 = torch.ops.aten.unsqueeze.default(unsqueeze_1704, -1);  unsqueeze_1704 = None
        unsqueeze_1706 = torch.ops.aten.unsqueeze.default(mul_689, -1);  mul_689 = None
        unsqueeze_1707 = torch.ops.aten.unsqueeze.default(unsqueeze_1706, -1);  unsqueeze_1706 = None
        sub_263 = torch.ops.aten.sub.Tensor(convolution_263, unsqueeze_1705);  convolution_263 = unsqueeze_1705 = None
        mul_690 = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_1707);  sub_263 = unsqueeze_1707 = None
        unsqueeze_1708 = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1709 = torch.ops.aten.unsqueeze.default(unsqueeze_1708, -1);  unsqueeze_1708 = None
        mul_691 = torch.ops.aten.mul.Tensor(mul_690, unsqueeze_1709);  mul_690 = unsqueeze_1709 = None
        unsqueeze_1710 = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
        unsqueeze_1711 = torch.ops.aten.unsqueeze.default(unsqueeze_1710, -1);  unsqueeze_1710 = None
        add_477 = torch.ops.aten.add.Tensor(mul_691, unsqueeze_1711);  mul_691 = unsqueeze_1711 = None
        relu_206 = torch.ops.aten.relu.default(add_477);  add_477 = None
        convolution_264 = torch.ops.aten.convolution.default(relu_206, arg427_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_206 = arg427_1 = None
        add_478 = torch.ops.aten.add.Tensor(arg429_1, 1e-05);  arg429_1 = None
        sqrt_214 = torch.ops.aten.sqrt.default(add_478);  add_478 = None
        reciprocal_214 = torch.ops.aten.reciprocal.default(sqrt_214);  sqrt_214 = None
        mul_692 = torch.ops.aten.mul.Tensor(reciprocal_214, 1);  reciprocal_214 = None
        unsqueeze_1712 = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1713 = torch.ops.aten.unsqueeze.default(unsqueeze_1712, -1);  unsqueeze_1712 = None
        unsqueeze_1714 = torch.ops.aten.unsqueeze.default(mul_692, -1);  mul_692 = None
        unsqueeze_1715 = torch.ops.aten.unsqueeze.default(unsqueeze_1714, -1);  unsqueeze_1714 = None
        sub_264 = torch.ops.aten.sub.Tensor(convolution_264, unsqueeze_1713);  convolution_264 = unsqueeze_1713 = None
        mul_693 = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_1715);  sub_264 = unsqueeze_1715 = None
        unsqueeze_1716 = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_1717 = torch.ops.aten.unsqueeze.default(unsqueeze_1716, -1);  unsqueeze_1716 = None
        mul_694 = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_1717);  mul_693 = unsqueeze_1717 = None
        unsqueeze_1718 = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_1719 = torch.ops.aten.unsqueeze.default(unsqueeze_1718, -1);  unsqueeze_1718 = None
        add_479 = torch.ops.aten.add.Tensor(mul_694, unsqueeze_1719);  mul_694 = unsqueeze_1719 = None
        relu_207 = torch.ops.aten.relu.default(add_479);  add_479 = None
        view_302 = torch.ops.aten.view.default(relu_207, [8, 2, 256, 16, 16]);  relu_207 = None
        sum_151 = torch.ops.aten.sum.dim_IntList(view_302, [1])
        mean_51 = torch.ops.aten.mean.dim(sum_151, [2, 3], True);  sum_151 = None
        convolution_265 = torch.ops.aten.convolution.default(mean_51, arg432_1, arg433_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_51 = arg432_1 = arg433_1 = None
        add_480 = torch.ops.aten.add.Tensor(arg435_1, 1e-05);  arg435_1 = None
        sqrt_215 = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_215 = torch.ops.aten.reciprocal.default(sqrt_215);  sqrt_215 = None
        mul_695 = torch.ops.aten.mul.Tensor(reciprocal_215, 1);  reciprocal_215 = None
        unsqueeze_1720 = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_1721 = torch.ops.aten.unsqueeze.default(unsqueeze_1720, -1);  unsqueeze_1720 = None
        unsqueeze_1722 = torch.ops.aten.unsqueeze.default(mul_695, -1);  mul_695 = None
        unsqueeze_1723 = torch.ops.aten.unsqueeze.default(unsqueeze_1722, -1);  unsqueeze_1722 = None
        sub_265 = torch.ops.aten.sub.Tensor(convolution_265, unsqueeze_1721);  convolution_265 = unsqueeze_1721 = None
        mul_696 = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_1723);  sub_265 = unsqueeze_1723 = None
        unsqueeze_1724 = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
        unsqueeze_1725 = torch.ops.aten.unsqueeze.default(unsqueeze_1724, -1);  unsqueeze_1724 = None
        mul_697 = torch.ops.aten.mul.Tensor(mul_696, unsqueeze_1725);  mul_696 = unsqueeze_1725 = None
        unsqueeze_1726 = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
        unsqueeze_1727 = torch.ops.aten.unsqueeze.default(unsqueeze_1726, -1);  unsqueeze_1726 = None
        add_481 = torch.ops.aten.add.Tensor(mul_697, unsqueeze_1727);  mul_697 = unsqueeze_1727 = None
        relu_208 = torch.ops.aten.relu.default(add_481);  add_481 = None
        convolution_266 = torch.ops.aten.convolution.default(relu_208, arg438_1, arg439_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_208 = arg438_1 = arg439_1 = None
        view_303 = torch.ops.aten.view.default(convolution_266, [8, 1, 2, -1]);  convolution_266 = None
        permute_51 = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        amax_50 = torch.ops.aten.amax.default(permute_51, [1], True)
        sub_266 = torch.ops.aten.sub.Tensor(permute_51, amax_50);  permute_51 = amax_50 = None
        exp_50 = torch.ops.aten.exp.default(sub_266);  sub_266 = None
        sum_152 = torch.ops.aten.sum.dim_IntList(exp_50, [1], True)
        div_50 = torch.ops.aten.div.Tensor(exp_50, sum_152);  exp_50 = sum_152 = None
        view_304 = torch.ops.aten.view.default(div_50, [8, -1]);  div_50 = None
        view_305 = torch.ops.aten.view.default(view_304, [8, -1, 1, 1]);  view_304 = None
        view_306 = torch.ops.aten.view.default(view_305, [8, 2, 256, 1, 1]);  view_305 = None
        mul_698 = torch.ops.aten.mul.Tensor(view_302, view_306);  view_302 = view_306 = None
        sum_153 = torch.ops.aten.sum.dim_IntList(mul_698, [1]);  mul_698 = None
        convolution_267 = torch.ops.aten.convolution.default(sum_153, arg440_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_153 = arg440_1 = None
        add_482 = torch.ops.aten.add.Tensor(arg442_1, 1e-05);  arg442_1 = None
        sqrt_216 = torch.ops.aten.sqrt.default(add_482);  add_482 = None
        reciprocal_216 = torch.ops.aten.reciprocal.default(sqrt_216);  sqrt_216 = None
        mul_699 = torch.ops.aten.mul.Tensor(reciprocal_216, 1);  reciprocal_216 = None
        unsqueeze_1728 = torch.ops.aten.unsqueeze.default(arg441_1, -1);  arg441_1 = None
        unsqueeze_1729 = torch.ops.aten.unsqueeze.default(unsqueeze_1728, -1);  unsqueeze_1728 = None
        unsqueeze_1730 = torch.ops.aten.unsqueeze.default(mul_699, -1);  mul_699 = None
        unsqueeze_1731 = torch.ops.aten.unsqueeze.default(unsqueeze_1730, -1);  unsqueeze_1730 = None
        sub_267 = torch.ops.aten.sub.Tensor(convolution_267, unsqueeze_1729);  convolution_267 = unsqueeze_1729 = None
        mul_700 = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_1731);  sub_267 = unsqueeze_1731 = None
        unsqueeze_1732 = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
        unsqueeze_1733 = torch.ops.aten.unsqueeze.default(unsqueeze_1732, -1);  unsqueeze_1732 = None
        mul_701 = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_1733);  mul_700 = unsqueeze_1733 = None
        unsqueeze_1734 = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_1735 = torch.ops.aten.unsqueeze.default(unsqueeze_1734, -1);  unsqueeze_1734 = None
        add_483 = torch.ops.aten.add.Tensor(mul_701, unsqueeze_1735);  mul_701 = unsqueeze_1735 = None
        add_484 = torch.ops.aten.add.Tensor(add_483, relu_205);  add_483 = relu_205 = None
        relu_209 = torch.ops.aten.relu.default(add_484);  add_484 = None
        convolution_268 = torch.ops.aten.convolution.default(relu_209, arg445_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg445_1 = None
        add_485 = torch.ops.aten.add.Tensor(arg447_1, 1e-05);  arg447_1 = None
        sqrt_217 = torch.ops.aten.sqrt.default(add_485);  add_485 = None
        reciprocal_217 = torch.ops.aten.reciprocal.default(sqrt_217);  sqrt_217 = None
        mul_702 = torch.ops.aten.mul.Tensor(reciprocal_217, 1);  reciprocal_217 = None
        unsqueeze_1736 = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
        unsqueeze_1737 = torch.ops.aten.unsqueeze.default(unsqueeze_1736, -1);  unsqueeze_1736 = None
        unsqueeze_1738 = torch.ops.aten.unsqueeze.default(mul_702, -1);  mul_702 = None
        unsqueeze_1739 = torch.ops.aten.unsqueeze.default(unsqueeze_1738, -1);  unsqueeze_1738 = None
        sub_268 = torch.ops.aten.sub.Tensor(convolution_268, unsqueeze_1737);  convolution_268 = unsqueeze_1737 = None
        mul_703 = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_1739);  sub_268 = unsqueeze_1739 = None
        unsqueeze_1740 = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
        unsqueeze_1741 = torch.ops.aten.unsqueeze.default(unsqueeze_1740, -1);  unsqueeze_1740 = None
        mul_704 = torch.ops.aten.mul.Tensor(mul_703, unsqueeze_1741);  mul_703 = unsqueeze_1741 = None
        unsqueeze_1742 = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_1743 = torch.ops.aten.unsqueeze.default(unsqueeze_1742, -1);  unsqueeze_1742 = None
        add_486 = torch.ops.aten.add.Tensor(mul_704, unsqueeze_1743);  mul_704 = unsqueeze_1743 = None
        relu_210 = torch.ops.aten.relu.default(add_486);  add_486 = None
        convolution_269 = torch.ops.aten.convolution.default(relu_210, arg450_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_210 = arg450_1 = None
        add_487 = torch.ops.aten.add.Tensor(arg452_1, 1e-05);  arg452_1 = None
        sqrt_218 = torch.ops.aten.sqrt.default(add_487);  add_487 = None
        reciprocal_218 = torch.ops.aten.reciprocal.default(sqrt_218);  sqrt_218 = None
        mul_705 = torch.ops.aten.mul.Tensor(reciprocal_218, 1);  reciprocal_218 = None
        unsqueeze_1744 = torch.ops.aten.unsqueeze.default(arg451_1, -1);  arg451_1 = None
        unsqueeze_1745 = torch.ops.aten.unsqueeze.default(unsqueeze_1744, -1);  unsqueeze_1744 = None
        unsqueeze_1746 = torch.ops.aten.unsqueeze.default(mul_705, -1);  mul_705 = None
        unsqueeze_1747 = torch.ops.aten.unsqueeze.default(unsqueeze_1746, -1);  unsqueeze_1746 = None
        sub_269 = torch.ops.aten.sub.Tensor(convolution_269, unsqueeze_1745);  convolution_269 = unsqueeze_1745 = None
        mul_706 = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_1747);  sub_269 = unsqueeze_1747 = None
        unsqueeze_1748 = torch.ops.aten.unsqueeze.default(arg453_1, -1);  arg453_1 = None
        unsqueeze_1749 = torch.ops.aten.unsqueeze.default(unsqueeze_1748, -1);  unsqueeze_1748 = None
        mul_707 = torch.ops.aten.mul.Tensor(mul_706, unsqueeze_1749);  mul_706 = unsqueeze_1749 = None
        unsqueeze_1750 = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_1751 = torch.ops.aten.unsqueeze.default(unsqueeze_1750, -1);  unsqueeze_1750 = None
        add_488 = torch.ops.aten.add.Tensor(mul_707, unsqueeze_1751);  mul_707 = unsqueeze_1751 = None
        relu_211 = torch.ops.aten.relu.default(add_488);  add_488 = None
        view_308 = torch.ops.aten.view.default(relu_211, [8, 2, 256, 16, 16]);  relu_211 = None
        sum_154 = torch.ops.aten.sum.dim_IntList(view_308, [1])
        mean_52 = torch.ops.aten.mean.dim(sum_154, [2, 3], True);  sum_154 = None
        convolution_270 = torch.ops.aten.convolution.default(mean_52, arg455_1, arg456_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_52 = arg455_1 = arg456_1 = None
        add_489 = torch.ops.aten.add.Tensor(arg458_1, 1e-05);  arg458_1 = None
        sqrt_219 = torch.ops.aten.sqrt.default(add_489);  add_489 = None
        reciprocal_219 = torch.ops.aten.reciprocal.default(sqrt_219);  sqrt_219 = None
        mul_708 = torch.ops.aten.mul.Tensor(reciprocal_219, 1);  reciprocal_219 = None
        unsqueeze_1752 = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
        unsqueeze_1753 = torch.ops.aten.unsqueeze.default(unsqueeze_1752, -1);  unsqueeze_1752 = None
        unsqueeze_1754 = torch.ops.aten.unsqueeze.default(mul_708, -1);  mul_708 = None
        unsqueeze_1755 = torch.ops.aten.unsqueeze.default(unsqueeze_1754, -1);  unsqueeze_1754 = None
        sub_270 = torch.ops.aten.sub.Tensor(convolution_270, unsqueeze_1753);  convolution_270 = unsqueeze_1753 = None
        mul_709 = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_1755);  sub_270 = unsqueeze_1755 = None
        unsqueeze_1756 = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_1757 = torch.ops.aten.unsqueeze.default(unsqueeze_1756, -1);  unsqueeze_1756 = None
        mul_710 = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_1757);  mul_709 = unsqueeze_1757 = None
        unsqueeze_1758 = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_1759 = torch.ops.aten.unsqueeze.default(unsqueeze_1758, -1);  unsqueeze_1758 = None
        add_490 = torch.ops.aten.add.Tensor(mul_710, unsqueeze_1759);  mul_710 = unsqueeze_1759 = None
        relu_212 = torch.ops.aten.relu.default(add_490);  add_490 = None
        convolution_271 = torch.ops.aten.convolution.default(relu_212, arg461_1, arg462_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_212 = arg461_1 = arg462_1 = None
        view_309 = torch.ops.aten.view.default(convolution_271, [8, 1, 2, -1]);  convolution_271 = None
        permute_52 = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
        amax_51 = torch.ops.aten.amax.default(permute_52, [1], True)
        sub_271 = torch.ops.aten.sub.Tensor(permute_52, amax_51);  permute_52 = amax_51 = None
        exp_51 = torch.ops.aten.exp.default(sub_271);  sub_271 = None
        sum_155 = torch.ops.aten.sum.dim_IntList(exp_51, [1], True)
        div_51 = torch.ops.aten.div.Tensor(exp_51, sum_155);  exp_51 = sum_155 = None
        view_310 = torch.ops.aten.view.default(div_51, [8, -1]);  div_51 = None
        view_311 = torch.ops.aten.view.default(view_310, [8, -1, 1, 1]);  view_310 = None
        view_312 = torch.ops.aten.view.default(view_311, [8, 2, 256, 1, 1]);  view_311 = None
        mul_711 = torch.ops.aten.mul.Tensor(view_308, view_312);  view_308 = view_312 = None
        sum_156 = torch.ops.aten.sum.dim_IntList(mul_711, [1]);  mul_711 = None
        convolution_272 = torch.ops.aten.convolution.default(sum_156, arg463_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_156 = arg463_1 = None
        add_491 = torch.ops.aten.add.Tensor(arg465_1, 1e-05);  arg465_1 = None
        sqrt_220 = torch.ops.aten.sqrt.default(add_491);  add_491 = None
        reciprocal_220 = torch.ops.aten.reciprocal.default(sqrt_220);  sqrt_220 = None
        mul_712 = torch.ops.aten.mul.Tensor(reciprocal_220, 1);  reciprocal_220 = None
        unsqueeze_1760 = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_1761 = torch.ops.aten.unsqueeze.default(unsqueeze_1760, -1);  unsqueeze_1760 = None
        unsqueeze_1762 = torch.ops.aten.unsqueeze.default(mul_712, -1);  mul_712 = None
        unsqueeze_1763 = torch.ops.aten.unsqueeze.default(unsqueeze_1762, -1);  unsqueeze_1762 = None
        sub_272 = torch.ops.aten.sub.Tensor(convolution_272, unsqueeze_1761);  convolution_272 = unsqueeze_1761 = None
        mul_713 = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_1763);  sub_272 = unsqueeze_1763 = None
        unsqueeze_1764 = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
        unsqueeze_1765 = torch.ops.aten.unsqueeze.default(unsqueeze_1764, -1);  unsqueeze_1764 = None
        mul_714 = torch.ops.aten.mul.Tensor(mul_713, unsqueeze_1765);  mul_713 = unsqueeze_1765 = None
        unsqueeze_1766 = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
        unsqueeze_1767 = torch.ops.aten.unsqueeze.default(unsqueeze_1766, -1);  unsqueeze_1766 = None
        add_492 = torch.ops.aten.add.Tensor(mul_714, unsqueeze_1767);  mul_714 = unsqueeze_1767 = None
        add_493 = torch.ops.aten.add.Tensor(add_492, relu_209);  add_492 = relu_209 = None
        relu_213 = torch.ops.aten.relu.default(add_493);  add_493 = None
        convolution_273 = torch.ops.aten.convolution.default(relu_213, arg468_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg468_1 = None
        add_494 = torch.ops.aten.add.Tensor(arg470_1, 1e-05);  arg470_1 = None
        sqrt_221 = torch.ops.aten.sqrt.default(add_494);  add_494 = None
        reciprocal_221 = torch.ops.aten.reciprocal.default(sqrt_221);  sqrt_221 = None
        mul_715 = torch.ops.aten.mul.Tensor(reciprocal_221, 1);  reciprocal_221 = None
        unsqueeze_1768 = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_1769 = torch.ops.aten.unsqueeze.default(unsqueeze_1768, -1);  unsqueeze_1768 = None
        unsqueeze_1770 = torch.ops.aten.unsqueeze.default(mul_715, -1);  mul_715 = None
        unsqueeze_1771 = torch.ops.aten.unsqueeze.default(unsqueeze_1770, -1);  unsqueeze_1770 = None
        sub_273 = torch.ops.aten.sub.Tensor(convolution_273, unsqueeze_1769);  convolution_273 = unsqueeze_1769 = None
        mul_716 = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_1771);  sub_273 = unsqueeze_1771 = None
        unsqueeze_1772 = torch.ops.aten.unsqueeze.default(arg471_1, -1);  arg471_1 = None
        unsqueeze_1773 = torch.ops.aten.unsqueeze.default(unsqueeze_1772, -1);  unsqueeze_1772 = None
        mul_717 = torch.ops.aten.mul.Tensor(mul_716, unsqueeze_1773);  mul_716 = unsqueeze_1773 = None
        unsqueeze_1774 = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
        unsqueeze_1775 = torch.ops.aten.unsqueeze.default(unsqueeze_1774, -1);  unsqueeze_1774 = None
        add_495 = torch.ops.aten.add.Tensor(mul_717, unsqueeze_1775);  mul_717 = unsqueeze_1775 = None
        relu_214 = torch.ops.aten.relu.default(add_495);  add_495 = None
        convolution_274 = torch.ops.aten.convolution.default(relu_214, arg473_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_214 = arg473_1 = None
        add_496 = torch.ops.aten.add.Tensor(arg475_1, 1e-05);  arg475_1 = None
        sqrt_222 = torch.ops.aten.sqrt.default(add_496);  add_496 = None
        reciprocal_222 = torch.ops.aten.reciprocal.default(sqrt_222);  sqrt_222 = None
        mul_718 = torch.ops.aten.mul.Tensor(reciprocal_222, 1);  reciprocal_222 = None
        unsqueeze_1776 = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_1777 = torch.ops.aten.unsqueeze.default(unsqueeze_1776, -1);  unsqueeze_1776 = None
        unsqueeze_1778 = torch.ops.aten.unsqueeze.default(mul_718, -1);  mul_718 = None
        unsqueeze_1779 = torch.ops.aten.unsqueeze.default(unsqueeze_1778, -1);  unsqueeze_1778 = None
        sub_274 = torch.ops.aten.sub.Tensor(convolution_274, unsqueeze_1777);  convolution_274 = unsqueeze_1777 = None
        mul_719 = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_1779);  sub_274 = unsqueeze_1779 = None
        unsqueeze_1780 = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
        unsqueeze_1781 = torch.ops.aten.unsqueeze.default(unsqueeze_1780, -1);  unsqueeze_1780 = None
        mul_720 = torch.ops.aten.mul.Tensor(mul_719, unsqueeze_1781);  mul_719 = unsqueeze_1781 = None
        unsqueeze_1782 = torch.ops.aten.unsqueeze.default(arg477_1, -1);  arg477_1 = None
        unsqueeze_1783 = torch.ops.aten.unsqueeze.default(unsqueeze_1782, -1);  unsqueeze_1782 = None
        add_497 = torch.ops.aten.add.Tensor(mul_720, unsqueeze_1783);  mul_720 = unsqueeze_1783 = None
        relu_215 = torch.ops.aten.relu.default(add_497);  add_497 = None
        view_314 = torch.ops.aten.view.default(relu_215, [8, 2, 256, 16, 16]);  relu_215 = None
        sum_157 = torch.ops.aten.sum.dim_IntList(view_314, [1])
        mean_53 = torch.ops.aten.mean.dim(sum_157, [2, 3], True);  sum_157 = None
        convolution_275 = torch.ops.aten.convolution.default(mean_53, arg478_1, arg479_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_53 = arg478_1 = arg479_1 = None
        add_498 = torch.ops.aten.add.Tensor(arg481_1, 1e-05);  arg481_1 = None
        sqrt_223 = torch.ops.aten.sqrt.default(add_498);  add_498 = None
        reciprocal_223 = torch.ops.aten.reciprocal.default(sqrt_223);  sqrt_223 = None
        mul_721 = torch.ops.aten.mul.Tensor(reciprocal_223, 1);  reciprocal_223 = None
        unsqueeze_1784 = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_1785 = torch.ops.aten.unsqueeze.default(unsqueeze_1784, -1);  unsqueeze_1784 = None
        unsqueeze_1786 = torch.ops.aten.unsqueeze.default(mul_721, -1);  mul_721 = None
        unsqueeze_1787 = torch.ops.aten.unsqueeze.default(unsqueeze_1786, -1);  unsqueeze_1786 = None
        sub_275 = torch.ops.aten.sub.Tensor(convolution_275, unsqueeze_1785);  convolution_275 = unsqueeze_1785 = None
        mul_722 = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_1787);  sub_275 = unsqueeze_1787 = None
        unsqueeze_1788 = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_1789 = torch.ops.aten.unsqueeze.default(unsqueeze_1788, -1);  unsqueeze_1788 = None
        mul_723 = torch.ops.aten.mul.Tensor(mul_722, unsqueeze_1789);  mul_722 = unsqueeze_1789 = None
        unsqueeze_1790 = torch.ops.aten.unsqueeze.default(arg483_1, -1);  arg483_1 = None
        unsqueeze_1791 = torch.ops.aten.unsqueeze.default(unsqueeze_1790, -1);  unsqueeze_1790 = None
        add_499 = torch.ops.aten.add.Tensor(mul_723, unsqueeze_1791);  mul_723 = unsqueeze_1791 = None
        relu_216 = torch.ops.aten.relu.default(add_499);  add_499 = None
        convolution_276 = torch.ops.aten.convolution.default(relu_216, arg484_1, arg485_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_216 = arg484_1 = arg485_1 = None
        view_315 = torch.ops.aten.view.default(convolution_276, [8, 1, 2, -1]);  convolution_276 = None
        permute_53 = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        amax_52 = torch.ops.aten.amax.default(permute_53, [1], True)
        sub_276 = torch.ops.aten.sub.Tensor(permute_53, amax_52);  permute_53 = amax_52 = None
        exp_52 = torch.ops.aten.exp.default(sub_276);  sub_276 = None
        sum_158 = torch.ops.aten.sum.dim_IntList(exp_52, [1], True)
        div_52 = torch.ops.aten.div.Tensor(exp_52, sum_158);  exp_52 = sum_158 = None
        view_316 = torch.ops.aten.view.default(div_52, [8, -1]);  div_52 = None
        view_317 = torch.ops.aten.view.default(view_316, [8, -1, 1, 1]);  view_316 = None
        view_318 = torch.ops.aten.view.default(view_317, [8, 2, 256, 1, 1]);  view_317 = None
        mul_724 = torch.ops.aten.mul.Tensor(view_314, view_318);  view_314 = view_318 = None
        sum_159 = torch.ops.aten.sum.dim_IntList(mul_724, [1]);  mul_724 = None
        convolution_277 = torch.ops.aten.convolution.default(sum_159, arg486_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_159 = arg486_1 = None
        add_500 = torch.ops.aten.add.Tensor(arg488_1, 1e-05);  arg488_1 = None
        sqrt_224 = torch.ops.aten.sqrt.default(add_500);  add_500 = None
        reciprocal_224 = torch.ops.aten.reciprocal.default(sqrt_224);  sqrt_224 = None
        mul_725 = torch.ops.aten.mul.Tensor(reciprocal_224, 1);  reciprocal_224 = None
        unsqueeze_1792 = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
        unsqueeze_1793 = torch.ops.aten.unsqueeze.default(unsqueeze_1792, -1);  unsqueeze_1792 = None
        unsqueeze_1794 = torch.ops.aten.unsqueeze.default(mul_725, -1);  mul_725 = None
        unsqueeze_1795 = torch.ops.aten.unsqueeze.default(unsqueeze_1794, -1);  unsqueeze_1794 = None
        sub_277 = torch.ops.aten.sub.Tensor(convolution_277, unsqueeze_1793);  convolution_277 = unsqueeze_1793 = None
        mul_726 = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_1795);  sub_277 = unsqueeze_1795 = None
        unsqueeze_1796 = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_1797 = torch.ops.aten.unsqueeze.default(unsqueeze_1796, -1);  unsqueeze_1796 = None
        mul_727 = torch.ops.aten.mul.Tensor(mul_726, unsqueeze_1797);  mul_726 = unsqueeze_1797 = None
        unsqueeze_1798 = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_1799 = torch.ops.aten.unsqueeze.default(unsqueeze_1798, -1);  unsqueeze_1798 = None
        add_501 = torch.ops.aten.add.Tensor(mul_727, unsqueeze_1799);  mul_727 = unsqueeze_1799 = None
        add_502 = torch.ops.aten.add.Tensor(add_501, relu_213);  add_501 = relu_213 = None
        relu_217 = torch.ops.aten.relu.default(add_502);  add_502 = None
        convolution_278 = torch.ops.aten.convolution.default(relu_217, arg491_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg491_1 = None
        add_503 = torch.ops.aten.add.Tensor(arg493_1, 1e-05);  arg493_1 = None
        sqrt_225 = torch.ops.aten.sqrt.default(add_503);  add_503 = None
        reciprocal_225 = torch.ops.aten.reciprocal.default(sqrt_225);  sqrt_225 = None
        mul_728 = torch.ops.aten.mul.Tensor(reciprocal_225, 1);  reciprocal_225 = None
        unsqueeze_1800 = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
        unsqueeze_1801 = torch.ops.aten.unsqueeze.default(unsqueeze_1800, -1);  unsqueeze_1800 = None
        unsqueeze_1802 = torch.ops.aten.unsqueeze.default(mul_728, -1);  mul_728 = None
        unsqueeze_1803 = torch.ops.aten.unsqueeze.default(unsqueeze_1802, -1);  unsqueeze_1802 = None
        sub_278 = torch.ops.aten.sub.Tensor(convolution_278, unsqueeze_1801);  convolution_278 = unsqueeze_1801 = None
        mul_729 = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_1803);  sub_278 = unsqueeze_1803 = None
        unsqueeze_1804 = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_1805 = torch.ops.aten.unsqueeze.default(unsqueeze_1804, -1);  unsqueeze_1804 = None
        mul_730 = torch.ops.aten.mul.Tensor(mul_729, unsqueeze_1805);  mul_729 = unsqueeze_1805 = None
        unsqueeze_1806 = torch.ops.aten.unsqueeze.default(arg495_1, -1);  arg495_1 = None
        unsqueeze_1807 = torch.ops.aten.unsqueeze.default(unsqueeze_1806, -1);  unsqueeze_1806 = None
        add_504 = torch.ops.aten.add.Tensor(mul_730, unsqueeze_1807);  mul_730 = unsqueeze_1807 = None
        relu_218 = torch.ops.aten.relu.default(add_504);  add_504 = None
        convolution_279 = torch.ops.aten.convolution.default(relu_218, arg496_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_218 = arg496_1 = None
        add_505 = torch.ops.aten.add.Tensor(arg498_1, 1e-05);  arg498_1 = None
        sqrt_226 = torch.ops.aten.sqrt.default(add_505);  add_505 = None
        reciprocal_226 = torch.ops.aten.reciprocal.default(sqrt_226);  sqrt_226 = None
        mul_731 = torch.ops.aten.mul.Tensor(reciprocal_226, 1);  reciprocal_226 = None
        unsqueeze_1808 = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
        unsqueeze_1809 = torch.ops.aten.unsqueeze.default(unsqueeze_1808, -1);  unsqueeze_1808 = None
        unsqueeze_1810 = torch.ops.aten.unsqueeze.default(mul_731, -1);  mul_731 = None
        unsqueeze_1811 = torch.ops.aten.unsqueeze.default(unsqueeze_1810, -1);  unsqueeze_1810 = None
        sub_279 = torch.ops.aten.sub.Tensor(convolution_279, unsqueeze_1809);  convolution_279 = unsqueeze_1809 = None
        mul_732 = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_1811);  sub_279 = unsqueeze_1811 = None
        unsqueeze_1812 = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_1813 = torch.ops.aten.unsqueeze.default(unsqueeze_1812, -1);  unsqueeze_1812 = None
        mul_733 = torch.ops.aten.mul.Tensor(mul_732, unsqueeze_1813);  mul_732 = unsqueeze_1813 = None
        unsqueeze_1814 = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
        unsqueeze_1815 = torch.ops.aten.unsqueeze.default(unsqueeze_1814, -1);  unsqueeze_1814 = None
        add_506 = torch.ops.aten.add.Tensor(mul_733, unsqueeze_1815);  mul_733 = unsqueeze_1815 = None
        relu_219 = torch.ops.aten.relu.default(add_506);  add_506 = None
        view_320 = torch.ops.aten.view.default(relu_219, [8, 2, 256, 16, 16]);  relu_219 = None
        sum_160 = torch.ops.aten.sum.dim_IntList(view_320, [1])
        mean_54 = torch.ops.aten.mean.dim(sum_160, [2, 3], True);  sum_160 = None
        convolution_280 = torch.ops.aten.convolution.default(mean_54, arg501_1, arg502_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_54 = arg501_1 = arg502_1 = None
        add_507 = torch.ops.aten.add.Tensor(arg504_1, 1e-05);  arg504_1 = None
        sqrt_227 = torch.ops.aten.sqrt.default(add_507);  add_507 = None
        reciprocal_227 = torch.ops.aten.reciprocal.default(sqrt_227);  sqrt_227 = None
        mul_734 = torch.ops.aten.mul.Tensor(reciprocal_227, 1);  reciprocal_227 = None
        unsqueeze_1816 = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
        unsqueeze_1817 = torch.ops.aten.unsqueeze.default(unsqueeze_1816, -1);  unsqueeze_1816 = None
        unsqueeze_1818 = torch.ops.aten.unsqueeze.default(mul_734, -1);  mul_734 = None
        unsqueeze_1819 = torch.ops.aten.unsqueeze.default(unsqueeze_1818, -1);  unsqueeze_1818 = None
        sub_280 = torch.ops.aten.sub.Tensor(convolution_280, unsqueeze_1817);  convolution_280 = unsqueeze_1817 = None
        mul_735 = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_1819);  sub_280 = unsqueeze_1819 = None
        unsqueeze_1820 = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
        unsqueeze_1821 = torch.ops.aten.unsqueeze.default(unsqueeze_1820, -1);  unsqueeze_1820 = None
        mul_736 = torch.ops.aten.mul.Tensor(mul_735, unsqueeze_1821);  mul_735 = unsqueeze_1821 = None
        unsqueeze_1822 = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
        unsqueeze_1823 = torch.ops.aten.unsqueeze.default(unsqueeze_1822, -1);  unsqueeze_1822 = None
        add_508 = torch.ops.aten.add.Tensor(mul_736, unsqueeze_1823);  mul_736 = unsqueeze_1823 = None
        relu_220 = torch.ops.aten.relu.default(add_508);  add_508 = None
        convolution_281 = torch.ops.aten.convolution.default(relu_220, arg507_1, arg508_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_220 = arg507_1 = arg508_1 = None
        view_321 = torch.ops.aten.view.default(convolution_281, [8, 1, 2, -1]);  convolution_281 = None
        permute_54 = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
        amax_53 = torch.ops.aten.amax.default(permute_54, [1], True)
        sub_281 = torch.ops.aten.sub.Tensor(permute_54, amax_53);  permute_54 = amax_53 = None
        exp_53 = torch.ops.aten.exp.default(sub_281);  sub_281 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(exp_53, [1], True)
        div_53 = torch.ops.aten.div.Tensor(exp_53, sum_161);  exp_53 = sum_161 = None
        view_322 = torch.ops.aten.view.default(div_53, [8, -1]);  div_53 = None
        view_323 = torch.ops.aten.view.default(view_322, [8, -1, 1, 1]);  view_322 = None
        view_324 = torch.ops.aten.view.default(view_323, [8, 2, 256, 1, 1]);  view_323 = None
        mul_737 = torch.ops.aten.mul.Tensor(view_320, view_324);  view_320 = view_324 = None
        sum_162 = torch.ops.aten.sum.dim_IntList(mul_737, [1]);  mul_737 = None
        convolution_282 = torch.ops.aten.convolution.default(sum_162, arg509_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_162 = arg509_1 = None
        add_509 = torch.ops.aten.add.Tensor(arg511_1, 1e-05);  arg511_1 = None
        sqrt_228 = torch.ops.aten.sqrt.default(add_509);  add_509 = None
        reciprocal_228 = torch.ops.aten.reciprocal.default(sqrt_228);  sqrt_228 = None
        mul_738 = torch.ops.aten.mul.Tensor(reciprocal_228, 1);  reciprocal_228 = None
        unsqueeze_1824 = torch.ops.aten.unsqueeze.default(arg510_1, -1);  arg510_1 = None
        unsqueeze_1825 = torch.ops.aten.unsqueeze.default(unsqueeze_1824, -1);  unsqueeze_1824 = None
        unsqueeze_1826 = torch.ops.aten.unsqueeze.default(mul_738, -1);  mul_738 = None
        unsqueeze_1827 = torch.ops.aten.unsqueeze.default(unsqueeze_1826, -1);  unsqueeze_1826 = None
        sub_282 = torch.ops.aten.sub.Tensor(convolution_282, unsqueeze_1825);  convolution_282 = unsqueeze_1825 = None
        mul_739 = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_1827);  sub_282 = unsqueeze_1827 = None
        unsqueeze_1828 = torch.ops.aten.unsqueeze.default(arg512_1, -1);  arg512_1 = None
        unsqueeze_1829 = torch.ops.aten.unsqueeze.default(unsqueeze_1828, -1);  unsqueeze_1828 = None
        mul_740 = torch.ops.aten.mul.Tensor(mul_739, unsqueeze_1829);  mul_739 = unsqueeze_1829 = None
        unsqueeze_1830 = torch.ops.aten.unsqueeze.default(arg513_1, -1);  arg513_1 = None
        unsqueeze_1831 = torch.ops.aten.unsqueeze.default(unsqueeze_1830, -1);  unsqueeze_1830 = None
        add_510 = torch.ops.aten.add.Tensor(mul_740, unsqueeze_1831);  mul_740 = unsqueeze_1831 = None
        add_511 = torch.ops.aten.add.Tensor(add_510, relu_217);  add_510 = relu_217 = None
        relu_221 = torch.ops.aten.relu.default(add_511);  add_511 = None
        convolution_283 = torch.ops.aten.convolution.default(relu_221, arg514_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg514_1 = None
        add_512 = torch.ops.aten.add.Tensor(arg516_1, 1e-05);  arg516_1 = None
        sqrt_229 = torch.ops.aten.sqrt.default(add_512);  add_512 = None
        reciprocal_229 = torch.ops.aten.reciprocal.default(sqrt_229);  sqrt_229 = None
        mul_741 = torch.ops.aten.mul.Tensor(reciprocal_229, 1);  reciprocal_229 = None
        unsqueeze_1832 = torch.ops.aten.unsqueeze.default(arg515_1, -1);  arg515_1 = None
        unsqueeze_1833 = torch.ops.aten.unsqueeze.default(unsqueeze_1832, -1);  unsqueeze_1832 = None
        unsqueeze_1834 = torch.ops.aten.unsqueeze.default(mul_741, -1);  mul_741 = None
        unsqueeze_1835 = torch.ops.aten.unsqueeze.default(unsqueeze_1834, -1);  unsqueeze_1834 = None
        sub_283 = torch.ops.aten.sub.Tensor(convolution_283, unsqueeze_1833);  convolution_283 = unsqueeze_1833 = None
        mul_742 = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_1835);  sub_283 = unsqueeze_1835 = None
        unsqueeze_1836 = torch.ops.aten.unsqueeze.default(arg517_1, -1);  arg517_1 = None
        unsqueeze_1837 = torch.ops.aten.unsqueeze.default(unsqueeze_1836, -1);  unsqueeze_1836 = None
        mul_743 = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_1837);  mul_742 = unsqueeze_1837 = None
        unsqueeze_1838 = torch.ops.aten.unsqueeze.default(arg518_1, -1);  arg518_1 = None
        unsqueeze_1839 = torch.ops.aten.unsqueeze.default(unsqueeze_1838, -1);  unsqueeze_1838 = None
        add_513 = torch.ops.aten.add.Tensor(mul_743, unsqueeze_1839);  mul_743 = unsqueeze_1839 = None
        relu_222 = torch.ops.aten.relu.default(add_513);  add_513 = None
        convolution_284 = torch.ops.aten.convolution.default(relu_222, arg519_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_222 = arg519_1 = None
        add_514 = torch.ops.aten.add.Tensor(arg521_1, 1e-05);  arg521_1 = None
        sqrt_230 = torch.ops.aten.sqrt.default(add_514);  add_514 = None
        reciprocal_230 = torch.ops.aten.reciprocal.default(sqrt_230);  sqrt_230 = None
        mul_744 = torch.ops.aten.mul.Tensor(reciprocal_230, 1);  reciprocal_230 = None
        unsqueeze_1840 = torch.ops.aten.unsqueeze.default(arg520_1, -1);  arg520_1 = None
        unsqueeze_1841 = torch.ops.aten.unsqueeze.default(unsqueeze_1840, -1);  unsqueeze_1840 = None
        unsqueeze_1842 = torch.ops.aten.unsqueeze.default(mul_744, -1);  mul_744 = None
        unsqueeze_1843 = torch.ops.aten.unsqueeze.default(unsqueeze_1842, -1);  unsqueeze_1842 = None
        sub_284 = torch.ops.aten.sub.Tensor(convolution_284, unsqueeze_1841);  convolution_284 = unsqueeze_1841 = None
        mul_745 = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_1843);  sub_284 = unsqueeze_1843 = None
        unsqueeze_1844 = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
        unsqueeze_1845 = torch.ops.aten.unsqueeze.default(unsqueeze_1844, -1);  unsqueeze_1844 = None
        mul_746 = torch.ops.aten.mul.Tensor(mul_745, unsqueeze_1845);  mul_745 = unsqueeze_1845 = None
        unsqueeze_1846 = torch.ops.aten.unsqueeze.default(arg523_1, -1);  arg523_1 = None
        unsqueeze_1847 = torch.ops.aten.unsqueeze.default(unsqueeze_1846, -1);  unsqueeze_1846 = None
        add_515 = torch.ops.aten.add.Tensor(mul_746, unsqueeze_1847);  mul_746 = unsqueeze_1847 = None
        relu_223 = torch.ops.aten.relu.default(add_515);  add_515 = None
        view_326 = torch.ops.aten.view.default(relu_223, [8, 2, 256, 16, 16]);  relu_223 = None
        sum_163 = torch.ops.aten.sum.dim_IntList(view_326, [1])
        mean_55 = torch.ops.aten.mean.dim(sum_163, [2, 3], True);  sum_163 = None
        convolution_285 = torch.ops.aten.convolution.default(mean_55, arg524_1, arg525_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_55 = arg524_1 = arg525_1 = None
        add_516 = torch.ops.aten.add.Tensor(arg527_1, 1e-05);  arg527_1 = None
        sqrt_231 = torch.ops.aten.sqrt.default(add_516);  add_516 = None
        reciprocal_231 = torch.ops.aten.reciprocal.default(sqrt_231);  sqrt_231 = None
        mul_747 = torch.ops.aten.mul.Tensor(reciprocal_231, 1);  reciprocal_231 = None
        unsqueeze_1848 = torch.ops.aten.unsqueeze.default(arg526_1, -1);  arg526_1 = None
        unsqueeze_1849 = torch.ops.aten.unsqueeze.default(unsqueeze_1848, -1);  unsqueeze_1848 = None
        unsqueeze_1850 = torch.ops.aten.unsqueeze.default(mul_747, -1);  mul_747 = None
        unsqueeze_1851 = torch.ops.aten.unsqueeze.default(unsqueeze_1850, -1);  unsqueeze_1850 = None
        sub_285 = torch.ops.aten.sub.Tensor(convolution_285, unsqueeze_1849);  convolution_285 = unsqueeze_1849 = None
        mul_748 = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_1851);  sub_285 = unsqueeze_1851 = None
        unsqueeze_1852 = torch.ops.aten.unsqueeze.default(arg528_1, -1);  arg528_1 = None
        unsqueeze_1853 = torch.ops.aten.unsqueeze.default(unsqueeze_1852, -1);  unsqueeze_1852 = None
        mul_749 = torch.ops.aten.mul.Tensor(mul_748, unsqueeze_1853);  mul_748 = unsqueeze_1853 = None
        unsqueeze_1854 = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_1855 = torch.ops.aten.unsqueeze.default(unsqueeze_1854, -1);  unsqueeze_1854 = None
        add_517 = torch.ops.aten.add.Tensor(mul_749, unsqueeze_1855);  mul_749 = unsqueeze_1855 = None
        relu_224 = torch.ops.aten.relu.default(add_517);  add_517 = None
        convolution_286 = torch.ops.aten.convolution.default(relu_224, arg530_1, arg531_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_224 = arg530_1 = arg531_1 = None
        view_327 = torch.ops.aten.view.default(convolution_286, [8, 1, 2, -1]);  convolution_286 = None
        permute_55 = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        amax_54 = torch.ops.aten.amax.default(permute_55, [1], True)
        sub_286 = torch.ops.aten.sub.Tensor(permute_55, amax_54);  permute_55 = amax_54 = None
        exp_54 = torch.ops.aten.exp.default(sub_286);  sub_286 = None
        sum_164 = torch.ops.aten.sum.dim_IntList(exp_54, [1], True)
        div_54 = torch.ops.aten.div.Tensor(exp_54, sum_164);  exp_54 = sum_164 = None
        view_328 = torch.ops.aten.view.default(div_54, [8, -1]);  div_54 = None
        view_329 = torch.ops.aten.view.default(view_328, [8, -1, 1, 1]);  view_328 = None
        view_330 = torch.ops.aten.view.default(view_329, [8, 2, 256, 1, 1]);  view_329 = None
        mul_750 = torch.ops.aten.mul.Tensor(view_326, view_330);  view_326 = view_330 = None
        sum_165 = torch.ops.aten.sum.dim_IntList(mul_750, [1]);  mul_750 = None
        convolution_287 = torch.ops.aten.convolution.default(sum_165, arg532_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_165 = arg532_1 = None
        add_518 = torch.ops.aten.add.Tensor(arg534_1, 1e-05);  arg534_1 = None
        sqrt_232 = torch.ops.aten.sqrt.default(add_518);  add_518 = None
        reciprocal_232 = torch.ops.aten.reciprocal.default(sqrt_232);  sqrt_232 = None
        mul_751 = torch.ops.aten.mul.Tensor(reciprocal_232, 1);  reciprocal_232 = None
        unsqueeze_1856 = torch.ops.aten.unsqueeze.default(arg533_1, -1);  arg533_1 = None
        unsqueeze_1857 = torch.ops.aten.unsqueeze.default(unsqueeze_1856, -1);  unsqueeze_1856 = None
        unsqueeze_1858 = torch.ops.aten.unsqueeze.default(mul_751, -1);  mul_751 = None
        unsqueeze_1859 = torch.ops.aten.unsqueeze.default(unsqueeze_1858, -1);  unsqueeze_1858 = None
        sub_287 = torch.ops.aten.sub.Tensor(convolution_287, unsqueeze_1857);  convolution_287 = unsqueeze_1857 = None
        mul_752 = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_1859);  sub_287 = unsqueeze_1859 = None
        unsqueeze_1860 = torch.ops.aten.unsqueeze.default(arg535_1, -1);  arg535_1 = None
        unsqueeze_1861 = torch.ops.aten.unsqueeze.default(unsqueeze_1860, -1);  unsqueeze_1860 = None
        mul_753 = torch.ops.aten.mul.Tensor(mul_752, unsqueeze_1861);  mul_752 = unsqueeze_1861 = None
        unsqueeze_1862 = torch.ops.aten.unsqueeze.default(arg536_1, -1);  arg536_1 = None
        unsqueeze_1863 = torch.ops.aten.unsqueeze.default(unsqueeze_1862, -1);  unsqueeze_1862 = None
        add_519 = torch.ops.aten.add.Tensor(mul_753, unsqueeze_1863);  mul_753 = unsqueeze_1863 = None
        add_520 = torch.ops.aten.add.Tensor(add_519, relu_221);  add_519 = relu_221 = None
        relu_225 = torch.ops.aten.relu.default(add_520);  add_520 = None
        convolution_288 = torch.ops.aten.convolution.default(relu_225, arg537_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg537_1 = None
        add_521 = torch.ops.aten.add.Tensor(arg539_1, 1e-05);  arg539_1 = None
        sqrt_233 = torch.ops.aten.sqrt.default(add_521);  add_521 = None
        reciprocal_233 = torch.ops.aten.reciprocal.default(sqrt_233);  sqrt_233 = None
        mul_754 = torch.ops.aten.mul.Tensor(reciprocal_233, 1);  reciprocal_233 = None
        unsqueeze_1864 = torch.ops.aten.unsqueeze.default(arg538_1, -1);  arg538_1 = None
        unsqueeze_1865 = torch.ops.aten.unsqueeze.default(unsqueeze_1864, -1);  unsqueeze_1864 = None
        unsqueeze_1866 = torch.ops.aten.unsqueeze.default(mul_754, -1);  mul_754 = None
        unsqueeze_1867 = torch.ops.aten.unsqueeze.default(unsqueeze_1866, -1);  unsqueeze_1866 = None
        sub_288 = torch.ops.aten.sub.Tensor(convolution_288, unsqueeze_1865);  convolution_288 = unsqueeze_1865 = None
        mul_755 = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_1867);  sub_288 = unsqueeze_1867 = None
        unsqueeze_1868 = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
        unsqueeze_1869 = torch.ops.aten.unsqueeze.default(unsqueeze_1868, -1);  unsqueeze_1868 = None
        mul_756 = torch.ops.aten.mul.Tensor(mul_755, unsqueeze_1869);  mul_755 = unsqueeze_1869 = None
        unsqueeze_1870 = torch.ops.aten.unsqueeze.default(arg541_1, -1);  arg541_1 = None
        unsqueeze_1871 = torch.ops.aten.unsqueeze.default(unsqueeze_1870, -1);  unsqueeze_1870 = None
        add_522 = torch.ops.aten.add.Tensor(mul_756, unsqueeze_1871);  mul_756 = unsqueeze_1871 = None
        relu_226 = torch.ops.aten.relu.default(add_522);  add_522 = None
        convolution_289 = torch.ops.aten.convolution.default(relu_226, arg542_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_226 = arg542_1 = None
        add_523 = torch.ops.aten.add.Tensor(arg544_1, 1e-05);  arg544_1 = None
        sqrt_234 = torch.ops.aten.sqrt.default(add_523);  add_523 = None
        reciprocal_234 = torch.ops.aten.reciprocal.default(sqrt_234);  sqrt_234 = None
        mul_757 = torch.ops.aten.mul.Tensor(reciprocal_234, 1);  reciprocal_234 = None
        unsqueeze_1872 = torch.ops.aten.unsqueeze.default(arg543_1, -1);  arg543_1 = None
        unsqueeze_1873 = torch.ops.aten.unsqueeze.default(unsqueeze_1872, -1);  unsqueeze_1872 = None
        unsqueeze_1874 = torch.ops.aten.unsqueeze.default(mul_757, -1);  mul_757 = None
        unsqueeze_1875 = torch.ops.aten.unsqueeze.default(unsqueeze_1874, -1);  unsqueeze_1874 = None
        sub_289 = torch.ops.aten.sub.Tensor(convolution_289, unsqueeze_1873);  convolution_289 = unsqueeze_1873 = None
        mul_758 = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_1875);  sub_289 = unsqueeze_1875 = None
        unsqueeze_1876 = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
        unsqueeze_1877 = torch.ops.aten.unsqueeze.default(unsqueeze_1876, -1);  unsqueeze_1876 = None
        mul_759 = torch.ops.aten.mul.Tensor(mul_758, unsqueeze_1877);  mul_758 = unsqueeze_1877 = None
        unsqueeze_1878 = torch.ops.aten.unsqueeze.default(arg546_1, -1);  arg546_1 = None
        unsqueeze_1879 = torch.ops.aten.unsqueeze.default(unsqueeze_1878, -1);  unsqueeze_1878 = None
        add_524 = torch.ops.aten.add.Tensor(mul_759, unsqueeze_1879);  mul_759 = unsqueeze_1879 = None
        relu_227 = torch.ops.aten.relu.default(add_524);  add_524 = None
        view_332 = torch.ops.aten.view.default(relu_227, [8, 2, 256, 16, 16]);  relu_227 = None
        sum_166 = torch.ops.aten.sum.dim_IntList(view_332, [1])
        mean_56 = torch.ops.aten.mean.dim(sum_166, [2, 3], True);  sum_166 = None
        convolution_290 = torch.ops.aten.convolution.default(mean_56, arg547_1, arg548_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_56 = arg547_1 = arg548_1 = None
        add_525 = torch.ops.aten.add.Tensor(arg550_1, 1e-05);  arg550_1 = None
        sqrt_235 = torch.ops.aten.sqrt.default(add_525);  add_525 = None
        reciprocal_235 = torch.ops.aten.reciprocal.default(sqrt_235);  sqrt_235 = None
        mul_760 = torch.ops.aten.mul.Tensor(reciprocal_235, 1);  reciprocal_235 = None
        unsqueeze_1880 = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_1881 = torch.ops.aten.unsqueeze.default(unsqueeze_1880, -1);  unsqueeze_1880 = None
        unsqueeze_1882 = torch.ops.aten.unsqueeze.default(mul_760, -1);  mul_760 = None
        unsqueeze_1883 = torch.ops.aten.unsqueeze.default(unsqueeze_1882, -1);  unsqueeze_1882 = None
        sub_290 = torch.ops.aten.sub.Tensor(convolution_290, unsqueeze_1881);  convolution_290 = unsqueeze_1881 = None
        mul_761 = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_1883);  sub_290 = unsqueeze_1883 = None
        unsqueeze_1884 = torch.ops.aten.unsqueeze.default(arg551_1, -1);  arg551_1 = None
        unsqueeze_1885 = torch.ops.aten.unsqueeze.default(unsqueeze_1884, -1);  unsqueeze_1884 = None
        mul_762 = torch.ops.aten.mul.Tensor(mul_761, unsqueeze_1885);  mul_761 = unsqueeze_1885 = None
        unsqueeze_1886 = torch.ops.aten.unsqueeze.default(arg552_1, -1);  arg552_1 = None
        unsqueeze_1887 = torch.ops.aten.unsqueeze.default(unsqueeze_1886, -1);  unsqueeze_1886 = None
        add_526 = torch.ops.aten.add.Tensor(mul_762, unsqueeze_1887);  mul_762 = unsqueeze_1887 = None
        relu_228 = torch.ops.aten.relu.default(add_526);  add_526 = None
        convolution_291 = torch.ops.aten.convolution.default(relu_228, arg553_1, arg554_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_228 = arg553_1 = arg554_1 = None
        view_333 = torch.ops.aten.view.default(convolution_291, [8, 1, 2, -1]);  convolution_291 = None
        permute_56 = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
        amax_55 = torch.ops.aten.amax.default(permute_56, [1], True)
        sub_291 = torch.ops.aten.sub.Tensor(permute_56, amax_55);  permute_56 = amax_55 = None
        exp_55 = torch.ops.aten.exp.default(sub_291);  sub_291 = None
        sum_167 = torch.ops.aten.sum.dim_IntList(exp_55, [1], True)
        div_55 = torch.ops.aten.div.Tensor(exp_55, sum_167);  exp_55 = sum_167 = None
        view_334 = torch.ops.aten.view.default(div_55, [8, -1]);  div_55 = None
        view_335 = torch.ops.aten.view.default(view_334, [8, -1, 1, 1]);  view_334 = None
        view_336 = torch.ops.aten.view.default(view_335, [8, 2, 256, 1, 1]);  view_335 = None
        mul_763 = torch.ops.aten.mul.Tensor(view_332, view_336);  view_332 = view_336 = None
        sum_168 = torch.ops.aten.sum.dim_IntList(mul_763, [1]);  mul_763 = None
        convolution_292 = torch.ops.aten.convolution.default(sum_168, arg555_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_168 = arg555_1 = None
        add_527 = torch.ops.aten.add.Tensor(arg557_1, 1e-05);  arg557_1 = None
        sqrt_236 = torch.ops.aten.sqrt.default(add_527);  add_527 = None
        reciprocal_236 = torch.ops.aten.reciprocal.default(sqrt_236);  sqrt_236 = None
        mul_764 = torch.ops.aten.mul.Tensor(reciprocal_236, 1);  reciprocal_236 = None
        unsqueeze_1888 = torch.ops.aten.unsqueeze.default(arg556_1, -1);  arg556_1 = None
        unsqueeze_1889 = torch.ops.aten.unsqueeze.default(unsqueeze_1888, -1);  unsqueeze_1888 = None
        unsqueeze_1890 = torch.ops.aten.unsqueeze.default(mul_764, -1);  mul_764 = None
        unsqueeze_1891 = torch.ops.aten.unsqueeze.default(unsqueeze_1890, -1);  unsqueeze_1890 = None
        sub_292 = torch.ops.aten.sub.Tensor(convolution_292, unsqueeze_1889);  convolution_292 = unsqueeze_1889 = None
        mul_765 = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_1891);  sub_292 = unsqueeze_1891 = None
        unsqueeze_1892 = torch.ops.aten.unsqueeze.default(arg558_1, -1);  arg558_1 = None
        unsqueeze_1893 = torch.ops.aten.unsqueeze.default(unsqueeze_1892, -1);  unsqueeze_1892 = None
        mul_766 = torch.ops.aten.mul.Tensor(mul_765, unsqueeze_1893);  mul_765 = unsqueeze_1893 = None
        unsqueeze_1894 = torch.ops.aten.unsqueeze.default(arg559_1, -1);  arg559_1 = None
        unsqueeze_1895 = torch.ops.aten.unsqueeze.default(unsqueeze_1894, -1);  unsqueeze_1894 = None
        add_528 = torch.ops.aten.add.Tensor(mul_766, unsqueeze_1895);  mul_766 = unsqueeze_1895 = None
        add_529 = torch.ops.aten.add.Tensor(add_528, relu_225);  add_528 = relu_225 = None
        relu_229 = torch.ops.aten.relu.default(add_529);  add_529 = None
        convolution_293 = torch.ops.aten.convolution.default(relu_229, arg560_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg560_1 = None
        add_530 = torch.ops.aten.add.Tensor(arg562_1, 1e-05);  arg562_1 = None
        sqrt_237 = torch.ops.aten.sqrt.default(add_530);  add_530 = None
        reciprocal_237 = torch.ops.aten.reciprocal.default(sqrt_237);  sqrt_237 = None
        mul_767 = torch.ops.aten.mul.Tensor(reciprocal_237, 1);  reciprocal_237 = None
        unsqueeze_1896 = torch.ops.aten.unsqueeze.default(arg561_1, -1);  arg561_1 = None
        unsqueeze_1897 = torch.ops.aten.unsqueeze.default(unsqueeze_1896, -1);  unsqueeze_1896 = None
        unsqueeze_1898 = torch.ops.aten.unsqueeze.default(mul_767, -1);  mul_767 = None
        unsqueeze_1899 = torch.ops.aten.unsqueeze.default(unsqueeze_1898, -1);  unsqueeze_1898 = None
        sub_293 = torch.ops.aten.sub.Tensor(convolution_293, unsqueeze_1897);  convolution_293 = unsqueeze_1897 = None
        mul_768 = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_1899);  sub_293 = unsqueeze_1899 = None
        unsqueeze_1900 = torch.ops.aten.unsqueeze.default(arg563_1, -1);  arg563_1 = None
        unsqueeze_1901 = torch.ops.aten.unsqueeze.default(unsqueeze_1900, -1);  unsqueeze_1900 = None
        mul_769 = torch.ops.aten.mul.Tensor(mul_768, unsqueeze_1901);  mul_768 = unsqueeze_1901 = None
        unsqueeze_1902 = torch.ops.aten.unsqueeze.default(arg564_1, -1);  arg564_1 = None
        unsqueeze_1903 = torch.ops.aten.unsqueeze.default(unsqueeze_1902, -1);  unsqueeze_1902 = None
        add_531 = torch.ops.aten.add.Tensor(mul_769, unsqueeze_1903);  mul_769 = unsqueeze_1903 = None
        relu_230 = torch.ops.aten.relu.default(add_531);  add_531 = None
        convolution_294 = torch.ops.aten.convolution.default(relu_230, arg565_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_230 = arg565_1 = None
        add_532 = torch.ops.aten.add.Tensor(arg567_1, 1e-05);  arg567_1 = None
        sqrt_238 = torch.ops.aten.sqrt.default(add_532);  add_532 = None
        reciprocal_238 = torch.ops.aten.reciprocal.default(sqrt_238);  sqrt_238 = None
        mul_770 = torch.ops.aten.mul.Tensor(reciprocal_238, 1);  reciprocal_238 = None
        unsqueeze_1904 = torch.ops.aten.unsqueeze.default(arg566_1, -1);  arg566_1 = None
        unsqueeze_1905 = torch.ops.aten.unsqueeze.default(unsqueeze_1904, -1);  unsqueeze_1904 = None
        unsqueeze_1906 = torch.ops.aten.unsqueeze.default(mul_770, -1);  mul_770 = None
        unsqueeze_1907 = torch.ops.aten.unsqueeze.default(unsqueeze_1906, -1);  unsqueeze_1906 = None
        sub_294 = torch.ops.aten.sub.Tensor(convolution_294, unsqueeze_1905);  convolution_294 = unsqueeze_1905 = None
        mul_771 = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_1907);  sub_294 = unsqueeze_1907 = None
        unsqueeze_1908 = torch.ops.aten.unsqueeze.default(arg568_1, -1);  arg568_1 = None
        unsqueeze_1909 = torch.ops.aten.unsqueeze.default(unsqueeze_1908, -1);  unsqueeze_1908 = None
        mul_772 = torch.ops.aten.mul.Tensor(mul_771, unsqueeze_1909);  mul_771 = unsqueeze_1909 = None
        unsqueeze_1910 = torch.ops.aten.unsqueeze.default(arg569_1, -1);  arg569_1 = None
        unsqueeze_1911 = torch.ops.aten.unsqueeze.default(unsqueeze_1910, -1);  unsqueeze_1910 = None
        add_533 = torch.ops.aten.add.Tensor(mul_772, unsqueeze_1911);  mul_772 = unsqueeze_1911 = None
        relu_231 = torch.ops.aten.relu.default(add_533);  add_533 = None
        view_338 = torch.ops.aten.view.default(relu_231, [8, 2, 256, 16, 16]);  relu_231 = None
        sum_169 = torch.ops.aten.sum.dim_IntList(view_338, [1])
        mean_57 = torch.ops.aten.mean.dim(sum_169, [2, 3], True);  sum_169 = None
        convolution_295 = torch.ops.aten.convolution.default(mean_57, arg570_1, arg571_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_57 = arg570_1 = arg571_1 = None
        add_534 = torch.ops.aten.add.Tensor(arg573_1, 1e-05);  arg573_1 = None
        sqrt_239 = torch.ops.aten.sqrt.default(add_534);  add_534 = None
        reciprocal_239 = torch.ops.aten.reciprocal.default(sqrt_239);  sqrt_239 = None
        mul_773 = torch.ops.aten.mul.Tensor(reciprocal_239, 1);  reciprocal_239 = None
        unsqueeze_1912 = torch.ops.aten.unsqueeze.default(arg572_1, -1);  arg572_1 = None
        unsqueeze_1913 = torch.ops.aten.unsqueeze.default(unsqueeze_1912, -1);  unsqueeze_1912 = None
        unsqueeze_1914 = torch.ops.aten.unsqueeze.default(mul_773, -1);  mul_773 = None
        unsqueeze_1915 = torch.ops.aten.unsqueeze.default(unsqueeze_1914, -1);  unsqueeze_1914 = None
        sub_295 = torch.ops.aten.sub.Tensor(convolution_295, unsqueeze_1913);  convolution_295 = unsqueeze_1913 = None
        mul_774 = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_1915);  sub_295 = unsqueeze_1915 = None
        unsqueeze_1916 = torch.ops.aten.unsqueeze.default(arg574_1, -1);  arg574_1 = None
        unsqueeze_1917 = torch.ops.aten.unsqueeze.default(unsqueeze_1916, -1);  unsqueeze_1916 = None
        mul_775 = torch.ops.aten.mul.Tensor(mul_774, unsqueeze_1917);  mul_774 = unsqueeze_1917 = None
        unsqueeze_1918 = torch.ops.aten.unsqueeze.default(arg575_1, -1);  arg575_1 = None
        unsqueeze_1919 = torch.ops.aten.unsqueeze.default(unsqueeze_1918, -1);  unsqueeze_1918 = None
        add_535 = torch.ops.aten.add.Tensor(mul_775, unsqueeze_1919);  mul_775 = unsqueeze_1919 = None
        relu_232 = torch.ops.aten.relu.default(add_535);  add_535 = None
        convolution_296 = torch.ops.aten.convolution.default(relu_232, arg576_1, arg577_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_232 = arg576_1 = arg577_1 = None
        view_339 = torch.ops.aten.view.default(convolution_296, [8, 1, 2, -1]);  convolution_296 = None
        permute_57 = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        amax_56 = torch.ops.aten.amax.default(permute_57, [1], True)
        sub_296 = torch.ops.aten.sub.Tensor(permute_57, amax_56);  permute_57 = amax_56 = None
        exp_56 = torch.ops.aten.exp.default(sub_296);  sub_296 = None
        sum_170 = torch.ops.aten.sum.dim_IntList(exp_56, [1], True)
        div_56 = torch.ops.aten.div.Tensor(exp_56, sum_170);  exp_56 = sum_170 = None
        view_340 = torch.ops.aten.view.default(div_56, [8, -1]);  div_56 = None
        view_341 = torch.ops.aten.view.default(view_340, [8, -1, 1, 1]);  view_340 = None
        view_342 = torch.ops.aten.view.default(view_341, [8, 2, 256, 1, 1]);  view_341 = None
        mul_776 = torch.ops.aten.mul.Tensor(view_338, view_342);  view_338 = view_342 = None
        sum_171 = torch.ops.aten.sum.dim_IntList(mul_776, [1]);  mul_776 = None
        convolution_297 = torch.ops.aten.convolution.default(sum_171, arg578_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_171 = arg578_1 = None
        add_536 = torch.ops.aten.add.Tensor(arg580_1, 1e-05);  arg580_1 = None
        sqrt_240 = torch.ops.aten.sqrt.default(add_536);  add_536 = None
        reciprocal_240 = torch.ops.aten.reciprocal.default(sqrt_240);  sqrt_240 = None
        mul_777 = torch.ops.aten.mul.Tensor(reciprocal_240, 1);  reciprocal_240 = None
        unsqueeze_1920 = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_1921 = torch.ops.aten.unsqueeze.default(unsqueeze_1920, -1);  unsqueeze_1920 = None
        unsqueeze_1922 = torch.ops.aten.unsqueeze.default(mul_777, -1);  mul_777 = None
        unsqueeze_1923 = torch.ops.aten.unsqueeze.default(unsqueeze_1922, -1);  unsqueeze_1922 = None
        sub_297 = torch.ops.aten.sub.Tensor(convolution_297, unsqueeze_1921);  convolution_297 = unsqueeze_1921 = None
        mul_778 = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1923);  sub_297 = unsqueeze_1923 = None
        unsqueeze_1924 = torch.ops.aten.unsqueeze.default(arg581_1, -1);  arg581_1 = None
        unsqueeze_1925 = torch.ops.aten.unsqueeze.default(unsqueeze_1924, -1);  unsqueeze_1924 = None
        mul_779 = torch.ops.aten.mul.Tensor(mul_778, unsqueeze_1925);  mul_778 = unsqueeze_1925 = None
        unsqueeze_1926 = torch.ops.aten.unsqueeze.default(arg582_1, -1);  arg582_1 = None
        unsqueeze_1927 = torch.ops.aten.unsqueeze.default(unsqueeze_1926, -1);  unsqueeze_1926 = None
        add_537 = torch.ops.aten.add.Tensor(mul_779, unsqueeze_1927);  mul_779 = unsqueeze_1927 = None
        add_538 = torch.ops.aten.add.Tensor(add_537, relu_229);  add_537 = relu_229 = None
        relu_233 = torch.ops.aten.relu.default(add_538);  add_538 = None
        convolution_298 = torch.ops.aten.convolution.default(relu_233, arg583_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg583_1 = None
        add_539 = torch.ops.aten.add.Tensor(arg585_1, 1e-05);  arg585_1 = None
        sqrt_241 = torch.ops.aten.sqrt.default(add_539);  add_539 = None
        reciprocal_241 = torch.ops.aten.reciprocal.default(sqrt_241);  sqrt_241 = None
        mul_780 = torch.ops.aten.mul.Tensor(reciprocal_241, 1);  reciprocal_241 = None
        unsqueeze_1928 = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
        unsqueeze_1929 = torch.ops.aten.unsqueeze.default(unsqueeze_1928, -1);  unsqueeze_1928 = None
        unsqueeze_1930 = torch.ops.aten.unsqueeze.default(mul_780, -1);  mul_780 = None
        unsqueeze_1931 = torch.ops.aten.unsqueeze.default(unsqueeze_1930, -1);  unsqueeze_1930 = None
        sub_298 = torch.ops.aten.sub.Tensor(convolution_298, unsqueeze_1929);  convolution_298 = unsqueeze_1929 = None
        mul_781 = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_1931);  sub_298 = unsqueeze_1931 = None
        unsqueeze_1932 = torch.ops.aten.unsqueeze.default(arg586_1, -1);  arg586_1 = None
        unsqueeze_1933 = torch.ops.aten.unsqueeze.default(unsqueeze_1932, -1);  unsqueeze_1932 = None
        mul_782 = torch.ops.aten.mul.Tensor(mul_781, unsqueeze_1933);  mul_781 = unsqueeze_1933 = None
        unsqueeze_1934 = torch.ops.aten.unsqueeze.default(arg587_1, -1);  arg587_1 = None
        unsqueeze_1935 = torch.ops.aten.unsqueeze.default(unsqueeze_1934, -1);  unsqueeze_1934 = None
        add_540 = torch.ops.aten.add.Tensor(mul_782, unsqueeze_1935);  mul_782 = unsqueeze_1935 = None
        relu_234 = torch.ops.aten.relu.default(add_540);  add_540 = None
        convolution_299 = torch.ops.aten.convolution.default(relu_234, arg588_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_234 = arg588_1 = None
        add_541 = torch.ops.aten.add.Tensor(arg590_1, 1e-05);  arg590_1 = None
        sqrt_242 = torch.ops.aten.sqrt.default(add_541);  add_541 = None
        reciprocal_242 = torch.ops.aten.reciprocal.default(sqrt_242);  sqrt_242 = None
        mul_783 = torch.ops.aten.mul.Tensor(reciprocal_242, 1);  reciprocal_242 = None
        unsqueeze_1936 = torch.ops.aten.unsqueeze.default(arg589_1, -1);  arg589_1 = None
        unsqueeze_1937 = torch.ops.aten.unsqueeze.default(unsqueeze_1936, -1);  unsqueeze_1936 = None
        unsqueeze_1938 = torch.ops.aten.unsqueeze.default(mul_783, -1);  mul_783 = None
        unsqueeze_1939 = torch.ops.aten.unsqueeze.default(unsqueeze_1938, -1);  unsqueeze_1938 = None
        sub_299 = torch.ops.aten.sub.Tensor(convolution_299, unsqueeze_1937);  convolution_299 = unsqueeze_1937 = None
        mul_784 = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_1939);  sub_299 = unsqueeze_1939 = None
        unsqueeze_1940 = torch.ops.aten.unsqueeze.default(arg591_1, -1);  arg591_1 = None
        unsqueeze_1941 = torch.ops.aten.unsqueeze.default(unsqueeze_1940, -1);  unsqueeze_1940 = None
        mul_785 = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_1941);  mul_784 = unsqueeze_1941 = None
        unsqueeze_1942 = torch.ops.aten.unsqueeze.default(arg592_1, -1);  arg592_1 = None
        unsqueeze_1943 = torch.ops.aten.unsqueeze.default(unsqueeze_1942, -1);  unsqueeze_1942 = None
        add_542 = torch.ops.aten.add.Tensor(mul_785, unsqueeze_1943);  mul_785 = unsqueeze_1943 = None
        relu_235 = torch.ops.aten.relu.default(add_542);  add_542 = None
        view_344 = torch.ops.aten.view.default(relu_235, [8, 2, 256, 16, 16]);  relu_235 = None
        sum_172 = torch.ops.aten.sum.dim_IntList(view_344, [1])
        mean_58 = torch.ops.aten.mean.dim(sum_172, [2, 3], True);  sum_172 = None
        convolution_300 = torch.ops.aten.convolution.default(mean_58, arg593_1, arg594_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_58 = arg593_1 = arg594_1 = None
        add_543 = torch.ops.aten.add.Tensor(arg596_1, 1e-05);  arg596_1 = None
        sqrt_243 = torch.ops.aten.sqrt.default(add_543);  add_543 = None
        reciprocal_243 = torch.ops.aten.reciprocal.default(sqrt_243);  sqrt_243 = None
        mul_786 = torch.ops.aten.mul.Tensor(reciprocal_243, 1);  reciprocal_243 = None
        unsqueeze_1944 = torch.ops.aten.unsqueeze.default(arg595_1, -1);  arg595_1 = None
        unsqueeze_1945 = torch.ops.aten.unsqueeze.default(unsqueeze_1944, -1);  unsqueeze_1944 = None
        unsqueeze_1946 = torch.ops.aten.unsqueeze.default(mul_786, -1);  mul_786 = None
        unsqueeze_1947 = torch.ops.aten.unsqueeze.default(unsqueeze_1946, -1);  unsqueeze_1946 = None
        sub_300 = torch.ops.aten.sub.Tensor(convolution_300, unsqueeze_1945);  convolution_300 = unsqueeze_1945 = None
        mul_787 = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1947);  sub_300 = unsqueeze_1947 = None
        unsqueeze_1948 = torch.ops.aten.unsqueeze.default(arg597_1, -1);  arg597_1 = None
        unsqueeze_1949 = torch.ops.aten.unsqueeze.default(unsqueeze_1948, -1);  unsqueeze_1948 = None
        mul_788 = torch.ops.aten.mul.Tensor(mul_787, unsqueeze_1949);  mul_787 = unsqueeze_1949 = None
        unsqueeze_1950 = torch.ops.aten.unsqueeze.default(arg598_1, -1);  arg598_1 = None
        unsqueeze_1951 = torch.ops.aten.unsqueeze.default(unsqueeze_1950, -1);  unsqueeze_1950 = None
        add_544 = torch.ops.aten.add.Tensor(mul_788, unsqueeze_1951);  mul_788 = unsqueeze_1951 = None
        relu_236 = torch.ops.aten.relu.default(add_544);  add_544 = None
        convolution_301 = torch.ops.aten.convolution.default(relu_236, arg599_1, arg600_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_236 = arg599_1 = arg600_1 = None
        view_345 = torch.ops.aten.view.default(convolution_301, [8, 1, 2, -1]);  convolution_301 = None
        permute_58 = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        amax_57 = torch.ops.aten.amax.default(permute_58, [1], True)
        sub_301 = torch.ops.aten.sub.Tensor(permute_58, amax_57);  permute_58 = amax_57 = None
        exp_57 = torch.ops.aten.exp.default(sub_301);  sub_301 = None
        sum_173 = torch.ops.aten.sum.dim_IntList(exp_57, [1], True)
        div_57 = torch.ops.aten.div.Tensor(exp_57, sum_173);  exp_57 = sum_173 = None
        view_346 = torch.ops.aten.view.default(div_57, [8, -1]);  div_57 = None
        view_347 = torch.ops.aten.view.default(view_346, [8, -1, 1, 1]);  view_346 = None
        view_348 = torch.ops.aten.view.default(view_347, [8, 2, 256, 1, 1]);  view_347 = None
        mul_789 = torch.ops.aten.mul.Tensor(view_344, view_348);  view_344 = view_348 = None
        sum_174 = torch.ops.aten.sum.dim_IntList(mul_789, [1]);  mul_789 = None
        convolution_302 = torch.ops.aten.convolution.default(sum_174, arg601_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_174 = arg601_1 = None
        add_545 = torch.ops.aten.add.Tensor(arg603_1, 1e-05);  arg603_1 = None
        sqrt_244 = torch.ops.aten.sqrt.default(add_545);  add_545 = None
        reciprocal_244 = torch.ops.aten.reciprocal.default(sqrt_244);  sqrt_244 = None
        mul_790 = torch.ops.aten.mul.Tensor(reciprocal_244, 1);  reciprocal_244 = None
        unsqueeze_1952 = torch.ops.aten.unsqueeze.default(arg602_1, -1);  arg602_1 = None
        unsqueeze_1953 = torch.ops.aten.unsqueeze.default(unsqueeze_1952, -1);  unsqueeze_1952 = None
        unsqueeze_1954 = torch.ops.aten.unsqueeze.default(mul_790, -1);  mul_790 = None
        unsqueeze_1955 = torch.ops.aten.unsqueeze.default(unsqueeze_1954, -1);  unsqueeze_1954 = None
        sub_302 = torch.ops.aten.sub.Tensor(convolution_302, unsqueeze_1953);  convolution_302 = unsqueeze_1953 = None
        mul_791 = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1955);  sub_302 = unsqueeze_1955 = None
        unsqueeze_1956 = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_1957 = torch.ops.aten.unsqueeze.default(unsqueeze_1956, -1);  unsqueeze_1956 = None
        mul_792 = torch.ops.aten.mul.Tensor(mul_791, unsqueeze_1957);  mul_791 = unsqueeze_1957 = None
        unsqueeze_1958 = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
        unsqueeze_1959 = torch.ops.aten.unsqueeze.default(unsqueeze_1958, -1);  unsqueeze_1958 = None
        add_546 = torch.ops.aten.add.Tensor(mul_792, unsqueeze_1959);  mul_792 = unsqueeze_1959 = None
        add_547 = torch.ops.aten.add.Tensor(add_546, relu_233);  add_546 = relu_233 = None
        relu_237 = torch.ops.aten.relu.default(add_547);  add_547 = None
        convolution_303 = torch.ops.aten.convolution.default(relu_237, arg606_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg606_1 = None
        add_548 = torch.ops.aten.add.Tensor(arg608_1, 1e-05);  arg608_1 = None
        sqrt_245 = torch.ops.aten.sqrt.default(add_548);  add_548 = None
        reciprocal_245 = torch.ops.aten.reciprocal.default(sqrt_245);  sqrt_245 = None
        mul_793 = torch.ops.aten.mul.Tensor(reciprocal_245, 1);  reciprocal_245 = None
        unsqueeze_1960 = torch.ops.aten.unsqueeze.default(arg607_1, -1);  arg607_1 = None
        unsqueeze_1961 = torch.ops.aten.unsqueeze.default(unsqueeze_1960, -1);  unsqueeze_1960 = None
        unsqueeze_1962 = torch.ops.aten.unsqueeze.default(mul_793, -1);  mul_793 = None
        unsqueeze_1963 = torch.ops.aten.unsqueeze.default(unsqueeze_1962, -1);  unsqueeze_1962 = None
        sub_303 = torch.ops.aten.sub.Tensor(convolution_303, unsqueeze_1961);  convolution_303 = unsqueeze_1961 = None
        mul_794 = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1963);  sub_303 = unsqueeze_1963 = None
        unsqueeze_1964 = torch.ops.aten.unsqueeze.default(arg609_1, -1);  arg609_1 = None
        unsqueeze_1965 = torch.ops.aten.unsqueeze.default(unsqueeze_1964, -1);  unsqueeze_1964 = None
        mul_795 = torch.ops.aten.mul.Tensor(mul_794, unsqueeze_1965);  mul_794 = unsqueeze_1965 = None
        unsqueeze_1966 = torch.ops.aten.unsqueeze.default(arg610_1, -1);  arg610_1 = None
        unsqueeze_1967 = torch.ops.aten.unsqueeze.default(unsqueeze_1966, -1);  unsqueeze_1966 = None
        add_549 = torch.ops.aten.add.Tensor(mul_795, unsqueeze_1967);  mul_795 = unsqueeze_1967 = None
        relu_238 = torch.ops.aten.relu.default(add_549);  add_549 = None
        convolution_304 = torch.ops.aten.convolution.default(relu_238, arg611_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_238 = arg611_1 = None
        add_550 = torch.ops.aten.add.Tensor(arg613_1, 1e-05);  arg613_1 = None
        sqrt_246 = torch.ops.aten.sqrt.default(add_550);  add_550 = None
        reciprocal_246 = torch.ops.aten.reciprocal.default(sqrt_246);  sqrt_246 = None
        mul_796 = torch.ops.aten.mul.Tensor(reciprocal_246, 1);  reciprocal_246 = None
        unsqueeze_1968 = torch.ops.aten.unsqueeze.default(arg612_1, -1);  arg612_1 = None
        unsqueeze_1969 = torch.ops.aten.unsqueeze.default(unsqueeze_1968, -1);  unsqueeze_1968 = None
        unsqueeze_1970 = torch.ops.aten.unsqueeze.default(mul_796, -1);  mul_796 = None
        unsqueeze_1971 = torch.ops.aten.unsqueeze.default(unsqueeze_1970, -1);  unsqueeze_1970 = None
        sub_304 = torch.ops.aten.sub.Tensor(convolution_304, unsqueeze_1969);  convolution_304 = unsqueeze_1969 = None
        mul_797 = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1971);  sub_304 = unsqueeze_1971 = None
        unsqueeze_1972 = torch.ops.aten.unsqueeze.default(arg614_1, -1);  arg614_1 = None
        unsqueeze_1973 = torch.ops.aten.unsqueeze.default(unsqueeze_1972, -1);  unsqueeze_1972 = None
        mul_798 = torch.ops.aten.mul.Tensor(mul_797, unsqueeze_1973);  mul_797 = unsqueeze_1973 = None
        unsqueeze_1974 = torch.ops.aten.unsqueeze.default(arg615_1, -1);  arg615_1 = None
        unsqueeze_1975 = torch.ops.aten.unsqueeze.default(unsqueeze_1974, -1);  unsqueeze_1974 = None
        add_551 = torch.ops.aten.add.Tensor(mul_798, unsqueeze_1975);  mul_798 = unsqueeze_1975 = None
        relu_239 = torch.ops.aten.relu.default(add_551);  add_551 = None
        view_350 = torch.ops.aten.view.default(relu_239, [8, 2, 256, 16, 16]);  relu_239 = None
        sum_175 = torch.ops.aten.sum.dim_IntList(view_350, [1])
        mean_59 = torch.ops.aten.mean.dim(sum_175, [2, 3], True);  sum_175 = None
        convolution_305 = torch.ops.aten.convolution.default(mean_59, arg616_1, arg617_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_59 = arg616_1 = arg617_1 = None
        add_552 = torch.ops.aten.add.Tensor(arg619_1, 1e-05);  arg619_1 = None
        sqrt_247 = torch.ops.aten.sqrt.default(add_552);  add_552 = None
        reciprocal_247 = torch.ops.aten.reciprocal.default(sqrt_247);  sqrt_247 = None
        mul_799 = torch.ops.aten.mul.Tensor(reciprocal_247, 1);  reciprocal_247 = None
        unsqueeze_1976 = torch.ops.aten.unsqueeze.default(arg618_1, -1);  arg618_1 = None
        unsqueeze_1977 = torch.ops.aten.unsqueeze.default(unsqueeze_1976, -1);  unsqueeze_1976 = None
        unsqueeze_1978 = torch.ops.aten.unsqueeze.default(mul_799, -1);  mul_799 = None
        unsqueeze_1979 = torch.ops.aten.unsqueeze.default(unsqueeze_1978, -1);  unsqueeze_1978 = None
        sub_305 = torch.ops.aten.sub.Tensor(convolution_305, unsqueeze_1977);  convolution_305 = unsqueeze_1977 = None
        mul_800 = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1979);  sub_305 = unsqueeze_1979 = None
        unsqueeze_1980 = torch.ops.aten.unsqueeze.default(arg620_1, -1);  arg620_1 = None
        unsqueeze_1981 = torch.ops.aten.unsqueeze.default(unsqueeze_1980, -1);  unsqueeze_1980 = None
        mul_801 = torch.ops.aten.mul.Tensor(mul_800, unsqueeze_1981);  mul_800 = unsqueeze_1981 = None
        unsqueeze_1982 = torch.ops.aten.unsqueeze.default(arg621_1, -1);  arg621_1 = None
        unsqueeze_1983 = torch.ops.aten.unsqueeze.default(unsqueeze_1982, -1);  unsqueeze_1982 = None
        add_553 = torch.ops.aten.add.Tensor(mul_801, unsqueeze_1983);  mul_801 = unsqueeze_1983 = None
        relu_240 = torch.ops.aten.relu.default(add_553);  add_553 = None
        convolution_306 = torch.ops.aten.convolution.default(relu_240, arg622_1, arg623_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_240 = arg622_1 = arg623_1 = None
        view_351 = torch.ops.aten.view.default(convolution_306, [8, 1, 2, -1]);  convolution_306 = None
        permute_59 = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        amax_58 = torch.ops.aten.amax.default(permute_59, [1], True)
        sub_306 = torch.ops.aten.sub.Tensor(permute_59, amax_58);  permute_59 = amax_58 = None
        exp_58 = torch.ops.aten.exp.default(sub_306);  sub_306 = None
        sum_176 = torch.ops.aten.sum.dim_IntList(exp_58, [1], True)
        div_58 = torch.ops.aten.div.Tensor(exp_58, sum_176);  exp_58 = sum_176 = None
        view_352 = torch.ops.aten.view.default(div_58, [8, -1]);  div_58 = None
        view_353 = torch.ops.aten.view.default(view_352, [8, -1, 1, 1]);  view_352 = None
        view_354 = torch.ops.aten.view.default(view_353, [8, 2, 256, 1, 1]);  view_353 = None
        mul_802 = torch.ops.aten.mul.Tensor(view_350, view_354);  view_350 = view_354 = None
        sum_177 = torch.ops.aten.sum.dim_IntList(mul_802, [1]);  mul_802 = None
        convolution_307 = torch.ops.aten.convolution.default(sum_177, arg624_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_177 = arg624_1 = None
        add_554 = torch.ops.aten.add.Tensor(arg626_1, 1e-05);  arg626_1 = None
        sqrt_248 = torch.ops.aten.sqrt.default(add_554);  add_554 = None
        reciprocal_248 = torch.ops.aten.reciprocal.default(sqrt_248);  sqrt_248 = None
        mul_803 = torch.ops.aten.mul.Tensor(reciprocal_248, 1);  reciprocal_248 = None
        unsqueeze_1984 = torch.ops.aten.unsqueeze.default(arg625_1, -1);  arg625_1 = None
        unsqueeze_1985 = torch.ops.aten.unsqueeze.default(unsqueeze_1984, -1);  unsqueeze_1984 = None
        unsqueeze_1986 = torch.ops.aten.unsqueeze.default(mul_803, -1);  mul_803 = None
        unsqueeze_1987 = torch.ops.aten.unsqueeze.default(unsqueeze_1986, -1);  unsqueeze_1986 = None
        sub_307 = torch.ops.aten.sub.Tensor(convolution_307, unsqueeze_1985);  convolution_307 = unsqueeze_1985 = None
        mul_804 = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1987);  sub_307 = unsqueeze_1987 = None
        unsqueeze_1988 = torch.ops.aten.unsqueeze.default(arg627_1, -1);  arg627_1 = None
        unsqueeze_1989 = torch.ops.aten.unsqueeze.default(unsqueeze_1988, -1);  unsqueeze_1988 = None
        mul_805 = torch.ops.aten.mul.Tensor(mul_804, unsqueeze_1989);  mul_804 = unsqueeze_1989 = None
        unsqueeze_1990 = torch.ops.aten.unsqueeze.default(arg628_1, -1);  arg628_1 = None
        unsqueeze_1991 = torch.ops.aten.unsqueeze.default(unsqueeze_1990, -1);  unsqueeze_1990 = None
        add_555 = torch.ops.aten.add.Tensor(mul_805, unsqueeze_1991);  mul_805 = unsqueeze_1991 = None
        add_556 = torch.ops.aten.add.Tensor(add_555, relu_237);  add_555 = relu_237 = None
        relu_241 = torch.ops.aten.relu.default(add_556);  add_556 = None
        convolution_308 = torch.ops.aten.convolution.default(relu_241, arg629_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg629_1 = None
        add_557 = torch.ops.aten.add.Tensor(arg631_1, 1e-05);  arg631_1 = None
        sqrt_249 = torch.ops.aten.sqrt.default(add_557);  add_557 = None
        reciprocal_249 = torch.ops.aten.reciprocal.default(sqrt_249);  sqrt_249 = None
        mul_806 = torch.ops.aten.mul.Tensor(reciprocal_249, 1);  reciprocal_249 = None
        unsqueeze_1992 = torch.ops.aten.unsqueeze.default(arg630_1, -1);  arg630_1 = None
        unsqueeze_1993 = torch.ops.aten.unsqueeze.default(unsqueeze_1992, -1);  unsqueeze_1992 = None
        unsqueeze_1994 = torch.ops.aten.unsqueeze.default(mul_806, -1);  mul_806 = None
        unsqueeze_1995 = torch.ops.aten.unsqueeze.default(unsqueeze_1994, -1);  unsqueeze_1994 = None
        sub_308 = torch.ops.aten.sub.Tensor(convolution_308, unsqueeze_1993);  convolution_308 = unsqueeze_1993 = None
        mul_807 = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1995);  sub_308 = unsqueeze_1995 = None
        unsqueeze_1996 = torch.ops.aten.unsqueeze.default(arg632_1, -1);  arg632_1 = None
        unsqueeze_1997 = torch.ops.aten.unsqueeze.default(unsqueeze_1996, -1);  unsqueeze_1996 = None
        mul_808 = torch.ops.aten.mul.Tensor(mul_807, unsqueeze_1997);  mul_807 = unsqueeze_1997 = None
        unsqueeze_1998 = torch.ops.aten.unsqueeze.default(arg633_1, -1);  arg633_1 = None
        unsqueeze_1999 = torch.ops.aten.unsqueeze.default(unsqueeze_1998, -1);  unsqueeze_1998 = None
        add_558 = torch.ops.aten.add.Tensor(mul_808, unsqueeze_1999);  mul_808 = unsqueeze_1999 = None
        relu_242 = torch.ops.aten.relu.default(add_558);  add_558 = None
        convolution_309 = torch.ops.aten.convolution.default(relu_242, arg634_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_242 = arg634_1 = None
        add_559 = torch.ops.aten.add.Tensor(arg636_1, 1e-05);  arg636_1 = None
        sqrt_250 = torch.ops.aten.sqrt.default(add_559);  add_559 = None
        reciprocal_250 = torch.ops.aten.reciprocal.default(sqrt_250);  sqrt_250 = None
        mul_809 = torch.ops.aten.mul.Tensor(reciprocal_250, 1);  reciprocal_250 = None
        unsqueeze_2000 = torch.ops.aten.unsqueeze.default(arg635_1, -1);  arg635_1 = None
        unsqueeze_2001 = torch.ops.aten.unsqueeze.default(unsqueeze_2000, -1);  unsqueeze_2000 = None
        unsqueeze_2002 = torch.ops.aten.unsqueeze.default(mul_809, -1);  mul_809 = None
        unsqueeze_2003 = torch.ops.aten.unsqueeze.default(unsqueeze_2002, -1);  unsqueeze_2002 = None
        sub_309 = torch.ops.aten.sub.Tensor(convolution_309, unsqueeze_2001);  convolution_309 = unsqueeze_2001 = None
        mul_810 = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_2003);  sub_309 = unsqueeze_2003 = None
        unsqueeze_2004 = torch.ops.aten.unsqueeze.default(arg637_1, -1);  arg637_1 = None
        unsqueeze_2005 = torch.ops.aten.unsqueeze.default(unsqueeze_2004, -1);  unsqueeze_2004 = None
        mul_811 = torch.ops.aten.mul.Tensor(mul_810, unsqueeze_2005);  mul_810 = unsqueeze_2005 = None
        unsqueeze_2006 = torch.ops.aten.unsqueeze.default(arg638_1, -1);  arg638_1 = None
        unsqueeze_2007 = torch.ops.aten.unsqueeze.default(unsqueeze_2006, -1);  unsqueeze_2006 = None
        add_560 = torch.ops.aten.add.Tensor(mul_811, unsqueeze_2007);  mul_811 = unsqueeze_2007 = None
        relu_243 = torch.ops.aten.relu.default(add_560);  add_560 = None
        view_356 = torch.ops.aten.view.default(relu_243, [8, 2, 256, 16, 16]);  relu_243 = None
        sum_178 = torch.ops.aten.sum.dim_IntList(view_356, [1])
        mean_60 = torch.ops.aten.mean.dim(sum_178, [2, 3], True);  sum_178 = None
        convolution_310 = torch.ops.aten.convolution.default(mean_60, arg639_1, arg640_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_60 = arg639_1 = arg640_1 = None
        add_561 = torch.ops.aten.add.Tensor(arg642_1, 1e-05);  arg642_1 = None
        sqrt_251 = torch.ops.aten.sqrt.default(add_561);  add_561 = None
        reciprocal_251 = torch.ops.aten.reciprocal.default(sqrt_251);  sqrt_251 = None
        mul_812 = torch.ops.aten.mul.Tensor(reciprocal_251, 1);  reciprocal_251 = None
        unsqueeze_2008 = torch.ops.aten.unsqueeze.default(arg641_1, -1);  arg641_1 = None
        unsqueeze_2009 = torch.ops.aten.unsqueeze.default(unsqueeze_2008, -1);  unsqueeze_2008 = None
        unsqueeze_2010 = torch.ops.aten.unsqueeze.default(mul_812, -1);  mul_812 = None
        unsqueeze_2011 = torch.ops.aten.unsqueeze.default(unsqueeze_2010, -1);  unsqueeze_2010 = None
        sub_310 = torch.ops.aten.sub.Tensor(convolution_310, unsqueeze_2009);  convolution_310 = unsqueeze_2009 = None
        mul_813 = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_2011);  sub_310 = unsqueeze_2011 = None
        unsqueeze_2012 = torch.ops.aten.unsqueeze.default(arg643_1, -1);  arg643_1 = None
        unsqueeze_2013 = torch.ops.aten.unsqueeze.default(unsqueeze_2012, -1);  unsqueeze_2012 = None
        mul_814 = torch.ops.aten.mul.Tensor(mul_813, unsqueeze_2013);  mul_813 = unsqueeze_2013 = None
        unsqueeze_2014 = torch.ops.aten.unsqueeze.default(arg644_1, -1);  arg644_1 = None
        unsqueeze_2015 = torch.ops.aten.unsqueeze.default(unsqueeze_2014, -1);  unsqueeze_2014 = None
        add_562 = torch.ops.aten.add.Tensor(mul_814, unsqueeze_2015);  mul_814 = unsqueeze_2015 = None
        relu_244 = torch.ops.aten.relu.default(add_562);  add_562 = None
        convolution_311 = torch.ops.aten.convolution.default(relu_244, arg645_1, arg646_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_244 = arg645_1 = arg646_1 = None
        view_357 = torch.ops.aten.view.default(convolution_311, [8, 1, 2, -1]);  convolution_311 = None
        permute_60 = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        amax_59 = torch.ops.aten.amax.default(permute_60, [1], True)
        sub_311 = torch.ops.aten.sub.Tensor(permute_60, amax_59);  permute_60 = amax_59 = None
        exp_59 = torch.ops.aten.exp.default(sub_311);  sub_311 = None
        sum_179 = torch.ops.aten.sum.dim_IntList(exp_59, [1], True)
        div_59 = torch.ops.aten.div.Tensor(exp_59, sum_179);  exp_59 = sum_179 = None
        view_358 = torch.ops.aten.view.default(div_59, [8, -1]);  div_59 = None
        view_359 = torch.ops.aten.view.default(view_358, [8, -1, 1, 1]);  view_358 = None
        view_360 = torch.ops.aten.view.default(view_359, [8, 2, 256, 1, 1]);  view_359 = None
        mul_815 = torch.ops.aten.mul.Tensor(view_356, view_360);  view_356 = view_360 = None
        sum_180 = torch.ops.aten.sum.dim_IntList(mul_815, [1]);  mul_815 = None
        convolution_312 = torch.ops.aten.convolution.default(sum_180, arg647_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_180 = arg647_1 = None
        add_563 = torch.ops.aten.add.Tensor(arg649_1, 1e-05);  arg649_1 = None
        sqrt_252 = torch.ops.aten.sqrt.default(add_563);  add_563 = None
        reciprocal_252 = torch.ops.aten.reciprocal.default(sqrt_252);  sqrt_252 = None
        mul_816 = torch.ops.aten.mul.Tensor(reciprocal_252, 1);  reciprocal_252 = None
        unsqueeze_2016 = torch.ops.aten.unsqueeze.default(arg648_1, -1);  arg648_1 = None
        unsqueeze_2017 = torch.ops.aten.unsqueeze.default(unsqueeze_2016, -1);  unsqueeze_2016 = None
        unsqueeze_2018 = torch.ops.aten.unsqueeze.default(mul_816, -1);  mul_816 = None
        unsqueeze_2019 = torch.ops.aten.unsqueeze.default(unsqueeze_2018, -1);  unsqueeze_2018 = None
        sub_312 = torch.ops.aten.sub.Tensor(convolution_312, unsqueeze_2017);  convolution_312 = unsqueeze_2017 = None
        mul_817 = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_2019);  sub_312 = unsqueeze_2019 = None
        unsqueeze_2020 = torch.ops.aten.unsqueeze.default(arg650_1, -1);  arg650_1 = None
        unsqueeze_2021 = torch.ops.aten.unsqueeze.default(unsqueeze_2020, -1);  unsqueeze_2020 = None
        mul_818 = torch.ops.aten.mul.Tensor(mul_817, unsqueeze_2021);  mul_817 = unsqueeze_2021 = None
        unsqueeze_2022 = torch.ops.aten.unsqueeze.default(arg651_1, -1);  arg651_1 = None
        unsqueeze_2023 = torch.ops.aten.unsqueeze.default(unsqueeze_2022, -1);  unsqueeze_2022 = None
        add_564 = torch.ops.aten.add.Tensor(mul_818, unsqueeze_2023);  mul_818 = unsqueeze_2023 = None
        add_565 = torch.ops.aten.add.Tensor(add_564, relu_241);  add_564 = relu_241 = None
        relu_245 = torch.ops.aten.relu.default(add_565);  add_565 = None
        convolution_313 = torch.ops.aten.convolution.default(relu_245, arg652_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg652_1 = None
        add_566 = torch.ops.aten.add.Tensor(arg654_1, 1e-05);  arg654_1 = None
        sqrt_253 = torch.ops.aten.sqrt.default(add_566);  add_566 = None
        reciprocal_253 = torch.ops.aten.reciprocal.default(sqrt_253);  sqrt_253 = None
        mul_819 = torch.ops.aten.mul.Tensor(reciprocal_253, 1);  reciprocal_253 = None
        unsqueeze_2024 = torch.ops.aten.unsqueeze.default(arg653_1, -1);  arg653_1 = None
        unsqueeze_2025 = torch.ops.aten.unsqueeze.default(unsqueeze_2024, -1);  unsqueeze_2024 = None
        unsqueeze_2026 = torch.ops.aten.unsqueeze.default(mul_819, -1);  mul_819 = None
        unsqueeze_2027 = torch.ops.aten.unsqueeze.default(unsqueeze_2026, -1);  unsqueeze_2026 = None
        sub_313 = torch.ops.aten.sub.Tensor(convolution_313, unsqueeze_2025);  convolution_313 = unsqueeze_2025 = None
        mul_820 = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_2027);  sub_313 = unsqueeze_2027 = None
        unsqueeze_2028 = torch.ops.aten.unsqueeze.default(arg655_1, -1);  arg655_1 = None
        unsqueeze_2029 = torch.ops.aten.unsqueeze.default(unsqueeze_2028, -1);  unsqueeze_2028 = None
        mul_821 = torch.ops.aten.mul.Tensor(mul_820, unsqueeze_2029);  mul_820 = unsqueeze_2029 = None
        unsqueeze_2030 = torch.ops.aten.unsqueeze.default(arg656_1, -1);  arg656_1 = None
        unsqueeze_2031 = torch.ops.aten.unsqueeze.default(unsqueeze_2030, -1);  unsqueeze_2030 = None
        add_567 = torch.ops.aten.add.Tensor(mul_821, unsqueeze_2031);  mul_821 = unsqueeze_2031 = None
        relu_246 = torch.ops.aten.relu.default(add_567);  add_567 = None
        convolution_314 = torch.ops.aten.convolution.default(relu_246, arg657_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_246 = arg657_1 = None
        add_568 = torch.ops.aten.add.Tensor(arg659_1, 1e-05);  arg659_1 = None
        sqrt_254 = torch.ops.aten.sqrt.default(add_568);  add_568 = None
        reciprocal_254 = torch.ops.aten.reciprocal.default(sqrt_254);  sqrt_254 = None
        mul_822 = torch.ops.aten.mul.Tensor(reciprocal_254, 1);  reciprocal_254 = None
        unsqueeze_2032 = torch.ops.aten.unsqueeze.default(arg658_1, -1);  arg658_1 = None
        unsqueeze_2033 = torch.ops.aten.unsqueeze.default(unsqueeze_2032, -1);  unsqueeze_2032 = None
        unsqueeze_2034 = torch.ops.aten.unsqueeze.default(mul_822, -1);  mul_822 = None
        unsqueeze_2035 = torch.ops.aten.unsqueeze.default(unsqueeze_2034, -1);  unsqueeze_2034 = None
        sub_314 = torch.ops.aten.sub.Tensor(convolution_314, unsqueeze_2033);  convolution_314 = unsqueeze_2033 = None
        mul_823 = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_2035);  sub_314 = unsqueeze_2035 = None
        unsqueeze_2036 = torch.ops.aten.unsqueeze.default(arg660_1, -1);  arg660_1 = None
        unsqueeze_2037 = torch.ops.aten.unsqueeze.default(unsqueeze_2036, -1);  unsqueeze_2036 = None
        mul_824 = torch.ops.aten.mul.Tensor(mul_823, unsqueeze_2037);  mul_823 = unsqueeze_2037 = None
        unsqueeze_2038 = torch.ops.aten.unsqueeze.default(arg661_1, -1);  arg661_1 = None
        unsqueeze_2039 = torch.ops.aten.unsqueeze.default(unsqueeze_2038, -1);  unsqueeze_2038 = None
        add_569 = torch.ops.aten.add.Tensor(mul_824, unsqueeze_2039);  mul_824 = unsqueeze_2039 = None
        relu_247 = torch.ops.aten.relu.default(add_569);  add_569 = None
        view_362 = torch.ops.aten.view.default(relu_247, [8, 2, 256, 16, 16]);  relu_247 = None
        sum_181 = torch.ops.aten.sum.dim_IntList(view_362, [1])
        mean_61 = torch.ops.aten.mean.dim(sum_181, [2, 3], True);  sum_181 = None
        convolution_315 = torch.ops.aten.convolution.default(mean_61, arg662_1, arg663_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_61 = arg662_1 = arg663_1 = None
        add_570 = torch.ops.aten.add.Tensor(arg665_1, 1e-05);  arg665_1 = None
        sqrt_255 = torch.ops.aten.sqrt.default(add_570);  add_570 = None
        reciprocal_255 = torch.ops.aten.reciprocal.default(sqrt_255);  sqrt_255 = None
        mul_825 = torch.ops.aten.mul.Tensor(reciprocal_255, 1);  reciprocal_255 = None
        unsqueeze_2040 = torch.ops.aten.unsqueeze.default(arg664_1, -1);  arg664_1 = None
        unsqueeze_2041 = torch.ops.aten.unsqueeze.default(unsqueeze_2040, -1);  unsqueeze_2040 = None
        unsqueeze_2042 = torch.ops.aten.unsqueeze.default(mul_825, -1);  mul_825 = None
        unsqueeze_2043 = torch.ops.aten.unsqueeze.default(unsqueeze_2042, -1);  unsqueeze_2042 = None
        sub_315 = torch.ops.aten.sub.Tensor(convolution_315, unsqueeze_2041);  convolution_315 = unsqueeze_2041 = None
        mul_826 = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_2043);  sub_315 = unsqueeze_2043 = None
        unsqueeze_2044 = torch.ops.aten.unsqueeze.default(arg666_1, -1);  arg666_1 = None
        unsqueeze_2045 = torch.ops.aten.unsqueeze.default(unsqueeze_2044, -1);  unsqueeze_2044 = None
        mul_827 = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_2045);  mul_826 = unsqueeze_2045 = None
        unsqueeze_2046 = torch.ops.aten.unsqueeze.default(arg667_1, -1);  arg667_1 = None
        unsqueeze_2047 = torch.ops.aten.unsqueeze.default(unsqueeze_2046, -1);  unsqueeze_2046 = None
        add_571 = torch.ops.aten.add.Tensor(mul_827, unsqueeze_2047);  mul_827 = unsqueeze_2047 = None
        relu_248 = torch.ops.aten.relu.default(add_571);  add_571 = None
        convolution_316 = torch.ops.aten.convolution.default(relu_248, arg668_1, arg669_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_248 = arg668_1 = arg669_1 = None
        view_363 = torch.ops.aten.view.default(convolution_316, [8, 1, 2, -1]);  convolution_316 = None
        permute_61 = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None
        amax_60 = torch.ops.aten.amax.default(permute_61, [1], True)
        sub_316 = torch.ops.aten.sub.Tensor(permute_61, amax_60);  permute_61 = amax_60 = None
        exp_60 = torch.ops.aten.exp.default(sub_316);  sub_316 = None
        sum_182 = torch.ops.aten.sum.dim_IntList(exp_60, [1], True)
        div_60 = torch.ops.aten.div.Tensor(exp_60, sum_182);  exp_60 = sum_182 = None
        view_364 = torch.ops.aten.view.default(div_60, [8, -1]);  div_60 = None
        view_365 = torch.ops.aten.view.default(view_364, [8, -1, 1, 1]);  view_364 = None
        view_366 = torch.ops.aten.view.default(view_365, [8, 2, 256, 1, 1]);  view_365 = None
        mul_828 = torch.ops.aten.mul.Tensor(view_362, view_366);  view_362 = view_366 = None
        sum_183 = torch.ops.aten.sum.dim_IntList(mul_828, [1]);  mul_828 = None
        convolution_317 = torch.ops.aten.convolution.default(sum_183, arg670_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_183 = arg670_1 = None
        add_572 = torch.ops.aten.add.Tensor(arg672_1, 1e-05);  arg672_1 = None
        sqrt_256 = torch.ops.aten.sqrt.default(add_572);  add_572 = None
        reciprocal_256 = torch.ops.aten.reciprocal.default(sqrt_256);  sqrt_256 = None
        mul_829 = torch.ops.aten.mul.Tensor(reciprocal_256, 1);  reciprocal_256 = None
        unsqueeze_2048 = torch.ops.aten.unsqueeze.default(arg671_1, -1);  arg671_1 = None
        unsqueeze_2049 = torch.ops.aten.unsqueeze.default(unsqueeze_2048, -1);  unsqueeze_2048 = None
        unsqueeze_2050 = torch.ops.aten.unsqueeze.default(mul_829, -1);  mul_829 = None
        unsqueeze_2051 = torch.ops.aten.unsqueeze.default(unsqueeze_2050, -1);  unsqueeze_2050 = None
        sub_317 = torch.ops.aten.sub.Tensor(convolution_317, unsqueeze_2049);  convolution_317 = unsqueeze_2049 = None
        mul_830 = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_2051);  sub_317 = unsqueeze_2051 = None
        unsqueeze_2052 = torch.ops.aten.unsqueeze.default(arg673_1, -1);  arg673_1 = None
        unsqueeze_2053 = torch.ops.aten.unsqueeze.default(unsqueeze_2052, -1);  unsqueeze_2052 = None
        mul_831 = torch.ops.aten.mul.Tensor(mul_830, unsqueeze_2053);  mul_830 = unsqueeze_2053 = None
        unsqueeze_2054 = torch.ops.aten.unsqueeze.default(arg674_1, -1);  arg674_1 = None
        unsqueeze_2055 = torch.ops.aten.unsqueeze.default(unsqueeze_2054, -1);  unsqueeze_2054 = None
        add_573 = torch.ops.aten.add.Tensor(mul_831, unsqueeze_2055);  mul_831 = unsqueeze_2055 = None
        add_574 = torch.ops.aten.add.Tensor(add_573, relu_245);  add_573 = relu_245 = None
        relu_249 = torch.ops.aten.relu.default(add_574);  add_574 = None
        convolution_318 = torch.ops.aten.convolution.default(relu_249, arg675_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg675_1 = None
        add_575 = torch.ops.aten.add.Tensor(arg677_1, 1e-05);  arg677_1 = None
        sqrt_257 = torch.ops.aten.sqrt.default(add_575);  add_575 = None
        reciprocal_257 = torch.ops.aten.reciprocal.default(sqrt_257);  sqrt_257 = None
        mul_832 = torch.ops.aten.mul.Tensor(reciprocal_257, 1);  reciprocal_257 = None
        unsqueeze_2056 = torch.ops.aten.unsqueeze.default(arg676_1, -1);  arg676_1 = None
        unsqueeze_2057 = torch.ops.aten.unsqueeze.default(unsqueeze_2056, -1);  unsqueeze_2056 = None
        unsqueeze_2058 = torch.ops.aten.unsqueeze.default(mul_832, -1);  mul_832 = None
        unsqueeze_2059 = torch.ops.aten.unsqueeze.default(unsqueeze_2058, -1);  unsqueeze_2058 = None
        sub_318 = torch.ops.aten.sub.Tensor(convolution_318, unsqueeze_2057);  convolution_318 = unsqueeze_2057 = None
        mul_833 = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_2059);  sub_318 = unsqueeze_2059 = None
        unsqueeze_2060 = torch.ops.aten.unsqueeze.default(arg678_1, -1);  arg678_1 = None
        unsqueeze_2061 = torch.ops.aten.unsqueeze.default(unsqueeze_2060, -1);  unsqueeze_2060 = None
        mul_834 = torch.ops.aten.mul.Tensor(mul_833, unsqueeze_2061);  mul_833 = unsqueeze_2061 = None
        unsqueeze_2062 = torch.ops.aten.unsqueeze.default(arg679_1, -1);  arg679_1 = None
        unsqueeze_2063 = torch.ops.aten.unsqueeze.default(unsqueeze_2062, -1);  unsqueeze_2062 = None
        add_576 = torch.ops.aten.add.Tensor(mul_834, unsqueeze_2063);  mul_834 = unsqueeze_2063 = None
        relu_250 = torch.ops.aten.relu.default(add_576);  add_576 = None
        convolution_319 = torch.ops.aten.convolution.default(relu_250, arg680_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_250 = arg680_1 = None
        add_577 = torch.ops.aten.add.Tensor(arg682_1, 1e-05);  arg682_1 = None
        sqrt_258 = torch.ops.aten.sqrt.default(add_577);  add_577 = None
        reciprocal_258 = torch.ops.aten.reciprocal.default(sqrt_258);  sqrt_258 = None
        mul_835 = torch.ops.aten.mul.Tensor(reciprocal_258, 1);  reciprocal_258 = None
        unsqueeze_2064 = torch.ops.aten.unsqueeze.default(arg681_1, -1);  arg681_1 = None
        unsqueeze_2065 = torch.ops.aten.unsqueeze.default(unsqueeze_2064, -1);  unsqueeze_2064 = None
        unsqueeze_2066 = torch.ops.aten.unsqueeze.default(mul_835, -1);  mul_835 = None
        unsqueeze_2067 = torch.ops.aten.unsqueeze.default(unsqueeze_2066, -1);  unsqueeze_2066 = None
        sub_319 = torch.ops.aten.sub.Tensor(convolution_319, unsqueeze_2065);  convolution_319 = unsqueeze_2065 = None
        mul_836 = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_2067);  sub_319 = unsqueeze_2067 = None
        unsqueeze_2068 = torch.ops.aten.unsqueeze.default(arg683_1, -1);  arg683_1 = None
        unsqueeze_2069 = torch.ops.aten.unsqueeze.default(unsqueeze_2068, -1);  unsqueeze_2068 = None
        mul_837 = torch.ops.aten.mul.Tensor(mul_836, unsqueeze_2069);  mul_836 = unsqueeze_2069 = None
        unsqueeze_2070 = torch.ops.aten.unsqueeze.default(arg684_1, -1);  arg684_1 = None
        unsqueeze_2071 = torch.ops.aten.unsqueeze.default(unsqueeze_2070, -1);  unsqueeze_2070 = None
        add_578 = torch.ops.aten.add.Tensor(mul_837, unsqueeze_2071);  mul_837 = unsqueeze_2071 = None
        relu_251 = torch.ops.aten.relu.default(add_578);  add_578 = None
        view_368 = torch.ops.aten.view.default(relu_251, [8, 2, 256, 16, 16]);  relu_251 = None
        sum_184 = torch.ops.aten.sum.dim_IntList(view_368, [1])
        mean_62 = torch.ops.aten.mean.dim(sum_184, [2, 3], True);  sum_184 = None
        convolution_320 = torch.ops.aten.convolution.default(mean_62, arg685_1, arg686_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_62 = arg685_1 = arg686_1 = None
        add_579 = torch.ops.aten.add.Tensor(arg688_1, 1e-05);  arg688_1 = None
        sqrt_259 = torch.ops.aten.sqrt.default(add_579);  add_579 = None
        reciprocal_259 = torch.ops.aten.reciprocal.default(sqrt_259);  sqrt_259 = None
        mul_838 = torch.ops.aten.mul.Tensor(reciprocal_259, 1);  reciprocal_259 = None
        unsqueeze_2072 = torch.ops.aten.unsqueeze.default(arg687_1, -1);  arg687_1 = None
        unsqueeze_2073 = torch.ops.aten.unsqueeze.default(unsqueeze_2072, -1);  unsqueeze_2072 = None
        unsqueeze_2074 = torch.ops.aten.unsqueeze.default(mul_838, -1);  mul_838 = None
        unsqueeze_2075 = torch.ops.aten.unsqueeze.default(unsqueeze_2074, -1);  unsqueeze_2074 = None
        sub_320 = torch.ops.aten.sub.Tensor(convolution_320, unsqueeze_2073);  convolution_320 = unsqueeze_2073 = None
        mul_839 = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_2075);  sub_320 = unsqueeze_2075 = None
        unsqueeze_2076 = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
        unsqueeze_2077 = torch.ops.aten.unsqueeze.default(unsqueeze_2076, -1);  unsqueeze_2076 = None
        mul_840 = torch.ops.aten.mul.Tensor(mul_839, unsqueeze_2077);  mul_839 = unsqueeze_2077 = None
        unsqueeze_2078 = torch.ops.aten.unsqueeze.default(arg690_1, -1);  arg690_1 = None
        unsqueeze_2079 = torch.ops.aten.unsqueeze.default(unsqueeze_2078, -1);  unsqueeze_2078 = None
        add_580 = torch.ops.aten.add.Tensor(mul_840, unsqueeze_2079);  mul_840 = unsqueeze_2079 = None
        relu_252 = torch.ops.aten.relu.default(add_580);  add_580 = None
        convolution_321 = torch.ops.aten.convolution.default(relu_252, arg691_1, arg692_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_252 = arg691_1 = arg692_1 = None
        view_369 = torch.ops.aten.view.default(convolution_321, [8, 1, 2, -1]);  convolution_321 = None
        permute_62 = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
        amax_61 = torch.ops.aten.amax.default(permute_62, [1], True)
        sub_321 = torch.ops.aten.sub.Tensor(permute_62, amax_61);  permute_62 = amax_61 = None
        exp_61 = torch.ops.aten.exp.default(sub_321);  sub_321 = None
        sum_185 = torch.ops.aten.sum.dim_IntList(exp_61, [1], True)
        div_61 = torch.ops.aten.div.Tensor(exp_61, sum_185);  exp_61 = sum_185 = None
        view_370 = torch.ops.aten.view.default(div_61, [8, -1]);  div_61 = None
        view_371 = torch.ops.aten.view.default(view_370, [8, -1, 1, 1]);  view_370 = None
        view_372 = torch.ops.aten.view.default(view_371, [8, 2, 256, 1, 1]);  view_371 = None
        mul_841 = torch.ops.aten.mul.Tensor(view_368, view_372);  view_368 = view_372 = None
        sum_186 = torch.ops.aten.sum.dim_IntList(mul_841, [1]);  mul_841 = None
        convolution_322 = torch.ops.aten.convolution.default(sum_186, arg693_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_186 = arg693_1 = None
        add_581 = torch.ops.aten.add.Tensor(arg695_1, 1e-05);  arg695_1 = None
        sqrt_260 = torch.ops.aten.sqrt.default(add_581);  add_581 = None
        reciprocal_260 = torch.ops.aten.reciprocal.default(sqrt_260);  sqrt_260 = None
        mul_842 = torch.ops.aten.mul.Tensor(reciprocal_260, 1);  reciprocal_260 = None
        unsqueeze_2080 = torch.ops.aten.unsqueeze.default(arg694_1, -1);  arg694_1 = None
        unsqueeze_2081 = torch.ops.aten.unsqueeze.default(unsqueeze_2080, -1);  unsqueeze_2080 = None
        unsqueeze_2082 = torch.ops.aten.unsqueeze.default(mul_842, -1);  mul_842 = None
        unsqueeze_2083 = torch.ops.aten.unsqueeze.default(unsqueeze_2082, -1);  unsqueeze_2082 = None
        sub_322 = torch.ops.aten.sub.Tensor(convolution_322, unsqueeze_2081);  convolution_322 = unsqueeze_2081 = None
        mul_843 = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_2083);  sub_322 = unsqueeze_2083 = None
        unsqueeze_2084 = torch.ops.aten.unsqueeze.default(arg696_1, -1);  arg696_1 = None
        unsqueeze_2085 = torch.ops.aten.unsqueeze.default(unsqueeze_2084, -1);  unsqueeze_2084 = None
        mul_844 = torch.ops.aten.mul.Tensor(mul_843, unsqueeze_2085);  mul_843 = unsqueeze_2085 = None
        unsqueeze_2086 = torch.ops.aten.unsqueeze.default(arg697_1, -1);  arg697_1 = None
        unsqueeze_2087 = torch.ops.aten.unsqueeze.default(unsqueeze_2086, -1);  unsqueeze_2086 = None
        add_582 = torch.ops.aten.add.Tensor(mul_844, unsqueeze_2087);  mul_844 = unsqueeze_2087 = None
        add_583 = torch.ops.aten.add.Tensor(add_582, relu_249);  add_582 = relu_249 = None
        relu_253 = torch.ops.aten.relu.default(add_583);  add_583 = None
        convolution_323 = torch.ops.aten.convolution.default(relu_253, arg698_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg698_1 = None
        add_584 = torch.ops.aten.add.Tensor(arg700_1, 1e-05);  arg700_1 = None
        sqrt_261 = torch.ops.aten.sqrt.default(add_584);  add_584 = None
        reciprocal_261 = torch.ops.aten.reciprocal.default(sqrt_261);  sqrt_261 = None
        mul_845 = torch.ops.aten.mul.Tensor(reciprocal_261, 1);  reciprocal_261 = None
        unsqueeze_2088 = torch.ops.aten.unsqueeze.default(arg699_1, -1);  arg699_1 = None
        unsqueeze_2089 = torch.ops.aten.unsqueeze.default(unsqueeze_2088, -1);  unsqueeze_2088 = None
        unsqueeze_2090 = torch.ops.aten.unsqueeze.default(mul_845, -1);  mul_845 = None
        unsqueeze_2091 = torch.ops.aten.unsqueeze.default(unsqueeze_2090, -1);  unsqueeze_2090 = None
        sub_323 = torch.ops.aten.sub.Tensor(convolution_323, unsqueeze_2089);  convolution_323 = unsqueeze_2089 = None
        mul_846 = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_2091);  sub_323 = unsqueeze_2091 = None
        unsqueeze_2092 = torch.ops.aten.unsqueeze.default(arg701_1, -1);  arg701_1 = None
        unsqueeze_2093 = torch.ops.aten.unsqueeze.default(unsqueeze_2092, -1);  unsqueeze_2092 = None
        mul_847 = torch.ops.aten.mul.Tensor(mul_846, unsqueeze_2093);  mul_846 = unsqueeze_2093 = None
        unsqueeze_2094 = torch.ops.aten.unsqueeze.default(arg702_1, -1);  arg702_1 = None
        unsqueeze_2095 = torch.ops.aten.unsqueeze.default(unsqueeze_2094, -1);  unsqueeze_2094 = None
        add_585 = torch.ops.aten.add.Tensor(mul_847, unsqueeze_2095);  mul_847 = unsqueeze_2095 = None
        relu_254 = torch.ops.aten.relu.default(add_585);  add_585 = None
        convolution_324 = torch.ops.aten.convolution.default(relu_254, arg703_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_254 = arg703_1 = None
        add_586 = torch.ops.aten.add.Tensor(arg705_1, 1e-05);  arg705_1 = None
        sqrt_262 = torch.ops.aten.sqrt.default(add_586);  add_586 = None
        reciprocal_262 = torch.ops.aten.reciprocal.default(sqrt_262);  sqrt_262 = None
        mul_848 = torch.ops.aten.mul.Tensor(reciprocal_262, 1);  reciprocal_262 = None
        unsqueeze_2096 = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
        unsqueeze_2097 = torch.ops.aten.unsqueeze.default(unsqueeze_2096, -1);  unsqueeze_2096 = None
        unsqueeze_2098 = torch.ops.aten.unsqueeze.default(mul_848, -1);  mul_848 = None
        unsqueeze_2099 = torch.ops.aten.unsqueeze.default(unsqueeze_2098, -1);  unsqueeze_2098 = None
        sub_324 = torch.ops.aten.sub.Tensor(convolution_324, unsqueeze_2097);  convolution_324 = unsqueeze_2097 = None
        mul_849 = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_2099);  sub_324 = unsqueeze_2099 = None
        unsqueeze_2100 = torch.ops.aten.unsqueeze.default(arg706_1, -1);  arg706_1 = None
        unsqueeze_2101 = torch.ops.aten.unsqueeze.default(unsqueeze_2100, -1);  unsqueeze_2100 = None
        mul_850 = torch.ops.aten.mul.Tensor(mul_849, unsqueeze_2101);  mul_849 = unsqueeze_2101 = None
        unsqueeze_2102 = torch.ops.aten.unsqueeze.default(arg707_1, -1);  arg707_1 = None
        unsqueeze_2103 = torch.ops.aten.unsqueeze.default(unsqueeze_2102, -1);  unsqueeze_2102 = None
        add_587 = torch.ops.aten.add.Tensor(mul_850, unsqueeze_2103);  mul_850 = unsqueeze_2103 = None
        relu_255 = torch.ops.aten.relu.default(add_587);  add_587 = None
        view_374 = torch.ops.aten.view.default(relu_255, [8, 2, 256, 16, 16]);  relu_255 = None
        sum_187 = torch.ops.aten.sum.dim_IntList(view_374, [1])
        mean_63 = torch.ops.aten.mean.dim(sum_187, [2, 3], True);  sum_187 = None
        convolution_325 = torch.ops.aten.convolution.default(mean_63, arg708_1, arg709_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_63 = arg708_1 = arg709_1 = None
        add_588 = torch.ops.aten.add.Tensor(arg711_1, 1e-05);  arg711_1 = None
        sqrt_263 = torch.ops.aten.sqrt.default(add_588);  add_588 = None
        reciprocal_263 = torch.ops.aten.reciprocal.default(sqrt_263);  sqrt_263 = None
        mul_851 = torch.ops.aten.mul.Tensor(reciprocal_263, 1);  reciprocal_263 = None
        unsqueeze_2104 = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
        unsqueeze_2105 = torch.ops.aten.unsqueeze.default(unsqueeze_2104, -1);  unsqueeze_2104 = None
        unsqueeze_2106 = torch.ops.aten.unsqueeze.default(mul_851, -1);  mul_851 = None
        unsqueeze_2107 = torch.ops.aten.unsqueeze.default(unsqueeze_2106, -1);  unsqueeze_2106 = None
        sub_325 = torch.ops.aten.sub.Tensor(convolution_325, unsqueeze_2105);  convolution_325 = unsqueeze_2105 = None
        mul_852 = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_2107);  sub_325 = unsqueeze_2107 = None
        unsqueeze_2108 = torch.ops.aten.unsqueeze.default(arg712_1, -1);  arg712_1 = None
        unsqueeze_2109 = torch.ops.aten.unsqueeze.default(unsqueeze_2108, -1);  unsqueeze_2108 = None
        mul_853 = torch.ops.aten.mul.Tensor(mul_852, unsqueeze_2109);  mul_852 = unsqueeze_2109 = None
        unsqueeze_2110 = torch.ops.aten.unsqueeze.default(arg713_1, -1);  arg713_1 = None
        unsqueeze_2111 = torch.ops.aten.unsqueeze.default(unsqueeze_2110, -1);  unsqueeze_2110 = None
        add_589 = torch.ops.aten.add.Tensor(mul_853, unsqueeze_2111);  mul_853 = unsqueeze_2111 = None
        relu_256 = torch.ops.aten.relu.default(add_589);  add_589 = None
        convolution_326 = torch.ops.aten.convolution.default(relu_256, arg714_1, arg715_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_256 = arg714_1 = arg715_1 = None
        view_375 = torch.ops.aten.view.default(convolution_326, [8, 1, 2, -1]);  convolution_326 = None
        permute_63 = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        amax_62 = torch.ops.aten.amax.default(permute_63, [1], True)
        sub_326 = torch.ops.aten.sub.Tensor(permute_63, amax_62);  permute_63 = amax_62 = None
        exp_62 = torch.ops.aten.exp.default(sub_326);  sub_326 = None
        sum_188 = torch.ops.aten.sum.dim_IntList(exp_62, [1], True)
        div_62 = torch.ops.aten.div.Tensor(exp_62, sum_188);  exp_62 = sum_188 = None
        view_376 = torch.ops.aten.view.default(div_62, [8, -1]);  div_62 = None
        view_377 = torch.ops.aten.view.default(view_376, [8, -1, 1, 1]);  view_376 = None
        view_378 = torch.ops.aten.view.default(view_377, [8, 2, 256, 1, 1]);  view_377 = None
        mul_854 = torch.ops.aten.mul.Tensor(view_374, view_378);  view_374 = view_378 = None
        sum_189 = torch.ops.aten.sum.dim_IntList(mul_854, [1]);  mul_854 = None
        convolution_327 = torch.ops.aten.convolution.default(sum_189, arg716_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_189 = arg716_1 = None
        add_590 = torch.ops.aten.add.Tensor(arg718_1, 1e-05);  arg718_1 = None
        sqrt_264 = torch.ops.aten.sqrt.default(add_590);  add_590 = None
        reciprocal_264 = torch.ops.aten.reciprocal.default(sqrt_264);  sqrt_264 = None
        mul_855 = torch.ops.aten.mul.Tensor(reciprocal_264, 1);  reciprocal_264 = None
        unsqueeze_2112 = torch.ops.aten.unsqueeze.default(arg717_1, -1);  arg717_1 = None
        unsqueeze_2113 = torch.ops.aten.unsqueeze.default(unsqueeze_2112, -1);  unsqueeze_2112 = None
        unsqueeze_2114 = torch.ops.aten.unsqueeze.default(mul_855, -1);  mul_855 = None
        unsqueeze_2115 = torch.ops.aten.unsqueeze.default(unsqueeze_2114, -1);  unsqueeze_2114 = None
        sub_327 = torch.ops.aten.sub.Tensor(convolution_327, unsqueeze_2113);  convolution_327 = unsqueeze_2113 = None
        mul_856 = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_2115);  sub_327 = unsqueeze_2115 = None
        unsqueeze_2116 = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_2117 = torch.ops.aten.unsqueeze.default(unsqueeze_2116, -1);  unsqueeze_2116 = None
        mul_857 = torch.ops.aten.mul.Tensor(mul_856, unsqueeze_2117);  mul_856 = unsqueeze_2117 = None
        unsqueeze_2118 = torch.ops.aten.unsqueeze.default(arg720_1, -1);  arg720_1 = None
        unsqueeze_2119 = torch.ops.aten.unsqueeze.default(unsqueeze_2118, -1);  unsqueeze_2118 = None
        add_591 = torch.ops.aten.add.Tensor(mul_857, unsqueeze_2119);  mul_857 = unsqueeze_2119 = None
        add_592 = torch.ops.aten.add.Tensor(add_591, relu_253);  add_591 = relu_253 = None
        relu_257 = torch.ops.aten.relu.default(add_592);  add_592 = None
        convolution_328 = torch.ops.aten.convolution.default(relu_257, arg721_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg721_1 = None
        add_593 = torch.ops.aten.add.Tensor(arg723_1, 1e-05);  arg723_1 = None
        sqrt_265 = torch.ops.aten.sqrt.default(add_593);  add_593 = None
        reciprocal_265 = torch.ops.aten.reciprocal.default(sqrt_265);  sqrt_265 = None
        mul_858 = torch.ops.aten.mul.Tensor(reciprocal_265, 1);  reciprocal_265 = None
        unsqueeze_2120 = torch.ops.aten.unsqueeze.default(arg722_1, -1);  arg722_1 = None
        unsqueeze_2121 = torch.ops.aten.unsqueeze.default(unsqueeze_2120, -1);  unsqueeze_2120 = None
        unsqueeze_2122 = torch.ops.aten.unsqueeze.default(mul_858, -1);  mul_858 = None
        unsqueeze_2123 = torch.ops.aten.unsqueeze.default(unsqueeze_2122, -1);  unsqueeze_2122 = None
        sub_328 = torch.ops.aten.sub.Tensor(convolution_328, unsqueeze_2121);  convolution_328 = unsqueeze_2121 = None
        mul_859 = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_2123);  sub_328 = unsqueeze_2123 = None
        unsqueeze_2124 = torch.ops.aten.unsqueeze.default(arg724_1, -1);  arg724_1 = None
        unsqueeze_2125 = torch.ops.aten.unsqueeze.default(unsqueeze_2124, -1);  unsqueeze_2124 = None
        mul_860 = torch.ops.aten.mul.Tensor(mul_859, unsqueeze_2125);  mul_859 = unsqueeze_2125 = None
        unsqueeze_2126 = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_2127 = torch.ops.aten.unsqueeze.default(unsqueeze_2126, -1);  unsqueeze_2126 = None
        add_594 = torch.ops.aten.add.Tensor(mul_860, unsqueeze_2127);  mul_860 = unsqueeze_2127 = None
        relu_258 = torch.ops.aten.relu.default(add_594);  add_594 = None
        convolution_329 = torch.ops.aten.convolution.default(relu_258, arg726_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_258 = arg726_1 = None
        add_595 = torch.ops.aten.add.Tensor(arg728_1, 1e-05);  arg728_1 = None
        sqrt_266 = torch.ops.aten.sqrt.default(add_595);  add_595 = None
        reciprocal_266 = torch.ops.aten.reciprocal.default(sqrt_266);  sqrt_266 = None
        mul_861 = torch.ops.aten.mul.Tensor(reciprocal_266, 1);  reciprocal_266 = None
        unsqueeze_2128 = torch.ops.aten.unsqueeze.default(arg727_1, -1);  arg727_1 = None
        unsqueeze_2129 = torch.ops.aten.unsqueeze.default(unsqueeze_2128, -1);  unsqueeze_2128 = None
        unsqueeze_2130 = torch.ops.aten.unsqueeze.default(mul_861, -1);  mul_861 = None
        unsqueeze_2131 = torch.ops.aten.unsqueeze.default(unsqueeze_2130, -1);  unsqueeze_2130 = None
        sub_329 = torch.ops.aten.sub.Tensor(convolution_329, unsqueeze_2129);  convolution_329 = unsqueeze_2129 = None
        mul_862 = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_2131);  sub_329 = unsqueeze_2131 = None
        unsqueeze_2132 = torch.ops.aten.unsqueeze.default(arg729_1, -1);  arg729_1 = None
        unsqueeze_2133 = torch.ops.aten.unsqueeze.default(unsqueeze_2132, -1);  unsqueeze_2132 = None
        mul_863 = torch.ops.aten.mul.Tensor(mul_862, unsqueeze_2133);  mul_862 = unsqueeze_2133 = None
        unsqueeze_2134 = torch.ops.aten.unsqueeze.default(arg730_1, -1);  arg730_1 = None
        unsqueeze_2135 = torch.ops.aten.unsqueeze.default(unsqueeze_2134, -1);  unsqueeze_2134 = None
        add_596 = torch.ops.aten.add.Tensor(mul_863, unsqueeze_2135);  mul_863 = unsqueeze_2135 = None
        relu_259 = torch.ops.aten.relu.default(add_596);  add_596 = None
        view_380 = torch.ops.aten.view.default(relu_259, [8, 2, 512, 16, 16]);  relu_259 = None
        sum_190 = torch.ops.aten.sum.dim_IntList(view_380, [1])
        mean_64 = torch.ops.aten.mean.dim(sum_190, [2, 3], True);  sum_190 = None
        convolution_330 = torch.ops.aten.convolution.default(mean_64, arg731_1, arg732_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_64 = arg731_1 = arg732_1 = None
        add_597 = torch.ops.aten.add.Tensor(arg734_1, 1e-05);  arg734_1 = None
        sqrt_267 = torch.ops.aten.sqrt.default(add_597);  add_597 = None
        reciprocal_267 = torch.ops.aten.reciprocal.default(sqrt_267);  sqrt_267 = None
        mul_864 = torch.ops.aten.mul.Tensor(reciprocal_267, 1);  reciprocal_267 = None
        unsqueeze_2136 = torch.ops.aten.unsqueeze.default(arg733_1, -1);  arg733_1 = None
        unsqueeze_2137 = torch.ops.aten.unsqueeze.default(unsqueeze_2136, -1);  unsqueeze_2136 = None
        unsqueeze_2138 = torch.ops.aten.unsqueeze.default(mul_864, -1);  mul_864 = None
        unsqueeze_2139 = torch.ops.aten.unsqueeze.default(unsqueeze_2138, -1);  unsqueeze_2138 = None
        sub_330 = torch.ops.aten.sub.Tensor(convolution_330, unsqueeze_2137);  convolution_330 = unsqueeze_2137 = None
        mul_865 = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_2139);  sub_330 = unsqueeze_2139 = None
        unsqueeze_2140 = torch.ops.aten.unsqueeze.default(arg735_1, -1);  arg735_1 = None
        unsqueeze_2141 = torch.ops.aten.unsqueeze.default(unsqueeze_2140, -1);  unsqueeze_2140 = None
        mul_866 = torch.ops.aten.mul.Tensor(mul_865, unsqueeze_2141);  mul_865 = unsqueeze_2141 = None
        unsqueeze_2142 = torch.ops.aten.unsqueeze.default(arg736_1, -1);  arg736_1 = None
        unsqueeze_2143 = torch.ops.aten.unsqueeze.default(unsqueeze_2142, -1);  unsqueeze_2142 = None
        add_598 = torch.ops.aten.add.Tensor(mul_866, unsqueeze_2143);  mul_866 = unsqueeze_2143 = None
        relu_260 = torch.ops.aten.relu.default(add_598);  add_598 = None
        convolution_331 = torch.ops.aten.convolution.default(relu_260, arg737_1, arg738_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_260 = arg737_1 = arg738_1 = None
        view_381 = torch.ops.aten.view.default(convolution_331, [8, 1, 2, -1]);  convolution_331 = None
        permute_64 = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        amax_63 = torch.ops.aten.amax.default(permute_64, [1], True)
        sub_331 = torch.ops.aten.sub.Tensor(permute_64, amax_63);  permute_64 = amax_63 = None
        exp_63 = torch.ops.aten.exp.default(sub_331);  sub_331 = None
        sum_191 = torch.ops.aten.sum.dim_IntList(exp_63, [1], True)
        div_63 = torch.ops.aten.div.Tensor(exp_63, sum_191);  exp_63 = sum_191 = None
        view_382 = torch.ops.aten.view.default(div_63, [8, -1]);  div_63 = None
        view_383 = torch.ops.aten.view.default(view_382, [8, -1, 1, 1]);  view_382 = None
        view_384 = torch.ops.aten.view.default(view_383, [8, 2, 512, 1, 1]);  view_383 = None
        mul_867 = torch.ops.aten.mul.Tensor(view_380, view_384);  view_380 = view_384 = None
        sum_192 = torch.ops.aten.sum.dim_IntList(mul_867, [1]);  mul_867 = None
        avg_pool2d_10 = torch.ops.aten.avg_pool2d.default(sum_192, [3, 3], [2, 2], [1, 1]);  sum_192 = None
        convolution_332 = torch.ops.aten.convolution.default(avg_pool2d_10, arg739_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_10 = arg739_1 = None
        add_599 = torch.ops.aten.add.Tensor(arg741_1, 1e-05);  arg741_1 = None
        sqrt_268 = torch.ops.aten.sqrt.default(add_599);  add_599 = None
        reciprocal_268 = torch.ops.aten.reciprocal.default(sqrt_268);  sqrt_268 = None
        mul_868 = torch.ops.aten.mul.Tensor(reciprocal_268, 1);  reciprocal_268 = None
        unsqueeze_2144 = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
        unsqueeze_2145 = torch.ops.aten.unsqueeze.default(unsqueeze_2144, -1);  unsqueeze_2144 = None
        unsqueeze_2146 = torch.ops.aten.unsqueeze.default(mul_868, -1);  mul_868 = None
        unsqueeze_2147 = torch.ops.aten.unsqueeze.default(unsqueeze_2146, -1);  unsqueeze_2146 = None
        sub_332 = torch.ops.aten.sub.Tensor(convolution_332, unsqueeze_2145);  convolution_332 = unsqueeze_2145 = None
        mul_869 = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_2147);  sub_332 = unsqueeze_2147 = None
        unsqueeze_2148 = torch.ops.aten.unsqueeze.default(arg742_1, -1);  arg742_1 = None
        unsqueeze_2149 = torch.ops.aten.unsqueeze.default(unsqueeze_2148, -1);  unsqueeze_2148 = None
        mul_870 = torch.ops.aten.mul.Tensor(mul_869, unsqueeze_2149);  mul_869 = unsqueeze_2149 = None
        unsqueeze_2150 = torch.ops.aten.unsqueeze.default(arg743_1, -1);  arg743_1 = None
        unsqueeze_2151 = torch.ops.aten.unsqueeze.default(unsqueeze_2150, -1);  unsqueeze_2150 = None
        add_600 = torch.ops.aten.add.Tensor(mul_870, unsqueeze_2151);  mul_870 = unsqueeze_2151 = None
        avg_pool2d_11 = torch.ops.aten.avg_pool2d.default(relu_257, [2, 2], [2, 2], [0, 0], True, False);  relu_257 = None
        convolution_333 = torch.ops.aten.convolution.default(avg_pool2d_11, arg744_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_11 = arg744_1 = None
        add_601 = torch.ops.aten.add.Tensor(arg746_1, 1e-05);  arg746_1 = None
        sqrt_269 = torch.ops.aten.sqrt.default(add_601);  add_601 = None
        reciprocal_269 = torch.ops.aten.reciprocal.default(sqrt_269);  sqrt_269 = None
        mul_871 = torch.ops.aten.mul.Tensor(reciprocal_269, 1);  reciprocal_269 = None
        unsqueeze_2152 = torch.ops.aten.unsqueeze.default(arg745_1, -1);  arg745_1 = None
        unsqueeze_2153 = torch.ops.aten.unsqueeze.default(unsqueeze_2152, -1);  unsqueeze_2152 = None
        unsqueeze_2154 = torch.ops.aten.unsqueeze.default(mul_871, -1);  mul_871 = None
        unsqueeze_2155 = torch.ops.aten.unsqueeze.default(unsqueeze_2154, -1);  unsqueeze_2154 = None
        sub_333 = torch.ops.aten.sub.Tensor(convolution_333, unsqueeze_2153);  convolution_333 = unsqueeze_2153 = None
        mul_872 = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_2155);  sub_333 = unsqueeze_2155 = None
        unsqueeze_2156 = torch.ops.aten.unsqueeze.default(arg747_1, -1);  arg747_1 = None
        unsqueeze_2157 = torch.ops.aten.unsqueeze.default(unsqueeze_2156, -1);  unsqueeze_2156 = None
        mul_873 = torch.ops.aten.mul.Tensor(mul_872, unsqueeze_2157);  mul_872 = unsqueeze_2157 = None
        unsqueeze_2158 = torch.ops.aten.unsqueeze.default(arg748_1, -1);  arg748_1 = None
        unsqueeze_2159 = torch.ops.aten.unsqueeze.default(unsqueeze_2158, -1);  unsqueeze_2158 = None
        add_602 = torch.ops.aten.add.Tensor(mul_873, unsqueeze_2159);  mul_873 = unsqueeze_2159 = None
        add_603 = torch.ops.aten.add.Tensor(add_600, add_602);  add_600 = add_602 = None
        relu_261 = torch.ops.aten.relu.default(add_603);  add_603 = None
        convolution_334 = torch.ops.aten.convolution.default(relu_261, arg749_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg749_1 = None
        add_604 = torch.ops.aten.add.Tensor(arg751_1, 1e-05);  arg751_1 = None
        sqrt_270 = torch.ops.aten.sqrt.default(add_604);  add_604 = None
        reciprocal_270 = torch.ops.aten.reciprocal.default(sqrt_270);  sqrt_270 = None
        mul_874 = torch.ops.aten.mul.Tensor(reciprocal_270, 1);  reciprocal_270 = None
        unsqueeze_2160 = torch.ops.aten.unsqueeze.default(arg750_1, -1);  arg750_1 = None
        unsqueeze_2161 = torch.ops.aten.unsqueeze.default(unsqueeze_2160, -1);  unsqueeze_2160 = None
        unsqueeze_2162 = torch.ops.aten.unsqueeze.default(mul_874, -1);  mul_874 = None
        unsqueeze_2163 = torch.ops.aten.unsqueeze.default(unsqueeze_2162, -1);  unsqueeze_2162 = None
        sub_334 = torch.ops.aten.sub.Tensor(convolution_334, unsqueeze_2161);  convolution_334 = unsqueeze_2161 = None
        mul_875 = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_2163);  sub_334 = unsqueeze_2163 = None
        unsqueeze_2164 = torch.ops.aten.unsqueeze.default(arg752_1, -1);  arg752_1 = None
        unsqueeze_2165 = torch.ops.aten.unsqueeze.default(unsqueeze_2164, -1);  unsqueeze_2164 = None
        mul_876 = torch.ops.aten.mul.Tensor(mul_875, unsqueeze_2165);  mul_875 = unsqueeze_2165 = None
        unsqueeze_2166 = torch.ops.aten.unsqueeze.default(arg753_1, -1);  arg753_1 = None
        unsqueeze_2167 = torch.ops.aten.unsqueeze.default(unsqueeze_2166, -1);  unsqueeze_2166 = None
        add_605 = torch.ops.aten.add.Tensor(mul_876, unsqueeze_2167);  mul_876 = unsqueeze_2167 = None
        relu_262 = torch.ops.aten.relu.default(add_605);  add_605 = None
        convolution_335 = torch.ops.aten.convolution.default(relu_262, arg754_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_262 = arg754_1 = None
        add_606 = torch.ops.aten.add.Tensor(arg756_1, 1e-05);  arg756_1 = None
        sqrt_271 = torch.ops.aten.sqrt.default(add_606);  add_606 = None
        reciprocal_271 = torch.ops.aten.reciprocal.default(sqrt_271);  sqrt_271 = None
        mul_877 = torch.ops.aten.mul.Tensor(reciprocal_271, 1);  reciprocal_271 = None
        unsqueeze_2168 = torch.ops.aten.unsqueeze.default(arg755_1, -1);  arg755_1 = None
        unsqueeze_2169 = torch.ops.aten.unsqueeze.default(unsqueeze_2168, -1);  unsqueeze_2168 = None
        unsqueeze_2170 = torch.ops.aten.unsqueeze.default(mul_877, -1);  mul_877 = None
        unsqueeze_2171 = torch.ops.aten.unsqueeze.default(unsqueeze_2170, -1);  unsqueeze_2170 = None
        sub_335 = torch.ops.aten.sub.Tensor(convolution_335, unsqueeze_2169);  convolution_335 = unsqueeze_2169 = None
        mul_878 = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_2171);  sub_335 = unsqueeze_2171 = None
        unsqueeze_2172 = torch.ops.aten.unsqueeze.default(arg757_1, -1);  arg757_1 = None
        unsqueeze_2173 = torch.ops.aten.unsqueeze.default(unsqueeze_2172, -1);  unsqueeze_2172 = None
        mul_879 = torch.ops.aten.mul.Tensor(mul_878, unsqueeze_2173);  mul_878 = unsqueeze_2173 = None
        unsqueeze_2174 = torch.ops.aten.unsqueeze.default(arg758_1, -1);  arg758_1 = None
        unsqueeze_2175 = torch.ops.aten.unsqueeze.default(unsqueeze_2174, -1);  unsqueeze_2174 = None
        add_607 = torch.ops.aten.add.Tensor(mul_879, unsqueeze_2175);  mul_879 = unsqueeze_2175 = None
        relu_263 = torch.ops.aten.relu.default(add_607);  add_607 = None
        view_386 = torch.ops.aten.view.default(relu_263, [8, 2, 512, 8, 8]);  relu_263 = None
        sum_193 = torch.ops.aten.sum.dim_IntList(view_386, [1])
        mean_65 = torch.ops.aten.mean.dim(sum_193, [2, 3], True);  sum_193 = None
        convolution_336 = torch.ops.aten.convolution.default(mean_65, arg759_1, arg760_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_65 = arg759_1 = arg760_1 = None
        add_608 = torch.ops.aten.add.Tensor(arg762_1, 1e-05);  arg762_1 = None
        sqrt_272 = torch.ops.aten.sqrt.default(add_608);  add_608 = None
        reciprocal_272 = torch.ops.aten.reciprocal.default(sqrt_272);  sqrt_272 = None
        mul_880 = torch.ops.aten.mul.Tensor(reciprocal_272, 1);  reciprocal_272 = None
        unsqueeze_2176 = torch.ops.aten.unsqueeze.default(arg761_1, -1);  arg761_1 = None
        unsqueeze_2177 = torch.ops.aten.unsqueeze.default(unsqueeze_2176, -1);  unsqueeze_2176 = None
        unsqueeze_2178 = torch.ops.aten.unsqueeze.default(mul_880, -1);  mul_880 = None
        unsqueeze_2179 = torch.ops.aten.unsqueeze.default(unsqueeze_2178, -1);  unsqueeze_2178 = None
        sub_336 = torch.ops.aten.sub.Tensor(convolution_336, unsqueeze_2177);  convolution_336 = unsqueeze_2177 = None
        mul_881 = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_2179);  sub_336 = unsqueeze_2179 = None
        unsqueeze_2180 = torch.ops.aten.unsqueeze.default(arg763_1, -1);  arg763_1 = None
        unsqueeze_2181 = torch.ops.aten.unsqueeze.default(unsqueeze_2180, -1);  unsqueeze_2180 = None
        mul_882 = torch.ops.aten.mul.Tensor(mul_881, unsqueeze_2181);  mul_881 = unsqueeze_2181 = None
        unsqueeze_2182 = torch.ops.aten.unsqueeze.default(arg764_1, -1);  arg764_1 = None
        unsqueeze_2183 = torch.ops.aten.unsqueeze.default(unsqueeze_2182, -1);  unsqueeze_2182 = None
        add_609 = torch.ops.aten.add.Tensor(mul_882, unsqueeze_2183);  mul_882 = unsqueeze_2183 = None
        relu_264 = torch.ops.aten.relu.default(add_609);  add_609 = None
        convolution_337 = torch.ops.aten.convolution.default(relu_264, arg765_1, arg766_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_264 = arg765_1 = arg766_1 = None
        view_387 = torch.ops.aten.view.default(convolution_337, [8, 1, 2, -1]);  convolution_337 = None
        permute_65 = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        amax_64 = torch.ops.aten.amax.default(permute_65, [1], True)
        sub_337 = torch.ops.aten.sub.Tensor(permute_65, amax_64);  permute_65 = amax_64 = None
        exp_64 = torch.ops.aten.exp.default(sub_337);  sub_337 = None
        sum_194 = torch.ops.aten.sum.dim_IntList(exp_64, [1], True)
        div_64 = torch.ops.aten.div.Tensor(exp_64, sum_194);  exp_64 = sum_194 = None
        view_388 = torch.ops.aten.view.default(div_64, [8, -1]);  div_64 = None
        view_389 = torch.ops.aten.view.default(view_388, [8, -1, 1, 1]);  view_388 = None
        view_390 = torch.ops.aten.view.default(view_389, [8, 2, 512, 1, 1]);  view_389 = None
        mul_883 = torch.ops.aten.mul.Tensor(view_386, view_390);  view_386 = view_390 = None
        sum_195 = torch.ops.aten.sum.dim_IntList(mul_883, [1]);  mul_883 = None
        convolution_338 = torch.ops.aten.convolution.default(sum_195, arg767_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_195 = arg767_1 = None
        add_610 = torch.ops.aten.add.Tensor(arg769_1, 1e-05);  arg769_1 = None
        sqrt_273 = torch.ops.aten.sqrt.default(add_610);  add_610 = None
        reciprocal_273 = torch.ops.aten.reciprocal.default(sqrt_273);  sqrt_273 = None
        mul_884 = torch.ops.aten.mul.Tensor(reciprocal_273, 1);  reciprocal_273 = None
        unsqueeze_2184 = torch.ops.aten.unsqueeze.default(arg768_1, -1);  arg768_1 = None
        unsqueeze_2185 = torch.ops.aten.unsqueeze.default(unsqueeze_2184, -1);  unsqueeze_2184 = None
        unsqueeze_2186 = torch.ops.aten.unsqueeze.default(mul_884, -1);  mul_884 = None
        unsqueeze_2187 = torch.ops.aten.unsqueeze.default(unsqueeze_2186, -1);  unsqueeze_2186 = None
        sub_338 = torch.ops.aten.sub.Tensor(convolution_338, unsqueeze_2185);  convolution_338 = unsqueeze_2185 = None
        mul_885 = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_2187);  sub_338 = unsqueeze_2187 = None
        unsqueeze_2188 = torch.ops.aten.unsqueeze.default(arg770_1, -1);  arg770_1 = None
        unsqueeze_2189 = torch.ops.aten.unsqueeze.default(unsqueeze_2188, -1);  unsqueeze_2188 = None
        mul_886 = torch.ops.aten.mul.Tensor(mul_885, unsqueeze_2189);  mul_885 = unsqueeze_2189 = None
        unsqueeze_2190 = torch.ops.aten.unsqueeze.default(arg771_1, -1);  arg771_1 = None
        unsqueeze_2191 = torch.ops.aten.unsqueeze.default(unsqueeze_2190, -1);  unsqueeze_2190 = None
        add_611 = torch.ops.aten.add.Tensor(mul_886, unsqueeze_2191);  mul_886 = unsqueeze_2191 = None
        add_612 = torch.ops.aten.add.Tensor(add_611, relu_261);  add_611 = relu_261 = None
        relu_265 = torch.ops.aten.relu.default(add_612);  add_612 = None
        convolution_339 = torch.ops.aten.convolution.default(relu_265, arg772_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg772_1 = None
        add_613 = torch.ops.aten.add.Tensor(arg774_1, 1e-05);  arg774_1 = None
        sqrt_274 = torch.ops.aten.sqrt.default(add_613);  add_613 = None
        reciprocal_274 = torch.ops.aten.reciprocal.default(sqrt_274);  sqrt_274 = None
        mul_887 = torch.ops.aten.mul.Tensor(reciprocal_274, 1);  reciprocal_274 = None
        unsqueeze_2192 = torch.ops.aten.unsqueeze.default(arg773_1, -1);  arg773_1 = None
        unsqueeze_2193 = torch.ops.aten.unsqueeze.default(unsqueeze_2192, -1);  unsqueeze_2192 = None
        unsqueeze_2194 = torch.ops.aten.unsqueeze.default(mul_887, -1);  mul_887 = None
        unsqueeze_2195 = torch.ops.aten.unsqueeze.default(unsqueeze_2194, -1);  unsqueeze_2194 = None
        sub_339 = torch.ops.aten.sub.Tensor(convolution_339, unsqueeze_2193);  convolution_339 = unsqueeze_2193 = None
        mul_888 = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_2195);  sub_339 = unsqueeze_2195 = None
        unsqueeze_2196 = torch.ops.aten.unsqueeze.default(arg775_1, -1);  arg775_1 = None
        unsqueeze_2197 = torch.ops.aten.unsqueeze.default(unsqueeze_2196, -1);  unsqueeze_2196 = None
        mul_889 = torch.ops.aten.mul.Tensor(mul_888, unsqueeze_2197);  mul_888 = unsqueeze_2197 = None
        unsqueeze_2198 = torch.ops.aten.unsqueeze.default(arg776_1, -1);  arg776_1 = None
        unsqueeze_2199 = torch.ops.aten.unsqueeze.default(unsqueeze_2198, -1);  unsqueeze_2198 = None
        add_614 = torch.ops.aten.add.Tensor(mul_889, unsqueeze_2199);  mul_889 = unsqueeze_2199 = None
        relu_266 = torch.ops.aten.relu.default(add_614);  add_614 = None
        convolution_340 = torch.ops.aten.convolution.default(relu_266, arg777_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_266 = arg777_1 = None
        add_615 = torch.ops.aten.add.Tensor(arg779_1, 1e-05);  arg779_1 = None
        sqrt_275 = torch.ops.aten.sqrt.default(add_615);  add_615 = None
        reciprocal_275 = torch.ops.aten.reciprocal.default(sqrt_275);  sqrt_275 = None
        mul_890 = torch.ops.aten.mul.Tensor(reciprocal_275, 1);  reciprocal_275 = None
        unsqueeze_2200 = torch.ops.aten.unsqueeze.default(arg778_1, -1);  arg778_1 = None
        unsqueeze_2201 = torch.ops.aten.unsqueeze.default(unsqueeze_2200, -1);  unsqueeze_2200 = None
        unsqueeze_2202 = torch.ops.aten.unsqueeze.default(mul_890, -1);  mul_890 = None
        unsqueeze_2203 = torch.ops.aten.unsqueeze.default(unsqueeze_2202, -1);  unsqueeze_2202 = None
        sub_340 = torch.ops.aten.sub.Tensor(convolution_340, unsqueeze_2201);  convolution_340 = unsqueeze_2201 = None
        mul_891 = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_2203);  sub_340 = unsqueeze_2203 = None
        unsqueeze_2204 = torch.ops.aten.unsqueeze.default(arg780_1, -1);  arg780_1 = None
        unsqueeze_2205 = torch.ops.aten.unsqueeze.default(unsqueeze_2204, -1);  unsqueeze_2204 = None
        mul_892 = torch.ops.aten.mul.Tensor(mul_891, unsqueeze_2205);  mul_891 = unsqueeze_2205 = None
        unsqueeze_2206 = torch.ops.aten.unsqueeze.default(arg781_1, -1);  arg781_1 = None
        unsqueeze_2207 = torch.ops.aten.unsqueeze.default(unsqueeze_2206, -1);  unsqueeze_2206 = None
        add_616 = torch.ops.aten.add.Tensor(mul_892, unsqueeze_2207);  mul_892 = unsqueeze_2207 = None
        relu_267 = torch.ops.aten.relu.default(add_616);  add_616 = None
        view_392 = torch.ops.aten.view.default(relu_267, [8, 2, 512, 8, 8]);  relu_267 = None
        sum_196 = torch.ops.aten.sum.dim_IntList(view_392, [1])
        mean_66 = torch.ops.aten.mean.dim(sum_196, [2, 3], True);  sum_196 = None
        convolution_341 = torch.ops.aten.convolution.default(mean_66, arg782_1, arg783_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_66 = arg782_1 = arg783_1 = None
        add_617 = torch.ops.aten.add.Tensor(arg785_1, 1e-05);  arg785_1 = None
        sqrt_276 = torch.ops.aten.sqrt.default(add_617);  add_617 = None
        reciprocal_276 = torch.ops.aten.reciprocal.default(sqrt_276);  sqrt_276 = None
        mul_893 = torch.ops.aten.mul.Tensor(reciprocal_276, 1);  reciprocal_276 = None
        unsqueeze_2208 = torch.ops.aten.unsqueeze.default(arg784_1, -1);  arg784_1 = None
        unsqueeze_2209 = torch.ops.aten.unsqueeze.default(unsqueeze_2208, -1);  unsqueeze_2208 = None
        unsqueeze_2210 = torch.ops.aten.unsqueeze.default(mul_893, -1);  mul_893 = None
        unsqueeze_2211 = torch.ops.aten.unsqueeze.default(unsqueeze_2210, -1);  unsqueeze_2210 = None
        sub_341 = torch.ops.aten.sub.Tensor(convolution_341, unsqueeze_2209);  convolution_341 = unsqueeze_2209 = None
        mul_894 = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_2211);  sub_341 = unsqueeze_2211 = None
        unsqueeze_2212 = torch.ops.aten.unsqueeze.default(arg786_1, -1);  arg786_1 = None
        unsqueeze_2213 = torch.ops.aten.unsqueeze.default(unsqueeze_2212, -1);  unsqueeze_2212 = None
        mul_895 = torch.ops.aten.mul.Tensor(mul_894, unsqueeze_2213);  mul_894 = unsqueeze_2213 = None
        unsqueeze_2214 = torch.ops.aten.unsqueeze.default(arg787_1, -1);  arg787_1 = None
        unsqueeze_2215 = torch.ops.aten.unsqueeze.default(unsqueeze_2214, -1);  unsqueeze_2214 = None
        add_618 = torch.ops.aten.add.Tensor(mul_895, unsqueeze_2215);  mul_895 = unsqueeze_2215 = None
        relu_268 = torch.ops.aten.relu.default(add_618);  add_618 = None
        convolution_342 = torch.ops.aten.convolution.default(relu_268, arg788_1, arg789_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_268 = arg788_1 = arg789_1 = None
        view_393 = torch.ops.aten.view.default(convolution_342, [8, 1, 2, -1]);  convolution_342 = None
        permute_66 = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        amax_65 = torch.ops.aten.amax.default(permute_66, [1], True)
        sub_342 = torch.ops.aten.sub.Tensor(permute_66, amax_65);  permute_66 = amax_65 = None
        exp_65 = torch.ops.aten.exp.default(sub_342);  sub_342 = None
        sum_197 = torch.ops.aten.sum.dim_IntList(exp_65, [1], True)
        div_65 = torch.ops.aten.div.Tensor(exp_65, sum_197);  exp_65 = sum_197 = None
        view_394 = torch.ops.aten.view.default(div_65, [8, -1]);  div_65 = None
        view_395 = torch.ops.aten.view.default(view_394, [8, -1, 1, 1]);  view_394 = None
        view_396 = torch.ops.aten.view.default(view_395, [8, 2, 512, 1, 1]);  view_395 = None
        mul_896 = torch.ops.aten.mul.Tensor(view_392, view_396);  view_392 = view_396 = None
        sum_198 = torch.ops.aten.sum.dim_IntList(mul_896, [1]);  mul_896 = None
        convolution_343 = torch.ops.aten.convolution.default(sum_198, arg790_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_198 = arg790_1 = None
        add_619 = torch.ops.aten.add.Tensor(arg792_1, 1e-05);  arg792_1 = None
        sqrt_277 = torch.ops.aten.sqrt.default(add_619);  add_619 = None
        reciprocal_277 = torch.ops.aten.reciprocal.default(sqrt_277);  sqrt_277 = None
        mul_897 = torch.ops.aten.mul.Tensor(reciprocal_277, 1);  reciprocal_277 = None
        unsqueeze_2216 = torch.ops.aten.unsqueeze.default(arg791_1, -1);  arg791_1 = None
        unsqueeze_2217 = torch.ops.aten.unsqueeze.default(unsqueeze_2216, -1);  unsqueeze_2216 = None
        unsqueeze_2218 = torch.ops.aten.unsqueeze.default(mul_897, -1);  mul_897 = None
        unsqueeze_2219 = torch.ops.aten.unsqueeze.default(unsqueeze_2218, -1);  unsqueeze_2218 = None
        sub_343 = torch.ops.aten.sub.Tensor(convolution_343, unsqueeze_2217);  convolution_343 = unsqueeze_2217 = None
        mul_898 = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_2219);  sub_343 = unsqueeze_2219 = None
        unsqueeze_2220 = torch.ops.aten.unsqueeze.default(arg793_1, -1);  arg793_1 = None
        unsqueeze_2221 = torch.ops.aten.unsqueeze.default(unsqueeze_2220, -1);  unsqueeze_2220 = None
        mul_899 = torch.ops.aten.mul.Tensor(mul_898, unsqueeze_2221);  mul_898 = unsqueeze_2221 = None
        unsqueeze_2222 = torch.ops.aten.unsqueeze.default(arg794_1, -1);  arg794_1 = None
        unsqueeze_2223 = torch.ops.aten.unsqueeze.default(unsqueeze_2222, -1);  unsqueeze_2222 = None
        add_620 = torch.ops.aten.add.Tensor(mul_899, unsqueeze_2223);  mul_899 = unsqueeze_2223 = None
        add_621 = torch.ops.aten.add.Tensor(add_620, relu_265);  add_620 = relu_265 = None
        relu_269 = torch.ops.aten.relu.default(add_621);  add_621 = None
        mean_67 = torch.ops.aten.mean.dim(relu_269, [-1, -2], True);  relu_269 = None
        view_397 = torch.ops.aten.view.default(mean_67, [8, 2048]);  mean_67 = None
        permute_67 = torch.ops.aten.permute.default(arg795_1, [1, 0]);  arg795_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg796_1, view_397, permute_67);  arg796_1 = view_397 = permute_67 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 256, 256), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 64, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128, 64, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64, 128, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf18, (64,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf19, (64,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf20, (64,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128, 32, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf24, (128,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf26, (32, 64, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf27, (32,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf28, (32,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf29, (32,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf30, (32,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf31, (32,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128, 32, 1, 1), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256, 64, 1, 1), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256, 128, 1, 1), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64, 256, 1, 1), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf49, (128, 32, 3, 3), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf50, (128,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf51, (128,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf52, (128,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf53, (128,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf54, (32, 64, 1, 1), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf55, (32,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf56, (32,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf57, (32,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf58, (32,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf59, (32,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf60, (128, 32, 1, 1), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf62, (256, 64, 1, 1), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf63, (256,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf64, (256,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf65, (256,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf67, (64, 256, 1, 1), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf68, (64,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf69, (64,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf70, (64,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf71, (64,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf72, (128, 32, 3, 3), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf73, (128,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf77, (32, 64, 1, 1), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf78, (32,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf79, (32,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf80, (32,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf81, (32,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf82, (32,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128, 32, 1, 1), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256, 64, 1, 1), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf86, (256,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf87, (256,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf88, (256,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf89, (256,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128, 256, 1, 1), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf94, (128,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf95, (256, 64, 3, 3), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf96, (256,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf97, (256,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf98, (256,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf99, (256,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf100, (64, 128, 1, 1), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf101, (64,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf102, (64,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf103, (64,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf104, (64,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf105, (64,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf106, (256, 64, 1, 1), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf107, (256,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf108, (512, 128, 1, 1), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512, 256, 1, 1), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf118, (128, 512, 1, 1), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf119, (128,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf120, (128,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf121, (128,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf122, (128,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256, 64, 3, 3), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf126, (256,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf128, (64, 128, 1, 1), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf129, (64,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf130, (64,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf131, (64,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf132, (64,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf133, (64,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256, 64, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf135, (256,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf136, (512, 128, 1, 1), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf141, (128, 512, 1, 1), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf142, (128,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf143, (128,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf144, (128,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf145, (128,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf146, (256, 64, 3, 3), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf147, (256,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf148, (256,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf149, (256,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf150, (256,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf151, (64, 128, 1, 1), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf152, (64,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf153, (64,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf154, (64,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf155, (64,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf156, (64,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf157, (256, 64, 1, 1), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf158, (256,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf159, (512, 128, 1, 1), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf160, (512,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf161, (512,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf164, (128, 512, 1, 1), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf165, (128,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf166, (128,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf167, (128,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf168, (128,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf169, (256, 64, 3, 3), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf170, (256,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf171, (256,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf172, (256,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf173, (256,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf174, (64, 128, 1, 1), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf175, (64,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf176, (64,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf177, (64,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf178, (64,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf179, (64,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf180, (256, 64, 1, 1), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf181, (256,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512, 128, 1, 1), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf183, (512,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf184, (512,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf185, (512,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf187, (256, 512, 1, 1), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf188, (256,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf189, (256,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf190, (256,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf191, (256,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512, 128, 3, 3), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf194, (512,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf195, (512,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf196, (512,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf197, (128, 256, 1, 1), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf198, (128,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf199, (128,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf200, (128,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf201, (128,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf202, (128,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf203, (512, 128, 1, 1), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1024, 256, 1, 1), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1024,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf207, (1024,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf208, (1024,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf209, (1024,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf210, (1024, 512, 1, 1), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf211, (1024,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1024,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1024,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1024,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf215, (256, 1024, 1, 1), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf216, (256,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf217, (256,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf218, (256,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf219, (256,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf220, (512, 128, 3, 3), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf222, (512,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf224, (512,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf225, (128, 256, 1, 1), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf226, (128,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf227, (128,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf228, (128,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf229, (128,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf230, (128,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf231, (512, 128, 1, 1), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf232, (512,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1024, 256, 1, 1), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf234, (1024,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf235, (1024,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf236, (1024,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf237, (1024,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf238, (256, 1024, 1, 1), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf239, (256,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf240, (256,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf241, (256,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf242, (256,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf243, (512, 128, 3, 3), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf244, (512,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf245, (512,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf246, (512,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf247, (512,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf248, (128, 256, 1, 1), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf249, (128,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf250, (128,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf251, (128,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf252, (128,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf253, (128,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf254, (512, 128, 1, 1), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf255, (512,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1024, 256, 1, 1), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf257, (1024,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf258, (1024,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1024,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1024,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf261, (256, 1024, 1, 1), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf262, (256,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf263, (256,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf264, (256,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf265, (256,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf266, (512, 128, 3, 3), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf267, (512,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf268, (512,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf269, (512,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf270, (512,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf271, (128, 256, 1, 1), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf272, (128,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf273, (128,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf274, (128,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf275, (128,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf276, (128,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf277, (512, 128, 1, 1), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf278, (512,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1024, 256, 1, 1), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf280, (1024,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf281, (1024,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf282, (1024,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf283, (1024,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf284, (256, 1024, 1, 1), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf285, (256,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf286, (256,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf287, (256,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf288, (256,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf289, (512, 128, 3, 3), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf290, (512,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf291, (512,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf292, (512,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf293, (512,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf294, (128, 256, 1, 1), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf295, (128,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf296, (128,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf297, (128,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf298, (128,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf299, (128,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf300, (512, 128, 1, 1), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf301, (512,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf302, (1024, 256, 1, 1), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1024,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf304, (1024,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1024,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1024,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf307, (256, 1024, 1, 1), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf308, (256,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf309, (256,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf310, (256,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf311, (256,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf312, (512, 128, 3, 3), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf313, (512,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf314, (512,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf315, (512,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf316, (512,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf317, (128, 256, 1, 1), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf318, (128,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf319, (128,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf320, (128,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf321, (128,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf322, (128,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf323, (512, 128, 1, 1), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf324, (512,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1024, 256, 1, 1), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf326, (1024,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf327, (1024,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf328, (1024,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf329, (1024,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf330, (256, 1024, 1, 1), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf331, (256,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf332, (256,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf333, (256,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf334, (256,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf335, (512, 128, 3, 3), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf336, (512,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf337, (512,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf338, (512,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf339, (512,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf340, (128, 256, 1, 1), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf341, (128,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf342, (128,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf343, (128,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf344, (128,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf345, (128,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf346, (512, 128, 1, 1), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf347, (512,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1024, 256, 1, 1), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1024,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1024,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1024,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf352, (1024,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf353, (256, 1024, 1, 1), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf354, (256,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf355, (256,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf356, (256,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf357, (256,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf358, (512, 128, 3, 3), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf359, (512,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf360, (512,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf361, (512,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf362, (512,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf363, (128, 256, 1, 1), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf364, (128,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf365, (128,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf366, (128,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf367, (128,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf368, (128,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf369, (512, 128, 1, 1), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf370, (512,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf371, (1024, 256, 1, 1), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf372, (1024,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf373, (1024,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf374, (1024,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf375, (1024,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf376, (256, 1024, 1, 1), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf377, (256,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf378, (256,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf379, (256,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf380, (256,), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf381, (512, 128, 3, 3), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf382, (512,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf383, (512,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf384, (512,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf385, (512,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf386, (128, 256, 1, 1), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf387, (128,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf388, (128,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf389, (128,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf390, (128,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf391, (128,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf392, (512, 128, 1, 1), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf393, (512,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf394, (1024, 256, 1, 1), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf395, (1024,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf396, (1024,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf397, (1024,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf398, (1024,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf399, (256, 1024, 1, 1), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf400, (256,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf401, (256,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf402, (256,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf403, (256,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf404, (512, 128, 3, 3), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf405, (512,), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf406, (512,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf407, (512,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf408, (512,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf409, (128, 256, 1, 1), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf410, (128,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf411, (128,), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf412, (128,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf413, (128,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf414, (128,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf415, (512, 128, 1, 1), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf416, (512,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf417, (1024, 256, 1, 1), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf418, (1024,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf419, (1024,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf420, (1024,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf421, (1024,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf422, (256, 1024, 1, 1), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf423, (256,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf424, (256,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf425, (256,), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf426, (256,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf427, (512, 128, 3, 3), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf428, (512,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf429, (512,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf430, (512,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf431, (512,), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf432, (128, 256, 1, 1), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf433, (128,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf434, (128,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf435, (128,), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf436, (128,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf437, (128,), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf438, (512, 128, 1, 1), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf439, (512,), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf440, (1024, 256, 1, 1), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf441, (1024,), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf442, (1024,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf443, (1024,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf444, (1024,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf445, (256, 1024, 1, 1), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf446, (256,), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf447, (256,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf448, (256,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf449, (256,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf450, (512, 128, 3, 3), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf451, (512,), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf452, (512,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf453, (512,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf454, (512,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf455, (128, 256, 1, 1), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf456, (128,), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf457, (128,), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf458, (128,), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf459, (128,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf460, (128,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf461, (512, 128, 1, 1), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf462, (512,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf463, (1024, 256, 1, 1), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf464, (1024,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf465, (1024,), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf466, (1024,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf467, (1024,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf468, (256, 1024, 1, 1), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf469, (256,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf470, (256,), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf471, (256,), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf472, (256,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf473, (512, 128, 3, 3), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf474, (512,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf475, (512,), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf476, (512,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf477, (512,), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf478, (128, 256, 1, 1), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf479, (128,), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf480, (128,), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf481, (128,), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf482, (128,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf483, (128,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf484, (512, 128, 1, 1), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf485, (512,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf486, (1024, 256, 1, 1), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf487, (1024,), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf488, (1024,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf489, (1024,), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf490, (1024,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf491, (256, 1024, 1, 1), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf492, (256,), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf493, (256,), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf494, (256,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf495, (256,), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf496, (512, 128, 3, 3), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf497, (512,), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf498, (512,), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf499, (512,), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf500, (512,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf501, (128, 256, 1, 1), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf502, (128,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf503, (128,), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf504, (128,), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf505, (128,), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf506, (128,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf507, (512, 128, 1, 1), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf508, (512,), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf509, (1024, 256, 1, 1), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf510, (1024,), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf511, (1024,), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf512, (1024,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf513, (1024,), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf514, (256, 1024, 1, 1), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf515, (256,), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf516, (256,), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf517, (256,), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf518, (256,), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf519, (512, 128, 3, 3), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf520, (512,), is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf521, (512,), is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf522, (512,), is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf523, (512,), is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf524, (128, 256, 1, 1), is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf525, (128,), is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf526, (128,), is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf527, (128,), is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf528, (128,), is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf529, (128,), is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf530, (512, 128, 1, 1), is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf531, (512,), is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf532, (1024, 256, 1, 1), is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf533, (1024,), is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf534, (1024,), is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf535, (1024,), is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf536, (1024,), is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf537, (256, 1024, 1, 1), is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf538, (256,), is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf539, (256,), is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf540, (256,), is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf541, (256,), is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf542, (512, 128, 3, 3), is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf543, (512,), is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf544, (512,), is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf545, (512,), is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf546, (512,), is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf547, (128, 256, 1, 1), is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf548, (128,), is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf549, (128,), is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf550, (128,), is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf551, (128,), is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf552, (128,), is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf553, (512, 128, 1, 1), is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf554, (512,), is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf555, (1024, 256, 1, 1), is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf556, (1024,), is_leaf=True)  # arg556_1
    buf557 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf557, (1024,), is_leaf=True)  # arg557_1
    buf558 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf558, (1024,), is_leaf=True)  # arg558_1
    buf559 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf559, (1024,), is_leaf=True)  # arg559_1
    buf560 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf560, (256, 1024, 1, 1), is_leaf=True)  # arg560_1
    buf561 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf561, (256,), is_leaf=True)  # arg561_1
    buf562 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf562, (256,), is_leaf=True)  # arg562_1
    buf563 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf563, (256,), is_leaf=True)  # arg563_1
    buf564 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf564, (256,), is_leaf=True)  # arg564_1
    buf565 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf565, (512, 128, 3, 3), is_leaf=True)  # arg565_1
    buf566 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf566, (512,), is_leaf=True)  # arg566_1
    buf567 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf567, (512,), is_leaf=True)  # arg567_1
    buf568 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf568, (512,), is_leaf=True)  # arg568_1
    buf569 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf569, (512,), is_leaf=True)  # arg569_1
    buf570 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf570, (128, 256, 1, 1), is_leaf=True)  # arg570_1
    buf571 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf571, (128,), is_leaf=True)  # arg571_1
    buf572 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf572, (128,), is_leaf=True)  # arg572_1
    buf573 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf573, (128,), is_leaf=True)  # arg573_1
    buf574 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf574, (128,), is_leaf=True)  # arg574_1
    buf575 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf575, (128,), is_leaf=True)  # arg575_1
    buf576 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf576, (512, 128, 1, 1), is_leaf=True)  # arg576_1
    buf577 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf577, (512,), is_leaf=True)  # arg577_1
    buf578 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf578, (1024, 256, 1, 1), is_leaf=True)  # arg578_1
    buf579 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf579, (1024,), is_leaf=True)  # arg579_1
    buf580 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf580, (1024,), is_leaf=True)  # arg580_1
    buf581 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf581, (1024,), is_leaf=True)  # arg581_1
    buf582 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf582, (1024,), is_leaf=True)  # arg582_1
    buf583 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf583, (256, 1024, 1, 1), is_leaf=True)  # arg583_1
    buf584 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf584, (256,), is_leaf=True)  # arg584_1
    buf585 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf585, (256,), is_leaf=True)  # arg585_1
    buf586 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf586, (256,), is_leaf=True)  # arg586_1
    buf587 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf587, (256,), is_leaf=True)  # arg587_1
    buf588 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf588, (512, 128, 3, 3), is_leaf=True)  # arg588_1
    buf589 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf589, (512,), is_leaf=True)  # arg589_1
    buf590 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf590, (512,), is_leaf=True)  # arg590_1
    buf591 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf591, (512,), is_leaf=True)  # arg591_1
    buf592 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf592, (512,), is_leaf=True)  # arg592_1
    buf593 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf593, (128, 256, 1, 1), is_leaf=True)  # arg593_1
    buf594 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf594, (128,), is_leaf=True)  # arg594_1
    buf595 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf595, (128,), is_leaf=True)  # arg595_1
    buf596 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf596, (128,), is_leaf=True)  # arg596_1
    buf597 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf597, (128,), is_leaf=True)  # arg597_1
    buf598 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf598, (128,), is_leaf=True)  # arg598_1
    buf599 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf599, (512, 128, 1, 1), is_leaf=True)  # arg599_1
    buf600 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf600, (512,), is_leaf=True)  # arg600_1
    buf601 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf601, (1024, 256, 1, 1), is_leaf=True)  # arg601_1
    buf602 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf602, (1024,), is_leaf=True)  # arg602_1
    buf603 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf603, (1024,), is_leaf=True)  # arg603_1
    buf604 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf604, (1024,), is_leaf=True)  # arg604_1
    buf605 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf605, (1024,), is_leaf=True)  # arg605_1
    buf606 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf606, (256, 1024, 1, 1), is_leaf=True)  # arg606_1
    buf607 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf607, (256,), is_leaf=True)  # arg607_1
    buf608 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf608, (256,), is_leaf=True)  # arg608_1
    buf609 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf609, (256,), is_leaf=True)  # arg609_1
    buf610 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf610, (256,), is_leaf=True)  # arg610_1
    buf611 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf611, (512, 128, 3, 3), is_leaf=True)  # arg611_1
    buf612 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf612, (512,), is_leaf=True)  # arg612_1
    buf613 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf613, (512,), is_leaf=True)  # arg613_1
    buf614 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf614, (512,), is_leaf=True)  # arg614_1
    buf615 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf615, (512,), is_leaf=True)  # arg615_1
    buf616 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf616, (128, 256, 1, 1), is_leaf=True)  # arg616_1
    buf617 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf617, (128,), is_leaf=True)  # arg617_1
    buf618 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf618, (128,), is_leaf=True)  # arg618_1
    buf619 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf619, (128,), is_leaf=True)  # arg619_1
    buf620 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf620, (128,), is_leaf=True)  # arg620_1
    buf621 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf621, (128,), is_leaf=True)  # arg621_1
    buf622 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf622, (512, 128, 1, 1), is_leaf=True)  # arg622_1
    buf623 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf623, (512,), is_leaf=True)  # arg623_1
    buf624 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf624, (1024, 256, 1, 1), is_leaf=True)  # arg624_1
    buf625 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf625, (1024,), is_leaf=True)  # arg625_1
    buf626 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf626, (1024,), is_leaf=True)  # arg626_1
    buf627 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf627, (1024,), is_leaf=True)  # arg627_1
    buf628 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf628, (1024,), is_leaf=True)  # arg628_1
    buf629 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf629, (256, 1024, 1, 1), is_leaf=True)  # arg629_1
    buf630 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf630, (256,), is_leaf=True)  # arg630_1
    buf631 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf631, (256,), is_leaf=True)  # arg631_1
    buf632 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf632, (256,), is_leaf=True)  # arg632_1
    buf633 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf633, (256,), is_leaf=True)  # arg633_1
    buf634 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf634, (512, 128, 3, 3), is_leaf=True)  # arg634_1
    buf635 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf635, (512,), is_leaf=True)  # arg635_1
    buf636 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf636, (512,), is_leaf=True)  # arg636_1
    buf637 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf637, (512,), is_leaf=True)  # arg637_1
    buf638 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf638, (512,), is_leaf=True)  # arg638_1
    buf639 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf639, (128, 256, 1, 1), is_leaf=True)  # arg639_1
    buf640 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf640, (128,), is_leaf=True)  # arg640_1
    buf641 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf641, (128,), is_leaf=True)  # arg641_1
    buf642 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf642, (128,), is_leaf=True)  # arg642_1
    buf643 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf643, (128,), is_leaf=True)  # arg643_1
    buf644 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf644, (128,), is_leaf=True)  # arg644_1
    buf645 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf645, (512, 128, 1, 1), is_leaf=True)  # arg645_1
    buf646 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf646, (512,), is_leaf=True)  # arg646_1
    buf647 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf647, (1024, 256, 1, 1), is_leaf=True)  # arg647_1
    buf648 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf648, (1024,), is_leaf=True)  # arg648_1
    buf649 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf649, (1024,), is_leaf=True)  # arg649_1
    buf650 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf650, (1024,), is_leaf=True)  # arg650_1
    buf651 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf651, (1024,), is_leaf=True)  # arg651_1
    buf652 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf652, (256, 1024, 1, 1), is_leaf=True)  # arg652_1
    buf653 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf653, (256,), is_leaf=True)  # arg653_1
    buf654 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf654, (256,), is_leaf=True)  # arg654_1
    buf655 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf655, (256,), is_leaf=True)  # arg655_1
    buf656 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf656, (256,), is_leaf=True)  # arg656_1
    buf657 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf657, (512, 128, 3, 3), is_leaf=True)  # arg657_1
    buf658 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf658, (512,), is_leaf=True)  # arg658_1
    buf659 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf659, (512,), is_leaf=True)  # arg659_1
    buf660 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf660, (512,), is_leaf=True)  # arg660_1
    buf661 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf661, (512,), is_leaf=True)  # arg661_1
    buf662 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf662, (128, 256, 1, 1), is_leaf=True)  # arg662_1
    buf663 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf663, (128,), is_leaf=True)  # arg663_1
    buf664 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf664, (128,), is_leaf=True)  # arg664_1
    buf665 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf665, (128,), is_leaf=True)  # arg665_1
    buf666 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf666, (128,), is_leaf=True)  # arg666_1
    buf667 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf667, (128,), is_leaf=True)  # arg667_1
    buf668 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf668, (512, 128, 1, 1), is_leaf=True)  # arg668_1
    buf669 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf669, (512,), is_leaf=True)  # arg669_1
    buf670 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf670, (1024, 256, 1, 1), is_leaf=True)  # arg670_1
    buf671 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf671, (1024,), is_leaf=True)  # arg671_1
    buf672 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf672, (1024,), is_leaf=True)  # arg672_1
    buf673 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf673, (1024,), is_leaf=True)  # arg673_1
    buf674 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf674, (1024,), is_leaf=True)  # arg674_1
    buf675 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf675, (256, 1024, 1, 1), is_leaf=True)  # arg675_1
    buf676 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf676, (256,), is_leaf=True)  # arg676_1
    buf677 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf677, (256,), is_leaf=True)  # arg677_1
    buf678 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf678, (256,), is_leaf=True)  # arg678_1
    buf679 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf679, (256,), is_leaf=True)  # arg679_1
    buf680 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf680, (512, 128, 3, 3), is_leaf=True)  # arg680_1
    buf681 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf681, (512,), is_leaf=True)  # arg681_1
    buf682 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf682, (512,), is_leaf=True)  # arg682_1
    buf683 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf683, (512,), is_leaf=True)  # arg683_1
    buf684 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf684, (512,), is_leaf=True)  # arg684_1
    buf685 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf685, (128, 256, 1, 1), is_leaf=True)  # arg685_1
    buf686 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf686, (128,), is_leaf=True)  # arg686_1
    buf687 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf687, (128,), is_leaf=True)  # arg687_1
    buf688 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf688, (128,), is_leaf=True)  # arg688_1
    buf689 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf689, (128,), is_leaf=True)  # arg689_1
    buf690 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf690, (128,), is_leaf=True)  # arg690_1
    buf691 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf691, (512, 128, 1, 1), is_leaf=True)  # arg691_1
    buf692 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf692, (512,), is_leaf=True)  # arg692_1
    buf693 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf693, (1024, 256, 1, 1), is_leaf=True)  # arg693_1
    buf694 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf694, (1024,), is_leaf=True)  # arg694_1
    buf695 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf695, (1024,), is_leaf=True)  # arg695_1
    buf696 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf696, (1024,), is_leaf=True)  # arg696_1
    buf697 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf697, (1024,), is_leaf=True)  # arg697_1
    buf698 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf698, (256, 1024, 1, 1), is_leaf=True)  # arg698_1
    buf699 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf699, (256,), is_leaf=True)  # arg699_1
    buf700 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf700, (256,), is_leaf=True)  # arg700_1
    buf701 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf701, (256,), is_leaf=True)  # arg701_1
    buf702 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf702, (256,), is_leaf=True)  # arg702_1
    buf703 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf703, (512, 128, 3, 3), is_leaf=True)  # arg703_1
    buf704 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf704, (512,), is_leaf=True)  # arg704_1
    buf705 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf705, (512,), is_leaf=True)  # arg705_1
    buf706 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf706, (512,), is_leaf=True)  # arg706_1
    buf707 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf707, (512,), is_leaf=True)  # arg707_1
    buf708 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf708, (128, 256, 1, 1), is_leaf=True)  # arg708_1
    buf709 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf709, (128,), is_leaf=True)  # arg709_1
    buf710 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf710, (128,), is_leaf=True)  # arg710_1
    buf711 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf711, (128,), is_leaf=True)  # arg711_1
    buf712 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf712, (128,), is_leaf=True)  # arg712_1
    buf713 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf713, (128,), is_leaf=True)  # arg713_1
    buf714 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf714, (512, 128, 1, 1), is_leaf=True)  # arg714_1
    buf715 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf715, (512,), is_leaf=True)  # arg715_1
    buf716 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf716, (1024, 256, 1, 1), is_leaf=True)  # arg716_1
    buf717 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf717, (1024,), is_leaf=True)  # arg717_1
    buf718 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf718, (1024,), is_leaf=True)  # arg718_1
    buf719 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf719, (1024,), is_leaf=True)  # arg719_1
    buf720 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf720, (1024,), is_leaf=True)  # arg720_1
    buf721 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf721, (512, 1024, 1, 1), is_leaf=True)  # arg721_1
    buf722 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf722, (512,), is_leaf=True)  # arg722_1
    buf723 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf723, (512,), is_leaf=True)  # arg723_1
    buf724 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf724, (512,), is_leaf=True)  # arg724_1
    buf725 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf725, (512,), is_leaf=True)  # arg725_1
    buf726 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf726, (1024, 256, 3, 3), is_leaf=True)  # arg726_1
    buf727 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf727, (1024,), is_leaf=True)  # arg727_1
    buf728 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf728, (1024,), is_leaf=True)  # arg728_1
    buf729 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf729, (1024,), is_leaf=True)  # arg729_1
    buf730 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf730, (1024,), is_leaf=True)  # arg730_1
    buf731 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf731, (256, 512, 1, 1), is_leaf=True)  # arg731_1
    buf732 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf732, (256,), is_leaf=True)  # arg732_1
    buf733 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf733, (256,), is_leaf=True)  # arg733_1
    buf734 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf734, (256,), is_leaf=True)  # arg734_1
    buf735 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf735, (256,), is_leaf=True)  # arg735_1
    buf736 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf736, (256,), is_leaf=True)  # arg736_1
    buf737 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf737, (1024, 256, 1, 1), is_leaf=True)  # arg737_1
    buf738 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf738, (1024,), is_leaf=True)  # arg738_1
    buf739 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf739, (2048, 512, 1, 1), is_leaf=True)  # arg739_1
    buf740 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf740, (2048,), is_leaf=True)  # arg740_1
    buf741 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf741, (2048,), is_leaf=True)  # arg741_1
    buf742 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf742, (2048,), is_leaf=True)  # arg742_1
    buf743 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf743, (2048,), is_leaf=True)  # arg743_1
    buf744 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf744, (2048, 1024, 1, 1), is_leaf=True)  # arg744_1
    buf745 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf745, (2048,), is_leaf=True)  # arg745_1
    buf746 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf746, (2048,), is_leaf=True)  # arg746_1
    buf747 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf747, (2048,), is_leaf=True)  # arg747_1
    buf748 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf748, (2048,), is_leaf=True)  # arg748_1
    buf749 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf749, (512, 2048, 1, 1), is_leaf=True)  # arg749_1
    buf750 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf750, (512,), is_leaf=True)  # arg750_1
    buf751 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf751, (512,), is_leaf=True)  # arg751_1
    buf752 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf752, (512,), is_leaf=True)  # arg752_1
    buf753 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf753, (512,), is_leaf=True)  # arg753_1
    buf754 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf754, (1024, 256, 3, 3), is_leaf=True)  # arg754_1
    buf755 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf755, (1024,), is_leaf=True)  # arg755_1
    buf756 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf756, (1024,), is_leaf=True)  # arg756_1
    buf757 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf757, (1024,), is_leaf=True)  # arg757_1
    buf758 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf758, (1024,), is_leaf=True)  # arg758_1
    buf759 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf759, (256, 512, 1, 1), is_leaf=True)  # arg759_1
    buf760 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf760, (256,), is_leaf=True)  # arg760_1
    buf761 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf761, (256,), is_leaf=True)  # arg761_1
    buf762 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf762, (256,), is_leaf=True)  # arg762_1
    buf763 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf763, (256,), is_leaf=True)  # arg763_1
    buf764 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf764, (256,), is_leaf=True)  # arg764_1
    buf765 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf765, (1024, 256, 1, 1), is_leaf=True)  # arg765_1
    buf766 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf766, (1024,), is_leaf=True)  # arg766_1
    buf767 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf767, (2048, 512, 1, 1), is_leaf=True)  # arg767_1
    buf768 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf768, (2048,), is_leaf=True)  # arg768_1
    buf769 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf769, (2048,), is_leaf=True)  # arg769_1
    buf770 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf770, (2048,), is_leaf=True)  # arg770_1
    buf771 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf771, (2048,), is_leaf=True)  # arg771_1
    buf772 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf772, (512, 2048, 1, 1), is_leaf=True)  # arg772_1
    buf773 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf773, (512,), is_leaf=True)  # arg773_1
    buf774 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf774, (512,), is_leaf=True)  # arg774_1
    buf775 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf775, (512,), is_leaf=True)  # arg775_1
    buf776 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf776, (512,), is_leaf=True)  # arg776_1
    buf777 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf777, (1024, 256, 3, 3), is_leaf=True)  # arg777_1
    buf778 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf778, (1024,), is_leaf=True)  # arg778_1
    buf779 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf779, (1024,), is_leaf=True)  # arg779_1
    buf780 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf780, (1024,), is_leaf=True)  # arg780_1
    buf781 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf781, (1024,), is_leaf=True)  # arg781_1
    buf782 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf782, (256, 512, 1, 1), is_leaf=True)  # arg782_1
    buf783 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf783, (256,), is_leaf=True)  # arg783_1
    buf784 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf784, (256,), is_leaf=True)  # arg784_1
    buf785 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf785, (256,), is_leaf=True)  # arg785_1
    buf786 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf786, (256,), is_leaf=True)  # arg786_1
    buf787 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf787, (256,), is_leaf=True)  # arg787_1
    buf788 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf788, (1024, 256, 1, 1), is_leaf=True)  # arg788_1
    buf789 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf789, (1024,), is_leaf=True)  # arg789_1
    buf790 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf790, (2048, 512, 1, 1), is_leaf=True)  # arg790_1
    buf791 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf791, (2048,), is_leaf=True)  # arg791_1
    buf792 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf792, (2048,), is_leaf=True)  # arg792_1
    buf793 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf793, (2048,), is_leaf=True)  # arg793_1
    buf794 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf794, (2048,), is_leaf=True)  # arg794_1
    buf795 = reader.storage(None, 8192000, device=device(type='cuda', index=0))
    reader.tensor(buf795, (1000, 2048), is_leaf=True)  # arg795_1
    buf796 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf796, (1000,), is_leaf=True)  # arg796_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)