
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1):
        convolution_373 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_472 = torch.ops.aten.add.Tensor(arg3_1, 0.001);  arg3_1 = None
        sqrt_201 = torch.ops.aten.sqrt.default(add_472);  add_472 = None
        reciprocal_201 = torch.ops.aten.reciprocal.default(sqrt_201);  sqrt_201 = None
        mul_603 = torch.ops.aten.mul.Tensor(reciprocal_201, 1);  reciprocal_201 = None
        unsqueeze_1608 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1609 = torch.ops.aten.unsqueeze.default(unsqueeze_1608, -1);  unsqueeze_1608 = None
        unsqueeze_1610 = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
        unsqueeze_1611 = torch.ops.aten.unsqueeze.default(unsqueeze_1610, -1);  unsqueeze_1610 = None
        sub_201 = torch.ops.aten.sub.Tensor(convolution_373, unsqueeze_1609);  convolution_373 = unsqueeze_1609 = None
        mul_604 = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_1611);  sub_201 = unsqueeze_1611 = None
        unsqueeze_1612 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_1613 = torch.ops.aten.unsqueeze.default(unsqueeze_1612, -1);  unsqueeze_1612 = None
        mul_605 = torch.ops.aten.mul.Tensor(mul_604, unsqueeze_1613);  mul_604 = unsqueeze_1613 = None
        unsqueeze_1614 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_1615 = torch.ops.aten.unsqueeze.default(unsqueeze_1614, -1);  unsqueeze_1614 = None
        add_473 = torch.ops.aten.add.Tensor(mul_605, unsqueeze_1615);  mul_605 = unsqueeze_1615 = None
        relu_200 = torch.ops.aten.relu.default(add_473)
        convolution_374 = torch.ops.aten.convolution.default(relu_200, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_200 = arg6_1 = None
        add_474 = torch.ops.aten.add.Tensor(arg8_1, 0.001);  arg8_1 = None
        sqrt_202 = torch.ops.aten.sqrt.default(add_474);  add_474 = None
        reciprocal_202 = torch.ops.aten.reciprocal.default(sqrt_202);  sqrt_202 = None
        mul_606 = torch.ops.aten.mul.Tensor(reciprocal_202, 1);  reciprocal_202 = None
        unsqueeze_1616 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_1617 = torch.ops.aten.unsqueeze.default(unsqueeze_1616, -1);  unsqueeze_1616 = None
        unsqueeze_1618 = torch.ops.aten.unsqueeze.default(mul_606, -1);  mul_606 = None
        unsqueeze_1619 = torch.ops.aten.unsqueeze.default(unsqueeze_1618, -1);  unsqueeze_1618 = None
        sub_202 = torch.ops.aten.sub.Tensor(convolution_374, unsqueeze_1617);  convolution_374 = unsqueeze_1617 = None
        mul_607 = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_1619);  sub_202 = unsqueeze_1619 = None
        unsqueeze_1620 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_1621 = torch.ops.aten.unsqueeze.default(unsqueeze_1620, -1);  unsqueeze_1620 = None
        mul_608 = torch.ops.aten.mul.Tensor(mul_607, unsqueeze_1621);  mul_607 = unsqueeze_1621 = None
        unsqueeze_1622 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_1623 = torch.ops.aten.unsqueeze.default(unsqueeze_1622, -1);  unsqueeze_1622 = None
        add_475 = torch.ops.aten.add.Tensor(mul_608, unsqueeze_1623);  mul_608 = unsqueeze_1623 = None
        relu_201 = torch.ops.aten.relu.default(add_473)
        constant_pad_nd_40 = torch.ops.aten.constant_pad_nd.default(relu_201, [2, 2, 2, 2], 0.0);  relu_201 = None
        convolution_375 = torch.ops.aten.convolution.default(constant_pad_nd_40, arg11_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96);  constant_pad_nd_40 = arg11_1 = None
        convolution_376 = torch.ops.aten.convolution.default(convolution_375, arg12_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_375 = arg12_1 = None
        add_476 = torch.ops.aten.add.Tensor(arg14_1, 0.001);  arg14_1 = None
        sqrt_203 = torch.ops.aten.sqrt.default(add_476);  add_476 = None
        reciprocal_203 = torch.ops.aten.reciprocal.default(sqrt_203);  sqrt_203 = None
        mul_609 = torch.ops.aten.mul.Tensor(reciprocal_203, 1);  reciprocal_203 = None
        unsqueeze_1624 = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_1625 = torch.ops.aten.unsqueeze.default(unsqueeze_1624, -1);  unsqueeze_1624 = None
        unsqueeze_1626 = torch.ops.aten.unsqueeze.default(mul_609, -1);  mul_609 = None
        unsqueeze_1627 = torch.ops.aten.unsqueeze.default(unsqueeze_1626, -1);  unsqueeze_1626 = None
        sub_203 = torch.ops.aten.sub.Tensor(convolution_376, unsqueeze_1625);  convolution_376 = unsqueeze_1625 = None
        mul_610 = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_1627);  sub_203 = unsqueeze_1627 = None
        unsqueeze_1628 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_1629 = torch.ops.aten.unsqueeze.default(unsqueeze_1628, -1);  unsqueeze_1628 = None
        mul_611 = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_1629);  mul_610 = unsqueeze_1629 = None
        unsqueeze_1630 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_1631 = torch.ops.aten.unsqueeze.default(unsqueeze_1630, -1);  unsqueeze_1630 = None
        add_477 = torch.ops.aten.add.Tensor(mul_611, unsqueeze_1631);  mul_611 = unsqueeze_1631 = None
        relu_202 = torch.ops.aten.relu.default(add_477);  add_477 = None
        convolution_377 = torch.ops.aten.convolution.default(relu_202, arg17_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 54);  relu_202 = arg17_1 = None
        convolution_378 = torch.ops.aten.convolution.default(convolution_377, arg18_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_377 = arg18_1 = None
        add_478 = torch.ops.aten.add.Tensor(arg20_1, 0.001);  arg20_1 = None
        sqrt_204 = torch.ops.aten.sqrt.default(add_478);  add_478 = None
        reciprocal_204 = torch.ops.aten.reciprocal.default(sqrt_204);  sqrt_204 = None
        mul_612 = torch.ops.aten.mul.Tensor(reciprocal_204, 1);  reciprocal_204 = None
        unsqueeze_1632 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_1633 = torch.ops.aten.unsqueeze.default(unsqueeze_1632, -1);  unsqueeze_1632 = None
        unsqueeze_1634 = torch.ops.aten.unsqueeze.default(mul_612, -1);  mul_612 = None
        unsqueeze_1635 = torch.ops.aten.unsqueeze.default(unsqueeze_1634, -1);  unsqueeze_1634 = None
        sub_204 = torch.ops.aten.sub.Tensor(convolution_378, unsqueeze_1633);  convolution_378 = unsqueeze_1633 = None
        mul_613 = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_1635);  sub_204 = unsqueeze_1635 = None
        unsqueeze_1636 = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_1637 = torch.ops.aten.unsqueeze.default(unsqueeze_1636, -1);  unsqueeze_1636 = None
        mul_614 = torch.ops.aten.mul.Tensor(mul_613, unsqueeze_1637);  mul_613 = unsqueeze_1637 = None
        unsqueeze_1638 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_1639 = torch.ops.aten.unsqueeze.default(unsqueeze_1638, -1);  unsqueeze_1638 = None
        add_479 = torch.ops.aten.add.Tensor(mul_614, unsqueeze_1639);  mul_614 = unsqueeze_1639 = None
        constant_pad_nd_41 = torch.ops.aten.constant_pad_nd.default(add_473, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_42 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_41, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_41 = None
        getitem_84 = _low_memory_max_pool2d_with_offsets_42[0];  _low_memory_max_pool2d_with_offsets_42 = None
        convolution_379 = torch.ops.aten.convolution.default(getitem_84, arg23_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_84 = arg23_1 = None
        add_480 = torch.ops.aten.add.Tensor(arg25_1, 0.001);  arg25_1 = None
        sqrt_205 = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_205 = torch.ops.aten.reciprocal.default(sqrt_205);  sqrt_205 = None
        mul_615 = torch.ops.aten.mul.Tensor(reciprocal_205, 1);  reciprocal_205 = None
        unsqueeze_1640 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_1641 = torch.ops.aten.unsqueeze.default(unsqueeze_1640, -1);  unsqueeze_1640 = None
        unsqueeze_1642 = torch.ops.aten.unsqueeze.default(mul_615, -1);  mul_615 = None
        unsqueeze_1643 = torch.ops.aten.unsqueeze.default(unsqueeze_1642, -1);  unsqueeze_1642 = None
        sub_205 = torch.ops.aten.sub.Tensor(convolution_379, unsqueeze_1641);  convolution_379 = unsqueeze_1641 = None
        mul_616 = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_1643);  sub_205 = unsqueeze_1643 = None
        unsqueeze_1644 = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_1645 = torch.ops.aten.unsqueeze.default(unsqueeze_1644, -1);  unsqueeze_1644 = None
        mul_617 = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_1645);  mul_616 = unsqueeze_1645 = None
        unsqueeze_1646 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_1647 = torch.ops.aten.unsqueeze.default(unsqueeze_1646, -1);  unsqueeze_1646 = None
        add_481 = torch.ops.aten.add.Tensor(mul_617, unsqueeze_1647);  mul_617 = unsqueeze_1647 = None
        add_482 = torch.ops.aten.add.Tensor(add_479, add_481);  add_479 = add_481 = None
        relu_203 = torch.ops.aten.relu.default(add_475)
        constant_pad_nd_42 = torch.ops.aten.constant_pad_nd.default(relu_203, [3, 3, 3, 3], 0.0);  relu_203 = None
        convolution_380 = torch.ops.aten.convolution.default(constant_pad_nd_42, arg28_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 54);  constant_pad_nd_42 = arg28_1 = None
        convolution_381 = torch.ops.aten.convolution.default(convolution_380, arg29_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_380 = arg29_1 = None
        add_483 = torch.ops.aten.add.Tensor(arg31_1, 0.001);  arg31_1 = None
        sqrt_206 = torch.ops.aten.sqrt.default(add_483);  add_483 = None
        reciprocal_206 = torch.ops.aten.reciprocal.default(sqrt_206);  sqrt_206 = None
        mul_618 = torch.ops.aten.mul.Tensor(reciprocal_206, 1);  reciprocal_206 = None
        unsqueeze_1648 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_1649 = torch.ops.aten.unsqueeze.default(unsqueeze_1648, -1);  unsqueeze_1648 = None
        unsqueeze_1650 = torch.ops.aten.unsqueeze.default(mul_618, -1);  mul_618 = None
        unsqueeze_1651 = torch.ops.aten.unsqueeze.default(unsqueeze_1650, -1);  unsqueeze_1650 = None
        sub_206 = torch.ops.aten.sub.Tensor(convolution_381, unsqueeze_1649);  convolution_381 = unsqueeze_1649 = None
        mul_619 = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_1651);  sub_206 = unsqueeze_1651 = None
        unsqueeze_1652 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_1653 = torch.ops.aten.unsqueeze.default(unsqueeze_1652, -1);  unsqueeze_1652 = None
        mul_620 = torch.ops.aten.mul.Tensor(mul_619, unsqueeze_1653);  mul_619 = unsqueeze_1653 = None
        unsqueeze_1654 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_1655 = torch.ops.aten.unsqueeze.default(unsqueeze_1654, -1);  unsqueeze_1654 = None
        add_484 = torch.ops.aten.add.Tensor(mul_620, unsqueeze_1655);  mul_620 = unsqueeze_1655 = None
        relu_204 = torch.ops.aten.relu.default(add_484);  add_484 = None
        convolution_382 = torch.ops.aten.convolution.default(relu_204, arg34_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 54);  relu_204 = arg34_1 = None
        convolution_383 = torch.ops.aten.convolution.default(convolution_382, arg35_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_382 = arg35_1 = None
        add_485 = torch.ops.aten.add.Tensor(arg37_1, 0.001);  arg37_1 = None
        sqrt_207 = torch.ops.aten.sqrt.default(add_485);  add_485 = None
        reciprocal_207 = torch.ops.aten.reciprocal.default(sqrt_207);  sqrt_207 = None
        mul_621 = torch.ops.aten.mul.Tensor(reciprocal_207, 1);  reciprocal_207 = None
        unsqueeze_1656 = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_1657 = torch.ops.aten.unsqueeze.default(unsqueeze_1656, -1);  unsqueeze_1656 = None
        unsqueeze_1658 = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
        unsqueeze_1659 = torch.ops.aten.unsqueeze.default(unsqueeze_1658, -1);  unsqueeze_1658 = None
        sub_207 = torch.ops.aten.sub.Tensor(convolution_383, unsqueeze_1657);  convolution_383 = unsqueeze_1657 = None
        mul_622 = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_1659);  sub_207 = unsqueeze_1659 = None
        unsqueeze_1660 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_1661 = torch.ops.aten.unsqueeze.default(unsqueeze_1660, -1);  unsqueeze_1660 = None
        mul_623 = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_1661);  mul_622 = unsqueeze_1661 = None
        unsqueeze_1662 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_1663 = torch.ops.aten.unsqueeze.default(unsqueeze_1662, -1);  unsqueeze_1662 = None
        add_486 = torch.ops.aten.add.Tensor(mul_623, unsqueeze_1663);  mul_623 = unsqueeze_1663 = None
        constant_pad_nd_43 = torch.ops.aten.constant_pad_nd.default(add_475, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_43 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_43, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_43 = None
        getitem_86 = _low_memory_max_pool2d_with_offsets_43[0];  _low_memory_max_pool2d_with_offsets_43 = None
        add_487 = torch.ops.aten.add.Tensor(add_486, getitem_86);  add_486 = getitem_86 = None
        relu_205 = torch.ops.aten.relu.default(add_475)
        constant_pad_nd_44 = torch.ops.aten.constant_pad_nd.default(relu_205, [2, 2, 2, 2], 0.0);  relu_205 = None
        convolution_384 = torch.ops.aten.convolution.default(constant_pad_nd_44, arg40_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 54);  constant_pad_nd_44 = arg40_1 = None
        convolution_385 = torch.ops.aten.convolution.default(convolution_384, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_384 = arg41_1 = None
        add_488 = torch.ops.aten.add.Tensor(arg43_1, 0.001);  arg43_1 = None
        sqrt_208 = torch.ops.aten.sqrt.default(add_488);  add_488 = None
        reciprocal_208 = torch.ops.aten.reciprocal.default(sqrt_208);  sqrt_208 = None
        mul_624 = torch.ops.aten.mul.Tensor(reciprocal_208, 1);  reciprocal_208 = None
        unsqueeze_1664 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_1665 = torch.ops.aten.unsqueeze.default(unsqueeze_1664, -1);  unsqueeze_1664 = None
        unsqueeze_1666 = torch.ops.aten.unsqueeze.default(mul_624, -1);  mul_624 = None
        unsqueeze_1667 = torch.ops.aten.unsqueeze.default(unsqueeze_1666, -1);  unsqueeze_1666 = None
        sub_208 = torch.ops.aten.sub.Tensor(convolution_385, unsqueeze_1665);  convolution_385 = unsqueeze_1665 = None
        mul_625 = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_1667);  sub_208 = unsqueeze_1667 = None
        unsqueeze_1668 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_1669 = torch.ops.aten.unsqueeze.default(unsqueeze_1668, -1);  unsqueeze_1668 = None
        mul_626 = torch.ops.aten.mul.Tensor(mul_625, unsqueeze_1669);  mul_625 = unsqueeze_1669 = None
        unsqueeze_1670 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_1671 = torch.ops.aten.unsqueeze.default(unsqueeze_1670, -1);  unsqueeze_1670 = None
        add_489 = torch.ops.aten.add.Tensor(mul_626, unsqueeze_1671);  mul_626 = unsqueeze_1671 = None
        relu_206 = torch.ops.aten.relu.default(add_489);  add_489 = None
        convolution_386 = torch.ops.aten.convolution.default(relu_206, arg46_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 54);  relu_206 = arg46_1 = None
        convolution_387 = torch.ops.aten.convolution.default(convolution_386, arg47_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_386 = arg47_1 = None
        add_490 = torch.ops.aten.add.Tensor(arg49_1, 0.001);  arg49_1 = None
        sqrt_209 = torch.ops.aten.sqrt.default(add_490);  add_490 = None
        reciprocal_209 = torch.ops.aten.reciprocal.default(sqrt_209);  sqrt_209 = None
        mul_627 = torch.ops.aten.mul.Tensor(reciprocal_209, 1);  reciprocal_209 = None
        unsqueeze_1672 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_1673 = torch.ops.aten.unsqueeze.default(unsqueeze_1672, -1);  unsqueeze_1672 = None
        unsqueeze_1674 = torch.ops.aten.unsqueeze.default(mul_627, -1);  mul_627 = None
        unsqueeze_1675 = torch.ops.aten.unsqueeze.default(unsqueeze_1674, -1);  unsqueeze_1674 = None
        sub_209 = torch.ops.aten.sub.Tensor(convolution_387, unsqueeze_1673);  convolution_387 = unsqueeze_1673 = None
        mul_628 = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_1675);  sub_209 = unsqueeze_1675 = None
        unsqueeze_1676 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_1677 = torch.ops.aten.unsqueeze.default(unsqueeze_1676, -1);  unsqueeze_1676 = None
        mul_629 = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_1677);  mul_628 = unsqueeze_1677 = None
        unsqueeze_1678 = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_1679 = torch.ops.aten.unsqueeze.default(unsqueeze_1678, -1);  unsqueeze_1678 = None
        add_491 = torch.ops.aten.add.Tensor(mul_629, unsqueeze_1679);  mul_629 = unsqueeze_1679 = None
        relu_207 = torch.ops.aten.relu.default(add_475)
        constant_pad_nd_45 = torch.ops.aten.constant_pad_nd.default(relu_207, [1, 1, 1, 1], 0.0);  relu_207 = None
        convolution_388 = torch.ops.aten.convolution.default(constant_pad_nd_45, arg52_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 54);  constant_pad_nd_45 = arg52_1 = None
        convolution_389 = torch.ops.aten.convolution.default(convolution_388, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_388 = arg53_1 = None
        add_492 = torch.ops.aten.add.Tensor(arg55_1, 0.001);  arg55_1 = None
        sqrt_210 = torch.ops.aten.sqrt.default(add_492);  add_492 = None
        reciprocal_210 = torch.ops.aten.reciprocal.default(sqrt_210);  sqrt_210 = None
        mul_630 = torch.ops.aten.mul.Tensor(reciprocal_210, 1);  reciprocal_210 = None
        unsqueeze_1680 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_1681 = torch.ops.aten.unsqueeze.default(unsqueeze_1680, -1);  unsqueeze_1680 = None
        unsqueeze_1682 = torch.ops.aten.unsqueeze.default(mul_630, -1);  mul_630 = None
        unsqueeze_1683 = torch.ops.aten.unsqueeze.default(unsqueeze_1682, -1);  unsqueeze_1682 = None
        sub_210 = torch.ops.aten.sub.Tensor(convolution_389, unsqueeze_1681);  convolution_389 = unsqueeze_1681 = None
        mul_631 = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_1683);  sub_210 = unsqueeze_1683 = None
        unsqueeze_1684 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_1685 = torch.ops.aten.unsqueeze.default(unsqueeze_1684, -1);  unsqueeze_1684 = None
        mul_632 = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_1685);  mul_631 = unsqueeze_1685 = None
        unsqueeze_1686 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_1687 = torch.ops.aten.unsqueeze.default(unsqueeze_1686, -1);  unsqueeze_1686 = None
        add_493 = torch.ops.aten.add.Tensor(mul_632, unsqueeze_1687);  mul_632 = unsqueeze_1687 = None
        relu_208 = torch.ops.aten.relu.default(add_493);  add_493 = None
        convolution_390 = torch.ops.aten.convolution.default(relu_208, arg58_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54);  relu_208 = arg58_1 = None
        convolution_391 = torch.ops.aten.convolution.default(convolution_390, arg59_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_390 = arg59_1 = None
        add_494 = torch.ops.aten.add.Tensor(arg61_1, 0.001);  arg61_1 = None
        sqrt_211 = torch.ops.aten.sqrt.default(add_494);  add_494 = None
        reciprocal_211 = torch.ops.aten.reciprocal.default(sqrt_211);  sqrt_211 = None
        mul_633 = torch.ops.aten.mul.Tensor(reciprocal_211, 1);  reciprocal_211 = None
        unsqueeze_1688 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_1689 = torch.ops.aten.unsqueeze.default(unsqueeze_1688, -1);  unsqueeze_1688 = None
        unsqueeze_1690 = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_1691 = torch.ops.aten.unsqueeze.default(unsqueeze_1690, -1);  unsqueeze_1690 = None
        sub_211 = torch.ops.aten.sub.Tensor(convolution_391, unsqueeze_1689);  convolution_391 = unsqueeze_1689 = None
        mul_634 = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_1691);  sub_211 = unsqueeze_1691 = None
        unsqueeze_1692 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_1693 = torch.ops.aten.unsqueeze.default(unsqueeze_1692, -1);  unsqueeze_1692 = None
        mul_635 = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_1693);  mul_634 = unsqueeze_1693 = None
        unsqueeze_1694 = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_1695 = torch.ops.aten.unsqueeze.default(unsqueeze_1694, -1);  unsqueeze_1694 = None
        add_495 = torch.ops.aten.add.Tensor(mul_635, unsqueeze_1695);  mul_635 = unsqueeze_1695 = None
        add_496 = torch.ops.aten.add.Tensor(add_491, add_495);  add_491 = add_495 = None
        relu_209 = torch.ops.aten.relu.default(add_496)
        convolution_392 = torch.ops.aten.convolution.default(relu_209, arg64_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54);  relu_209 = arg64_1 = None
        convolution_393 = torch.ops.aten.convolution.default(convolution_392, arg65_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_392 = arg65_1 = None
        add_497 = torch.ops.aten.add.Tensor(arg67_1, 0.001);  arg67_1 = None
        sqrt_212 = torch.ops.aten.sqrt.default(add_497);  add_497 = None
        reciprocal_212 = torch.ops.aten.reciprocal.default(sqrt_212);  sqrt_212 = None
        mul_636 = torch.ops.aten.mul.Tensor(reciprocal_212, 1);  reciprocal_212 = None
        unsqueeze_1696 = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_1697 = torch.ops.aten.unsqueeze.default(unsqueeze_1696, -1);  unsqueeze_1696 = None
        unsqueeze_1698 = torch.ops.aten.unsqueeze.default(mul_636, -1);  mul_636 = None
        unsqueeze_1699 = torch.ops.aten.unsqueeze.default(unsqueeze_1698, -1);  unsqueeze_1698 = None
        sub_212 = torch.ops.aten.sub.Tensor(convolution_393, unsqueeze_1697);  convolution_393 = unsqueeze_1697 = None
        mul_637 = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_1699);  sub_212 = unsqueeze_1699 = None
        unsqueeze_1700 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_1701 = torch.ops.aten.unsqueeze.default(unsqueeze_1700, -1);  unsqueeze_1700 = None
        mul_638 = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_1701);  mul_637 = unsqueeze_1701 = None
        unsqueeze_1702 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_1703 = torch.ops.aten.unsqueeze.default(unsqueeze_1702, -1);  unsqueeze_1702 = None
        add_498 = torch.ops.aten.add.Tensor(mul_638, unsqueeze_1703);  mul_638 = unsqueeze_1703 = None
        relu_210 = torch.ops.aten.relu.default(add_498);  add_498 = None
        convolution_394 = torch.ops.aten.convolution.default(relu_210, arg70_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54);  relu_210 = arg70_1 = None
        convolution_395 = torch.ops.aten.convolution.default(convolution_394, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_394 = arg71_1 = None
        add_499 = torch.ops.aten.add.Tensor(arg73_1, 0.001);  arg73_1 = None
        sqrt_213 = torch.ops.aten.sqrt.default(add_499);  add_499 = None
        reciprocal_213 = torch.ops.aten.reciprocal.default(sqrt_213);  sqrt_213 = None
        mul_639 = torch.ops.aten.mul.Tensor(reciprocal_213, 1);  reciprocal_213 = None
        unsqueeze_1704 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_1705 = torch.ops.aten.unsqueeze.default(unsqueeze_1704, -1);  unsqueeze_1704 = None
        unsqueeze_1706 = torch.ops.aten.unsqueeze.default(mul_639, -1);  mul_639 = None
        unsqueeze_1707 = torch.ops.aten.unsqueeze.default(unsqueeze_1706, -1);  unsqueeze_1706 = None
        sub_213 = torch.ops.aten.sub.Tensor(convolution_395, unsqueeze_1705);  convolution_395 = unsqueeze_1705 = None
        mul_640 = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_1707);  sub_213 = unsqueeze_1707 = None
        unsqueeze_1708 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_1709 = torch.ops.aten.unsqueeze.default(unsqueeze_1708, -1);  unsqueeze_1708 = None
        mul_641 = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_1709);  mul_640 = unsqueeze_1709 = None
        unsqueeze_1710 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_1711 = torch.ops.aten.unsqueeze.default(unsqueeze_1710, -1);  unsqueeze_1710 = None
        add_500 = torch.ops.aten.add.Tensor(mul_641, unsqueeze_1711);  mul_641 = unsqueeze_1711 = None
        constant_pad_nd_46 = torch.ops.aten.constant_pad_nd.default(add_475, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_44 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_46, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_46 = None
        getitem_88 = _low_memory_max_pool2d_with_offsets_44[0];  _low_memory_max_pool2d_with_offsets_44 = None
        add_501 = torch.ops.aten.add.Tensor(add_500, getitem_88);  add_500 = getitem_88 = None
        relu_211 = torch.ops.aten.relu.default(add_473)
        constant_pad_nd_47 = torch.ops.aten.constant_pad_nd.default(relu_211, [1, 1, 1, 1], 0.0);  relu_211 = None
        convolution_396 = torch.ops.aten.convolution.default(constant_pad_nd_47, arg76_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96);  constant_pad_nd_47 = arg76_1 = None
        convolution_397 = torch.ops.aten.convolution.default(convolution_396, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_396 = arg77_1 = None
        add_502 = torch.ops.aten.add.Tensor(arg79_1, 0.001);  arg79_1 = None
        sqrt_214 = torch.ops.aten.sqrt.default(add_502);  add_502 = None
        reciprocal_214 = torch.ops.aten.reciprocal.default(sqrt_214);  sqrt_214 = None
        mul_642 = torch.ops.aten.mul.Tensor(reciprocal_214, 1);  reciprocal_214 = None
        unsqueeze_1712 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_1713 = torch.ops.aten.unsqueeze.default(unsqueeze_1712, -1);  unsqueeze_1712 = None
        unsqueeze_1714 = torch.ops.aten.unsqueeze.default(mul_642, -1);  mul_642 = None
        unsqueeze_1715 = torch.ops.aten.unsqueeze.default(unsqueeze_1714, -1);  unsqueeze_1714 = None
        sub_214 = torch.ops.aten.sub.Tensor(convolution_397, unsqueeze_1713);  convolution_397 = unsqueeze_1713 = None
        mul_643 = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_1715);  sub_214 = unsqueeze_1715 = None
        unsqueeze_1716 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_1717 = torch.ops.aten.unsqueeze.default(unsqueeze_1716, -1);  unsqueeze_1716 = None
        mul_644 = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_1717);  mul_643 = unsqueeze_1717 = None
        unsqueeze_1718 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_1719 = torch.ops.aten.unsqueeze.default(unsqueeze_1718, -1);  unsqueeze_1718 = None
        add_503 = torch.ops.aten.add.Tensor(mul_644, unsqueeze_1719);  mul_644 = unsqueeze_1719 = None
        relu_212 = torch.ops.aten.relu.default(add_503);  add_503 = None
        convolution_398 = torch.ops.aten.convolution.default(relu_212, arg82_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54);  relu_212 = arg82_1 = None
        convolution_399 = torch.ops.aten.convolution.default(convolution_398, arg83_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_398 = arg83_1 = None
        add_504 = torch.ops.aten.add.Tensor(arg85_1, 0.001);  arg85_1 = None
        sqrt_215 = torch.ops.aten.sqrt.default(add_504);  add_504 = None
        reciprocal_215 = torch.ops.aten.reciprocal.default(sqrt_215);  sqrt_215 = None
        mul_645 = torch.ops.aten.mul.Tensor(reciprocal_215, 1);  reciprocal_215 = None
        unsqueeze_1720 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_1721 = torch.ops.aten.unsqueeze.default(unsqueeze_1720, -1);  unsqueeze_1720 = None
        unsqueeze_1722 = torch.ops.aten.unsqueeze.default(mul_645, -1);  mul_645 = None
        unsqueeze_1723 = torch.ops.aten.unsqueeze.default(unsqueeze_1722, -1);  unsqueeze_1722 = None
        sub_215 = torch.ops.aten.sub.Tensor(convolution_399, unsqueeze_1721);  convolution_399 = unsqueeze_1721 = None
        mul_646 = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_1723);  sub_215 = unsqueeze_1723 = None
        unsqueeze_1724 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_1725 = torch.ops.aten.unsqueeze.default(unsqueeze_1724, -1);  unsqueeze_1724 = None
        mul_647 = torch.ops.aten.mul.Tensor(mul_646, unsqueeze_1725);  mul_646 = unsqueeze_1725 = None
        unsqueeze_1726 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_1727 = torch.ops.aten.unsqueeze.default(unsqueeze_1726, -1);  unsqueeze_1726 = None
        add_505 = torch.ops.aten.add.Tensor(mul_647, unsqueeze_1727);  mul_647 = unsqueeze_1727 = None
        relu_213 = torch.ops.aten.relu.default(add_475);  add_475 = None
        convolution_400 = torch.ops.aten.convolution.default(relu_213, arg88_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_213 = arg88_1 = None
        add_506 = torch.ops.aten.add.Tensor(arg90_1, 0.001);  arg90_1 = None
        sqrt_216 = torch.ops.aten.sqrt.default(add_506);  add_506 = None
        reciprocal_216 = torch.ops.aten.reciprocal.default(sqrt_216);  sqrt_216 = None
        mul_648 = torch.ops.aten.mul.Tensor(reciprocal_216, 1);  reciprocal_216 = None
        unsqueeze_1728 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_1729 = torch.ops.aten.unsqueeze.default(unsqueeze_1728, -1);  unsqueeze_1728 = None
        unsqueeze_1730 = torch.ops.aten.unsqueeze.default(mul_648, -1);  mul_648 = None
        unsqueeze_1731 = torch.ops.aten.unsqueeze.default(unsqueeze_1730, -1);  unsqueeze_1730 = None
        sub_216 = torch.ops.aten.sub.Tensor(convolution_400, unsqueeze_1729);  convolution_400 = unsqueeze_1729 = None
        mul_649 = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_1731);  sub_216 = unsqueeze_1731 = None
        unsqueeze_1732 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_1733 = torch.ops.aten.unsqueeze.default(unsqueeze_1732, -1);  unsqueeze_1732 = None
        mul_650 = torch.ops.aten.mul.Tensor(mul_649, unsqueeze_1733);  mul_649 = unsqueeze_1733 = None
        unsqueeze_1734 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_1735 = torch.ops.aten.unsqueeze.default(unsqueeze_1734, -1);  unsqueeze_1734 = None
        add_507 = torch.ops.aten.add.Tensor(mul_650, unsqueeze_1735);  mul_650 = unsqueeze_1735 = None
        add_508 = torch.ops.aten.add.Tensor(add_505, add_507);  add_505 = add_507 = None
        cat_18 = torch.ops.aten.cat.default([add_482, add_487, add_496, add_501, add_508], 1);  add_482 = add_487 = add_496 = add_501 = add_508 = None
        relu_214 = torch.ops.aten.relu.default(add_473);  add_473 = None
        avg_pool2d_8 = torch.ops.aten.avg_pool2d.default(relu_214, [1, 1], [2, 2], [0, 0], False, False)
        convolution_401 = torch.ops.aten.convolution.default(avg_pool2d_8, arg93_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_8 = arg93_1 = None
        constant_pad_nd_49 = torch.ops.aten.constant_pad_nd.default(relu_214, [-1, 1, -1, 1], 0.0);  relu_214 = None
        avg_pool2d_9 = torch.ops.aten.avg_pool2d.default(constant_pad_nd_49, [1, 1], [2, 2], [0, 0], False, False);  constant_pad_nd_49 = None
        convolution_402 = torch.ops.aten.convolution.default(avg_pool2d_9, arg94_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_9 = arg94_1 = None
        cat_19 = torch.ops.aten.cat.default([convolution_401, convolution_402], 1);  convolution_401 = convolution_402 = None
        add_509 = torch.ops.aten.add.Tensor(arg96_1, 0.001);  arg96_1 = None
        sqrt_217 = torch.ops.aten.sqrt.default(add_509);  add_509 = None
        reciprocal_217 = torch.ops.aten.reciprocal.default(sqrt_217);  sqrt_217 = None
        mul_651 = torch.ops.aten.mul.Tensor(reciprocal_217, 1);  reciprocal_217 = None
        unsqueeze_1736 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_1737 = torch.ops.aten.unsqueeze.default(unsqueeze_1736, -1);  unsqueeze_1736 = None
        unsqueeze_1738 = torch.ops.aten.unsqueeze.default(mul_651, -1);  mul_651 = None
        unsqueeze_1739 = torch.ops.aten.unsqueeze.default(unsqueeze_1738, -1);  unsqueeze_1738 = None
        sub_217 = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_1737);  cat_19 = unsqueeze_1737 = None
        mul_652 = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_1739);  sub_217 = unsqueeze_1739 = None
        unsqueeze_1740 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_1741 = torch.ops.aten.unsqueeze.default(unsqueeze_1740, -1);  unsqueeze_1740 = None
        mul_653 = torch.ops.aten.mul.Tensor(mul_652, unsqueeze_1741);  mul_652 = unsqueeze_1741 = None
        unsqueeze_1742 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_1743 = torch.ops.aten.unsqueeze.default(unsqueeze_1742, -1);  unsqueeze_1742 = None
        add_510 = torch.ops.aten.add.Tensor(mul_653, unsqueeze_1743);  mul_653 = unsqueeze_1743 = None
        relu_215 = torch.ops.aten.relu.default(cat_18)
        convolution_403 = torch.ops.aten.convolution.default(relu_215, arg99_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_215 = arg99_1 = None
        add_511 = torch.ops.aten.add.Tensor(arg101_1, 0.001);  arg101_1 = None
        sqrt_218 = torch.ops.aten.sqrt.default(add_511);  add_511 = None
        reciprocal_218 = torch.ops.aten.reciprocal.default(sqrt_218);  sqrt_218 = None
        mul_654 = torch.ops.aten.mul.Tensor(reciprocal_218, 1);  reciprocal_218 = None
        unsqueeze_1744 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_1745 = torch.ops.aten.unsqueeze.default(unsqueeze_1744, -1);  unsqueeze_1744 = None
        unsqueeze_1746 = torch.ops.aten.unsqueeze.default(mul_654, -1);  mul_654 = None
        unsqueeze_1747 = torch.ops.aten.unsqueeze.default(unsqueeze_1746, -1);  unsqueeze_1746 = None
        sub_218 = torch.ops.aten.sub.Tensor(convolution_403, unsqueeze_1745);  convolution_403 = unsqueeze_1745 = None
        mul_655 = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_1747);  sub_218 = unsqueeze_1747 = None
        unsqueeze_1748 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_1749 = torch.ops.aten.unsqueeze.default(unsqueeze_1748, -1);  unsqueeze_1748 = None
        mul_656 = torch.ops.aten.mul.Tensor(mul_655, unsqueeze_1749);  mul_655 = unsqueeze_1749 = None
        unsqueeze_1750 = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_1751 = torch.ops.aten.unsqueeze.default(unsqueeze_1750, -1);  unsqueeze_1750 = None
        add_512 = torch.ops.aten.add.Tensor(mul_656, unsqueeze_1751);  mul_656 = unsqueeze_1751 = None
        relu_216 = torch.ops.aten.relu.default(add_510)
        constant_pad_nd_50 = torch.ops.aten.constant_pad_nd.default(relu_216, [2, 2, 2, 2], 0.0);  relu_216 = None
        convolution_404 = torch.ops.aten.convolution.default(constant_pad_nd_50, arg104_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108);  constant_pad_nd_50 = arg104_1 = None
        convolution_405 = torch.ops.aten.convolution.default(convolution_404, arg105_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_404 = arg105_1 = None
        add_513 = torch.ops.aten.add.Tensor(arg107_1, 0.001);  arg107_1 = None
        sqrt_219 = torch.ops.aten.sqrt.default(add_513);  add_513 = None
        reciprocal_219 = torch.ops.aten.reciprocal.default(sqrt_219);  sqrt_219 = None
        mul_657 = torch.ops.aten.mul.Tensor(reciprocal_219, 1);  reciprocal_219 = None
        unsqueeze_1752 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_1753 = torch.ops.aten.unsqueeze.default(unsqueeze_1752, -1);  unsqueeze_1752 = None
        unsqueeze_1754 = torch.ops.aten.unsqueeze.default(mul_657, -1);  mul_657 = None
        unsqueeze_1755 = torch.ops.aten.unsqueeze.default(unsqueeze_1754, -1);  unsqueeze_1754 = None
        sub_219 = torch.ops.aten.sub.Tensor(convolution_405, unsqueeze_1753);  convolution_405 = unsqueeze_1753 = None
        mul_658 = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_1755);  sub_219 = unsqueeze_1755 = None
        unsqueeze_1756 = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_1757 = torch.ops.aten.unsqueeze.default(unsqueeze_1756, -1);  unsqueeze_1756 = None
        mul_659 = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_1757);  mul_658 = unsqueeze_1757 = None
        unsqueeze_1758 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_1759 = torch.ops.aten.unsqueeze.default(unsqueeze_1758, -1);  unsqueeze_1758 = None
        add_514 = torch.ops.aten.add.Tensor(mul_659, unsqueeze_1759);  mul_659 = unsqueeze_1759 = None
        relu_217 = torch.ops.aten.relu.default(add_514);  add_514 = None
        convolution_406 = torch.ops.aten.convolution.default(relu_217, arg110_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 108);  relu_217 = arg110_1 = None
        convolution_407 = torch.ops.aten.convolution.default(convolution_406, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_406 = arg111_1 = None
        add_515 = torch.ops.aten.add.Tensor(arg113_1, 0.001);  arg113_1 = None
        sqrt_220 = torch.ops.aten.sqrt.default(add_515);  add_515 = None
        reciprocal_220 = torch.ops.aten.reciprocal.default(sqrt_220);  sqrt_220 = None
        mul_660 = torch.ops.aten.mul.Tensor(reciprocal_220, 1);  reciprocal_220 = None
        unsqueeze_1760 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_1761 = torch.ops.aten.unsqueeze.default(unsqueeze_1760, -1);  unsqueeze_1760 = None
        unsqueeze_1762 = torch.ops.aten.unsqueeze.default(mul_660, -1);  mul_660 = None
        unsqueeze_1763 = torch.ops.aten.unsqueeze.default(unsqueeze_1762, -1);  unsqueeze_1762 = None
        sub_220 = torch.ops.aten.sub.Tensor(convolution_407, unsqueeze_1761);  convolution_407 = unsqueeze_1761 = None
        mul_661 = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_1763);  sub_220 = unsqueeze_1763 = None
        unsqueeze_1764 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_1765 = torch.ops.aten.unsqueeze.default(unsqueeze_1764, -1);  unsqueeze_1764 = None
        mul_662 = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_1765);  mul_661 = unsqueeze_1765 = None
        unsqueeze_1766 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_1767 = torch.ops.aten.unsqueeze.default(unsqueeze_1766, -1);  unsqueeze_1766 = None
        add_516 = torch.ops.aten.add.Tensor(mul_662, unsqueeze_1767);  mul_662 = unsqueeze_1767 = None
        constant_pad_nd_51 = torch.ops.aten.constant_pad_nd.default(add_510, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_45 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_51, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_51 = None
        getitem_90 = _low_memory_max_pool2d_with_offsets_45[0];  _low_memory_max_pool2d_with_offsets_45 = None
        add_517 = torch.ops.aten.add.Tensor(add_516, getitem_90);  add_516 = getitem_90 = None
        relu_218 = torch.ops.aten.relu.default(add_512)
        constant_pad_nd_52 = torch.ops.aten.constant_pad_nd.default(relu_218, [3, 3, 3, 3], 0.0);  relu_218 = None
        convolution_408 = torch.ops.aten.convolution.default(constant_pad_nd_52, arg116_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108);  constant_pad_nd_52 = arg116_1 = None
        convolution_409 = torch.ops.aten.convolution.default(convolution_408, arg117_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_408 = arg117_1 = None
        add_518 = torch.ops.aten.add.Tensor(arg119_1, 0.001);  arg119_1 = None
        sqrt_221 = torch.ops.aten.sqrt.default(add_518);  add_518 = None
        reciprocal_221 = torch.ops.aten.reciprocal.default(sqrt_221);  sqrt_221 = None
        mul_663 = torch.ops.aten.mul.Tensor(reciprocal_221, 1);  reciprocal_221 = None
        unsqueeze_1768 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_1769 = torch.ops.aten.unsqueeze.default(unsqueeze_1768, -1);  unsqueeze_1768 = None
        unsqueeze_1770 = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_1771 = torch.ops.aten.unsqueeze.default(unsqueeze_1770, -1);  unsqueeze_1770 = None
        sub_221 = torch.ops.aten.sub.Tensor(convolution_409, unsqueeze_1769);  convolution_409 = unsqueeze_1769 = None
        mul_664 = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_1771);  sub_221 = unsqueeze_1771 = None
        unsqueeze_1772 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_1773 = torch.ops.aten.unsqueeze.default(unsqueeze_1772, -1);  unsqueeze_1772 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_1773);  mul_664 = unsqueeze_1773 = None
        unsqueeze_1774 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_1775 = torch.ops.aten.unsqueeze.default(unsqueeze_1774, -1);  unsqueeze_1774 = None
        add_519 = torch.ops.aten.add.Tensor(mul_665, unsqueeze_1775);  mul_665 = unsqueeze_1775 = None
        relu_219 = torch.ops.aten.relu.default(add_519);  add_519 = None
        convolution_410 = torch.ops.aten.convolution.default(relu_219, arg122_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 108);  relu_219 = arg122_1 = None
        convolution_411 = torch.ops.aten.convolution.default(convolution_410, arg123_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_410 = arg123_1 = None
        add_520 = torch.ops.aten.add.Tensor(arg125_1, 0.001);  arg125_1 = None
        sqrt_222 = torch.ops.aten.sqrt.default(add_520);  add_520 = None
        reciprocal_222 = torch.ops.aten.reciprocal.default(sqrt_222);  sqrt_222 = None
        mul_666 = torch.ops.aten.mul.Tensor(reciprocal_222, 1);  reciprocal_222 = None
        unsqueeze_1776 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_1777 = torch.ops.aten.unsqueeze.default(unsqueeze_1776, -1);  unsqueeze_1776 = None
        unsqueeze_1778 = torch.ops.aten.unsqueeze.default(mul_666, -1);  mul_666 = None
        unsqueeze_1779 = torch.ops.aten.unsqueeze.default(unsqueeze_1778, -1);  unsqueeze_1778 = None
        sub_222 = torch.ops.aten.sub.Tensor(convolution_411, unsqueeze_1777);  convolution_411 = unsqueeze_1777 = None
        mul_667 = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_1779);  sub_222 = unsqueeze_1779 = None
        unsqueeze_1780 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_1781 = torch.ops.aten.unsqueeze.default(unsqueeze_1780, -1);  unsqueeze_1780 = None
        mul_668 = torch.ops.aten.mul.Tensor(mul_667, unsqueeze_1781);  mul_667 = unsqueeze_1781 = None
        unsqueeze_1782 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_1783 = torch.ops.aten.unsqueeze.default(unsqueeze_1782, -1);  unsqueeze_1782 = None
        add_521 = torch.ops.aten.add.Tensor(mul_668, unsqueeze_1783);  mul_668 = unsqueeze_1783 = None
        constant_pad_nd_53 = torch.ops.aten.constant_pad_nd.default(add_512, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_46 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_53, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_53 = None
        getitem_92 = _low_memory_max_pool2d_with_offsets_46[0];  _low_memory_max_pool2d_with_offsets_46 = None
        add_522 = torch.ops.aten.add.Tensor(add_521, getitem_92);  add_521 = getitem_92 = None
        relu_220 = torch.ops.aten.relu.default(add_512)
        constant_pad_nd_54 = torch.ops.aten.constant_pad_nd.default(relu_220, [2, 2, 2, 2], 0.0);  relu_220 = None
        convolution_412 = torch.ops.aten.convolution.default(constant_pad_nd_54, arg128_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108);  constant_pad_nd_54 = arg128_1 = None
        convolution_413 = torch.ops.aten.convolution.default(convolution_412, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_412 = arg129_1 = None
        add_523 = torch.ops.aten.add.Tensor(arg131_1, 0.001);  arg131_1 = None
        sqrt_223 = torch.ops.aten.sqrt.default(add_523);  add_523 = None
        reciprocal_223 = torch.ops.aten.reciprocal.default(sqrt_223);  sqrt_223 = None
        mul_669 = torch.ops.aten.mul.Tensor(reciprocal_223, 1);  reciprocal_223 = None
        unsqueeze_1784 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_1785 = torch.ops.aten.unsqueeze.default(unsqueeze_1784, -1);  unsqueeze_1784 = None
        unsqueeze_1786 = torch.ops.aten.unsqueeze.default(mul_669, -1);  mul_669 = None
        unsqueeze_1787 = torch.ops.aten.unsqueeze.default(unsqueeze_1786, -1);  unsqueeze_1786 = None
        sub_223 = torch.ops.aten.sub.Tensor(convolution_413, unsqueeze_1785);  convolution_413 = unsqueeze_1785 = None
        mul_670 = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_1787);  sub_223 = unsqueeze_1787 = None
        unsqueeze_1788 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_1789 = torch.ops.aten.unsqueeze.default(unsqueeze_1788, -1);  unsqueeze_1788 = None
        mul_671 = torch.ops.aten.mul.Tensor(mul_670, unsqueeze_1789);  mul_670 = unsqueeze_1789 = None
        unsqueeze_1790 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_1791 = torch.ops.aten.unsqueeze.default(unsqueeze_1790, -1);  unsqueeze_1790 = None
        add_524 = torch.ops.aten.add.Tensor(mul_671, unsqueeze_1791);  mul_671 = unsqueeze_1791 = None
        relu_221 = torch.ops.aten.relu.default(add_524);  add_524 = None
        convolution_414 = torch.ops.aten.convolution.default(relu_221, arg134_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 108);  relu_221 = arg134_1 = None
        convolution_415 = torch.ops.aten.convolution.default(convolution_414, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_414 = arg135_1 = None
        add_525 = torch.ops.aten.add.Tensor(arg137_1, 0.001);  arg137_1 = None
        sqrt_224 = torch.ops.aten.sqrt.default(add_525);  add_525 = None
        reciprocal_224 = torch.ops.aten.reciprocal.default(sqrt_224);  sqrt_224 = None
        mul_672 = torch.ops.aten.mul.Tensor(reciprocal_224, 1);  reciprocal_224 = None
        unsqueeze_1792 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_1793 = torch.ops.aten.unsqueeze.default(unsqueeze_1792, -1);  unsqueeze_1792 = None
        unsqueeze_1794 = torch.ops.aten.unsqueeze.default(mul_672, -1);  mul_672 = None
        unsqueeze_1795 = torch.ops.aten.unsqueeze.default(unsqueeze_1794, -1);  unsqueeze_1794 = None
        sub_224 = torch.ops.aten.sub.Tensor(convolution_415, unsqueeze_1793);  convolution_415 = unsqueeze_1793 = None
        mul_673 = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_1795);  sub_224 = unsqueeze_1795 = None
        unsqueeze_1796 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_1797 = torch.ops.aten.unsqueeze.default(unsqueeze_1796, -1);  unsqueeze_1796 = None
        mul_674 = torch.ops.aten.mul.Tensor(mul_673, unsqueeze_1797);  mul_673 = unsqueeze_1797 = None
        unsqueeze_1798 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_1799 = torch.ops.aten.unsqueeze.default(unsqueeze_1798, -1);  unsqueeze_1798 = None
        add_526 = torch.ops.aten.add.Tensor(mul_674, unsqueeze_1799);  mul_674 = unsqueeze_1799 = None
        relu_222 = torch.ops.aten.relu.default(add_512)
        constant_pad_nd_55 = torch.ops.aten.constant_pad_nd.default(relu_222, [1, 1, 1, 1], 0.0);  relu_222 = None
        convolution_416 = torch.ops.aten.convolution.default(constant_pad_nd_55, arg140_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108);  constant_pad_nd_55 = arg140_1 = None
        convolution_417 = torch.ops.aten.convolution.default(convolution_416, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_416 = arg141_1 = None
        add_527 = torch.ops.aten.add.Tensor(arg143_1, 0.001);  arg143_1 = None
        sqrt_225 = torch.ops.aten.sqrt.default(add_527);  add_527 = None
        reciprocal_225 = torch.ops.aten.reciprocal.default(sqrt_225);  sqrt_225 = None
        mul_675 = torch.ops.aten.mul.Tensor(reciprocal_225, 1);  reciprocal_225 = None
        unsqueeze_1800 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_1801 = torch.ops.aten.unsqueeze.default(unsqueeze_1800, -1);  unsqueeze_1800 = None
        unsqueeze_1802 = torch.ops.aten.unsqueeze.default(mul_675, -1);  mul_675 = None
        unsqueeze_1803 = torch.ops.aten.unsqueeze.default(unsqueeze_1802, -1);  unsqueeze_1802 = None
        sub_225 = torch.ops.aten.sub.Tensor(convolution_417, unsqueeze_1801);  convolution_417 = unsqueeze_1801 = None
        mul_676 = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_1803);  sub_225 = unsqueeze_1803 = None
        unsqueeze_1804 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_1805 = torch.ops.aten.unsqueeze.default(unsqueeze_1804, -1);  unsqueeze_1804 = None
        mul_677 = torch.ops.aten.mul.Tensor(mul_676, unsqueeze_1805);  mul_676 = unsqueeze_1805 = None
        unsqueeze_1806 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_1807 = torch.ops.aten.unsqueeze.default(unsqueeze_1806, -1);  unsqueeze_1806 = None
        add_528 = torch.ops.aten.add.Tensor(mul_677, unsqueeze_1807);  mul_677 = unsqueeze_1807 = None
        relu_223 = torch.ops.aten.relu.default(add_528);  add_528 = None
        convolution_418 = torch.ops.aten.convolution.default(relu_223, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108);  relu_223 = arg146_1 = None
        convolution_419 = torch.ops.aten.convolution.default(convolution_418, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_418 = arg147_1 = None
        add_529 = torch.ops.aten.add.Tensor(arg149_1, 0.001);  arg149_1 = None
        sqrt_226 = torch.ops.aten.sqrt.default(add_529);  add_529 = None
        reciprocal_226 = torch.ops.aten.reciprocal.default(sqrt_226);  sqrt_226 = None
        mul_678 = torch.ops.aten.mul.Tensor(reciprocal_226, 1);  reciprocal_226 = None
        unsqueeze_1808 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_1809 = torch.ops.aten.unsqueeze.default(unsqueeze_1808, -1);  unsqueeze_1808 = None
        unsqueeze_1810 = torch.ops.aten.unsqueeze.default(mul_678, -1);  mul_678 = None
        unsqueeze_1811 = torch.ops.aten.unsqueeze.default(unsqueeze_1810, -1);  unsqueeze_1810 = None
        sub_226 = torch.ops.aten.sub.Tensor(convolution_419, unsqueeze_1809);  convolution_419 = unsqueeze_1809 = None
        mul_679 = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_1811);  sub_226 = unsqueeze_1811 = None
        unsqueeze_1812 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_1813 = torch.ops.aten.unsqueeze.default(unsqueeze_1812, -1);  unsqueeze_1812 = None
        mul_680 = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_1813);  mul_679 = unsqueeze_1813 = None
        unsqueeze_1814 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_1815 = torch.ops.aten.unsqueeze.default(unsqueeze_1814, -1);  unsqueeze_1814 = None
        add_530 = torch.ops.aten.add.Tensor(mul_680, unsqueeze_1815);  mul_680 = unsqueeze_1815 = None
        add_531 = torch.ops.aten.add.Tensor(add_526, add_530);  add_526 = add_530 = None
        relu_224 = torch.ops.aten.relu.default(add_531)
        convolution_420 = torch.ops.aten.convolution.default(relu_224, arg152_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108);  relu_224 = arg152_1 = None
        convolution_421 = torch.ops.aten.convolution.default(convolution_420, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_420 = arg153_1 = None
        add_532 = torch.ops.aten.add.Tensor(arg155_1, 0.001);  arg155_1 = None
        sqrt_227 = torch.ops.aten.sqrt.default(add_532);  add_532 = None
        reciprocal_227 = torch.ops.aten.reciprocal.default(sqrt_227);  sqrt_227 = None
        mul_681 = torch.ops.aten.mul.Tensor(reciprocal_227, 1);  reciprocal_227 = None
        unsqueeze_1816 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_1817 = torch.ops.aten.unsqueeze.default(unsqueeze_1816, -1);  unsqueeze_1816 = None
        unsqueeze_1818 = torch.ops.aten.unsqueeze.default(mul_681, -1);  mul_681 = None
        unsqueeze_1819 = torch.ops.aten.unsqueeze.default(unsqueeze_1818, -1);  unsqueeze_1818 = None
        sub_227 = torch.ops.aten.sub.Tensor(convolution_421, unsqueeze_1817);  convolution_421 = unsqueeze_1817 = None
        mul_682 = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_1819);  sub_227 = unsqueeze_1819 = None
        unsqueeze_1820 = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_1821 = torch.ops.aten.unsqueeze.default(unsqueeze_1820, -1);  unsqueeze_1820 = None
        mul_683 = torch.ops.aten.mul.Tensor(mul_682, unsqueeze_1821);  mul_682 = unsqueeze_1821 = None
        unsqueeze_1822 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_1823 = torch.ops.aten.unsqueeze.default(unsqueeze_1822, -1);  unsqueeze_1822 = None
        add_533 = torch.ops.aten.add.Tensor(mul_683, unsqueeze_1823);  mul_683 = unsqueeze_1823 = None
        relu_225 = torch.ops.aten.relu.default(add_533);  add_533 = None
        convolution_422 = torch.ops.aten.convolution.default(relu_225, arg158_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108);  relu_225 = arg158_1 = None
        convolution_423 = torch.ops.aten.convolution.default(convolution_422, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_422 = arg159_1 = None
        add_534 = torch.ops.aten.add.Tensor(arg161_1, 0.001);  arg161_1 = None
        sqrt_228 = torch.ops.aten.sqrt.default(add_534);  add_534 = None
        reciprocal_228 = torch.ops.aten.reciprocal.default(sqrt_228);  sqrt_228 = None
        mul_684 = torch.ops.aten.mul.Tensor(reciprocal_228, 1);  reciprocal_228 = None
        unsqueeze_1824 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_1825 = torch.ops.aten.unsqueeze.default(unsqueeze_1824, -1);  unsqueeze_1824 = None
        unsqueeze_1826 = torch.ops.aten.unsqueeze.default(mul_684, -1);  mul_684 = None
        unsqueeze_1827 = torch.ops.aten.unsqueeze.default(unsqueeze_1826, -1);  unsqueeze_1826 = None
        sub_228 = torch.ops.aten.sub.Tensor(convolution_423, unsqueeze_1825);  convolution_423 = unsqueeze_1825 = None
        mul_685 = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_1827);  sub_228 = unsqueeze_1827 = None
        unsqueeze_1828 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_1829 = torch.ops.aten.unsqueeze.default(unsqueeze_1828, -1);  unsqueeze_1828 = None
        mul_686 = torch.ops.aten.mul.Tensor(mul_685, unsqueeze_1829);  mul_685 = unsqueeze_1829 = None
        unsqueeze_1830 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_1831 = torch.ops.aten.unsqueeze.default(unsqueeze_1830, -1);  unsqueeze_1830 = None
        add_535 = torch.ops.aten.add.Tensor(mul_686, unsqueeze_1831);  mul_686 = unsqueeze_1831 = None
        constant_pad_nd_56 = torch.ops.aten.constant_pad_nd.default(add_512, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_47 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_56, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_56 = None
        getitem_94 = _low_memory_max_pool2d_with_offsets_47[0];  _low_memory_max_pool2d_with_offsets_47 = None
        add_536 = torch.ops.aten.add.Tensor(add_535, getitem_94);  add_535 = getitem_94 = None
        relu_226 = torch.ops.aten.relu.default(add_510);  add_510 = None
        constant_pad_nd_57 = torch.ops.aten.constant_pad_nd.default(relu_226, [1, 1, 1, 1], 0.0);  relu_226 = None
        convolution_424 = torch.ops.aten.convolution.default(constant_pad_nd_57, arg164_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108);  constant_pad_nd_57 = arg164_1 = None
        convolution_425 = torch.ops.aten.convolution.default(convolution_424, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_424 = arg165_1 = None
        add_537 = torch.ops.aten.add.Tensor(arg167_1, 0.001);  arg167_1 = None
        sqrt_229 = torch.ops.aten.sqrt.default(add_537);  add_537 = None
        reciprocal_229 = torch.ops.aten.reciprocal.default(sqrt_229);  sqrt_229 = None
        mul_687 = torch.ops.aten.mul.Tensor(reciprocal_229, 1);  reciprocal_229 = None
        unsqueeze_1832 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_1833 = torch.ops.aten.unsqueeze.default(unsqueeze_1832, -1);  unsqueeze_1832 = None
        unsqueeze_1834 = torch.ops.aten.unsqueeze.default(mul_687, -1);  mul_687 = None
        unsqueeze_1835 = torch.ops.aten.unsqueeze.default(unsqueeze_1834, -1);  unsqueeze_1834 = None
        sub_229 = torch.ops.aten.sub.Tensor(convolution_425, unsqueeze_1833);  convolution_425 = unsqueeze_1833 = None
        mul_688 = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_1835);  sub_229 = unsqueeze_1835 = None
        unsqueeze_1836 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_1837 = torch.ops.aten.unsqueeze.default(unsqueeze_1836, -1);  unsqueeze_1836 = None
        mul_689 = torch.ops.aten.mul.Tensor(mul_688, unsqueeze_1837);  mul_688 = unsqueeze_1837 = None
        unsqueeze_1838 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_1839 = torch.ops.aten.unsqueeze.default(unsqueeze_1838, -1);  unsqueeze_1838 = None
        add_538 = torch.ops.aten.add.Tensor(mul_689, unsqueeze_1839);  mul_689 = unsqueeze_1839 = None
        relu_227 = torch.ops.aten.relu.default(add_538);  add_538 = None
        convolution_426 = torch.ops.aten.convolution.default(relu_227, arg170_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108);  relu_227 = arg170_1 = None
        convolution_427 = torch.ops.aten.convolution.default(convolution_426, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_426 = arg171_1 = None
        add_539 = torch.ops.aten.add.Tensor(arg173_1, 0.001);  arg173_1 = None
        sqrt_230 = torch.ops.aten.sqrt.default(add_539);  add_539 = None
        reciprocal_230 = torch.ops.aten.reciprocal.default(sqrt_230);  sqrt_230 = None
        mul_690 = torch.ops.aten.mul.Tensor(reciprocal_230, 1);  reciprocal_230 = None
        unsqueeze_1840 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_1841 = torch.ops.aten.unsqueeze.default(unsqueeze_1840, -1);  unsqueeze_1840 = None
        unsqueeze_1842 = torch.ops.aten.unsqueeze.default(mul_690, -1);  mul_690 = None
        unsqueeze_1843 = torch.ops.aten.unsqueeze.default(unsqueeze_1842, -1);  unsqueeze_1842 = None
        sub_230 = torch.ops.aten.sub.Tensor(convolution_427, unsqueeze_1841);  convolution_427 = unsqueeze_1841 = None
        mul_691 = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_1843);  sub_230 = unsqueeze_1843 = None
        unsqueeze_1844 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_1845 = torch.ops.aten.unsqueeze.default(unsqueeze_1844, -1);  unsqueeze_1844 = None
        mul_692 = torch.ops.aten.mul.Tensor(mul_691, unsqueeze_1845);  mul_691 = unsqueeze_1845 = None
        unsqueeze_1846 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_1847 = torch.ops.aten.unsqueeze.default(unsqueeze_1846, -1);  unsqueeze_1846 = None
        add_540 = torch.ops.aten.add.Tensor(mul_692, unsqueeze_1847);  mul_692 = unsqueeze_1847 = None
        relu_228 = torch.ops.aten.relu.default(add_512);  add_512 = None
        convolution_428 = torch.ops.aten.convolution.default(relu_228, arg176_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_228 = arg176_1 = None
        add_541 = torch.ops.aten.add.Tensor(arg178_1, 0.001);  arg178_1 = None
        sqrt_231 = torch.ops.aten.sqrt.default(add_541);  add_541 = None
        reciprocal_231 = torch.ops.aten.reciprocal.default(sqrt_231);  sqrt_231 = None
        mul_693 = torch.ops.aten.mul.Tensor(reciprocal_231, 1);  reciprocal_231 = None
        unsqueeze_1848 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_1849 = torch.ops.aten.unsqueeze.default(unsqueeze_1848, -1);  unsqueeze_1848 = None
        unsqueeze_1850 = torch.ops.aten.unsqueeze.default(mul_693, -1);  mul_693 = None
        unsqueeze_1851 = torch.ops.aten.unsqueeze.default(unsqueeze_1850, -1);  unsqueeze_1850 = None
        sub_231 = torch.ops.aten.sub.Tensor(convolution_428, unsqueeze_1849);  convolution_428 = unsqueeze_1849 = None
        mul_694 = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_1851);  sub_231 = unsqueeze_1851 = None
        unsqueeze_1852 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_1853 = torch.ops.aten.unsqueeze.default(unsqueeze_1852, -1);  unsqueeze_1852 = None
        mul_695 = torch.ops.aten.mul.Tensor(mul_694, unsqueeze_1853);  mul_694 = unsqueeze_1853 = None
        unsqueeze_1854 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_1855 = torch.ops.aten.unsqueeze.default(unsqueeze_1854, -1);  unsqueeze_1854 = None
        add_542 = torch.ops.aten.add.Tensor(mul_695, unsqueeze_1855);  mul_695 = unsqueeze_1855 = None
        add_543 = torch.ops.aten.add.Tensor(add_540, add_542);  add_540 = add_542 = None
        cat_20 = torch.ops.aten.cat.default([add_517, add_522, add_531, add_536, add_543], 1);  add_517 = add_522 = add_531 = add_536 = add_543 = None
        relu_229 = torch.ops.aten.relu.default(cat_18);  cat_18 = None
        avg_pool2d_10 = torch.ops.aten.avg_pool2d.default(relu_229, [1, 1], [2, 2], [0, 0], False, False)
        convolution_429 = torch.ops.aten.convolution.default(avg_pool2d_10, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_10 = arg181_1 = None
        constant_pad_nd_59 = torch.ops.aten.constant_pad_nd.default(relu_229, [-1, 1, -1, 1], 0.0);  relu_229 = None
        avg_pool2d_11 = torch.ops.aten.avg_pool2d.default(constant_pad_nd_59, [1, 1], [2, 2], [0, 0], False, False);  constant_pad_nd_59 = None
        convolution_430 = torch.ops.aten.convolution.default(avg_pool2d_11, arg182_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_11 = arg182_1 = None
        cat_21 = torch.ops.aten.cat.default([convolution_429, convolution_430], 1);  convolution_429 = convolution_430 = None
        add_544 = torch.ops.aten.add.Tensor(arg184_1, 0.001);  arg184_1 = None
        sqrt_232 = torch.ops.aten.sqrt.default(add_544);  add_544 = None
        reciprocal_232 = torch.ops.aten.reciprocal.default(sqrt_232);  sqrt_232 = None
        mul_696 = torch.ops.aten.mul.Tensor(reciprocal_232, 1);  reciprocal_232 = None
        unsqueeze_1856 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_1857 = torch.ops.aten.unsqueeze.default(unsqueeze_1856, -1);  unsqueeze_1856 = None
        unsqueeze_1858 = torch.ops.aten.unsqueeze.default(mul_696, -1);  mul_696 = None
        unsqueeze_1859 = torch.ops.aten.unsqueeze.default(unsqueeze_1858, -1);  unsqueeze_1858 = None
        sub_232 = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_1857);  cat_21 = unsqueeze_1857 = None
        mul_697 = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_1859);  sub_232 = unsqueeze_1859 = None
        unsqueeze_1860 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_1861 = torch.ops.aten.unsqueeze.default(unsqueeze_1860, -1);  unsqueeze_1860 = None
        mul_698 = torch.ops.aten.mul.Tensor(mul_697, unsqueeze_1861);  mul_697 = unsqueeze_1861 = None
        unsqueeze_1862 = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_1863 = torch.ops.aten.unsqueeze.default(unsqueeze_1862, -1);  unsqueeze_1862 = None
        add_545 = torch.ops.aten.add.Tensor(mul_698, unsqueeze_1863);  mul_698 = unsqueeze_1863 = None
        relu_230 = torch.ops.aten.relu.default(cat_20)
        convolution_431 = torch.ops.aten.convolution.default(relu_230, arg187_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_230 = arg187_1 = None
        add_546 = torch.ops.aten.add.Tensor(arg189_1, 0.001);  arg189_1 = None
        sqrt_233 = torch.ops.aten.sqrt.default(add_546);  add_546 = None
        reciprocal_233 = torch.ops.aten.reciprocal.default(sqrt_233);  sqrt_233 = None
        mul_699 = torch.ops.aten.mul.Tensor(reciprocal_233, 1);  reciprocal_233 = None
        unsqueeze_1864 = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_1865 = torch.ops.aten.unsqueeze.default(unsqueeze_1864, -1);  unsqueeze_1864 = None
        unsqueeze_1866 = torch.ops.aten.unsqueeze.default(mul_699, -1);  mul_699 = None
        unsqueeze_1867 = torch.ops.aten.unsqueeze.default(unsqueeze_1866, -1);  unsqueeze_1866 = None
        sub_233 = torch.ops.aten.sub.Tensor(convolution_431, unsqueeze_1865);  convolution_431 = unsqueeze_1865 = None
        mul_700 = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_1867);  sub_233 = unsqueeze_1867 = None
        unsqueeze_1868 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_1869 = torch.ops.aten.unsqueeze.default(unsqueeze_1868, -1);  unsqueeze_1868 = None
        mul_701 = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_1869);  mul_700 = unsqueeze_1869 = None
        unsqueeze_1870 = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_1871 = torch.ops.aten.unsqueeze.default(unsqueeze_1870, -1);  unsqueeze_1870 = None
        add_547 = torch.ops.aten.add.Tensor(mul_701, unsqueeze_1871);  mul_701 = unsqueeze_1871 = None
        relu_231 = torch.ops.aten.relu.default(add_545)
        convolution_432 = torch.ops.aten.convolution.default(relu_231, arg192_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_231 = arg192_1 = None
        convolution_433 = torch.ops.aten.convolution.default(convolution_432, arg193_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_432 = arg193_1 = None
        add_548 = torch.ops.aten.add.Tensor(arg195_1, 0.001);  arg195_1 = None
        sqrt_234 = torch.ops.aten.sqrt.default(add_548);  add_548 = None
        reciprocal_234 = torch.ops.aten.reciprocal.default(sqrt_234);  sqrt_234 = None
        mul_702 = torch.ops.aten.mul.Tensor(reciprocal_234, 1);  reciprocal_234 = None
        unsqueeze_1872 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_1873 = torch.ops.aten.unsqueeze.default(unsqueeze_1872, -1);  unsqueeze_1872 = None
        unsqueeze_1874 = torch.ops.aten.unsqueeze.default(mul_702, -1);  mul_702 = None
        unsqueeze_1875 = torch.ops.aten.unsqueeze.default(unsqueeze_1874, -1);  unsqueeze_1874 = None
        sub_234 = torch.ops.aten.sub.Tensor(convolution_433, unsqueeze_1873);  convolution_433 = unsqueeze_1873 = None
        mul_703 = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_1875);  sub_234 = unsqueeze_1875 = None
        unsqueeze_1876 = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_1877 = torch.ops.aten.unsqueeze.default(unsqueeze_1876, -1);  unsqueeze_1876 = None
        mul_704 = torch.ops.aten.mul.Tensor(mul_703, unsqueeze_1877);  mul_703 = unsqueeze_1877 = None
        unsqueeze_1878 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_1879 = torch.ops.aten.unsqueeze.default(unsqueeze_1878, -1);  unsqueeze_1878 = None
        add_549 = torch.ops.aten.add.Tensor(mul_704, unsqueeze_1879);  mul_704 = unsqueeze_1879 = None
        relu_232 = torch.ops.aten.relu.default(add_549);  add_549 = None
        convolution_434 = torch.ops.aten.convolution.default(relu_232, arg198_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_232 = arg198_1 = None
        convolution_435 = torch.ops.aten.convolution.default(convolution_434, arg199_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_434 = arg199_1 = None
        add_550 = torch.ops.aten.add.Tensor(arg201_1, 0.001);  arg201_1 = None
        sqrt_235 = torch.ops.aten.sqrt.default(add_550);  add_550 = None
        reciprocal_235 = torch.ops.aten.reciprocal.default(sqrt_235);  sqrt_235 = None
        mul_705 = torch.ops.aten.mul.Tensor(reciprocal_235, 1);  reciprocal_235 = None
        unsqueeze_1880 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_1881 = torch.ops.aten.unsqueeze.default(unsqueeze_1880, -1);  unsqueeze_1880 = None
        unsqueeze_1882 = torch.ops.aten.unsqueeze.default(mul_705, -1);  mul_705 = None
        unsqueeze_1883 = torch.ops.aten.unsqueeze.default(unsqueeze_1882, -1);  unsqueeze_1882 = None
        sub_235 = torch.ops.aten.sub.Tensor(convolution_435, unsqueeze_1881);  convolution_435 = unsqueeze_1881 = None
        mul_706 = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_1883);  sub_235 = unsqueeze_1883 = None
        unsqueeze_1884 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_1885 = torch.ops.aten.unsqueeze.default(unsqueeze_1884, -1);  unsqueeze_1884 = None
        mul_707 = torch.ops.aten.mul.Tensor(mul_706, unsqueeze_1885);  mul_706 = unsqueeze_1885 = None
        unsqueeze_1886 = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_1887 = torch.ops.aten.unsqueeze.default(unsqueeze_1886, -1);  unsqueeze_1886 = None
        add_551 = torch.ops.aten.add.Tensor(mul_707, unsqueeze_1887);  mul_707 = unsqueeze_1887 = None
        _low_memory_max_pool2d_with_offsets_48 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_545, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_96 = _low_memory_max_pool2d_with_offsets_48[0];  _low_memory_max_pool2d_with_offsets_48 = None
        add_552 = torch.ops.aten.add.Tensor(add_551, getitem_96);  add_551 = getitem_96 = None
        relu_233 = torch.ops.aten.relu.default(add_547)
        convolution_436 = torch.ops.aten.convolution.default(relu_233, arg204_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_233 = arg204_1 = None
        convolution_437 = torch.ops.aten.convolution.default(convolution_436, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_436 = arg205_1 = None
        add_553 = torch.ops.aten.add.Tensor(arg207_1, 0.001);  arg207_1 = None
        sqrt_236 = torch.ops.aten.sqrt.default(add_553);  add_553 = None
        reciprocal_236 = torch.ops.aten.reciprocal.default(sqrt_236);  sqrt_236 = None
        mul_708 = torch.ops.aten.mul.Tensor(reciprocal_236, 1);  reciprocal_236 = None
        unsqueeze_1888 = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_1889 = torch.ops.aten.unsqueeze.default(unsqueeze_1888, -1);  unsqueeze_1888 = None
        unsqueeze_1890 = torch.ops.aten.unsqueeze.default(mul_708, -1);  mul_708 = None
        unsqueeze_1891 = torch.ops.aten.unsqueeze.default(unsqueeze_1890, -1);  unsqueeze_1890 = None
        sub_236 = torch.ops.aten.sub.Tensor(convolution_437, unsqueeze_1889);  convolution_437 = unsqueeze_1889 = None
        mul_709 = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_1891);  sub_236 = unsqueeze_1891 = None
        unsqueeze_1892 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_1893 = torch.ops.aten.unsqueeze.default(unsqueeze_1892, -1);  unsqueeze_1892 = None
        mul_710 = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_1893);  mul_709 = unsqueeze_1893 = None
        unsqueeze_1894 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1895 = torch.ops.aten.unsqueeze.default(unsqueeze_1894, -1);  unsqueeze_1894 = None
        add_554 = torch.ops.aten.add.Tensor(mul_710, unsqueeze_1895);  mul_710 = unsqueeze_1895 = None
        relu_234 = torch.ops.aten.relu.default(add_554);  add_554 = None
        convolution_438 = torch.ops.aten.convolution.default(relu_234, arg210_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_234 = arg210_1 = None
        convolution_439 = torch.ops.aten.convolution.default(convolution_438, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_438 = arg211_1 = None
        add_555 = torch.ops.aten.add.Tensor(arg213_1, 0.001);  arg213_1 = None
        sqrt_237 = torch.ops.aten.sqrt.default(add_555);  add_555 = None
        reciprocal_237 = torch.ops.aten.reciprocal.default(sqrt_237);  sqrt_237 = None
        mul_711 = torch.ops.aten.mul.Tensor(reciprocal_237, 1);  reciprocal_237 = None
        unsqueeze_1896 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_1897 = torch.ops.aten.unsqueeze.default(unsqueeze_1896, -1);  unsqueeze_1896 = None
        unsqueeze_1898 = torch.ops.aten.unsqueeze.default(mul_711, -1);  mul_711 = None
        unsqueeze_1899 = torch.ops.aten.unsqueeze.default(unsqueeze_1898, -1);  unsqueeze_1898 = None
        sub_237 = torch.ops.aten.sub.Tensor(convolution_439, unsqueeze_1897);  convolution_439 = unsqueeze_1897 = None
        mul_712 = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_1899);  sub_237 = unsqueeze_1899 = None
        unsqueeze_1900 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1901 = torch.ops.aten.unsqueeze.default(unsqueeze_1900, -1);  unsqueeze_1900 = None
        mul_713 = torch.ops.aten.mul.Tensor(mul_712, unsqueeze_1901);  mul_712 = unsqueeze_1901 = None
        unsqueeze_1902 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_1903 = torch.ops.aten.unsqueeze.default(unsqueeze_1902, -1);  unsqueeze_1902 = None
        add_556 = torch.ops.aten.add.Tensor(mul_713, unsqueeze_1903);  mul_713 = unsqueeze_1903 = None
        _low_memory_max_pool2d_with_offsets_49 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_547, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_98 = _low_memory_max_pool2d_with_offsets_49[0];  _low_memory_max_pool2d_with_offsets_49 = None
        add_557 = torch.ops.aten.add.Tensor(add_556, getitem_98);  add_556 = getitem_98 = None
        relu_235 = torch.ops.aten.relu.default(add_547)
        convolution_440 = torch.ops.aten.convolution.default(relu_235, arg216_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_235 = arg216_1 = None
        convolution_441 = torch.ops.aten.convolution.default(convolution_440, arg217_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_440 = arg217_1 = None
        add_558 = torch.ops.aten.add.Tensor(arg219_1, 0.001);  arg219_1 = None
        sqrt_238 = torch.ops.aten.sqrt.default(add_558);  add_558 = None
        reciprocal_238 = torch.ops.aten.reciprocal.default(sqrt_238);  sqrt_238 = None
        mul_714 = torch.ops.aten.mul.Tensor(reciprocal_238, 1);  reciprocal_238 = None
        unsqueeze_1904 = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_1905 = torch.ops.aten.unsqueeze.default(unsqueeze_1904, -1);  unsqueeze_1904 = None
        unsqueeze_1906 = torch.ops.aten.unsqueeze.default(mul_714, -1);  mul_714 = None
        unsqueeze_1907 = torch.ops.aten.unsqueeze.default(unsqueeze_1906, -1);  unsqueeze_1906 = None
        sub_238 = torch.ops.aten.sub.Tensor(convolution_441, unsqueeze_1905);  convolution_441 = unsqueeze_1905 = None
        mul_715 = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_1907);  sub_238 = unsqueeze_1907 = None
        unsqueeze_1908 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_1909 = torch.ops.aten.unsqueeze.default(unsqueeze_1908, -1);  unsqueeze_1908 = None
        mul_716 = torch.ops.aten.mul.Tensor(mul_715, unsqueeze_1909);  mul_715 = unsqueeze_1909 = None
        unsqueeze_1910 = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_1911 = torch.ops.aten.unsqueeze.default(unsqueeze_1910, -1);  unsqueeze_1910 = None
        add_559 = torch.ops.aten.add.Tensor(mul_716, unsqueeze_1911);  mul_716 = unsqueeze_1911 = None
        relu_236 = torch.ops.aten.relu.default(add_559);  add_559 = None
        convolution_442 = torch.ops.aten.convolution.default(relu_236, arg222_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_236 = arg222_1 = None
        convolution_443 = torch.ops.aten.convolution.default(convolution_442, arg223_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_442 = arg223_1 = None
        add_560 = torch.ops.aten.add.Tensor(arg225_1, 0.001);  arg225_1 = None
        sqrt_239 = torch.ops.aten.sqrt.default(add_560);  add_560 = None
        reciprocal_239 = torch.ops.aten.reciprocal.default(sqrt_239);  sqrt_239 = None
        mul_717 = torch.ops.aten.mul.Tensor(reciprocal_239, 1);  reciprocal_239 = None
        unsqueeze_1912 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1913 = torch.ops.aten.unsqueeze.default(unsqueeze_1912, -1);  unsqueeze_1912 = None
        unsqueeze_1914 = torch.ops.aten.unsqueeze.default(mul_717, -1);  mul_717 = None
        unsqueeze_1915 = torch.ops.aten.unsqueeze.default(unsqueeze_1914, -1);  unsqueeze_1914 = None
        sub_239 = torch.ops.aten.sub.Tensor(convolution_443, unsqueeze_1913);  convolution_443 = unsqueeze_1913 = None
        mul_718 = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_1915);  sub_239 = unsqueeze_1915 = None
        unsqueeze_1916 = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_1917 = torch.ops.aten.unsqueeze.default(unsqueeze_1916, -1);  unsqueeze_1916 = None
        mul_719 = torch.ops.aten.mul.Tensor(mul_718, unsqueeze_1917);  mul_718 = unsqueeze_1917 = None
        unsqueeze_1918 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1919 = torch.ops.aten.unsqueeze.default(unsqueeze_1918, -1);  unsqueeze_1918 = None
        add_561 = torch.ops.aten.add.Tensor(mul_719, unsqueeze_1919);  mul_719 = unsqueeze_1919 = None
        relu_237 = torch.ops.aten.relu.default(add_547)
        convolution_444 = torch.ops.aten.convolution.default(relu_237, arg228_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_237 = arg228_1 = None
        convolution_445 = torch.ops.aten.convolution.default(convolution_444, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_444 = arg229_1 = None
        add_562 = torch.ops.aten.add.Tensor(arg231_1, 0.001);  arg231_1 = None
        sqrt_240 = torch.ops.aten.sqrt.default(add_562);  add_562 = None
        reciprocal_240 = torch.ops.aten.reciprocal.default(sqrt_240);  sqrt_240 = None
        mul_720 = torch.ops.aten.mul.Tensor(reciprocal_240, 1);  reciprocal_240 = None
        unsqueeze_1920 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1921 = torch.ops.aten.unsqueeze.default(unsqueeze_1920, -1);  unsqueeze_1920 = None
        unsqueeze_1922 = torch.ops.aten.unsqueeze.default(mul_720, -1);  mul_720 = None
        unsqueeze_1923 = torch.ops.aten.unsqueeze.default(unsqueeze_1922, -1);  unsqueeze_1922 = None
        sub_240 = torch.ops.aten.sub.Tensor(convolution_445, unsqueeze_1921);  convolution_445 = unsqueeze_1921 = None
        mul_721 = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_1923);  sub_240 = unsqueeze_1923 = None
        unsqueeze_1924 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_1925 = torch.ops.aten.unsqueeze.default(unsqueeze_1924, -1);  unsqueeze_1924 = None
        mul_722 = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_1925);  mul_721 = unsqueeze_1925 = None
        unsqueeze_1926 = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_1927 = torch.ops.aten.unsqueeze.default(unsqueeze_1926, -1);  unsqueeze_1926 = None
        add_563 = torch.ops.aten.add.Tensor(mul_722, unsqueeze_1927);  mul_722 = unsqueeze_1927 = None
        relu_238 = torch.ops.aten.relu.default(add_563);  add_563 = None
        convolution_446 = torch.ops.aten.convolution.default(relu_238, arg234_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_238 = arg234_1 = None
        convolution_447 = torch.ops.aten.convolution.default(convolution_446, arg235_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_446 = arg235_1 = None
        add_564 = torch.ops.aten.add.Tensor(arg237_1, 0.001);  arg237_1 = None
        sqrt_241 = torch.ops.aten.sqrt.default(add_564);  add_564 = None
        reciprocal_241 = torch.ops.aten.reciprocal.default(sqrt_241);  sqrt_241 = None
        mul_723 = torch.ops.aten.mul.Tensor(reciprocal_241, 1);  reciprocal_241 = None
        unsqueeze_1928 = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_1929 = torch.ops.aten.unsqueeze.default(unsqueeze_1928, -1);  unsqueeze_1928 = None
        unsqueeze_1930 = torch.ops.aten.unsqueeze.default(mul_723, -1);  mul_723 = None
        unsqueeze_1931 = torch.ops.aten.unsqueeze.default(unsqueeze_1930, -1);  unsqueeze_1930 = None
        sub_241 = torch.ops.aten.sub.Tensor(convolution_447, unsqueeze_1929);  convolution_447 = unsqueeze_1929 = None
        mul_724 = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_1931);  sub_241 = unsqueeze_1931 = None
        unsqueeze_1932 = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_1933 = torch.ops.aten.unsqueeze.default(unsqueeze_1932, -1);  unsqueeze_1932 = None
        mul_725 = torch.ops.aten.mul.Tensor(mul_724, unsqueeze_1933);  mul_724 = unsqueeze_1933 = None
        unsqueeze_1934 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1935 = torch.ops.aten.unsqueeze.default(unsqueeze_1934, -1);  unsqueeze_1934 = None
        add_565 = torch.ops.aten.add.Tensor(mul_725, unsqueeze_1935);  mul_725 = unsqueeze_1935 = None
        add_566 = torch.ops.aten.add.Tensor(add_561, add_565);  add_561 = add_565 = None
        relu_239 = torch.ops.aten.relu.default(add_566)
        convolution_448 = torch.ops.aten.convolution.default(relu_239, arg240_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_239 = arg240_1 = None
        convolution_449 = torch.ops.aten.convolution.default(convolution_448, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_448 = arg241_1 = None
        add_567 = torch.ops.aten.add.Tensor(arg243_1, 0.001);  arg243_1 = None
        sqrt_242 = torch.ops.aten.sqrt.default(add_567);  add_567 = None
        reciprocal_242 = torch.ops.aten.reciprocal.default(sqrt_242);  sqrt_242 = None
        mul_726 = torch.ops.aten.mul.Tensor(reciprocal_242, 1);  reciprocal_242 = None
        unsqueeze_1936 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1937 = torch.ops.aten.unsqueeze.default(unsqueeze_1936, -1);  unsqueeze_1936 = None
        unsqueeze_1938 = torch.ops.aten.unsqueeze.default(mul_726, -1);  mul_726 = None
        unsqueeze_1939 = torch.ops.aten.unsqueeze.default(unsqueeze_1938, -1);  unsqueeze_1938 = None
        sub_242 = torch.ops.aten.sub.Tensor(convolution_449, unsqueeze_1937);  convolution_449 = unsqueeze_1937 = None
        mul_727 = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_1939);  sub_242 = unsqueeze_1939 = None
        unsqueeze_1940 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1941 = torch.ops.aten.unsqueeze.default(unsqueeze_1940, -1);  unsqueeze_1940 = None
        mul_728 = torch.ops.aten.mul.Tensor(mul_727, unsqueeze_1941);  mul_727 = unsqueeze_1941 = None
        unsqueeze_1942 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1943 = torch.ops.aten.unsqueeze.default(unsqueeze_1942, -1);  unsqueeze_1942 = None
        add_568 = torch.ops.aten.add.Tensor(mul_728, unsqueeze_1943);  mul_728 = unsqueeze_1943 = None
        relu_240 = torch.ops.aten.relu.default(add_568);  add_568 = None
        convolution_450 = torch.ops.aten.convolution.default(relu_240, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_240 = arg246_1 = None
        convolution_451 = torch.ops.aten.convolution.default(convolution_450, arg247_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_450 = arg247_1 = None
        add_569 = torch.ops.aten.add.Tensor(arg249_1, 0.001);  arg249_1 = None
        sqrt_243 = torch.ops.aten.sqrt.default(add_569);  add_569 = None
        reciprocal_243 = torch.ops.aten.reciprocal.default(sqrt_243);  sqrt_243 = None
        mul_729 = torch.ops.aten.mul.Tensor(reciprocal_243, 1);  reciprocal_243 = None
        unsqueeze_1944 = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_1945 = torch.ops.aten.unsqueeze.default(unsqueeze_1944, -1);  unsqueeze_1944 = None
        unsqueeze_1946 = torch.ops.aten.unsqueeze.default(mul_729, -1);  mul_729 = None
        unsqueeze_1947 = torch.ops.aten.unsqueeze.default(unsqueeze_1946, -1);  unsqueeze_1946 = None
        sub_243 = torch.ops.aten.sub.Tensor(convolution_451, unsqueeze_1945);  convolution_451 = unsqueeze_1945 = None
        mul_730 = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_1947);  sub_243 = unsqueeze_1947 = None
        unsqueeze_1948 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1949 = torch.ops.aten.unsqueeze.default(unsqueeze_1948, -1);  unsqueeze_1948 = None
        mul_731 = torch.ops.aten.mul.Tensor(mul_730, unsqueeze_1949);  mul_730 = unsqueeze_1949 = None
        unsqueeze_1950 = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_1951 = torch.ops.aten.unsqueeze.default(unsqueeze_1950, -1);  unsqueeze_1950 = None
        add_570 = torch.ops.aten.add.Tensor(mul_731, unsqueeze_1951);  mul_731 = unsqueeze_1951 = None
        _low_memory_max_pool2d_with_offsets_50 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_547, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_100 = _low_memory_max_pool2d_with_offsets_50[0];  _low_memory_max_pool2d_with_offsets_50 = None
        add_571 = torch.ops.aten.add.Tensor(add_570, getitem_100);  add_570 = getitem_100 = None
        relu_241 = torch.ops.aten.relu.default(add_545);  add_545 = None
        convolution_452 = torch.ops.aten.convolution.default(relu_241, arg252_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_241 = arg252_1 = None
        convolution_453 = torch.ops.aten.convolution.default(convolution_452, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_452 = arg253_1 = None
        add_572 = torch.ops.aten.add.Tensor(arg255_1, 0.001);  arg255_1 = None
        sqrt_244 = torch.ops.aten.sqrt.default(add_572);  add_572 = None
        reciprocal_244 = torch.ops.aten.reciprocal.default(sqrt_244);  sqrt_244 = None
        mul_732 = torch.ops.aten.mul.Tensor(reciprocal_244, 1);  reciprocal_244 = None
        unsqueeze_1952 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1953 = torch.ops.aten.unsqueeze.default(unsqueeze_1952, -1);  unsqueeze_1952 = None
        unsqueeze_1954 = torch.ops.aten.unsqueeze.default(mul_732, -1);  mul_732 = None
        unsqueeze_1955 = torch.ops.aten.unsqueeze.default(unsqueeze_1954, -1);  unsqueeze_1954 = None
        sub_244 = torch.ops.aten.sub.Tensor(convolution_453, unsqueeze_1953);  convolution_453 = unsqueeze_1953 = None
        mul_733 = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_1955);  sub_244 = unsqueeze_1955 = None
        unsqueeze_1956 = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_1957 = torch.ops.aten.unsqueeze.default(unsqueeze_1956, -1);  unsqueeze_1956 = None
        mul_734 = torch.ops.aten.mul.Tensor(mul_733, unsqueeze_1957);  mul_733 = unsqueeze_1957 = None
        unsqueeze_1958 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1959 = torch.ops.aten.unsqueeze.default(unsqueeze_1958, -1);  unsqueeze_1958 = None
        add_573 = torch.ops.aten.add.Tensor(mul_734, unsqueeze_1959);  mul_734 = unsqueeze_1959 = None
        relu_242 = torch.ops.aten.relu.default(add_573);  add_573 = None
        convolution_454 = torch.ops.aten.convolution.default(relu_242, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_242 = arg258_1 = None
        convolution_455 = torch.ops.aten.convolution.default(convolution_454, arg259_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_454 = arg259_1 = None
        add_574 = torch.ops.aten.add.Tensor(arg261_1, 0.001);  arg261_1 = None
        sqrt_245 = torch.ops.aten.sqrt.default(add_574);  add_574 = None
        reciprocal_245 = torch.ops.aten.reciprocal.default(sqrt_245);  sqrt_245 = None
        mul_735 = torch.ops.aten.mul.Tensor(reciprocal_245, 1);  reciprocal_245 = None
        unsqueeze_1960 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1961 = torch.ops.aten.unsqueeze.default(unsqueeze_1960, -1);  unsqueeze_1960 = None
        unsqueeze_1962 = torch.ops.aten.unsqueeze.default(mul_735, -1);  mul_735 = None
        unsqueeze_1963 = torch.ops.aten.unsqueeze.default(unsqueeze_1962, -1);  unsqueeze_1962 = None
        sub_245 = torch.ops.aten.sub.Tensor(convolution_455, unsqueeze_1961);  convolution_455 = unsqueeze_1961 = None
        mul_736 = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_1963);  sub_245 = unsqueeze_1963 = None
        unsqueeze_1964 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1965 = torch.ops.aten.unsqueeze.default(unsqueeze_1964, -1);  unsqueeze_1964 = None
        mul_737 = torch.ops.aten.mul.Tensor(mul_736, unsqueeze_1965);  mul_736 = unsqueeze_1965 = None
        unsqueeze_1966 = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_1967 = torch.ops.aten.unsqueeze.default(unsqueeze_1966, -1);  unsqueeze_1966 = None
        add_575 = torch.ops.aten.add.Tensor(mul_737, unsqueeze_1967);  mul_737 = unsqueeze_1967 = None
        add_576 = torch.ops.aten.add.Tensor(add_575, add_547);  add_575 = add_547 = None
        cat_22 = torch.ops.aten.cat.default([add_552, add_557, add_566, add_571, add_576], 1);  add_552 = add_557 = add_566 = add_571 = add_576 = None
        relu_243 = torch.ops.aten.relu.default(cat_20);  cat_20 = None
        convolution_456 = torch.ops.aten.convolution.default(relu_243, arg264_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_243 = arg264_1 = None
        add_577 = torch.ops.aten.add.Tensor(arg266_1, 0.001);  arg266_1 = None
        sqrt_246 = torch.ops.aten.sqrt.default(add_577);  add_577 = None
        reciprocal_246 = torch.ops.aten.reciprocal.default(sqrt_246);  sqrt_246 = None
        mul_738 = torch.ops.aten.mul.Tensor(reciprocal_246, 1);  reciprocal_246 = None
        unsqueeze_1968 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1969 = torch.ops.aten.unsqueeze.default(unsqueeze_1968, -1);  unsqueeze_1968 = None
        unsqueeze_1970 = torch.ops.aten.unsqueeze.default(mul_738, -1);  mul_738 = None
        unsqueeze_1971 = torch.ops.aten.unsqueeze.default(unsqueeze_1970, -1);  unsqueeze_1970 = None
        sub_246 = torch.ops.aten.sub.Tensor(convolution_456, unsqueeze_1969);  convolution_456 = unsqueeze_1969 = None
        mul_739 = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_1971);  sub_246 = unsqueeze_1971 = None
        unsqueeze_1972 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1973 = torch.ops.aten.unsqueeze.default(unsqueeze_1972, -1);  unsqueeze_1972 = None
        mul_740 = torch.ops.aten.mul.Tensor(mul_739, unsqueeze_1973);  mul_739 = unsqueeze_1973 = None
        unsqueeze_1974 = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_1975 = torch.ops.aten.unsqueeze.default(unsqueeze_1974, -1);  unsqueeze_1974 = None
        add_578 = torch.ops.aten.add.Tensor(mul_740, unsqueeze_1975);  mul_740 = unsqueeze_1975 = None
        relu_244 = torch.ops.aten.relu.default(cat_22)
        convolution_457 = torch.ops.aten.convolution.default(relu_244, arg269_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_244 = arg269_1 = None
        add_579 = torch.ops.aten.add.Tensor(arg271_1, 0.001);  arg271_1 = None
        sqrt_247 = torch.ops.aten.sqrt.default(add_579);  add_579 = None
        reciprocal_247 = torch.ops.aten.reciprocal.default(sqrt_247);  sqrt_247 = None
        mul_741 = torch.ops.aten.mul.Tensor(reciprocal_247, 1);  reciprocal_247 = None
        unsqueeze_1976 = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_1977 = torch.ops.aten.unsqueeze.default(unsqueeze_1976, -1);  unsqueeze_1976 = None
        unsqueeze_1978 = torch.ops.aten.unsqueeze.default(mul_741, -1);  mul_741 = None
        unsqueeze_1979 = torch.ops.aten.unsqueeze.default(unsqueeze_1978, -1);  unsqueeze_1978 = None
        sub_247 = torch.ops.aten.sub.Tensor(convolution_457, unsqueeze_1977);  convolution_457 = unsqueeze_1977 = None
        mul_742 = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_1979);  sub_247 = unsqueeze_1979 = None
        unsqueeze_1980 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_1981 = torch.ops.aten.unsqueeze.default(unsqueeze_1980, -1);  unsqueeze_1980 = None
        mul_743 = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_1981);  mul_742 = unsqueeze_1981 = None
        unsqueeze_1982 = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_1983 = torch.ops.aten.unsqueeze.default(unsqueeze_1982, -1);  unsqueeze_1982 = None
        add_580 = torch.ops.aten.add.Tensor(mul_743, unsqueeze_1983);  mul_743 = unsqueeze_1983 = None
        relu_245 = torch.ops.aten.relu.default(add_578)
        convolution_458 = torch.ops.aten.convolution.default(relu_245, arg274_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_245 = arg274_1 = None
        convolution_459 = torch.ops.aten.convolution.default(convolution_458, arg275_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_458 = arg275_1 = None
        add_581 = torch.ops.aten.add.Tensor(arg277_1, 0.001);  arg277_1 = None
        sqrt_248 = torch.ops.aten.sqrt.default(add_581);  add_581 = None
        reciprocal_248 = torch.ops.aten.reciprocal.default(sqrt_248);  sqrt_248 = None
        mul_744 = torch.ops.aten.mul.Tensor(reciprocal_248, 1);  reciprocal_248 = None
        unsqueeze_1984 = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1985 = torch.ops.aten.unsqueeze.default(unsqueeze_1984, -1);  unsqueeze_1984 = None
        unsqueeze_1986 = torch.ops.aten.unsqueeze.default(mul_744, -1);  mul_744 = None
        unsqueeze_1987 = torch.ops.aten.unsqueeze.default(unsqueeze_1986, -1);  unsqueeze_1986 = None
        sub_248 = torch.ops.aten.sub.Tensor(convolution_459, unsqueeze_1985);  convolution_459 = unsqueeze_1985 = None
        mul_745 = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_1987);  sub_248 = unsqueeze_1987 = None
        unsqueeze_1988 = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_1989 = torch.ops.aten.unsqueeze.default(unsqueeze_1988, -1);  unsqueeze_1988 = None
        mul_746 = torch.ops.aten.mul.Tensor(mul_745, unsqueeze_1989);  mul_745 = unsqueeze_1989 = None
        unsqueeze_1990 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1991 = torch.ops.aten.unsqueeze.default(unsqueeze_1990, -1);  unsqueeze_1990 = None
        add_582 = torch.ops.aten.add.Tensor(mul_746, unsqueeze_1991);  mul_746 = unsqueeze_1991 = None
        relu_246 = torch.ops.aten.relu.default(add_582);  add_582 = None
        convolution_460 = torch.ops.aten.convolution.default(relu_246, arg280_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_246 = arg280_1 = None
        convolution_461 = torch.ops.aten.convolution.default(convolution_460, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_460 = arg281_1 = None
        add_583 = torch.ops.aten.add.Tensor(arg283_1, 0.001);  arg283_1 = None
        sqrt_249 = torch.ops.aten.sqrt.default(add_583);  add_583 = None
        reciprocal_249 = torch.ops.aten.reciprocal.default(sqrt_249);  sqrt_249 = None
        mul_747 = torch.ops.aten.mul.Tensor(reciprocal_249, 1);  reciprocal_249 = None
        unsqueeze_1992 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1993 = torch.ops.aten.unsqueeze.default(unsqueeze_1992, -1);  unsqueeze_1992 = None
        unsqueeze_1994 = torch.ops.aten.unsqueeze.default(mul_747, -1);  mul_747 = None
        unsqueeze_1995 = torch.ops.aten.unsqueeze.default(unsqueeze_1994, -1);  unsqueeze_1994 = None
        sub_249 = torch.ops.aten.sub.Tensor(convolution_461, unsqueeze_1993);  convolution_461 = unsqueeze_1993 = None
        mul_748 = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_1995);  sub_249 = unsqueeze_1995 = None
        unsqueeze_1996 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1997 = torch.ops.aten.unsqueeze.default(unsqueeze_1996, -1);  unsqueeze_1996 = None
        mul_749 = torch.ops.aten.mul.Tensor(mul_748, unsqueeze_1997);  mul_748 = unsqueeze_1997 = None
        unsqueeze_1998 = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_1999 = torch.ops.aten.unsqueeze.default(unsqueeze_1998, -1);  unsqueeze_1998 = None
        add_584 = torch.ops.aten.add.Tensor(mul_749, unsqueeze_1999);  mul_749 = unsqueeze_1999 = None
        _low_memory_max_pool2d_with_offsets_51 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_578, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_102 = _low_memory_max_pool2d_with_offsets_51[0];  _low_memory_max_pool2d_with_offsets_51 = None
        add_585 = torch.ops.aten.add.Tensor(add_584, getitem_102);  add_584 = getitem_102 = None
        relu_247 = torch.ops.aten.relu.default(add_580)
        convolution_462 = torch.ops.aten.convolution.default(relu_247, arg286_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_247 = arg286_1 = None
        convolution_463 = torch.ops.aten.convolution.default(convolution_462, arg287_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_462 = arg287_1 = None
        add_586 = torch.ops.aten.add.Tensor(arg289_1, 0.001);  arg289_1 = None
        sqrt_250 = torch.ops.aten.sqrt.default(add_586);  add_586 = None
        reciprocal_250 = torch.ops.aten.reciprocal.default(sqrt_250);  sqrt_250 = None
        mul_750 = torch.ops.aten.mul.Tensor(reciprocal_250, 1);  reciprocal_250 = None
        unsqueeze_2000 = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_2001 = torch.ops.aten.unsqueeze.default(unsqueeze_2000, -1);  unsqueeze_2000 = None
        unsqueeze_2002 = torch.ops.aten.unsqueeze.default(mul_750, -1);  mul_750 = None
        unsqueeze_2003 = torch.ops.aten.unsqueeze.default(unsqueeze_2002, -1);  unsqueeze_2002 = None
        sub_250 = torch.ops.aten.sub.Tensor(convolution_463, unsqueeze_2001);  convolution_463 = unsqueeze_2001 = None
        mul_751 = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_2003);  sub_250 = unsqueeze_2003 = None
        unsqueeze_2004 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_2005 = torch.ops.aten.unsqueeze.default(unsqueeze_2004, -1);  unsqueeze_2004 = None
        mul_752 = torch.ops.aten.mul.Tensor(mul_751, unsqueeze_2005);  mul_751 = unsqueeze_2005 = None
        unsqueeze_2006 = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_2007 = torch.ops.aten.unsqueeze.default(unsqueeze_2006, -1);  unsqueeze_2006 = None
        add_587 = torch.ops.aten.add.Tensor(mul_752, unsqueeze_2007);  mul_752 = unsqueeze_2007 = None
        relu_248 = torch.ops.aten.relu.default(add_587);  add_587 = None
        convolution_464 = torch.ops.aten.convolution.default(relu_248, arg292_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_248 = arg292_1 = None
        convolution_465 = torch.ops.aten.convolution.default(convolution_464, arg293_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_464 = arg293_1 = None
        add_588 = torch.ops.aten.add.Tensor(arg295_1, 0.001);  arg295_1 = None
        sqrt_251 = torch.ops.aten.sqrt.default(add_588);  add_588 = None
        reciprocal_251 = torch.ops.aten.reciprocal.default(sqrt_251);  sqrt_251 = None
        mul_753 = torch.ops.aten.mul.Tensor(reciprocal_251, 1);  reciprocal_251 = None
        unsqueeze_2008 = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_2009 = torch.ops.aten.unsqueeze.default(unsqueeze_2008, -1);  unsqueeze_2008 = None
        unsqueeze_2010 = torch.ops.aten.unsqueeze.default(mul_753, -1);  mul_753 = None
        unsqueeze_2011 = torch.ops.aten.unsqueeze.default(unsqueeze_2010, -1);  unsqueeze_2010 = None
        sub_251 = torch.ops.aten.sub.Tensor(convolution_465, unsqueeze_2009);  convolution_465 = unsqueeze_2009 = None
        mul_754 = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_2011);  sub_251 = unsqueeze_2011 = None
        unsqueeze_2012 = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_2013 = torch.ops.aten.unsqueeze.default(unsqueeze_2012, -1);  unsqueeze_2012 = None
        mul_755 = torch.ops.aten.mul.Tensor(mul_754, unsqueeze_2013);  mul_754 = unsqueeze_2013 = None
        unsqueeze_2014 = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_2015 = torch.ops.aten.unsqueeze.default(unsqueeze_2014, -1);  unsqueeze_2014 = None
        add_589 = torch.ops.aten.add.Tensor(mul_755, unsqueeze_2015);  mul_755 = unsqueeze_2015 = None
        _low_memory_max_pool2d_with_offsets_52 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_580, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_104 = _low_memory_max_pool2d_with_offsets_52[0];  _low_memory_max_pool2d_with_offsets_52 = None
        add_590 = torch.ops.aten.add.Tensor(add_589, getitem_104);  add_589 = getitem_104 = None
        relu_249 = torch.ops.aten.relu.default(add_580)
        convolution_466 = torch.ops.aten.convolution.default(relu_249, arg298_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_249 = arg298_1 = None
        convolution_467 = torch.ops.aten.convolution.default(convolution_466, arg299_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_466 = arg299_1 = None
        add_591 = torch.ops.aten.add.Tensor(arg301_1, 0.001);  arg301_1 = None
        sqrt_252 = torch.ops.aten.sqrt.default(add_591);  add_591 = None
        reciprocal_252 = torch.ops.aten.reciprocal.default(sqrt_252);  sqrt_252 = None
        mul_756 = torch.ops.aten.mul.Tensor(reciprocal_252, 1);  reciprocal_252 = None
        unsqueeze_2016 = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_2017 = torch.ops.aten.unsqueeze.default(unsqueeze_2016, -1);  unsqueeze_2016 = None
        unsqueeze_2018 = torch.ops.aten.unsqueeze.default(mul_756, -1);  mul_756 = None
        unsqueeze_2019 = torch.ops.aten.unsqueeze.default(unsqueeze_2018, -1);  unsqueeze_2018 = None
        sub_252 = torch.ops.aten.sub.Tensor(convolution_467, unsqueeze_2017);  convolution_467 = unsqueeze_2017 = None
        mul_757 = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_2019);  sub_252 = unsqueeze_2019 = None
        unsqueeze_2020 = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_2021 = torch.ops.aten.unsqueeze.default(unsqueeze_2020, -1);  unsqueeze_2020 = None
        mul_758 = torch.ops.aten.mul.Tensor(mul_757, unsqueeze_2021);  mul_757 = unsqueeze_2021 = None
        unsqueeze_2022 = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_2023 = torch.ops.aten.unsqueeze.default(unsqueeze_2022, -1);  unsqueeze_2022 = None
        add_592 = torch.ops.aten.add.Tensor(mul_758, unsqueeze_2023);  mul_758 = unsqueeze_2023 = None
        relu_250 = torch.ops.aten.relu.default(add_592);  add_592 = None
        convolution_468 = torch.ops.aten.convolution.default(relu_250, arg304_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_250 = arg304_1 = None
        convolution_469 = torch.ops.aten.convolution.default(convolution_468, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_468 = arg305_1 = None
        add_593 = torch.ops.aten.add.Tensor(arg307_1, 0.001);  arg307_1 = None
        sqrt_253 = torch.ops.aten.sqrt.default(add_593);  add_593 = None
        reciprocal_253 = torch.ops.aten.reciprocal.default(sqrt_253);  sqrt_253 = None
        mul_759 = torch.ops.aten.mul.Tensor(reciprocal_253, 1);  reciprocal_253 = None
        unsqueeze_2024 = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_2025 = torch.ops.aten.unsqueeze.default(unsqueeze_2024, -1);  unsqueeze_2024 = None
        unsqueeze_2026 = torch.ops.aten.unsqueeze.default(mul_759, -1);  mul_759 = None
        unsqueeze_2027 = torch.ops.aten.unsqueeze.default(unsqueeze_2026, -1);  unsqueeze_2026 = None
        sub_253 = torch.ops.aten.sub.Tensor(convolution_469, unsqueeze_2025);  convolution_469 = unsqueeze_2025 = None
        mul_760 = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_2027);  sub_253 = unsqueeze_2027 = None
        unsqueeze_2028 = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_2029 = torch.ops.aten.unsqueeze.default(unsqueeze_2028, -1);  unsqueeze_2028 = None
        mul_761 = torch.ops.aten.mul.Tensor(mul_760, unsqueeze_2029);  mul_760 = unsqueeze_2029 = None
        unsqueeze_2030 = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_2031 = torch.ops.aten.unsqueeze.default(unsqueeze_2030, -1);  unsqueeze_2030 = None
        add_594 = torch.ops.aten.add.Tensor(mul_761, unsqueeze_2031);  mul_761 = unsqueeze_2031 = None
        relu_251 = torch.ops.aten.relu.default(add_580)
        convolution_470 = torch.ops.aten.convolution.default(relu_251, arg310_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_251 = arg310_1 = None
        convolution_471 = torch.ops.aten.convolution.default(convolution_470, arg311_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_470 = arg311_1 = None
        add_595 = torch.ops.aten.add.Tensor(arg313_1, 0.001);  arg313_1 = None
        sqrt_254 = torch.ops.aten.sqrt.default(add_595);  add_595 = None
        reciprocal_254 = torch.ops.aten.reciprocal.default(sqrt_254);  sqrt_254 = None
        mul_762 = torch.ops.aten.mul.Tensor(reciprocal_254, 1);  reciprocal_254 = None
        unsqueeze_2032 = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_2033 = torch.ops.aten.unsqueeze.default(unsqueeze_2032, -1);  unsqueeze_2032 = None
        unsqueeze_2034 = torch.ops.aten.unsqueeze.default(mul_762, -1);  mul_762 = None
        unsqueeze_2035 = torch.ops.aten.unsqueeze.default(unsqueeze_2034, -1);  unsqueeze_2034 = None
        sub_254 = torch.ops.aten.sub.Tensor(convolution_471, unsqueeze_2033);  convolution_471 = unsqueeze_2033 = None
        mul_763 = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_2035);  sub_254 = unsqueeze_2035 = None
        unsqueeze_2036 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_2037 = torch.ops.aten.unsqueeze.default(unsqueeze_2036, -1);  unsqueeze_2036 = None
        mul_764 = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_2037);  mul_763 = unsqueeze_2037 = None
        unsqueeze_2038 = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_2039 = torch.ops.aten.unsqueeze.default(unsqueeze_2038, -1);  unsqueeze_2038 = None
        add_596 = torch.ops.aten.add.Tensor(mul_764, unsqueeze_2039);  mul_764 = unsqueeze_2039 = None
        relu_252 = torch.ops.aten.relu.default(add_596);  add_596 = None
        convolution_472 = torch.ops.aten.convolution.default(relu_252, arg316_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_252 = arg316_1 = None
        convolution_473 = torch.ops.aten.convolution.default(convolution_472, arg317_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_472 = arg317_1 = None
        add_597 = torch.ops.aten.add.Tensor(arg319_1, 0.001);  arg319_1 = None
        sqrt_255 = torch.ops.aten.sqrt.default(add_597);  add_597 = None
        reciprocal_255 = torch.ops.aten.reciprocal.default(sqrt_255);  sqrt_255 = None
        mul_765 = torch.ops.aten.mul.Tensor(reciprocal_255, 1);  reciprocal_255 = None
        unsqueeze_2040 = torch.ops.aten.unsqueeze.default(arg318_1, -1);  arg318_1 = None
        unsqueeze_2041 = torch.ops.aten.unsqueeze.default(unsqueeze_2040, -1);  unsqueeze_2040 = None
        unsqueeze_2042 = torch.ops.aten.unsqueeze.default(mul_765, -1);  mul_765 = None
        unsqueeze_2043 = torch.ops.aten.unsqueeze.default(unsqueeze_2042, -1);  unsqueeze_2042 = None
        sub_255 = torch.ops.aten.sub.Tensor(convolution_473, unsqueeze_2041);  convolution_473 = unsqueeze_2041 = None
        mul_766 = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_2043);  sub_255 = unsqueeze_2043 = None
        unsqueeze_2044 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_2045 = torch.ops.aten.unsqueeze.default(unsqueeze_2044, -1);  unsqueeze_2044 = None
        mul_767 = torch.ops.aten.mul.Tensor(mul_766, unsqueeze_2045);  mul_766 = unsqueeze_2045 = None
        unsqueeze_2046 = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_2047 = torch.ops.aten.unsqueeze.default(unsqueeze_2046, -1);  unsqueeze_2046 = None
        add_598 = torch.ops.aten.add.Tensor(mul_767, unsqueeze_2047);  mul_767 = unsqueeze_2047 = None
        add_599 = torch.ops.aten.add.Tensor(add_594, add_598);  add_594 = add_598 = None
        relu_253 = torch.ops.aten.relu.default(add_599)
        convolution_474 = torch.ops.aten.convolution.default(relu_253, arg322_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_253 = arg322_1 = None
        convolution_475 = torch.ops.aten.convolution.default(convolution_474, arg323_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_474 = arg323_1 = None
        add_600 = torch.ops.aten.add.Tensor(arg325_1, 0.001);  arg325_1 = None
        sqrt_256 = torch.ops.aten.sqrt.default(add_600);  add_600 = None
        reciprocal_256 = torch.ops.aten.reciprocal.default(sqrt_256);  sqrt_256 = None
        mul_768 = torch.ops.aten.mul.Tensor(reciprocal_256, 1);  reciprocal_256 = None
        unsqueeze_2048 = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_2049 = torch.ops.aten.unsqueeze.default(unsqueeze_2048, -1);  unsqueeze_2048 = None
        unsqueeze_2050 = torch.ops.aten.unsqueeze.default(mul_768, -1);  mul_768 = None
        unsqueeze_2051 = torch.ops.aten.unsqueeze.default(unsqueeze_2050, -1);  unsqueeze_2050 = None
        sub_256 = torch.ops.aten.sub.Tensor(convolution_475, unsqueeze_2049);  convolution_475 = unsqueeze_2049 = None
        mul_769 = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_2051);  sub_256 = unsqueeze_2051 = None
        unsqueeze_2052 = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
        unsqueeze_2053 = torch.ops.aten.unsqueeze.default(unsqueeze_2052, -1);  unsqueeze_2052 = None
        mul_770 = torch.ops.aten.mul.Tensor(mul_769, unsqueeze_2053);  mul_769 = unsqueeze_2053 = None
        unsqueeze_2054 = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_2055 = torch.ops.aten.unsqueeze.default(unsqueeze_2054, -1);  unsqueeze_2054 = None
        add_601 = torch.ops.aten.add.Tensor(mul_770, unsqueeze_2055);  mul_770 = unsqueeze_2055 = None
        relu_254 = torch.ops.aten.relu.default(add_601);  add_601 = None
        convolution_476 = torch.ops.aten.convolution.default(relu_254, arg328_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_254 = arg328_1 = None
        convolution_477 = torch.ops.aten.convolution.default(convolution_476, arg329_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_476 = arg329_1 = None
        add_602 = torch.ops.aten.add.Tensor(arg331_1, 0.001);  arg331_1 = None
        sqrt_257 = torch.ops.aten.sqrt.default(add_602);  add_602 = None
        reciprocal_257 = torch.ops.aten.reciprocal.default(sqrt_257);  sqrt_257 = None
        mul_771 = torch.ops.aten.mul.Tensor(reciprocal_257, 1);  reciprocal_257 = None
        unsqueeze_2056 = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_2057 = torch.ops.aten.unsqueeze.default(unsqueeze_2056, -1);  unsqueeze_2056 = None
        unsqueeze_2058 = torch.ops.aten.unsqueeze.default(mul_771, -1);  mul_771 = None
        unsqueeze_2059 = torch.ops.aten.unsqueeze.default(unsqueeze_2058, -1);  unsqueeze_2058 = None
        sub_257 = torch.ops.aten.sub.Tensor(convolution_477, unsqueeze_2057);  convolution_477 = unsqueeze_2057 = None
        mul_772 = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_2059);  sub_257 = unsqueeze_2059 = None
        unsqueeze_2060 = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_2061 = torch.ops.aten.unsqueeze.default(unsqueeze_2060, -1);  unsqueeze_2060 = None
        mul_773 = torch.ops.aten.mul.Tensor(mul_772, unsqueeze_2061);  mul_772 = unsqueeze_2061 = None
        unsqueeze_2062 = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_2063 = torch.ops.aten.unsqueeze.default(unsqueeze_2062, -1);  unsqueeze_2062 = None
        add_603 = torch.ops.aten.add.Tensor(mul_773, unsqueeze_2063);  mul_773 = unsqueeze_2063 = None
        _low_memory_max_pool2d_with_offsets_53 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_580, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_106 = _low_memory_max_pool2d_with_offsets_53[0];  _low_memory_max_pool2d_with_offsets_53 = None
        add_604 = torch.ops.aten.add.Tensor(add_603, getitem_106);  add_603 = getitem_106 = None
        relu_255 = torch.ops.aten.relu.default(add_578);  add_578 = None
        convolution_478 = torch.ops.aten.convolution.default(relu_255, arg334_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_255 = arg334_1 = None
        convolution_479 = torch.ops.aten.convolution.default(convolution_478, arg335_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_478 = arg335_1 = None
        add_605 = torch.ops.aten.add.Tensor(arg337_1, 0.001);  arg337_1 = None
        sqrt_258 = torch.ops.aten.sqrt.default(add_605);  add_605 = None
        reciprocal_258 = torch.ops.aten.reciprocal.default(sqrt_258);  sqrt_258 = None
        mul_774 = torch.ops.aten.mul.Tensor(reciprocal_258, 1);  reciprocal_258 = None
        unsqueeze_2064 = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_2065 = torch.ops.aten.unsqueeze.default(unsqueeze_2064, -1);  unsqueeze_2064 = None
        unsqueeze_2066 = torch.ops.aten.unsqueeze.default(mul_774, -1);  mul_774 = None
        unsqueeze_2067 = torch.ops.aten.unsqueeze.default(unsqueeze_2066, -1);  unsqueeze_2066 = None
        sub_258 = torch.ops.aten.sub.Tensor(convolution_479, unsqueeze_2065);  convolution_479 = unsqueeze_2065 = None
        mul_775 = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_2067);  sub_258 = unsqueeze_2067 = None
        unsqueeze_2068 = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
        unsqueeze_2069 = torch.ops.aten.unsqueeze.default(unsqueeze_2068, -1);  unsqueeze_2068 = None
        mul_776 = torch.ops.aten.mul.Tensor(mul_775, unsqueeze_2069);  mul_775 = unsqueeze_2069 = None
        unsqueeze_2070 = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_2071 = torch.ops.aten.unsqueeze.default(unsqueeze_2070, -1);  unsqueeze_2070 = None
        add_606 = torch.ops.aten.add.Tensor(mul_776, unsqueeze_2071);  mul_776 = unsqueeze_2071 = None
        relu_256 = torch.ops.aten.relu.default(add_606);  add_606 = None
        convolution_480 = torch.ops.aten.convolution.default(relu_256, arg340_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_256 = arg340_1 = None
        convolution_481 = torch.ops.aten.convolution.default(convolution_480, arg341_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_480 = arg341_1 = None
        add_607 = torch.ops.aten.add.Tensor(arg343_1, 0.001);  arg343_1 = None
        sqrt_259 = torch.ops.aten.sqrt.default(add_607);  add_607 = None
        reciprocal_259 = torch.ops.aten.reciprocal.default(sqrt_259);  sqrt_259 = None
        mul_777 = torch.ops.aten.mul.Tensor(reciprocal_259, 1);  reciprocal_259 = None
        unsqueeze_2072 = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_2073 = torch.ops.aten.unsqueeze.default(unsqueeze_2072, -1);  unsqueeze_2072 = None
        unsqueeze_2074 = torch.ops.aten.unsqueeze.default(mul_777, -1);  mul_777 = None
        unsqueeze_2075 = torch.ops.aten.unsqueeze.default(unsqueeze_2074, -1);  unsqueeze_2074 = None
        sub_259 = torch.ops.aten.sub.Tensor(convolution_481, unsqueeze_2073);  convolution_481 = unsqueeze_2073 = None
        mul_778 = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_2075);  sub_259 = unsqueeze_2075 = None
        unsqueeze_2076 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_2077 = torch.ops.aten.unsqueeze.default(unsqueeze_2076, -1);  unsqueeze_2076 = None
        mul_779 = torch.ops.aten.mul.Tensor(mul_778, unsqueeze_2077);  mul_778 = unsqueeze_2077 = None
        unsqueeze_2078 = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_2079 = torch.ops.aten.unsqueeze.default(unsqueeze_2078, -1);  unsqueeze_2078 = None
        add_608 = torch.ops.aten.add.Tensor(mul_779, unsqueeze_2079);  mul_779 = unsqueeze_2079 = None
        add_609 = torch.ops.aten.add.Tensor(add_608, add_580);  add_608 = add_580 = None
        cat_23 = torch.ops.aten.cat.default([add_585, add_590, add_599, add_604, add_609], 1);  add_585 = add_590 = add_599 = add_604 = add_609 = None
        relu_257 = torch.ops.aten.relu.default(cat_22);  cat_22 = None
        convolution_482 = torch.ops.aten.convolution.default(relu_257, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_257 = arg346_1 = None
        add_610 = torch.ops.aten.add.Tensor(arg348_1, 0.001);  arg348_1 = None
        sqrt_260 = torch.ops.aten.sqrt.default(add_610);  add_610 = None
        reciprocal_260 = torch.ops.aten.reciprocal.default(sqrt_260);  sqrt_260 = None
        mul_780 = torch.ops.aten.mul.Tensor(reciprocal_260, 1);  reciprocal_260 = None
        unsqueeze_2080 = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_2081 = torch.ops.aten.unsqueeze.default(unsqueeze_2080, -1);  unsqueeze_2080 = None
        unsqueeze_2082 = torch.ops.aten.unsqueeze.default(mul_780, -1);  mul_780 = None
        unsqueeze_2083 = torch.ops.aten.unsqueeze.default(unsqueeze_2082, -1);  unsqueeze_2082 = None
        sub_260 = torch.ops.aten.sub.Tensor(convolution_482, unsqueeze_2081);  convolution_482 = unsqueeze_2081 = None
        mul_781 = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_2083);  sub_260 = unsqueeze_2083 = None
        unsqueeze_2084 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_2085 = torch.ops.aten.unsqueeze.default(unsqueeze_2084, -1);  unsqueeze_2084 = None
        mul_782 = torch.ops.aten.mul.Tensor(mul_781, unsqueeze_2085);  mul_781 = unsqueeze_2085 = None
        unsqueeze_2086 = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_2087 = torch.ops.aten.unsqueeze.default(unsqueeze_2086, -1);  unsqueeze_2086 = None
        add_611 = torch.ops.aten.add.Tensor(mul_782, unsqueeze_2087);  mul_782 = unsqueeze_2087 = None
        relu_258 = torch.ops.aten.relu.default(cat_23)
        convolution_483 = torch.ops.aten.convolution.default(relu_258, arg351_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_258 = arg351_1 = None
        add_612 = torch.ops.aten.add.Tensor(arg353_1, 0.001);  arg353_1 = None
        sqrt_261 = torch.ops.aten.sqrt.default(add_612);  add_612 = None
        reciprocal_261 = torch.ops.aten.reciprocal.default(sqrt_261);  sqrt_261 = None
        mul_783 = torch.ops.aten.mul.Tensor(reciprocal_261, 1);  reciprocal_261 = None
        unsqueeze_2088 = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_2089 = torch.ops.aten.unsqueeze.default(unsqueeze_2088, -1);  unsqueeze_2088 = None
        unsqueeze_2090 = torch.ops.aten.unsqueeze.default(mul_783, -1);  mul_783 = None
        unsqueeze_2091 = torch.ops.aten.unsqueeze.default(unsqueeze_2090, -1);  unsqueeze_2090 = None
        sub_261 = torch.ops.aten.sub.Tensor(convolution_483, unsqueeze_2089);  convolution_483 = unsqueeze_2089 = None
        mul_784 = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_2091);  sub_261 = unsqueeze_2091 = None
        unsqueeze_2092 = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_2093 = torch.ops.aten.unsqueeze.default(unsqueeze_2092, -1);  unsqueeze_2092 = None
        mul_785 = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_2093);  mul_784 = unsqueeze_2093 = None
        unsqueeze_2094 = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_2095 = torch.ops.aten.unsqueeze.default(unsqueeze_2094, -1);  unsqueeze_2094 = None
        add_613 = torch.ops.aten.add.Tensor(mul_785, unsqueeze_2095);  mul_785 = unsqueeze_2095 = None
        relu_259 = torch.ops.aten.relu.default(add_611)
        convolution_484 = torch.ops.aten.convolution.default(relu_259, arg356_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_259 = arg356_1 = None
        convolution_485 = torch.ops.aten.convolution.default(convolution_484, arg357_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_484 = arg357_1 = None
        add_614 = torch.ops.aten.add.Tensor(arg359_1, 0.001);  arg359_1 = None
        sqrt_262 = torch.ops.aten.sqrt.default(add_614);  add_614 = None
        reciprocal_262 = torch.ops.aten.reciprocal.default(sqrt_262);  sqrt_262 = None
        mul_786 = torch.ops.aten.mul.Tensor(reciprocal_262, 1);  reciprocal_262 = None
        unsqueeze_2096 = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
        unsqueeze_2097 = torch.ops.aten.unsqueeze.default(unsqueeze_2096, -1);  unsqueeze_2096 = None
        unsqueeze_2098 = torch.ops.aten.unsqueeze.default(mul_786, -1);  mul_786 = None
        unsqueeze_2099 = torch.ops.aten.unsqueeze.default(unsqueeze_2098, -1);  unsqueeze_2098 = None
        sub_262 = torch.ops.aten.sub.Tensor(convolution_485, unsqueeze_2097);  convolution_485 = unsqueeze_2097 = None
        mul_787 = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_2099);  sub_262 = unsqueeze_2099 = None
        unsqueeze_2100 = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_2101 = torch.ops.aten.unsqueeze.default(unsqueeze_2100, -1);  unsqueeze_2100 = None
        mul_788 = torch.ops.aten.mul.Tensor(mul_787, unsqueeze_2101);  mul_787 = unsqueeze_2101 = None
        unsqueeze_2102 = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
        unsqueeze_2103 = torch.ops.aten.unsqueeze.default(unsqueeze_2102, -1);  unsqueeze_2102 = None
        add_615 = torch.ops.aten.add.Tensor(mul_788, unsqueeze_2103);  mul_788 = unsqueeze_2103 = None
        relu_260 = torch.ops.aten.relu.default(add_615);  add_615 = None
        convolution_486 = torch.ops.aten.convolution.default(relu_260, arg362_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_260 = arg362_1 = None
        convolution_487 = torch.ops.aten.convolution.default(convolution_486, arg363_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_486 = arg363_1 = None
        add_616 = torch.ops.aten.add.Tensor(arg365_1, 0.001);  arg365_1 = None
        sqrt_263 = torch.ops.aten.sqrt.default(add_616);  add_616 = None
        reciprocal_263 = torch.ops.aten.reciprocal.default(sqrt_263);  sqrt_263 = None
        mul_789 = torch.ops.aten.mul.Tensor(reciprocal_263, 1);  reciprocal_263 = None
        unsqueeze_2104 = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_2105 = torch.ops.aten.unsqueeze.default(unsqueeze_2104, -1);  unsqueeze_2104 = None
        unsqueeze_2106 = torch.ops.aten.unsqueeze.default(mul_789, -1);  mul_789 = None
        unsqueeze_2107 = torch.ops.aten.unsqueeze.default(unsqueeze_2106, -1);  unsqueeze_2106 = None
        sub_263 = torch.ops.aten.sub.Tensor(convolution_487, unsqueeze_2105);  convolution_487 = unsqueeze_2105 = None
        mul_790 = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_2107);  sub_263 = unsqueeze_2107 = None
        unsqueeze_2108 = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
        unsqueeze_2109 = torch.ops.aten.unsqueeze.default(unsqueeze_2108, -1);  unsqueeze_2108 = None
        mul_791 = torch.ops.aten.mul.Tensor(mul_790, unsqueeze_2109);  mul_790 = unsqueeze_2109 = None
        unsqueeze_2110 = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_2111 = torch.ops.aten.unsqueeze.default(unsqueeze_2110, -1);  unsqueeze_2110 = None
        add_617 = torch.ops.aten.add.Tensor(mul_791, unsqueeze_2111);  mul_791 = unsqueeze_2111 = None
        _low_memory_max_pool2d_with_offsets_54 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_611, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_108 = _low_memory_max_pool2d_with_offsets_54[0];  _low_memory_max_pool2d_with_offsets_54 = None
        add_618 = torch.ops.aten.add.Tensor(add_617, getitem_108);  add_617 = getitem_108 = None
        relu_261 = torch.ops.aten.relu.default(add_613)
        convolution_488 = torch.ops.aten.convolution.default(relu_261, arg368_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_261 = arg368_1 = None
        convolution_489 = torch.ops.aten.convolution.default(convolution_488, arg369_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_488 = arg369_1 = None
        add_619 = torch.ops.aten.add.Tensor(arg371_1, 0.001);  arg371_1 = None
        sqrt_264 = torch.ops.aten.sqrt.default(add_619);  add_619 = None
        reciprocal_264 = torch.ops.aten.reciprocal.default(sqrt_264);  sqrt_264 = None
        mul_792 = torch.ops.aten.mul.Tensor(reciprocal_264, 1);  reciprocal_264 = None
        unsqueeze_2112 = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_2113 = torch.ops.aten.unsqueeze.default(unsqueeze_2112, -1);  unsqueeze_2112 = None
        unsqueeze_2114 = torch.ops.aten.unsqueeze.default(mul_792, -1);  mul_792 = None
        unsqueeze_2115 = torch.ops.aten.unsqueeze.default(unsqueeze_2114, -1);  unsqueeze_2114 = None
        sub_264 = torch.ops.aten.sub.Tensor(convolution_489, unsqueeze_2113);  convolution_489 = unsqueeze_2113 = None
        mul_793 = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_2115);  sub_264 = unsqueeze_2115 = None
        unsqueeze_2116 = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_2117 = torch.ops.aten.unsqueeze.default(unsqueeze_2116, -1);  unsqueeze_2116 = None
        mul_794 = torch.ops.aten.mul.Tensor(mul_793, unsqueeze_2117);  mul_793 = unsqueeze_2117 = None
        unsqueeze_2118 = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_2119 = torch.ops.aten.unsqueeze.default(unsqueeze_2118, -1);  unsqueeze_2118 = None
        add_620 = torch.ops.aten.add.Tensor(mul_794, unsqueeze_2119);  mul_794 = unsqueeze_2119 = None
        relu_262 = torch.ops.aten.relu.default(add_620);  add_620 = None
        convolution_490 = torch.ops.aten.convolution.default(relu_262, arg374_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_262 = arg374_1 = None
        convolution_491 = torch.ops.aten.convolution.default(convolution_490, arg375_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_490 = arg375_1 = None
        add_621 = torch.ops.aten.add.Tensor(arg377_1, 0.001);  arg377_1 = None
        sqrt_265 = torch.ops.aten.sqrt.default(add_621);  add_621 = None
        reciprocal_265 = torch.ops.aten.reciprocal.default(sqrt_265);  sqrt_265 = None
        mul_795 = torch.ops.aten.mul.Tensor(reciprocal_265, 1);  reciprocal_265 = None
        unsqueeze_2120 = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
        unsqueeze_2121 = torch.ops.aten.unsqueeze.default(unsqueeze_2120, -1);  unsqueeze_2120 = None
        unsqueeze_2122 = torch.ops.aten.unsqueeze.default(mul_795, -1);  mul_795 = None
        unsqueeze_2123 = torch.ops.aten.unsqueeze.default(unsqueeze_2122, -1);  unsqueeze_2122 = None
        sub_265 = torch.ops.aten.sub.Tensor(convolution_491, unsqueeze_2121);  convolution_491 = unsqueeze_2121 = None
        mul_796 = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_2123);  sub_265 = unsqueeze_2123 = None
        unsqueeze_2124 = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_2125 = torch.ops.aten.unsqueeze.default(unsqueeze_2124, -1);  unsqueeze_2124 = None
        mul_797 = torch.ops.aten.mul.Tensor(mul_796, unsqueeze_2125);  mul_796 = unsqueeze_2125 = None
        unsqueeze_2126 = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_2127 = torch.ops.aten.unsqueeze.default(unsqueeze_2126, -1);  unsqueeze_2126 = None
        add_622 = torch.ops.aten.add.Tensor(mul_797, unsqueeze_2127);  mul_797 = unsqueeze_2127 = None
        _low_memory_max_pool2d_with_offsets_55 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_613, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_110 = _low_memory_max_pool2d_with_offsets_55[0];  _low_memory_max_pool2d_with_offsets_55 = None
        add_623 = torch.ops.aten.add.Tensor(add_622, getitem_110);  add_622 = getitem_110 = None
        relu_263 = torch.ops.aten.relu.default(add_613)
        convolution_492 = torch.ops.aten.convolution.default(relu_263, arg380_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_263 = arg380_1 = None
        convolution_493 = torch.ops.aten.convolution.default(convolution_492, arg381_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_492 = arg381_1 = None
        add_624 = torch.ops.aten.add.Tensor(arg383_1, 0.001);  arg383_1 = None
        sqrt_266 = torch.ops.aten.sqrt.default(add_624);  add_624 = None
        reciprocal_266 = torch.ops.aten.reciprocal.default(sqrt_266);  sqrt_266 = None
        mul_798 = torch.ops.aten.mul.Tensor(reciprocal_266, 1);  reciprocal_266 = None
        unsqueeze_2128 = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
        unsqueeze_2129 = torch.ops.aten.unsqueeze.default(unsqueeze_2128, -1);  unsqueeze_2128 = None
        unsqueeze_2130 = torch.ops.aten.unsqueeze.default(mul_798, -1);  mul_798 = None
        unsqueeze_2131 = torch.ops.aten.unsqueeze.default(unsqueeze_2130, -1);  unsqueeze_2130 = None
        sub_266 = torch.ops.aten.sub.Tensor(convolution_493, unsqueeze_2129);  convolution_493 = unsqueeze_2129 = None
        mul_799 = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_2131);  sub_266 = unsqueeze_2131 = None
        unsqueeze_2132 = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_2133 = torch.ops.aten.unsqueeze.default(unsqueeze_2132, -1);  unsqueeze_2132 = None
        mul_800 = torch.ops.aten.mul.Tensor(mul_799, unsqueeze_2133);  mul_799 = unsqueeze_2133 = None
        unsqueeze_2134 = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_2135 = torch.ops.aten.unsqueeze.default(unsqueeze_2134, -1);  unsqueeze_2134 = None
        add_625 = torch.ops.aten.add.Tensor(mul_800, unsqueeze_2135);  mul_800 = unsqueeze_2135 = None
        relu_264 = torch.ops.aten.relu.default(add_625);  add_625 = None
        convolution_494 = torch.ops.aten.convolution.default(relu_264, arg386_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_264 = arg386_1 = None
        convolution_495 = torch.ops.aten.convolution.default(convolution_494, arg387_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_494 = arg387_1 = None
        add_626 = torch.ops.aten.add.Tensor(arg389_1, 0.001);  arg389_1 = None
        sqrt_267 = torch.ops.aten.sqrt.default(add_626);  add_626 = None
        reciprocal_267 = torch.ops.aten.reciprocal.default(sqrt_267);  sqrt_267 = None
        mul_801 = torch.ops.aten.mul.Tensor(reciprocal_267, 1);  reciprocal_267 = None
        unsqueeze_2136 = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_2137 = torch.ops.aten.unsqueeze.default(unsqueeze_2136, -1);  unsqueeze_2136 = None
        unsqueeze_2138 = torch.ops.aten.unsqueeze.default(mul_801, -1);  mul_801 = None
        unsqueeze_2139 = torch.ops.aten.unsqueeze.default(unsqueeze_2138, -1);  unsqueeze_2138 = None
        sub_267 = torch.ops.aten.sub.Tensor(convolution_495, unsqueeze_2137);  convolution_495 = unsqueeze_2137 = None
        mul_802 = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_2139);  sub_267 = unsqueeze_2139 = None
        unsqueeze_2140 = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_2141 = torch.ops.aten.unsqueeze.default(unsqueeze_2140, -1);  unsqueeze_2140 = None
        mul_803 = torch.ops.aten.mul.Tensor(mul_802, unsqueeze_2141);  mul_802 = unsqueeze_2141 = None
        unsqueeze_2142 = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_2143 = torch.ops.aten.unsqueeze.default(unsqueeze_2142, -1);  unsqueeze_2142 = None
        add_627 = torch.ops.aten.add.Tensor(mul_803, unsqueeze_2143);  mul_803 = unsqueeze_2143 = None
        relu_265 = torch.ops.aten.relu.default(add_613)
        convolution_496 = torch.ops.aten.convolution.default(relu_265, arg392_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_265 = arg392_1 = None
        convolution_497 = torch.ops.aten.convolution.default(convolution_496, arg393_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_496 = arg393_1 = None
        add_628 = torch.ops.aten.add.Tensor(arg395_1, 0.001);  arg395_1 = None
        sqrt_268 = torch.ops.aten.sqrt.default(add_628);  add_628 = None
        reciprocal_268 = torch.ops.aten.reciprocal.default(sqrt_268);  sqrt_268 = None
        mul_804 = torch.ops.aten.mul.Tensor(reciprocal_268, 1);  reciprocal_268 = None
        unsqueeze_2144 = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_2145 = torch.ops.aten.unsqueeze.default(unsqueeze_2144, -1);  unsqueeze_2144 = None
        unsqueeze_2146 = torch.ops.aten.unsqueeze.default(mul_804, -1);  mul_804 = None
        unsqueeze_2147 = torch.ops.aten.unsqueeze.default(unsqueeze_2146, -1);  unsqueeze_2146 = None
        sub_268 = torch.ops.aten.sub.Tensor(convolution_497, unsqueeze_2145);  convolution_497 = unsqueeze_2145 = None
        mul_805 = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_2147);  sub_268 = unsqueeze_2147 = None
        unsqueeze_2148 = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
        unsqueeze_2149 = torch.ops.aten.unsqueeze.default(unsqueeze_2148, -1);  unsqueeze_2148 = None
        mul_806 = torch.ops.aten.mul.Tensor(mul_805, unsqueeze_2149);  mul_805 = unsqueeze_2149 = None
        unsqueeze_2150 = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_2151 = torch.ops.aten.unsqueeze.default(unsqueeze_2150, -1);  unsqueeze_2150 = None
        add_629 = torch.ops.aten.add.Tensor(mul_806, unsqueeze_2151);  mul_806 = unsqueeze_2151 = None
        relu_266 = torch.ops.aten.relu.default(add_629);  add_629 = None
        convolution_498 = torch.ops.aten.convolution.default(relu_266, arg398_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_266 = arg398_1 = None
        convolution_499 = torch.ops.aten.convolution.default(convolution_498, arg399_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_498 = arg399_1 = None
        add_630 = torch.ops.aten.add.Tensor(arg401_1, 0.001);  arg401_1 = None
        sqrt_269 = torch.ops.aten.sqrt.default(add_630);  add_630 = None
        reciprocal_269 = torch.ops.aten.reciprocal.default(sqrt_269);  sqrt_269 = None
        mul_807 = torch.ops.aten.mul.Tensor(reciprocal_269, 1);  reciprocal_269 = None
        unsqueeze_2152 = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_2153 = torch.ops.aten.unsqueeze.default(unsqueeze_2152, -1);  unsqueeze_2152 = None
        unsqueeze_2154 = torch.ops.aten.unsqueeze.default(mul_807, -1);  mul_807 = None
        unsqueeze_2155 = torch.ops.aten.unsqueeze.default(unsqueeze_2154, -1);  unsqueeze_2154 = None
        sub_269 = torch.ops.aten.sub.Tensor(convolution_499, unsqueeze_2153);  convolution_499 = unsqueeze_2153 = None
        mul_808 = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_2155);  sub_269 = unsqueeze_2155 = None
        unsqueeze_2156 = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
        unsqueeze_2157 = torch.ops.aten.unsqueeze.default(unsqueeze_2156, -1);  unsqueeze_2156 = None
        mul_809 = torch.ops.aten.mul.Tensor(mul_808, unsqueeze_2157);  mul_808 = unsqueeze_2157 = None
        unsqueeze_2158 = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_2159 = torch.ops.aten.unsqueeze.default(unsqueeze_2158, -1);  unsqueeze_2158 = None
        add_631 = torch.ops.aten.add.Tensor(mul_809, unsqueeze_2159);  mul_809 = unsqueeze_2159 = None
        add_632 = torch.ops.aten.add.Tensor(add_627, add_631);  add_627 = add_631 = None
        relu_267 = torch.ops.aten.relu.default(add_632)
        convolution_500 = torch.ops.aten.convolution.default(relu_267, arg404_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_267 = arg404_1 = None
        convolution_501 = torch.ops.aten.convolution.default(convolution_500, arg405_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_500 = arg405_1 = None
        add_633 = torch.ops.aten.add.Tensor(arg407_1, 0.001);  arg407_1 = None
        sqrt_270 = torch.ops.aten.sqrt.default(add_633);  add_633 = None
        reciprocal_270 = torch.ops.aten.reciprocal.default(sqrt_270);  sqrt_270 = None
        mul_810 = torch.ops.aten.mul.Tensor(reciprocal_270, 1);  reciprocal_270 = None
        unsqueeze_2160 = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
        unsqueeze_2161 = torch.ops.aten.unsqueeze.default(unsqueeze_2160, -1);  unsqueeze_2160 = None
        unsqueeze_2162 = torch.ops.aten.unsqueeze.default(mul_810, -1);  mul_810 = None
        unsqueeze_2163 = torch.ops.aten.unsqueeze.default(unsqueeze_2162, -1);  unsqueeze_2162 = None
        sub_270 = torch.ops.aten.sub.Tensor(convolution_501, unsqueeze_2161);  convolution_501 = unsqueeze_2161 = None
        mul_811 = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_2163);  sub_270 = unsqueeze_2163 = None
        unsqueeze_2164 = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
        unsqueeze_2165 = torch.ops.aten.unsqueeze.default(unsqueeze_2164, -1);  unsqueeze_2164 = None
        mul_812 = torch.ops.aten.mul.Tensor(mul_811, unsqueeze_2165);  mul_811 = unsqueeze_2165 = None
        unsqueeze_2166 = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_2167 = torch.ops.aten.unsqueeze.default(unsqueeze_2166, -1);  unsqueeze_2166 = None
        add_634 = torch.ops.aten.add.Tensor(mul_812, unsqueeze_2167);  mul_812 = unsqueeze_2167 = None
        relu_268 = torch.ops.aten.relu.default(add_634);  add_634 = None
        convolution_502 = torch.ops.aten.convolution.default(relu_268, arg410_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_268 = arg410_1 = None
        convolution_503 = torch.ops.aten.convolution.default(convolution_502, arg411_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_502 = arg411_1 = None
        add_635 = torch.ops.aten.add.Tensor(arg413_1, 0.001);  arg413_1 = None
        sqrt_271 = torch.ops.aten.sqrt.default(add_635);  add_635 = None
        reciprocal_271 = torch.ops.aten.reciprocal.default(sqrt_271);  sqrt_271 = None
        mul_813 = torch.ops.aten.mul.Tensor(reciprocal_271, 1);  reciprocal_271 = None
        unsqueeze_2168 = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
        unsqueeze_2169 = torch.ops.aten.unsqueeze.default(unsqueeze_2168, -1);  unsqueeze_2168 = None
        unsqueeze_2170 = torch.ops.aten.unsqueeze.default(mul_813, -1);  mul_813 = None
        unsqueeze_2171 = torch.ops.aten.unsqueeze.default(unsqueeze_2170, -1);  unsqueeze_2170 = None
        sub_271 = torch.ops.aten.sub.Tensor(convolution_503, unsqueeze_2169);  convolution_503 = unsqueeze_2169 = None
        mul_814 = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_2171);  sub_271 = unsqueeze_2171 = None
        unsqueeze_2172 = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_2173 = torch.ops.aten.unsqueeze.default(unsqueeze_2172, -1);  unsqueeze_2172 = None
        mul_815 = torch.ops.aten.mul.Tensor(mul_814, unsqueeze_2173);  mul_814 = unsqueeze_2173 = None
        unsqueeze_2174 = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_2175 = torch.ops.aten.unsqueeze.default(unsqueeze_2174, -1);  unsqueeze_2174 = None
        add_636 = torch.ops.aten.add.Tensor(mul_815, unsqueeze_2175);  mul_815 = unsqueeze_2175 = None
        _low_memory_max_pool2d_with_offsets_56 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_613, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_112 = _low_memory_max_pool2d_with_offsets_56[0];  _low_memory_max_pool2d_with_offsets_56 = None
        add_637 = torch.ops.aten.add.Tensor(add_636, getitem_112);  add_636 = getitem_112 = None
        relu_269 = torch.ops.aten.relu.default(add_611);  add_611 = None
        convolution_504 = torch.ops.aten.convolution.default(relu_269, arg416_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_269 = arg416_1 = None
        convolution_505 = torch.ops.aten.convolution.default(convolution_504, arg417_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_504 = arg417_1 = None
        add_638 = torch.ops.aten.add.Tensor(arg419_1, 0.001);  arg419_1 = None
        sqrt_272 = torch.ops.aten.sqrt.default(add_638);  add_638 = None
        reciprocal_272 = torch.ops.aten.reciprocal.default(sqrt_272);  sqrt_272 = None
        mul_816 = torch.ops.aten.mul.Tensor(reciprocal_272, 1);  reciprocal_272 = None
        unsqueeze_2176 = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_2177 = torch.ops.aten.unsqueeze.default(unsqueeze_2176, -1);  unsqueeze_2176 = None
        unsqueeze_2178 = torch.ops.aten.unsqueeze.default(mul_816, -1);  mul_816 = None
        unsqueeze_2179 = torch.ops.aten.unsqueeze.default(unsqueeze_2178, -1);  unsqueeze_2178 = None
        sub_272 = torch.ops.aten.sub.Tensor(convolution_505, unsqueeze_2177);  convolution_505 = unsqueeze_2177 = None
        mul_817 = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_2179);  sub_272 = unsqueeze_2179 = None
        unsqueeze_2180 = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_2181 = torch.ops.aten.unsqueeze.default(unsqueeze_2180, -1);  unsqueeze_2180 = None
        mul_818 = torch.ops.aten.mul.Tensor(mul_817, unsqueeze_2181);  mul_817 = unsqueeze_2181 = None
        unsqueeze_2182 = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
        unsqueeze_2183 = torch.ops.aten.unsqueeze.default(unsqueeze_2182, -1);  unsqueeze_2182 = None
        add_639 = torch.ops.aten.add.Tensor(mul_818, unsqueeze_2183);  mul_818 = unsqueeze_2183 = None
        relu_270 = torch.ops.aten.relu.default(add_639);  add_639 = None
        convolution_506 = torch.ops.aten.convolution.default(relu_270, arg422_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_270 = arg422_1 = None
        convolution_507 = torch.ops.aten.convolution.default(convolution_506, arg423_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_506 = arg423_1 = None
        add_640 = torch.ops.aten.add.Tensor(arg425_1, 0.001);  arg425_1 = None
        sqrt_273 = torch.ops.aten.sqrt.default(add_640);  add_640 = None
        reciprocal_273 = torch.ops.aten.reciprocal.default(sqrt_273);  sqrt_273 = None
        mul_819 = torch.ops.aten.mul.Tensor(reciprocal_273, 1);  reciprocal_273 = None
        unsqueeze_2184 = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_2185 = torch.ops.aten.unsqueeze.default(unsqueeze_2184, -1);  unsqueeze_2184 = None
        unsqueeze_2186 = torch.ops.aten.unsqueeze.default(mul_819, -1);  mul_819 = None
        unsqueeze_2187 = torch.ops.aten.unsqueeze.default(unsqueeze_2186, -1);  unsqueeze_2186 = None
        sub_273 = torch.ops.aten.sub.Tensor(convolution_507, unsqueeze_2185);  convolution_507 = unsqueeze_2185 = None
        mul_820 = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_2187);  sub_273 = unsqueeze_2187 = None
        unsqueeze_2188 = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
        unsqueeze_2189 = torch.ops.aten.unsqueeze.default(unsqueeze_2188, -1);  unsqueeze_2188 = None
        mul_821 = torch.ops.aten.mul.Tensor(mul_820, unsqueeze_2189);  mul_820 = unsqueeze_2189 = None
        unsqueeze_2190 = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
        unsqueeze_2191 = torch.ops.aten.unsqueeze.default(unsqueeze_2190, -1);  unsqueeze_2190 = None
        add_641 = torch.ops.aten.add.Tensor(mul_821, unsqueeze_2191);  mul_821 = unsqueeze_2191 = None
        add_642 = torch.ops.aten.add.Tensor(add_641, add_613);  add_641 = add_613 = None
        cat_24 = torch.ops.aten.cat.default([add_618, add_623, add_632, add_637, add_642], 1);  add_618 = add_623 = add_632 = add_637 = add_642 = None
        relu_271 = torch.ops.aten.relu.default(cat_23);  cat_23 = None
        convolution_508 = torch.ops.aten.convolution.default(relu_271, arg428_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_271 = arg428_1 = None
        add_643 = torch.ops.aten.add.Tensor(arg430_1, 0.001);  arg430_1 = None
        sqrt_274 = torch.ops.aten.sqrt.default(add_643);  add_643 = None
        reciprocal_274 = torch.ops.aten.reciprocal.default(sqrt_274);  sqrt_274 = None
        mul_822 = torch.ops.aten.mul.Tensor(reciprocal_274, 1);  reciprocal_274 = None
        unsqueeze_2192 = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_2193 = torch.ops.aten.unsqueeze.default(unsqueeze_2192, -1);  unsqueeze_2192 = None
        unsqueeze_2194 = torch.ops.aten.unsqueeze.default(mul_822, -1);  mul_822 = None
        unsqueeze_2195 = torch.ops.aten.unsqueeze.default(unsqueeze_2194, -1);  unsqueeze_2194 = None
        sub_274 = torch.ops.aten.sub.Tensor(convolution_508, unsqueeze_2193);  convolution_508 = unsqueeze_2193 = None
        mul_823 = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_2195);  sub_274 = unsqueeze_2195 = None
        unsqueeze_2196 = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_2197 = torch.ops.aten.unsqueeze.default(unsqueeze_2196, -1);  unsqueeze_2196 = None
        mul_824 = torch.ops.aten.mul.Tensor(mul_823, unsqueeze_2197);  mul_823 = unsqueeze_2197 = None
        unsqueeze_2198 = torch.ops.aten.unsqueeze.default(arg432_1, -1);  arg432_1 = None
        unsqueeze_2199 = torch.ops.aten.unsqueeze.default(unsqueeze_2198, -1);  unsqueeze_2198 = None
        add_644 = torch.ops.aten.add.Tensor(mul_824, unsqueeze_2199);  mul_824 = unsqueeze_2199 = None
        relu_272 = torch.ops.aten.relu.default(cat_24)
        convolution_509 = torch.ops.aten.convolution.default(relu_272, arg433_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_272 = arg433_1 = None
        add_645 = torch.ops.aten.add.Tensor(arg435_1, 0.001);  arg435_1 = None
        sqrt_275 = torch.ops.aten.sqrt.default(add_645);  add_645 = None
        reciprocal_275 = torch.ops.aten.reciprocal.default(sqrt_275);  sqrt_275 = None
        mul_825 = torch.ops.aten.mul.Tensor(reciprocal_275, 1);  reciprocal_275 = None
        unsqueeze_2200 = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_2201 = torch.ops.aten.unsqueeze.default(unsqueeze_2200, -1);  unsqueeze_2200 = None
        unsqueeze_2202 = torch.ops.aten.unsqueeze.default(mul_825, -1);  mul_825 = None
        unsqueeze_2203 = torch.ops.aten.unsqueeze.default(unsqueeze_2202, -1);  unsqueeze_2202 = None
        sub_275 = torch.ops.aten.sub.Tensor(convolution_509, unsqueeze_2201);  convolution_509 = unsqueeze_2201 = None
        mul_826 = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_2203);  sub_275 = unsqueeze_2203 = None
        unsqueeze_2204 = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
        unsqueeze_2205 = torch.ops.aten.unsqueeze.default(unsqueeze_2204, -1);  unsqueeze_2204 = None
        mul_827 = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_2205);  mul_826 = unsqueeze_2205 = None
        unsqueeze_2206 = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
        unsqueeze_2207 = torch.ops.aten.unsqueeze.default(unsqueeze_2206, -1);  unsqueeze_2206 = None
        add_646 = torch.ops.aten.add.Tensor(mul_827, unsqueeze_2207);  mul_827 = unsqueeze_2207 = None
        relu_273 = torch.ops.aten.relu.default(add_644)
        convolution_510 = torch.ops.aten.convolution.default(relu_273, arg438_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_273 = arg438_1 = None
        convolution_511 = torch.ops.aten.convolution.default(convolution_510, arg439_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_510 = arg439_1 = None
        add_647 = torch.ops.aten.add.Tensor(arg441_1, 0.001);  arg441_1 = None
        sqrt_276 = torch.ops.aten.sqrt.default(add_647);  add_647 = None
        reciprocal_276 = torch.ops.aten.reciprocal.default(sqrt_276);  sqrt_276 = None
        mul_828 = torch.ops.aten.mul.Tensor(reciprocal_276, 1);  reciprocal_276 = None
        unsqueeze_2208 = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
        unsqueeze_2209 = torch.ops.aten.unsqueeze.default(unsqueeze_2208, -1);  unsqueeze_2208 = None
        unsqueeze_2210 = torch.ops.aten.unsqueeze.default(mul_828, -1);  mul_828 = None
        unsqueeze_2211 = torch.ops.aten.unsqueeze.default(unsqueeze_2210, -1);  unsqueeze_2210 = None
        sub_276 = torch.ops.aten.sub.Tensor(convolution_511, unsqueeze_2209);  convolution_511 = unsqueeze_2209 = None
        mul_829 = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_2211);  sub_276 = unsqueeze_2211 = None
        unsqueeze_2212 = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
        unsqueeze_2213 = torch.ops.aten.unsqueeze.default(unsqueeze_2212, -1);  unsqueeze_2212 = None
        mul_830 = torch.ops.aten.mul.Tensor(mul_829, unsqueeze_2213);  mul_829 = unsqueeze_2213 = None
        unsqueeze_2214 = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
        unsqueeze_2215 = torch.ops.aten.unsqueeze.default(unsqueeze_2214, -1);  unsqueeze_2214 = None
        add_648 = torch.ops.aten.add.Tensor(mul_830, unsqueeze_2215);  mul_830 = unsqueeze_2215 = None
        relu_274 = torch.ops.aten.relu.default(add_648);  add_648 = None
        convolution_512 = torch.ops.aten.convolution.default(relu_274, arg444_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_274 = arg444_1 = None
        convolution_513 = torch.ops.aten.convolution.default(convolution_512, arg445_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_512 = arg445_1 = None
        add_649 = torch.ops.aten.add.Tensor(arg447_1, 0.001);  arg447_1 = None
        sqrt_277 = torch.ops.aten.sqrt.default(add_649);  add_649 = None
        reciprocal_277 = torch.ops.aten.reciprocal.default(sqrt_277);  sqrt_277 = None
        mul_831 = torch.ops.aten.mul.Tensor(reciprocal_277, 1);  reciprocal_277 = None
        unsqueeze_2216 = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
        unsqueeze_2217 = torch.ops.aten.unsqueeze.default(unsqueeze_2216, -1);  unsqueeze_2216 = None
        unsqueeze_2218 = torch.ops.aten.unsqueeze.default(mul_831, -1);  mul_831 = None
        unsqueeze_2219 = torch.ops.aten.unsqueeze.default(unsqueeze_2218, -1);  unsqueeze_2218 = None
        sub_277 = torch.ops.aten.sub.Tensor(convolution_513, unsqueeze_2217);  convolution_513 = unsqueeze_2217 = None
        mul_832 = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_2219);  sub_277 = unsqueeze_2219 = None
        unsqueeze_2220 = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
        unsqueeze_2221 = torch.ops.aten.unsqueeze.default(unsqueeze_2220, -1);  unsqueeze_2220 = None
        mul_833 = torch.ops.aten.mul.Tensor(mul_832, unsqueeze_2221);  mul_832 = unsqueeze_2221 = None
        unsqueeze_2222 = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_2223 = torch.ops.aten.unsqueeze.default(unsqueeze_2222, -1);  unsqueeze_2222 = None
        add_650 = torch.ops.aten.add.Tensor(mul_833, unsqueeze_2223);  mul_833 = unsqueeze_2223 = None
        _low_memory_max_pool2d_with_offsets_57 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_644, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_114 = _low_memory_max_pool2d_with_offsets_57[0];  _low_memory_max_pool2d_with_offsets_57 = None
        add_651 = torch.ops.aten.add.Tensor(add_650, getitem_114);  add_650 = getitem_114 = None
        relu_275 = torch.ops.aten.relu.default(add_646)
        convolution_514 = torch.ops.aten.convolution.default(relu_275, arg450_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_275 = arg450_1 = None
        convolution_515 = torch.ops.aten.convolution.default(convolution_514, arg451_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_514 = arg451_1 = None
        add_652 = torch.ops.aten.add.Tensor(arg453_1, 0.001);  arg453_1 = None
        sqrt_278 = torch.ops.aten.sqrt.default(add_652);  add_652 = None
        reciprocal_278 = torch.ops.aten.reciprocal.default(sqrt_278);  sqrt_278 = None
        mul_834 = torch.ops.aten.mul.Tensor(reciprocal_278, 1);  reciprocal_278 = None
        unsqueeze_2224 = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
        unsqueeze_2225 = torch.ops.aten.unsqueeze.default(unsqueeze_2224, -1);  unsqueeze_2224 = None
        unsqueeze_2226 = torch.ops.aten.unsqueeze.default(mul_834, -1);  mul_834 = None
        unsqueeze_2227 = torch.ops.aten.unsqueeze.default(unsqueeze_2226, -1);  unsqueeze_2226 = None
        sub_278 = torch.ops.aten.sub.Tensor(convolution_515, unsqueeze_2225);  convolution_515 = unsqueeze_2225 = None
        mul_835 = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_2227);  sub_278 = unsqueeze_2227 = None
        unsqueeze_2228 = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_2229 = torch.ops.aten.unsqueeze.default(unsqueeze_2228, -1);  unsqueeze_2228 = None
        mul_836 = torch.ops.aten.mul.Tensor(mul_835, unsqueeze_2229);  mul_835 = unsqueeze_2229 = None
        unsqueeze_2230 = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_2231 = torch.ops.aten.unsqueeze.default(unsqueeze_2230, -1);  unsqueeze_2230 = None
        add_653 = torch.ops.aten.add.Tensor(mul_836, unsqueeze_2231);  mul_836 = unsqueeze_2231 = None
        relu_276 = torch.ops.aten.relu.default(add_653);  add_653 = None
        convolution_516 = torch.ops.aten.convolution.default(relu_276, arg456_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216);  relu_276 = arg456_1 = None
        convolution_517 = torch.ops.aten.convolution.default(convolution_516, arg457_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_516 = arg457_1 = None
        add_654 = torch.ops.aten.add.Tensor(arg459_1, 0.001);  arg459_1 = None
        sqrt_279 = torch.ops.aten.sqrt.default(add_654);  add_654 = None
        reciprocal_279 = torch.ops.aten.reciprocal.default(sqrt_279);  sqrt_279 = None
        mul_837 = torch.ops.aten.mul.Tensor(reciprocal_279, 1);  reciprocal_279 = None
        unsqueeze_2232 = torch.ops.aten.unsqueeze.default(arg458_1, -1);  arg458_1 = None
        unsqueeze_2233 = torch.ops.aten.unsqueeze.default(unsqueeze_2232, -1);  unsqueeze_2232 = None
        unsqueeze_2234 = torch.ops.aten.unsqueeze.default(mul_837, -1);  mul_837 = None
        unsqueeze_2235 = torch.ops.aten.unsqueeze.default(unsqueeze_2234, -1);  unsqueeze_2234 = None
        sub_279 = torch.ops.aten.sub.Tensor(convolution_517, unsqueeze_2233);  convolution_517 = unsqueeze_2233 = None
        mul_838 = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_2235);  sub_279 = unsqueeze_2235 = None
        unsqueeze_2236 = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_2237 = torch.ops.aten.unsqueeze.default(unsqueeze_2236, -1);  unsqueeze_2236 = None
        mul_839 = torch.ops.aten.mul.Tensor(mul_838, unsqueeze_2237);  mul_838 = unsqueeze_2237 = None
        unsqueeze_2238 = torch.ops.aten.unsqueeze.default(arg461_1, -1);  arg461_1 = None
        unsqueeze_2239 = torch.ops.aten.unsqueeze.default(unsqueeze_2238, -1);  unsqueeze_2238 = None
        add_655 = torch.ops.aten.add.Tensor(mul_839, unsqueeze_2239);  mul_839 = unsqueeze_2239 = None
        _low_memory_max_pool2d_with_offsets_58 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_646, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_116 = _low_memory_max_pool2d_with_offsets_58[0];  _low_memory_max_pool2d_with_offsets_58 = None
        add_656 = torch.ops.aten.add.Tensor(add_655, getitem_116);  add_655 = getitem_116 = None
        relu_277 = torch.ops.aten.relu.default(add_646)
        convolution_518 = torch.ops.aten.convolution.default(relu_277, arg462_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_277 = arg462_1 = None
        convolution_519 = torch.ops.aten.convolution.default(convolution_518, arg463_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_518 = arg463_1 = None
        add_657 = torch.ops.aten.add.Tensor(arg465_1, 0.001);  arg465_1 = None
        sqrt_280 = torch.ops.aten.sqrt.default(add_657);  add_657 = None
        reciprocal_280 = torch.ops.aten.reciprocal.default(sqrt_280);  sqrt_280 = None
        mul_840 = torch.ops.aten.mul.Tensor(reciprocal_280, 1);  reciprocal_280 = None
        unsqueeze_2240 = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_2241 = torch.ops.aten.unsqueeze.default(unsqueeze_2240, -1);  unsqueeze_2240 = None
        unsqueeze_2242 = torch.ops.aten.unsqueeze.default(mul_840, -1);  mul_840 = None
        unsqueeze_2243 = torch.ops.aten.unsqueeze.default(unsqueeze_2242, -1);  unsqueeze_2242 = None
        sub_280 = torch.ops.aten.sub.Tensor(convolution_519, unsqueeze_2241);  convolution_519 = unsqueeze_2241 = None
        mul_841 = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_2243);  sub_280 = unsqueeze_2243 = None
        unsqueeze_2244 = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
        unsqueeze_2245 = torch.ops.aten.unsqueeze.default(unsqueeze_2244, -1);  unsqueeze_2244 = None
        mul_842 = torch.ops.aten.mul.Tensor(mul_841, unsqueeze_2245);  mul_841 = unsqueeze_2245 = None
        unsqueeze_2246 = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
        unsqueeze_2247 = torch.ops.aten.unsqueeze.default(unsqueeze_2246, -1);  unsqueeze_2246 = None
        add_658 = torch.ops.aten.add.Tensor(mul_842, unsqueeze_2247);  mul_842 = unsqueeze_2247 = None
        relu_278 = torch.ops.aten.relu.default(add_658);  add_658 = None
        convolution_520 = torch.ops.aten.convolution.default(relu_278, arg468_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216);  relu_278 = arg468_1 = None
        convolution_521 = torch.ops.aten.convolution.default(convolution_520, arg469_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_520 = arg469_1 = None
        add_659 = torch.ops.aten.add.Tensor(arg471_1, 0.001);  arg471_1 = None
        sqrt_281 = torch.ops.aten.sqrt.default(add_659);  add_659 = None
        reciprocal_281 = torch.ops.aten.reciprocal.default(sqrt_281);  sqrt_281 = None
        mul_843 = torch.ops.aten.mul.Tensor(reciprocal_281, 1);  reciprocal_281 = None
        unsqueeze_2248 = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
        unsqueeze_2249 = torch.ops.aten.unsqueeze.default(unsqueeze_2248, -1);  unsqueeze_2248 = None
        unsqueeze_2250 = torch.ops.aten.unsqueeze.default(mul_843, -1);  mul_843 = None
        unsqueeze_2251 = torch.ops.aten.unsqueeze.default(unsqueeze_2250, -1);  unsqueeze_2250 = None
        sub_281 = torch.ops.aten.sub.Tensor(convolution_521, unsqueeze_2249);  convolution_521 = unsqueeze_2249 = None
        mul_844 = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_2251);  sub_281 = unsqueeze_2251 = None
        unsqueeze_2252 = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
        unsqueeze_2253 = torch.ops.aten.unsqueeze.default(unsqueeze_2252, -1);  unsqueeze_2252 = None
        mul_845 = torch.ops.aten.mul.Tensor(mul_844, unsqueeze_2253);  mul_844 = unsqueeze_2253 = None
        unsqueeze_2254 = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
        unsqueeze_2255 = torch.ops.aten.unsqueeze.default(unsqueeze_2254, -1);  unsqueeze_2254 = None
        add_660 = torch.ops.aten.add.Tensor(mul_845, unsqueeze_2255);  mul_845 = unsqueeze_2255 = None
        relu_279 = torch.ops.aten.relu.default(add_646)
        convolution_522 = torch.ops.aten.convolution.default(relu_279, arg474_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_279 = arg474_1 = None
        convolution_523 = torch.ops.aten.convolution.default(convolution_522, arg475_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_522 = arg475_1 = None
        add_661 = torch.ops.aten.add.Tensor(arg477_1, 0.001);  arg477_1 = None
        sqrt_282 = torch.ops.aten.sqrt.default(add_661);  add_661 = None
        reciprocal_282 = torch.ops.aten.reciprocal.default(sqrt_282);  sqrt_282 = None
        mul_846 = torch.ops.aten.mul.Tensor(reciprocal_282, 1);  reciprocal_282 = None
        unsqueeze_2256 = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
        unsqueeze_2257 = torch.ops.aten.unsqueeze.default(unsqueeze_2256, -1);  unsqueeze_2256 = None
        unsqueeze_2258 = torch.ops.aten.unsqueeze.default(mul_846, -1);  mul_846 = None
        unsqueeze_2259 = torch.ops.aten.unsqueeze.default(unsqueeze_2258, -1);  unsqueeze_2258 = None
        sub_282 = torch.ops.aten.sub.Tensor(convolution_523, unsqueeze_2257);  convolution_523 = unsqueeze_2257 = None
        mul_847 = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_2259);  sub_282 = unsqueeze_2259 = None
        unsqueeze_2260 = torch.ops.aten.unsqueeze.default(arg478_1, -1);  arg478_1 = None
        unsqueeze_2261 = torch.ops.aten.unsqueeze.default(unsqueeze_2260, -1);  unsqueeze_2260 = None
        mul_848 = torch.ops.aten.mul.Tensor(mul_847, unsqueeze_2261);  mul_847 = unsqueeze_2261 = None
        unsqueeze_2262 = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
        unsqueeze_2263 = torch.ops.aten.unsqueeze.default(unsqueeze_2262, -1);  unsqueeze_2262 = None
        add_662 = torch.ops.aten.add.Tensor(mul_848, unsqueeze_2263);  mul_848 = unsqueeze_2263 = None
        relu_280 = torch.ops.aten.relu.default(add_662);  add_662 = None
        convolution_524 = torch.ops.aten.convolution.default(relu_280, arg480_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_280 = arg480_1 = None
        convolution_525 = torch.ops.aten.convolution.default(convolution_524, arg481_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_524 = arg481_1 = None
        add_663 = torch.ops.aten.add.Tensor(arg483_1, 0.001);  arg483_1 = None
        sqrt_283 = torch.ops.aten.sqrt.default(add_663);  add_663 = None
        reciprocal_283 = torch.ops.aten.reciprocal.default(sqrt_283);  sqrt_283 = None
        mul_849 = torch.ops.aten.mul.Tensor(reciprocal_283, 1);  reciprocal_283 = None
        unsqueeze_2264 = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_2265 = torch.ops.aten.unsqueeze.default(unsqueeze_2264, -1);  unsqueeze_2264 = None
        unsqueeze_2266 = torch.ops.aten.unsqueeze.default(mul_849, -1);  mul_849 = None
        unsqueeze_2267 = torch.ops.aten.unsqueeze.default(unsqueeze_2266, -1);  unsqueeze_2266 = None
        sub_283 = torch.ops.aten.sub.Tensor(convolution_525, unsqueeze_2265);  convolution_525 = unsqueeze_2265 = None
        mul_850 = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_2267);  sub_283 = unsqueeze_2267 = None
        unsqueeze_2268 = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
        unsqueeze_2269 = torch.ops.aten.unsqueeze.default(unsqueeze_2268, -1);  unsqueeze_2268 = None
        mul_851 = torch.ops.aten.mul.Tensor(mul_850, unsqueeze_2269);  mul_850 = unsqueeze_2269 = None
        unsqueeze_2270 = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
        unsqueeze_2271 = torch.ops.aten.unsqueeze.default(unsqueeze_2270, -1);  unsqueeze_2270 = None
        add_664 = torch.ops.aten.add.Tensor(mul_851, unsqueeze_2271);  mul_851 = unsqueeze_2271 = None
        add_665 = torch.ops.aten.add.Tensor(add_660, add_664);  add_660 = add_664 = None
        relu_281 = torch.ops.aten.relu.default(add_665)
        convolution_526 = torch.ops.aten.convolution.default(relu_281, arg486_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_281 = arg486_1 = None
        convolution_527 = torch.ops.aten.convolution.default(convolution_526, arg487_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_526 = arg487_1 = None
        add_666 = torch.ops.aten.add.Tensor(arg489_1, 0.001);  arg489_1 = None
        sqrt_284 = torch.ops.aten.sqrt.default(add_666);  add_666 = None
        reciprocal_284 = torch.ops.aten.reciprocal.default(sqrt_284);  sqrt_284 = None
        mul_852 = torch.ops.aten.mul.Tensor(reciprocal_284, 1);  reciprocal_284 = None
        unsqueeze_2272 = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
        unsqueeze_2273 = torch.ops.aten.unsqueeze.default(unsqueeze_2272, -1);  unsqueeze_2272 = None
        unsqueeze_2274 = torch.ops.aten.unsqueeze.default(mul_852, -1);  mul_852 = None
        unsqueeze_2275 = torch.ops.aten.unsqueeze.default(unsqueeze_2274, -1);  unsqueeze_2274 = None
        sub_284 = torch.ops.aten.sub.Tensor(convolution_527, unsqueeze_2273);  convolution_527 = unsqueeze_2273 = None
        mul_853 = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_2275);  sub_284 = unsqueeze_2275 = None
        unsqueeze_2276 = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_2277 = torch.ops.aten.unsqueeze.default(unsqueeze_2276, -1);  unsqueeze_2276 = None
        mul_854 = torch.ops.aten.mul.Tensor(mul_853, unsqueeze_2277);  mul_853 = unsqueeze_2277 = None
        unsqueeze_2278 = torch.ops.aten.unsqueeze.default(arg491_1, -1);  arg491_1 = None
        unsqueeze_2279 = torch.ops.aten.unsqueeze.default(unsqueeze_2278, -1);  unsqueeze_2278 = None
        add_667 = torch.ops.aten.add.Tensor(mul_854, unsqueeze_2279);  mul_854 = unsqueeze_2279 = None
        relu_282 = torch.ops.aten.relu.default(add_667);  add_667 = None
        convolution_528 = torch.ops.aten.convolution.default(relu_282, arg492_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_282 = arg492_1 = None
        convolution_529 = torch.ops.aten.convolution.default(convolution_528, arg493_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_528 = arg493_1 = None
        add_668 = torch.ops.aten.add.Tensor(arg495_1, 0.001);  arg495_1 = None
        sqrt_285 = torch.ops.aten.sqrt.default(add_668);  add_668 = None
        reciprocal_285 = torch.ops.aten.reciprocal.default(sqrt_285);  sqrt_285 = None
        mul_855 = torch.ops.aten.mul.Tensor(reciprocal_285, 1);  reciprocal_285 = None
        unsqueeze_2280 = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_2281 = torch.ops.aten.unsqueeze.default(unsqueeze_2280, -1);  unsqueeze_2280 = None
        unsqueeze_2282 = torch.ops.aten.unsqueeze.default(mul_855, -1);  mul_855 = None
        unsqueeze_2283 = torch.ops.aten.unsqueeze.default(unsqueeze_2282, -1);  unsqueeze_2282 = None
        sub_285 = torch.ops.aten.sub.Tensor(convolution_529, unsqueeze_2281);  convolution_529 = unsqueeze_2281 = None
        mul_856 = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_2283);  sub_285 = unsqueeze_2283 = None
        unsqueeze_2284 = torch.ops.aten.unsqueeze.default(arg496_1, -1);  arg496_1 = None
        unsqueeze_2285 = torch.ops.aten.unsqueeze.default(unsqueeze_2284, -1);  unsqueeze_2284 = None
        mul_857 = torch.ops.aten.mul.Tensor(mul_856, unsqueeze_2285);  mul_856 = unsqueeze_2285 = None
        unsqueeze_2286 = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
        unsqueeze_2287 = torch.ops.aten.unsqueeze.default(unsqueeze_2286, -1);  unsqueeze_2286 = None
        add_669 = torch.ops.aten.add.Tensor(mul_857, unsqueeze_2287);  mul_857 = unsqueeze_2287 = None
        _low_memory_max_pool2d_with_offsets_59 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_646, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_118 = _low_memory_max_pool2d_with_offsets_59[0];  _low_memory_max_pool2d_with_offsets_59 = None
        add_670 = torch.ops.aten.add.Tensor(add_669, getitem_118);  add_669 = getitem_118 = None
        relu_283 = torch.ops.aten.relu.default(add_644);  add_644 = None
        convolution_530 = torch.ops.aten.convolution.default(relu_283, arg498_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_283 = arg498_1 = None
        convolution_531 = torch.ops.aten.convolution.default(convolution_530, arg499_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_530 = arg499_1 = None
        add_671 = torch.ops.aten.add.Tensor(arg501_1, 0.001);  arg501_1 = None
        sqrt_286 = torch.ops.aten.sqrt.default(add_671);  add_671 = None
        reciprocal_286 = torch.ops.aten.reciprocal.default(sqrt_286);  sqrt_286 = None
        mul_858 = torch.ops.aten.mul.Tensor(reciprocal_286, 1);  reciprocal_286 = None
        unsqueeze_2288 = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
        unsqueeze_2289 = torch.ops.aten.unsqueeze.default(unsqueeze_2288, -1);  unsqueeze_2288 = None
        unsqueeze_2290 = torch.ops.aten.unsqueeze.default(mul_858, -1);  mul_858 = None
        unsqueeze_2291 = torch.ops.aten.unsqueeze.default(unsqueeze_2290, -1);  unsqueeze_2290 = None
        sub_286 = torch.ops.aten.sub.Tensor(convolution_531, unsqueeze_2289);  convolution_531 = unsqueeze_2289 = None
        mul_859 = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_2291);  sub_286 = unsqueeze_2291 = None
        unsqueeze_2292 = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_2293 = torch.ops.aten.unsqueeze.default(unsqueeze_2292, -1);  unsqueeze_2292 = None
        mul_860 = torch.ops.aten.mul.Tensor(mul_859, unsqueeze_2293);  mul_859 = unsqueeze_2293 = None
        unsqueeze_2294 = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
        unsqueeze_2295 = torch.ops.aten.unsqueeze.default(unsqueeze_2294, -1);  unsqueeze_2294 = None
        add_672 = torch.ops.aten.add.Tensor(mul_860, unsqueeze_2295);  mul_860 = unsqueeze_2295 = None
        relu_284 = torch.ops.aten.relu.default(add_672);  add_672 = None
        convolution_532 = torch.ops.aten.convolution.default(relu_284, arg504_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  relu_284 = arg504_1 = None
        convolution_533 = torch.ops.aten.convolution.default(convolution_532, arg505_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_532 = arg505_1 = None
        add_673 = torch.ops.aten.add.Tensor(arg507_1, 0.001);  arg507_1 = None
        sqrt_287 = torch.ops.aten.sqrt.default(add_673);  add_673 = None
        reciprocal_287 = torch.ops.aten.reciprocal.default(sqrt_287);  sqrt_287 = None
        mul_861 = torch.ops.aten.mul.Tensor(reciprocal_287, 1);  reciprocal_287 = None
        unsqueeze_2296 = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
        unsqueeze_2297 = torch.ops.aten.unsqueeze.default(unsqueeze_2296, -1);  unsqueeze_2296 = None
        unsqueeze_2298 = torch.ops.aten.unsqueeze.default(mul_861, -1);  mul_861 = None
        unsqueeze_2299 = torch.ops.aten.unsqueeze.default(unsqueeze_2298, -1);  unsqueeze_2298 = None
        sub_287 = torch.ops.aten.sub.Tensor(convolution_533, unsqueeze_2297);  convolution_533 = unsqueeze_2297 = None
        mul_862 = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_2299);  sub_287 = unsqueeze_2299 = None
        unsqueeze_2300 = torch.ops.aten.unsqueeze.default(arg508_1, -1);  arg508_1 = None
        unsqueeze_2301 = torch.ops.aten.unsqueeze.default(unsqueeze_2300, -1);  unsqueeze_2300 = None
        mul_863 = torch.ops.aten.mul.Tensor(mul_862, unsqueeze_2301);  mul_862 = unsqueeze_2301 = None
        unsqueeze_2302 = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
        unsqueeze_2303 = torch.ops.aten.unsqueeze.default(unsqueeze_2302, -1);  unsqueeze_2302 = None
        add_674 = torch.ops.aten.add.Tensor(mul_863, unsqueeze_2303);  mul_863 = unsqueeze_2303 = None
        add_675 = torch.ops.aten.add.Tensor(add_674, add_646);  add_674 = add_646 = None
        cat_25 = torch.ops.aten.cat.default([add_651, add_656, add_665, add_670, add_675], 1);  add_651 = add_656 = add_665 = add_670 = add_675 = None
        relu_285 = torch.ops.aten.relu.default(cat_24);  cat_24 = None
        convolution_534 = torch.ops.aten.convolution.default(relu_285, arg510_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_285 = arg510_1 = None
        add_676 = torch.ops.aten.add.Tensor(arg512_1, 0.001);  arg512_1 = None
        sqrt_288 = torch.ops.aten.sqrt.default(add_676);  add_676 = None
        reciprocal_288 = torch.ops.aten.reciprocal.default(sqrt_288);  sqrt_288 = None
        mul_864 = torch.ops.aten.mul.Tensor(reciprocal_288, 1);  reciprocal_288 = None
        unsqueeze_2304 = torch.ops.aten.unsqueeze.default(arg511_1, -1);  arg511_1 = None
        unsqueeze_2305 = torch.ops.aten.unsqueeze.default(unsqueeze_2304, -1);  unsqueeze_2304 = None
        unsqueeze_2306 = torch.ops.aten.unsqueeze.default(mul_864, -1);  mul_864 = None
        unsqueeze_2307 = torch.ops.aten.unsqueeze.default(unsqueeze_2306, -1);  unsqueeze_2306 = None
        sub_288 = torch.ops.aten.sub.Tensor(convolution_534, unsqueeze_2305);  convolution_534 = unsqueeze_2305 = None
        mul_865 = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_2307);  sub_288 = unsqueeze_2307 = None
        unsqueeze_2308 = torch.ops.aten.unsqueeze.default(arg513_1, -1);  arg513_1 = None
        unsqueeze_2309 = torch.ops.aten.unsqueeze.default(unsqueeze_2308, -1);  unsqueeze_2308 = None
        mul_866 = torch.ops.aten.mul.Tensor(mul_865, unsqueeze_2309);  mul_865 = unsqueeze_2309 = None
        unsqueeze_2310 = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
        unsqueeze_2311 = torch.ops.aten.unsqueeze.default(unsqueeze_2310, -1);  unsqueeze_2310 = None
        add_677 = torch.ops.aten.add.Tensor(mul_866, unsqueeze_2311);  mul_866 = unsqueeze_2311 = None
        relu_286 = torch.ops.aten.relu.default(cat_25)
        convolution_535 = torch.ops.aten.convolution.default(relu_286, arg515_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_286 = arg515_1 = None
        add_678 = torch.ops.aten.add.Tensor(arg517_1, 0.001);  arg517_1 = None
        sqrt_289 = torch.ops.aten.sqrt.default(add_678);  add_678 = None
        reciprocal_289 = torch.ops.aten.reciprocal.default(sqrt_289);  sqrt_289 = None
        mul_867 = torch.ops.aten.mul.Tensor(reciprocal_289, 1);  reciprocal_289 = None
        unsqueeze_2312 = torch.ops.aten.unsqueeze.default(arg516_1, -1);  arg516_1 = None
        unsqueeze_2313 = torch.ops.aten.unsqueeze.default(unsqueeze_2312, -1);  unsqueeze_2312 = None
        unsqueeze_2314 = torch.ops.aten.unsqueeze.default(mul_867, -1);  mul_867 = None
        unsqueeze_2315 = torch.ops.aten.unsqueeze.default(unsqueeze_2314, -1);  unsqueeze_2314 = None
        sub_289 = torch.ops.aten.sub.Tensor(convolution_535, unsqueeze_2313);  convolution_535 = unsqueeze_2313 = None
        mul_868 = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_2315);  sub_289 = unsqueeze_2315 = None
        unsqueeze_2316 = torch.ops.aten.unsqueeze.default(arg518_1, -1);  arg518_1 = None
        unsqueeze_2317 = torch.ops.aten.unsqueeze.default(unsqueeze_2316, -1);  unsqueeze_2316 = None
        mul_869 = torch.ops.aten.mul.Tensor(mul_868, unsqueeze_2317);  mul_868 = unsqueeze_2317 = None
        unsqueeze_2318 = torch.ops.aten.unsqueeze.default(arg519_1, -1);  arg519_1 = None
        unsqueeze_2319 = torch.ops.aten.unsqueeze.default(unsqueeze_2318, -1);  unsqueeze_2318 = None
        add_679 = torch.ops.aten.add.Tensor(mul_869, unsqueeze_2319);  mul_869 = unsqueeze_2319 = None
        relu_287 = torch.ops.aten.relu.default(add_677)
        constant_pad_nd_60 = torch.ops.aten.constant_pad_nd.default(relu_287, [1, 2, 1, 2], 0.0);  relu_287 = None
        convolution_536 = torch.ops.aten.convolution.default(constant_pad_nd_60, arg520_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432);  constant_pad_nd_60 = arg520_1 = None
        convolution_537 = torch.ops.aten.convolution.default(convolution_536, arg521_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_536 = arg521_1 = None
        add_680 = torch.ops.aten.add.Tensor(arg523_1, 0.001);  arg523_1 = None
        sqrt_290 = torch.ops.aten.sqrt.default(add_680);  add_680 = None
        reciprocal_290 = torch.ops.aten.reciprocal.default(sqrt_290);  sqrt_290 = None
        mul_870 = torch.ops.aten.mul.Tensor(reciprocal_290, 1);  reciprocal_290 = None
        unsqueeze_2320 = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
        unsqueeze_2321 = torch.ops.aten.unsqueeze.default(unsqueeze_2320, -1);  unsqueeze_2320 = None
        unsqueeze_2322 = torch.ops.aten.unsqueeze.default(mul_870, -1);  mul_870 = None
        unsqueeze_2323 = torch.ops.aten.unsqueeze.default(unsqueeze_2322, -1);  unsqueeze_2322 = None
        sub_290 = torch.ops.aten.sub.Tensor(convolution_537, unsqueeze_2321);  convolution_537 = unsqueeze_2321 = None
        mul_871 = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_2323);  sub_290 = unsqueeze_2323 = None
        unsqueeze_2324 = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
        unsqueeze_2325 = torch.ops.aten.unsqueeze.default(unsqueeze_2324, -1);  unsqueeze_2324 = None
        mul_872 = torch.ops.aten.mul.Tensor(mul_871, unsqueeze_2325);  mul_871 = unsqueeze_2325 = None
        unsqueeze_2326 = torch.ops.aten.unsqueeze.default(arg525_1, -1);  arg525_1 = None
        unsqueeze_2327 = torch.ops.aten.unsqueeze.default(unsqueeze_2326, -1);  unsqueeze_2326 = None
        add_681 = torch.ops.aten.add.Tensor(mul_872, unsqueeze_2327);  mul_872 = unsqueeze_2327 = None
        relu_288 = torch.ops.aten.relu.default(add_681);  add_681 = None
        convolution_538 = torch.ops.aten.convolution.default(relu_288, arg526_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_288 = arg526_1 = None
        convolution_539 = torch.ops.aten.convolution.default(convolution_538, arg527_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_538 = arg527_1 = None
        add_682 = torch.ops.aten.add.Tensor(arg529_1, 0.001);  arg529_1 = None
        sqrt_291 = torch.ops.aten.sqrt.default(add_682);  add_682 = None
        reciprocal_291 = torch.ops.aten.reciprocal.default(sqrt_291);  sqrt_291 = None
        mul_873 = torch.ops.aten.mul.Tensor(reciprocal_291, 1);  reciprocal_291 = None
        unsqueeze_2328 = torch.ops.aten.unsqueeze.default(arg528_1, -1);  arg528_1 = None
        unsqueeze_2329 = torch.ops.aten.unsqueeze.default(unsqueeze_2328, -1);  unsqueeze_2328 = None
        unsqueeze_2330 = torch.ops.aten.unsqueeze.default(mul_873, -1);  mul_873 = None
        unsqueeze_2331 = torch.ops.aten.unsqueeze.default(unsqueeze_2330, -1);  unsqueeze_2330 = None
        sub_291 = torch.ops.aten.sub.Tensor(convolution_539, unsqueeze_2329);  convolution_539 = unsqueeze_2329 = None
        mul_874 = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_2331);  sub_291 = unsqueeze_2331 = None
        unsqueeze_2332 = torch.ops.aten.unsqueeze.default(arg530_1, -1);  arg530_1 = None
        unsqueeze_2333 = torch.ops.aten.unsqueeze.default(unsqueeze_2332, -1);  unsqueeze_2332 = None
        mul_875 = torch.ops.aten.mul.Tensor(mul_874, unsqueeze_2333);  mul_874 = unsqueeze_2333 = None
        unsqueeze_2334 = torch.ops.aten.unsqueeze.default(arg531_1, -1);  arg531_1 = None
        unsqueeze_2335 = torch.ops.aten.unsqueeze.default(unsqueeze_2334, -1);  unsqueeze_2334 = None
        add_683 = torch.ops.aten.add.Tensor(mul_875, unsqueeze_2335);  mul_875 = unsqueeze_2335 = None
        constant_pad_nd_61 = torch.ops.aten.constant_pad_nd.default(add_677, [0, 1, 0, 1], -inf)
        _low_memory_max_pool2d_with_offsets_60 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_61, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_61 = None
        getitem_120 = _low_memory_max_pool2d_with_offsets_60[0];  _low_memory_max_pool2d_with_offsets_60 = None
        add_684 = torch.ops.aten.add.Tensor(add_683, getitem_120);  add_683 = getitem_120 = None
        relu_289 = torch.ops.aten.relu.default(add_679)
        constant_pad_nd_62 = torch.ops.aten.constant_pad_nd.default(relu_289, [2, 3, 2, 3], 0.0);  relu_289 = None
        convolution_540 = torch.ops.aten.convolution.default(constant_pad_nd_62, arg532_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432);  constant_pad_nd_62 = arg532_1 = None
        convolution_541 = torch.ops.aten.convolution.default(convolution_540, arg533_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_540 = arg533_1 = None
        add_685 = torch.ops.aten.add.Tensor(arg535_1, 0.001);  arg535_1 = None
        sqrt_292 = torch.ops.aten.sqrt.default(add_685);  add_685 = None
        reciprocal_292 = torch.ops.aten.reciprocal.default(sqrt_292);  sqrt_292 = None
        mul_876 = torch.ops.aten.mul.Tensor(reciprocal_292, 1);  reciprocal_292 = None
        unsqueeze_2336 = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
        unsqueeze_2337 = torch.ops.aten.unsqueeze.default(unsqueeze_2336, -1);  unsqueeze_2336 = None
        unsqueeze_2338 = torch.ops.aten.unsqueeze.default(mul_876, -1);  mul_876 = None
        unsqueeze_2339 = torch.ops.aten.unsqueeze.default(unsqueeze_2338, -1);  unsqueeze_2338 = None
        sub_292 = torch.ops.aten.sub.Tensor(convolution_541, unsqueeze_2337);  convolution_541 = unsqueeze_2337 = None
        mul_877 = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_2339);  sub_292 = unsqueeze_2339 = None
        unsqueeze_2340 = torch.ops.aten.unsqueeze.default(arg536_1, -1);  arg536_1 = None
        unsqueeze_2341 = torch.ops.aten.unsqueeze.default(unsqueeze_2340, -1);  unsqueeze_2340 = None
        mul_878 = torch.ops.aten.mul.Tensor(mul_877, unsqueeze_2341);  mul_877 = unsqueeze_2341 = None
        unsqueeze_2342 = torch.ops.aten.unsqueeze.default(arg537_1, -1);  arg537_1 = None
        unsqueeze_2343 = torch.ops.aten.unsqueeze.default(unsqueeze_2342, -1);  unsqueeze_2342 = None
        add_686 = torch.ops.aten.add.Tensor(mul_878, unsqueeze_2343);  mul_878 = unsqueeze_2343 = None
        relu_290 = torch.ops.aten.relu.default(add_686);  add_686 = None
        convolution_542 = torch.ops.aten.convolution.default(relu_290, arg538_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432);  relu_290 = arg538_1 = None
        convolution_543 = torch.ops.aten.convolution.default(convolution_542, arg539_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_542 = arg539_1 = None
        add_687 = torch.ops.aten.add.Tensor(arg541_1, 0.001);  arg541_1 = None
        sqrt_293 = torch.ops.aten.sqrt.default(add_687);  add_687 = None
        reciprocal_293 = torch.ops.aten.reciprocal.default(sqrt_293);  sqrt_293 = None
        mul_879 = torch.ops.aten.mul.Tensor(reciprocal_293, 1);  reciprocal_293 = None
        unsqueeze_2344 = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
        unsqueeze_2345 = torch.ops.aten.unsqueeze.default(unsqueeze_2344, -1);  unsqueeze_2344 = None
        unsqueeze_2346 = torch.ops.aten.unsqueeze.default(mul_879, -1);  mul_879 = None
        unsqueeze_2347 = torch.ops.aten.unsqueeze.default(unsqueeze_2346, -1);  unsqueeze_2346 = None
        sub_293 = torch.ops.aten.sub.Tensor(convolution_543, unsqueeze_2345);  convolution_543 = unsqueeze_2345 = None
        mul_880 = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_2347);  sub_293 = unsqueeze_2347 = None
        unsqueeze_2348 = torch.ops.aten.unsqueeze.default(arg542_1, -1);  arg542_1 = None
        unsqueeze_2349 = torch.ops.aten.unsqueeze.default(unsqueeze_2348, -1);  unsqueeze_2348 = None
        mul_881 = torch.ops.aten.mul.Tensor(mul_880, unsqueeze_2349);  mul_880 = unsqueeze_2349 = None
        unsqueeze_2350 = torch.ops.aten.unsqueeze.default(arg543_1, -1);  arg543_1 = None
        unsqueeze_2351 = torch.ops.aten.unsqueeze.default(unsqueeze_2350, -1);  unsqueeze_2350 = None
        add_688 = torch.ops.aten.add.Tensor(mul_881, unsqueeze_2351);  mul_881 = unsqueeze_2351 = None
        constant_pad_nd_63 = torch.ops.aten.constant_pad_nd.default(add_679, [0, 1, 0, 1], -inf)
        _low_memory_max_pool2d_with_offsets_61 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_63, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_63 = None
        getitem_122 = _low_memory_max_pool2d_with_offsets_61[0];  _low_memory_max_pool2d_with_offsets_61 = None
        add_689 = torch.ops.aten.add.Tensor(add_688, getitem_122);  add_688 = getitem_122 = None
        relu_291 = torch.ops.aten.relu.default(add_679)
        constant_pad_nd_64 = torch.ops.aten.constant_pad_nd.default(relu_291, [1, 2, 1, 2], 0.0);  relu_291 = None
        convolution_544 = torch.ops.aten.convolution.default(constant_pad_nd_64, arg544_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432);  constant_pad_nd_64 = arg544_1 = None
        convolution_545 = torch.ops.aten.convolution.default(convolution_544, arg545_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_544 = arg545_1 = None
        add_690 = torch.ops.aten.add.Tensor(arg547_1, 0.001);  arg547_1 = None
        sqrt_294 = torch.ops.aten.sqrt.default(add_690);  add_690 = None
        reciprocal_294 = torch.ops.aten.reciprocal.default(sqrt_294);  sqrt_294 = None
        mul_882 = torch.ops.aten.mul.Tensor(reciprocal_294, 1);  reciprocal_294 = None
        unsqueeze_2352 = torch.ops.aten.unsqueeze.default(arg546_1, -1);  arg546_1 = None
        unsqueeze_2353 = torch.ops.aten.unsqueeze.default(unsqueeze_2352, -1);  unsqueeze_2352 = None
        unsqueeze_2354 = torch.ops.aten.unsqueeze.default(mul_882, -1);  mul_882 = None
        unsqueeze_2355 = torch.ops.aten.unsqueeze.default(unsqueeze_2354, -1);  unsqueeze_2354 = None
        sub_294 = torch.ops.aten.sub.Tensor(convolution_545, unsqueeze_2353);  convolution_545 = unsqueeze_2353 = None
        mul_883 = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_2355);  sub_294 = unsqueeze_2355 = None
        unsqueeze_2356 = torch.ops.aten.unsqueeze.default(arg548_1, -1);  arg548_1 = None
        unsqueeze_2357 = torch.ops.aten.unsqueeze.default(unsqueeze_2356, -1);  unsqueeze_2356 = None
        mul_884 = torch.ops.aten.mul.Tensor(mul_883, unsqueeze_2357);  mul_883 = unsqueeze_2357 = None
        unsqueeze_2358 = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_2359 = torch.ops.aten.unsqueeze.default(unsqueeze_2358, -1);  unsqueeze_2358 = None
        add_691 = torch.ops.aten.add.Tensor(mul_884, unsqueeze_2359);  mul_884 = unsqueeze_2359 = None
        relu_292 = torch.ops.aten.relu.default(add_691);  add_691 = None
        convolution_546 = torch.ops.aten.convolution.default(relu_292, arg550_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_292 = arg550_1 = None
        convolution_547 = torch.ops.aten.convolution.default(convolution_546, arg551_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_546 = arg551_1 = None
        add_692 = torch.ops.aten.add.Tensor(arg553_1, 0.001);  arg553_1 = None
        sqrt_295 = torch.ops.aten.sqrt.default(add_692);  add_692 = None
        reciprocal_295 = torch.ops.aten.reciprocal.default(sqrt_295);  sqrt_295 = None
        mul_885 = torch.ops.aten.mul.Tensor(reciprocal_295, 1);  reciprocal_295 = None
        unsqueeze_2360 = torch.ops.aten.unsqueeze.default(arg552_1, -1);  arg552_1 = None
        unsqueeze_2361 = torch.ops.aten.unsqueeze.default(unsqueeze_2360, -1);  unsqueeze_2360 = None
        unsqueeze_2362 = torch.ops.aten.unsqueeze.default(mul_885, -1);  mul_885 = None
        unsqueeze_2363 = torch.ops.aten.unsqueeze.default(unsqueeze_2362, -1);  unsqueeze_2362 = None
        sub_295 = torch.ops.aten.sub.Tensor(convolution_547, unsqueeze_2361);  convolution_547 = unsqueeze_2361 = None
        mul_886 = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_2363);  sub_295 = unsqueeze_2363 = None
        unsqueeze_2364 = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_2365 = torch.ops.aten.unsqueeze.default(unsqueeze_2364, -1);  unsqueeze_2364 = None
        mul_887 = torch.ops.aten.mul.Tensor(mul_886, unsqueeze_2365);  mul_886 = unsqueeze_2365 = None
        unsqueeze_2366 = torch.ops.aten.unsqueeze.default(arg555_1, -1);  arg555_1 = None
        unsqueeze_2367 = torch.ops.aten.unsqueeze.default(unsqueeze_2366, -1);  unsqueeze_2366 = None
        add_693 = torch.ops.aten.add.Tensor(mul_887, unsqueeze_2367);  mul_887 = unsqueeze_2367 = None
        relu_293 = torch.ops.aten.relu.default(add_679)
        constant_pad_nd_65 = torch.ops.aten.constant_pad_nd.default(relu_293, [0, 1, 0, 1], 0.0);  relu_293 = None
        convolution_548 = torch.ops.aten.convolution.default(constant_pad_nd_65, arg556_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432);  constant_pad_nd_65 = arg556_1 = None
        convolution_549 = torch.ops.aten.convolution.default(convolution_548, arg557_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_548 = arg557_1 = None
        add_694 = torch.ops.aten.add.Tensor(arg559_1, 0.001);  arg559_1 = None
        sqrt_296 = torch.ops.aten.sqrt.default(add_694);  add_694 = None
        reciprocal_296 = torch.ops.aten.reciprocal.default(sqrt_296);  sqrt_296 = None
        mul_888 = torch.ops.aten.mul.Tensor(reciprocal_296, 1);  reciprocal_296 = None
        unsqueeze_2368 = torch.ops.aten.unsqueeze.default(arg558_1, -1);  arg558_1 = None
        unsqueeze_2369 = torch.ops.aten.unsqueeze.default(unsqueeze_2368, -1);  unsqueeze_2368 = None
        unsqueeze_2370 = torch.ops.aten.unsqueeze.default(mul_888, -1);  mul_888 = None
        unsqueeze_2371 = torch.ops.aten.unsqueeze.default(unsqueeze_2370, -1);  unsqueeze_2370 = None
        sub_296 = torch.ops.aten.sub.Tensor(convolution_549, unsqueeze_2369);  convolution_549 = unsqueeze_2369 = None
        mul_889 = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_2371);  sub_296 = unsqueeze_2371 = None
        unsqueeze_2372 = torch.ops.aten.unsqueeze.default(arg560_1, -1);  arg560_1 = None
        unsqueeze_2373 = torch.ops.aten.unsqueeze.default(unsqueeze_2372, -1);  unsqueeze_2372 = None
        mul_890 = torch.ops.aten.mul.Tensor(mul_889, unsqueeze_2373);  mul_889 = unsqueeze_2373 = None
        unsqueeze_2374 = torch.ops.aten.unsqueeze.default(arg561_1, -1);  arg561_1 = None
        unsqueeze_2375 = torch.ops.aten.unsqueeze.default(unsqueeze_2374, -1);  unsqueeze_2374 = None
        add_695 = torch.ops.aten.add.Tensor(mul_890, unsqueeze_2375);  mul_890 = unsqueeze_2375 = None
        relu_294 = torch.ops.aten.relu.default(add_695);  add_695 = None
        convolution_550 = torch.ops.aten.convolution.default(relu_294, arg562_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_294 = arg562_1 = None
        convolution_551 = torch.ops.aten.convolution.default(convolution_550, arg563_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_550 = arg563_1 = None
        add_696 = torch.ops.aten.add.Tensor(arg565_1, 0.001);  arg565_1 = None
        sqrt_297 = torch.ops.aten.sqrt.default(add_696);  add_696 = None
        reciprocal_297 = torch.ops.aten.reciprocal.default(sqrt_297);  sqrt_297 = None
        mul_891 = torch.ops.aten.mul.Tensor(reciprocal_297, 1);  reciprocal_297 = None
        unsqueeze_2376 = torch.ops.aten.unsqueeze.default(arg564_1, -1);  arg564_1 = None
        unsqueeze_2377 = torch.ops.aten.unsqueeze.default(unsqueeze_2376, -1);  unsqueeze_2376 = None
        unsqueeze_2378 = torch.ops.aten.unsqueeze.default(mul_891, -1);  mul_891 = None
        unsqueeze_2379 = torch.ops.aten.unsqueeze.default(unsqueeze_2378, -1);  unsqueeze_2378 = None
        sub_297 = torch.ops.aten.sub.Tensor(convolution_551, unsqueeze_2377);  convolution_551 = unsqueeze_2377 = None
        mul_892 = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_2379);  sub_297 = unsqueeze_2379 = None
        unsqueeze_2380 = torch.ops.aten.unsqueeze.default(arg566_1, -1);  arg566_1 = None
        unsqueeze_2381 = torch.ops.aten.unsqueeze.default(unsqueeze_2380, -1);  unsqueeze_2380 = None
        mul_893 = torch.ops.aten.mul.Tensor(mul_892, unsqueeze_2381);  mul_892 = unsqueeze_2381 = None
        unsqueeze_2382 = torch.ops.aten.unsqueeze.default(arg567_1, -1);  arg567_1 = None
        unsqueeze_2383 = torch.ops.aten.unsqueeze.default(unsqueeze_2382, -1);  unsqueeze_2382 = None
        add_697 = torch.ops.aten.add.Tensor(mul_893, unsqueeze_2383);  mul_893 = unsqueeze_2383 = None
        add_698 = torch.ops.aten.add.Tensor(add_693, add_697);  add_693 = add_697 = None
        relu_295 = torch.ops.aten.relu.default(add_698)
        convolution_552 = torch.ops.aten.convolution.default(relu_295, arg568_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_295 = arg568_1 = None
        convolution_553 = torch.ops.aten.convolution.default(convolution_552, arg569_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_552 = arg569_1 = None
        add_699 = torch.ops.aten.add.Tensor(arg571_1, 0.001);  arg571_1 = None
        sqrt_298 = torch.ops.aten.sqrt.default(add_699);  add_699 = None
        reciprocal_298 = torch.ops.aten.reciprocal.default(sqrt_298);  sqrt_298 = None
        mul_894 = torch.ops.aten.mul.Tensor(reciprocal_298, 1);  reciprocal_298 = None
        unsqueeze_2384 = torch.ops.aten.unsqueeze.default(arg570_1, -1);  arg570_1 = None
        unsqueeze_2385 = torch.ops.aten.unsqueeze.default(unsqueeze_2384, -1);  unsqueeze_2384 = None
        unsqueeze_2386 = torch.ops.aten.unsqueeze.default(mul_894, -1);  mul_894 = None
        unsqueeze_2387 = torch.ops.aten.unsqueeze.default(unsqueeze_2386, -1);  unsqueeze_2386 = None
        sub_298 = torch.ops.aten.sub.Tensor(convolution_553, unsqueeze_2385);  convolution_553 = unsqueeze_2385 = None
        mul_895 = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_2387);  sub_298 = unsqueeze_2387 = None
        unsqueeze_2388 = torch.ops.aten.unsqueeze.default(arg572_1, -1);  arg572_1 = None
        unsqueeze_2389 = torch.ops.aten.unsqueeze.default(unsqueeze_2388, -1);  unsqueeze_2388 = None
        mul_896 = torch.ops.aten.mul.Tensor(mul_895, unsqueeze_2389);  mul_895 = unsqueeze_2389 = None
        unsqueeze_2390 = torch.ops.aten.unsqueeze.default(arg573_1, -1);  arg573_1 = None
        unsqueeze_2391 = torch.ops.aten.unsqueeze.default(unsqueeze_2390, -1);  unsqueeze_2390 = None
        add_700 = torch.ops.aten.add.Tensor(mul_896, unsqueeze_2391);  mul_896 = unsqueeze_2391 = None
        relu_296 = torch.ops.aten.relu.default(add_700);  add_700 = None
        convolution_554 = torch.ops.aten.convolution.default(relu_296, arg574_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_296 = arg574_1 = None
        convolution_555 = torch.ops.aten.convolution.default(convolution_554, arg575_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_554 = arg575_1 = None
        add_701 = torch.ops.aten.add.Tensor(arg577_1, 0.001);  arg577_1 = None
        sqrt_299 = torch.ops.aten.sqrt.default(add_701);  add_701 = None
        reciprocal_299 = torch.ops.aten.reciprocal.default(sqrt_299);  sqrt_299 = None
        mul_897 = torch.ops.aten.mul.Tensor(reciprocal_299, 1);  reciprocal_299 = None
        unsqueeze_2392 = torch.ops.aten.unsqueeze.default(arg576_1, -1);  arg576_1 = None
        unsqueeze_2393 = torch.ops.aten.unsqueeze.default(unsqueeze_2392, -1);  unsqueeze_2392 = None
        unsqueeze_2394 = torch.ops.aten.unsqueeze.default(mul_897, -1);  mul_897 = None
        unsqueeze_2395 = torch.ops.aten.unsqueeze.default(unsqueeze_2394, -1);  unsqueeze_2394 = None
        sub_299 = torch.ops.aten.sub.Tensor(convolution_555, unsqueeze_2393);  convolution_555 = unsqueeze_2393 = None
        mul_898 = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_2395);  sub_299 = unsqueeze_2395 = None
        unsqueeze_2396 = torch.ops.aten.unsqueeze.default(arg578_1, -1);  arg578_1 = None
        unsqueeze_2397 = torch.ops.aten.unsqueeze.default(unsqueeze_2396, -1);  unsqueeze_2396 = None
        mul_899 = torch.ops.aten.mul.Tensor(mul_898, unsqueeze_2397);  mul_898 = unsqueeze_2397 = None
        unsqueeze_2398 = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_2399 = torch.ops.aten.unsqueeze.default(unsqueeze_2398, -1);  unsqueeze_2398 = None
        add_702 = torch.ops.aten.add.Tensor(mul_899, unsqueeze_2399);  mul_899 = unsqueeze_2399 = None
        constant_pad_nd_66 = torch.ops.aten.constant_pad_nd.default(add_679, [0, 1, 0, 1], -inf)
        _low_memory_max_pool2d_with_offsets_62 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_66, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_66 = None
        getitem_124 = _low_memory_max_pool2d_with_offsets_62[0];  _low_memory_max_pool2d_with_offsets_62 = None
        add_703 = torch.ops.aten.add.Tensor(add_702, getitem_124);  add_702 = getitem_124 = None
        relu_297 = torch.ops.aten.relu.default(add_677);  add_677 = None
        constant_pad_nd_67 = torch.ops.aten.constant_pad_nd.default(relu_297, [0, 1, 0, 1], 0.0);  relu_297 = None
        convolution_556 = torch.ops.aten.convolution.default(constant_pad_nd_67, arg580_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432);  constant_pad_nd_67 = arg580_1 = None
        convolution_557 = torch.ops.aten.convolution.default(convolution_556, arg581_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_556 = arg581_1 = None
        add_704 = torch.ops.aten.add.Tensor(arg583_1, 0.001);  arg583_1 = None
        sqrt_300 = torch.ops.aten.sqrt.default(add_704);  add_704 = None
        reciprocal_300 = torch.ops.aten.reciprocal.default(sqrt_300);  sqrt_300 = None
        mul_900 = torch.ops.aten.mul.Tensor(reciprocal_300, 1);  reciprocal_300 = None
        unsqueeze_2400 = torch.ops.aten.unsqueeze.default(arg582_1, -1);  arg582_1 = None
        unsqueeze_2401 = torch.ops.aten.unsqueeze.default(unsqueeze_2400, -1);  unsqueeze_2400 = None
        unsqueeze_2402 = torch.ops.aten.unsqueeze.default(mul_900, -1);  mul_900 = None
        unsqueeze_2403 = torch.ops.aten.unsqueeze.default(unsqueeze_2402, -1);  unsqueeze_2402 = None
        sub_300 = torch.ops.aten.sub.Tensor(convolution_557, unsqueeze_2401);  convolution_557 = unsqueeze_2401 = None
        mul_901 = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_2403);  sub_300 = unsqueeze_2403 = None
        unsqueeze_2404 = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
        unsqueeze_2405 = torch.ops.aten.unsqueeze.default(unsqueeze_2404, -1);  unsqueeze_2404 = None
        mul_902 = torch.ops.aten.mul.Tensor(mul_901, unsqueeze_2405);  mul_901 = unsqueeze_2405 = None
        unsqueeze_2406 = torch.ops.aten.unsqueeze.default(arg585_1, -1);  arg585_1 = None
        unsqueeze_2407 = torch.ops.aten.unsqueeze.default(unsqueeze_2406, -1);  unsqueeze_2406 = None
        add_705 = torch.ops.aten.add.Tensor(mul_902, unsqueeze_2407);  mul_902 = unsqueeze_2407 = None
        relu_298 = torch.ops.aten.relu.default(add_705);  add_705 = None
        convolution_558 = torch.ops.aten.convolution.default(relu_298, arg586_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_298 = arg586_1 = None
        convolution_559 = torch.ops.aten.convolution.default(convolution_558, arg587_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_558 = arg587_1 = None
        add_706 = torch.ops.aten.add.Tensor(arg589_1, 0.001);  arg589_1 = None
        sqrt_301 = torch.ops.aten.sqrt.default(add_706);  add_706 = None
        reciprocal_301 = torch.ops.aten.reciprocal.default(sqrt_301);  sqrt_301 = None
        mul_903 = torch.ops.aten.mul.Tensor(reciprocal_301, 1);  reciprocal_301 = None
        unsqueeze_2408 = torch.ops.aten.unsqueeze.default(arg588_1, -1);  arg588_1 = None
        unsqueeze_2409 = torch.ops.aten.unsqueeze.default(unsqueeze_2408, -1);  unsqueeze_2408 = None
        unsqueeze_2410 = torch.ops.aten.unsqueeze.default(mul_903, -1);  mul_903 = None
        unsqueeze_2411 = torch.ops.aten.unsqueeze.default(unsqueeze_2410, -1);  unsqueeze_2410 = None
        sub_301 = torch.ops.aten.sub.Tensor(convolution_559, unsqueeze_2409);  convolution_559 = unsqueeze_2409 = None
        mul_904 = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_2411);  sub_301 = unsqueeze_2411 = None
        unsqueeze_2412 = torch.ops.aten.unsqueeze.default(arg590_1, -1);  arg590_1 = None
        unsqueeze_2413 = torch.ops.aten.unsqueeze.default(unsqueeze_2412, -1);  unsqueeze_2412 = None
        mul_905 = torch.ops.aten.mul.Tensor(mul_904, unsqueeze_2413);  mul_904 = unsqueeze_2413 = None
        unsqueeze_2414 = torch.ops.aten.unsqueeze.default(arg591_1, -1);  arg591_1 = None
        unsqueeze_2415 = torch.ops.aten.unsqueeze.default(unsqueeze_2414, -1);  unsqueeze_2414 = None
        add_707 = torch.ops.aten.add.Tensor(mul_905, unsqueeze_2415);  mul_905 = unsqueeze_2415 = None
        relu_299 = torch.ops.aten.relu.default(add_679);  add_679 = None
        convolution_560 = torch.ops.aten.convolution.default(relu_299, arg592_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_299 = arg592_1 = None
        add_708 = torch.ops.aten.add.Tensor(arg594_1, 0.001);  arg594_1 = None
        sqrt_302 = torch.ops.aten.sqrt.default(add_708);  add_708 = None
        reciprocal_302 = torch.ops.aten.reciprocal.default(sqrt_302);  sqrt_302 = None
        mul_906 = torch.ops.aten.mul.Tensor(reciprocal_302, 1);  reciprocal_302 = None
        unsqueeze_2416 = torch.ops.aten.unsqueeze.default(arg593_1, -1);  arg593_1 = None
        unsqueeze_2417 = torch.ops.aten.unsqueeze.default(unsqueeze_2416, -1);  unsqueeze_2416 = None
        unsqueeze_2418 = torch.ops.aten.unsqueeze.default(mul_906, -1);  mul_906 = None
        unsqueeze_2419 = torch.ops.aten.unsqueeze.default(unsqueeze_2418, -1);  unsqueeze_2418 = None
        sub_302 = torch.ops.aten.sub.Tensor(convolution_560, unsqueeze_2417);  convolution_560 = unsqueeze_2417 = None
        mul_907 = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_2419);  sub_302 = unsqueeze_2419 = None
        unsqueeze_2420 = torch.ops.aten.unsqueeze.default(arg595_1, -1);  arg595_1 = None
        unsqueeze_2421 = torch.ops.aten.unsqueeze.default(unsqueeze_2420, -1);  unsqueeze_2420 = None
        mul_908 = torch.ops.aten.mul.Tensor(mul_907, unsqueeze_2421);  mul_907 = unsqueeze_2421 = None
        unsqueeze_2422 = torch.ops.aten.unsqueeze.default(arg596_1, -1);  arg596_1 = None
        unsqueeze_2423 = torch.ops.aten.unsqueeze.default(unsqueeze_2422, -1);  unsqueeze_2422 = None
        add_709 = torch.ops.aten.add.Tensor(mul_908, unsqueeze_2423);  mul_908 = unsqueeze_2423 = None
        add_710 = torch.ops.aten.add.Tensor(add_707, add_709);  add_707 = add_709 = None
        cat_26 = torch.ops.aten.cat.default([add_684, add_689, add_698, add_703, add_710], 1);  add_684 = add_689 = add_698 = add_703 = add_710 = None
        relu_300 = torch.ops.aten.relu.default(cat_25);  cat_25 = None
        avg_pool2d_12 = torch.ops.aten.avg_pool2d.default(relu_300, [1, 1], [2, 2], [0, 0], False, False)
        convolution_561 = torch.ops.aten.convolution.default(avg_pool2d_12, arg597_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_12 = arg597_1 = None
        constant_pad_nd_69 = torch.ops.aten.constant_pad_nd.default(relu_300, [-1, 1, -1, 1], 0.0);  relu_300 = None
        avg_pool2d_13 = torch.ops.aten.avg_pool2d.default(constant_pad_nd_69, [1, 1], [2, 2], [0, 0], False, False);  constant_pad_nd_69 = None
        convolution_562 = torch.ops.aten.convolution.default(avg_pool2d_13, arg598_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_13 = arg598_1 = None
        cat_27 = torch.ops.aten.cat.default([convolution_561, convolution_562], 1);  convolution_561 = convolution_562 = None
        add_711 = torch.ops.aten.add.Tensor(arg600_1, 0.001);  arg600_1 = None
        sqrt_303 = torch.ops.aten.sqrt.default(add_711);  add_711 = None
        reciprocal_303 = torch.ops.aten.reciprocal.default(sqrt_303);  sqrt_303 = None
        mul_909 = torch.ops.aten.mul.Tensor(reciprocal_303, 1);  reciprocal_303 = None
        unsqueeze_2424 = torch.ops.aten.unsqueeze.default(arg599_1, -1);  arg599_1 = None
        unsqueeze_2425 = torch.ops.aten.unsqueeze.default(unsqueeze_2424, -1);  unsqueeze_2424 = None
        unsqueeze_2426 = torch.ops.aten.unsqueeze.default(mul_909, -1);  mul_909 = None
        unsqueeze_2427 = torch.ops.aten.unsqueeze.default(unsqueeze_2426, -1);  unsqueeze_2426 = None
        sub_303 = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_2425);  cat_27 = unsqueeze_2425 = None
        mul_910 = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_2427);  sub_303 = unsqueeze_2427 = None
        unsqueeze_2428 = torch.ops.aten.unsqueeze.default(arg601_1, -1);  arg601_1 = None
        unsqueeze_2429 = torch.ops.aten.unsqueeze.default(unsqueeze_2428, -1);  unsqueeze_2428 = None
        mul_911 = torch.ops.aten.mul.Tensor(mul_910, unsqueeze_2429);  mul_910 = unsqueeze_2429 = None
        unsqueeze_2430 = torch.ops.aten.unsqueeze.default(arg602_1, -1);  arg602_1 = None
        unsqueeze_2431 = torch.ops.aten.unsqueeze.default(unsqueeze_2430, -1);  unsqueeze_2430 = None
        add_712 = torch.ops.aten.add.Tensor(mul_911, unsqueeze_2431);  mul_911 = unsqueeze_2431 = None
        relu_301 = torch.ops.aten.relu.default(cat_26)
        convolution_563 = torch.ops.aten.convolution.default(relu_301, arg603_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_301 = arg603_1 = None
        add_713 = torch.ops.aten.add.Tensor(arg605_1, 0.001);  arg605_1 = None
        sqrt_304 = torch.ops.aten.sqrt.default(add_713);  add_713 = None
        reciprocal_304 = torch.ops.aten.reciprocal.default(sqrt_304);  sqrt_304 = None
        mul_912 = torch.ops.aten.mul.Tensor(reciprocal_304, 1);  reciprocal_304 = None
        unsqueeze_2432 = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_2433 = torch.ops.aten.unsqueeze.default(unsqueeze_2432, -1);  unsqueeze_2432 = None
        unsqueeze_2434 = torch.ops.aten.unsqueeze.default(mul_912, -1);  mul_912 = None
        unsqueeze_2435 = torch.ops.aten.unsqueeze.default(unsqueeze_2434, -1);  unsqueeze_2434 = None
        sub_304 = torch.ops.aten.sub.Tensor(convolution_563, unsqueeze_2433);  convolution_563 = unsqueeze_2433 = None
        mul_913 = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_2435);  sub_304 = unsqueeze_2435 = None
        unsqueeze_2436 = torch.ops.aten.unsqueeze.default(arg606_1, -1);  arg606_1 = None
        unsqueeze_2437 = torch.ops.aten.unsqueeze.default(unsqueeze_2436, -1);  unsqueeze_2436 = None
        mul_914 = torch.ops.aten.mul.Tensor(mul_913, unsqueeze_2437);  mul_913 = unsqueeze_2437 = None
        unsqueeze_2438 = torch.ops.aten.unsqueeze.default(arg607_1, -1);  arg607_1 = None
        unsqueeze_2439 = torch.ops.aten.unsqueeze.default(unsqueeze_2438, -1);  unsqueeze_2438 = None
        add_714 = torch.ops.aten.add.Tensor(mul_914, unsqueeze_2439);  mul_914 = unsqueeze_2439 = None
        relu_302 = torch.ops.aten.relu.default(add_712)
        convolution_564 = torch.ops.aten.convolution.default(relu_302, arg608_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_302 = arg608_1 = None
        convolution_565 = torch.ops.aten.convolution.default(convolution_564, arg609_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_564 = arg609_1 = None
        add_715 = torch.ops.aten.add.Tensor(arg611_1, 0.001);  arg611_1 = None
        sqrt_305 = torch.ops.aten.sqrt.default(add_715);  add_715 = None
        reciprocal_305 = torch.ops.aten.reciprocal.default(sqrt_305);  sqrt_305 = None
        mul_915 = torch.ops.aten.mul.Tensor(reciprocal_305, 1);  reciprocal_305 = None
        unsqueeze_2440 = torch.ops.aten.unsqueeze.default(arg610_1, -1);  arg610_1 = None
        unsqueeze_2441 = torch.ops.aten.unsqueeze.default(unsqueeze_2440, -1);  unsqueeze_2440 = None
        unsqueeze_2442 = torch.ops.aten.unsqueeze.default(mul_915, -1);  mul_915 = None
        unsqueeze_2443 = torch.ops.aten.unsqueeze.default(unsqueeze_2442, -1);  unsqueeze_2442 = None
        sub_305 = torch.ops.aten.sub.Tensor(convolution_565, unsqueeze_2441);  convolution_565 = unsqueeze_2441 = None
        mul_916 = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_2443);  sub_305 = unsqueeze_2443 = None
        unsqueeze_2444 = torch.ops.aten.unsqueeze.default(arg612_1, -1);  arg612_1 = None
        unsqueeze_2445 = torch.ops.aten.unsqueeze.default(unsqueeze_2444, -1);  unsqueeze_2444 = None
        mul_917 = torch.ops.aten.mul.Tensor(mul_916, unsqueeze_2445);  mul_916 = unsqueeze_2445 = None
        unsqueeze_2446 = torch.ops.aten.unsqueeze.default(arg613_1, -1);  arg613_1 = None
        unsqueeze_2447 = torch.ops.aten.unsqueeze.default(unsqueeze_2446, -1);  unsqueeze_2446 = None
        add_716 = torch.ops.aten.add.Tensor(mul_917, unsqueeze_2447);  mul_917 = unsqueeze_2447 = None
        relu_303 = torch.ops.aten.relu.default(add_716);  add_716 = None
        convolution_566 = torch.ops.aten.convolution.default(relu_303, arg614_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_303 = arg614_1 = None
        convolution_567 = torch.ops.aten.convolution.default(convolution_566, arg615_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_566 = arg615_1 = None
        add_717 = torch.ops.aten.add.Tensor(arg617_1, 0.001);  arg617_1 = None
        sqrt_306 = torch.ops.aten.sqrt.default(add_717);  add_717 = None
        reciprocal_306 = torch.ops.aten.reciprocal.default(sqrt_306);  sqrt_306 = None
        mul_918 = torch.ops.aten.mul.Tensor(reciprocal_306, 1);  reciprocal_306 = None
        unsqueeze_2448 = torch.ops.aten.unsqueeze.default(arg616_1, -1);  arg616_1 = None
        unsqueeze_2449 = torch.ops.aten.unsqueeze.default(unsqueeze_2448, -1);  unsqueeze_2448 = None
        unsqueeze_2450 = torch.ops.aten.unsqueeze.default(mul_918, -1);  mul_918 = None
        unsqueeze_2451 = torch.ops.aten.unsqueeze.default(unsqueeze_2450, -1);  unsqueeze_2450 = None
        sub_306 = torch.ops.aten.sub.Tensor(convolution_567, unsqueeze_2449);  convolution_567 = unsqueeze_2449 = None
        mul_919 = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_2451);  sub_306 = unsqueeze_2451 = None
        unsqueeze_2452 = torch.ops.aten.unsqueeze.default(arg618_1, -1);  arg618_1 = None
        unsqueeze_2453 = torch.ops.aten.unsqueeze.default(unsqueeze_2452, -1);  unsqueeze_2452 = None
        mul_920 = torch.ops.aten.mul.Tensor(mul_919, unsqueeze_2453);  mul_919 = unsqueeze_2453 = None
        unsqueeze_2454 = torch.ops.aten.unsqueeze.default(arg619_1, -1);  arg619_1 = None
        unsqueeze_2455 = torch.ops.aten.unsqueeze.default(unsqueeze_2454, -1);  unsqueeze_2454 = None
        add_718 = torch.ops.aten.add.Tensor(mul_920, unsqueeze_2455);  mul_920 = unsqueeze_2455 = None
        _low_memory_max_pool2d_with_offsets_63 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_712, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_126 = _low_memory_max_pool2d_with_offsets_63[0];  _low_memory_max_pool2d_with_offsets_63 = None
        add_719 = torch.ops.aten.add.Tensor(add_718, getitem_126);  add_718 = getitem_126 = None
        relu_304 = torch.ops.aten.relu.default(add_714)
        convolution_568 = torch.ops.aten.convolution.default(relu_304, arg620_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432);  relu_304 = arg620_1 = None
        convolution_569 = torch.ops.aten.convolution.default(convolution_568, arg621_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_568 = arg621_1 = None
        add_720 = torch.ops.aten.add.Tensor(arg623_1, 0.001);  arg623_1 = None
        sqrt_307 = torch.ops.aten.sqrt.default(add_720);  add_720 = None
        reciprocal_307 = torch.ops.aten.reciprocal.default(sqrt_307);  sqrt_307 = None
        mul_921 = torch.ops.aten.mul.Tensor(reciprocal_307, 1);  reciprocal_307 = None
        unsqueeze_2456 = torch.ops.aten.unsqueeze.default(arg622_1, -1);  arg622_1 = None
        unsqueeze_2457 = torch.ops.aten.unsqueeze.default(unsqueeze_2456, -1);  unsqueeze_2456 = None
        unsqueeze_2458 = torch.ops.aten.unsqueeze.default(mul_921, -1);  mul_921 = None
        unsqueeze_2459 = torch.ops.aten.unsqueeze.default(unsqueeze_2458, -1);  unsqueeze_2458 = None
        sub_307 = torch.ops.aten.sub.Tensor(convolution_569, unsqueeze_2457);  convolution_569 = unsqueeze_2457 = None
        mul_922 = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_2459);  sub_307 = unsqueeze_2459 = None
        unsqueeze_2460 = torch.ops.aten.unsqueeze.default(arg624_1, -1);  arg624_1 = None
        unsqueeze_2461 = torch.ops.aten.unsqueeze.default(unsqueeze_2460, -1);  unsqueeze_2460 = None
        mul_923 = torch.ops.aten.mul.Tensor(mul_922, unsqueeze_2461);  mul_922 = unsqueeze_2461 = None
        unsqueeze_2462 = torch.ops.aten.unsqueeze.default(arg625_1, -1);  arg625_1 = None
        unsqueeze_2463 = torch.ops.aten.unsqueeze.default(unsqueeze_2462, -1);  unsqueeze_2462 = None
        add_721 = torch.ops.aten.add.Tensor(mul_923, unsqueeze_2463);  mul_923 = unsqueeze_2463 = None
        relu_305 = torch.ops.aten.relu.default(add_721);  add_721 = None
        convolution_570 = torch.ops.aten.convolution.default(relu_305, arg626_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432);  relu_305 = arg626_1 = None
        convolution_571 = torch.ops.aten.convolution.default(convolution_570, arg627_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_570 = arg627_1 = None
        add_722 = torch.ops.aten.add.Tensor(arg629_1, 0.001);  arg629_1 = None
        sqrt_308 = torch.ops.aten.sqrt.default(add_722);  add_722 = None
        reciprocal_308 = torch.ops.aten.reciprocal.default(sqrt_308);  sqrt_308 = None
        mul_924 = torch.ops.aten.mul.Tensor(reciprocal_308, 1);  reciprocal_308 = None
        unsqueeze_2464 = torch.ops.aten.unsqueeze.default(arg628_1, -1);  arg628_1 = None
        unsqueeze_2465 = torch.ops.aten.unsqueeze.default(unsqueeze_2464, -1);  unsqueeze_2464 = None
        unsqueeze_2466 = torch.ops.aten.unsqueeze.default(mul_924, -1);  mul_924 = None
        unsqueeze_2467 = torch.ops.aten.unsqueeze.default(unsqueeze_2466, -1);  unsqueeze_2466 = None
        sub_308 = torch.ops.aten.sub.Tensor(convolution_571, unsqueeze_2465);  convolution_571 = unsqueeze_2465 = None
        mul_925 = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_2467);  sub_308 = unsqueeze_2467 = None
        unsqueeze_2468 = torch.ops.aten.unsqueeze.default(arg630_1, -1);  arg630_1 = None
        unsqueeze_2469 = torch.ops.aten.unsqueeze.default(unsqueeze_2468, -1);  unsqueeze_2468 = None
        mul_926 = torch.ops.aten.mul.Tensor(mul_925, unsqueeze_2469);  mul_925 = unsqueeze_2469 = None
        unsqueeze_2470 = torch.ops.aten.unsqueeze.default(arg631_1, -1);  arg631_1 = None
        unsqueeze_2471 = torch.ops.aten.unsqueeze.default(unsqueeze_2470, -1);  unsqueeze_2470 = None
        add_723 = torch.ops.aten.add.Tensor(mul_926, unsqueeze_2471);  mul_926 = unsqueeze_2471 = None
        _low_memory_max_pool2d_with_offsets_64 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_714, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_128 = _low_memory_max_pool2d_with_offsets_64[0];  _low_memory_max_pool2d_with_offsets_64 = None
        add_724 = torch.ops.aten.add.Tensor(add_723, getitem_128);  add_723 = getitem_128 = None
        relu_306 = torch.ops.aten.relu.default(add_714)
        convolution_572 = torch.ops.aten.convolution.default(relu_306, arg632_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_306 = arg632_1 = None
        convolution_573 = torch.ops.aten.convolution.default(convolution_572, arg633_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_572 = arg633_1 = None
        add_725 = torch.ops.aten.add.Tensor(arg635_1, 0.001);  arg635_1 = None
        sqrt_309 = torch.ops.aten.sqrt.default(add_725);  add_725 = None
        reciprocal_309 = torch.ops.aten.reciprocal.default(sqrt_309);  sqrt_309 = None
        mul_927 = torch.ops.aten.mul.Tensor(reciprocal_309, 1);  reciprocal_309 = None
        unsqueeze_2472 = torch.ops.aten.unsqueeze.default(arg634_1, -1);  arg634_1 = None
        unsqueeze_2473 = torch.ops.aten.unsqueeze.default(unsqueeze_2472, -1);  unsqueeze_2472 = None
        unsqueeze_2474 = torch.ops.aten.unsqueeze.default(mul_927, -1);  mul_927 = None
        unsqueeze_2475 = torch.ops.aten.unsqueeze.default(unsqueeze_2474, -1);  unsqueeze_2474 = None
        sub_309 = torch.ops.aten.sub.Tensor(convolution_573, unsqueeze_2473);  convolution_573 = unsqueeze_2473 = None
        mul_928 = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_2475);  sub_309 = unsqueeze_2475 = None
        unsqueeze_2476 = torch.ops.aten.unsqueeze.default(arg636_1, -1);  arg636_1 = None
        unsqueeze_2477 = torch.ops.aten.unsqueeze.default(unsqueeze_2476, -1);  unsqueeze_2476 = None
        mul_929 = torch.ops.aten.mul.Tensor(mul_928, unsqueeze_2477);  mul_928 = unsqueeze_2477 = None
        unsqueeze_2478 = torch.ops.aten.unsqueeze.default(arg637_1, -1);  arg637_1 = None
        unsqueeze_2479 = torch.ops.aten.unsqueeze.default(unsqueeze_2478, -1);  unsqueeze_2478 = None
        add_726 = torch.ops.aten.add.Tensor(mul_929, unsqueeze_2479);  mul_929 = unsqueeze_2479 = None
        relu_307 = torch.ops.aten.relu.default(add_726);  add_726 = None
        convolution_574 = torch.ops.aten.convolution.default(relu_307, arg638_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_307 = arg638_1 = None
        convolution_575 = torch.ops.aten.convolution.default(convolution_574, arg639_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_574 = arg639_1 = None
        add_727 = torch.ops.aten.add.Tensor(arg641_1, 0.001);  arg641_1 = None
        sqrt_310 = torch.ops.aten.sqrt.default(add_727);  add_727 = None
        reciprocal_310 = torch.ops.aten.reciprocal.default(sqrt_310);  sqrt_310 = None
        mul_930 = torch.ops.aten.mul.Tensor(reciprocal_310, 1);  reciprocal_310 = None
        unsqueeze_2480 = torch.ops.aten.unsqueeze.default(arg640_1, -1);  arg640_1 = None
        unsqueeze_2481 = torch.ops.aten.unsqueeze.default(unsqueeze_2480, -1);  unsqueeze_2480 = None
        unsqueeze_2482 = torch.ops.aten.unsqueeze.default(mul_930, -1);  mul_930 = None
        unsqueeze_2483 = torch.ops.aten.unsqueeze.default(unsqueeze_2482, -1);  unsqueeze_2482 = None
        sub_310 = torch.ops.aten.sub.Tensor(convolution_575, unsqueeze_2481);  convolution_575 = unsqueeze_2481 = None
        mul_931 = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_2483);  sub_310 = unsqueeze_2483 = None
        unsqueeze_2484 = torch.ops.aten.unsqueeze.default(arg642_1, -1);  arg642_1 = None
        unsqueeze_2485 = torch.ops.aten.unsqueeze.default(unsqueeze_2484, -1);  unsqueeze_2484 = None
        mul_932 = torch.ops.aten.mul.Tensor(mul_931, unsqueeze_2485);  mul_931 = unsqueeze_2485 = None
        unsqueeze_2486 = torch.ops.aten.unsqueeze.default(arg643_1, -1);  arg643_1 = None
        unsqueeze_2487 = torch.ops.aten.unsqueeze.default(unsqueeze_2486, -1);  unsqueeze_2486 = None
        add_728 = torch.ops.aten.add.Tensor(mul_932, unsqueeze_2487);  mul_932 = unsqueeze_2487 = None
        relu_308 = torch.ops.aten.relu.default(add_714)
        convolution_576 = torch.ops.aten.convolution.default(relu_308, arg644_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_308 = arg644_1 = None
        convolution_577 = torch.ops.aten.convolution.default(convolution_576, arg645_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_576 = arg645_1 = None
        add_729 = torch.ops.aten.add.Tensor(arg647_1, 0.001);  arg647_1 = None
        sqrt_311 = torch.ops.aten.sqrt.default(add_729);  add_729 = None
        reciprocal_311 = torch.ops.aten.reciprocal.default(sqrt_311);  sqrt_311 = None
        mul_933 = torch.ops.aten.mul.Tensor(reciprocal_311, 1);  reciprocal_311 = None
        unsqueeze_2488 = torch.ops.aten.unsqueeze.default(arg646_1, -1);  arg646_1 = None
        unsqueeze_2489 = torch.ops.aten.unsqueeze.default(unsqueeze_2488, -1);  unsqueeze_2488 = None
        unsqueeze_2490 = torch.ops.aten.unsqueeze.default(mul_933, -1);  mul_933 = None
        unsqueeze_2491 = torch.ops.aten.unsqueeze.default(unsqueeze_2490, -1);  unsqueeze_2490 = None
        sub_311 = torch.ops.aten.sub.Tensor(convolution_577, unsqueeze_2489);  convolution_577 = unsqueeze_2489 = None
        mul_934 = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_2491);  sub_311 = unsqueeze_2491 = None
        unsqueeze_2492 = torch.ops.aten.unsqueeze.default(arg648_1, -1);  arg648_1 = None
        unsqueeze_2493 = torch.ops.aten.unsqueeze.default(unsqueeze_2492, -1);  unsqueeze_2492 = None
        mul_935 = torch.ops.aten.mul.Tensor(mul_934, unsqueeze_2493);  mul_934 = unsqueeze_2493 = None
        unsqueeze_2494 = torch.ops.aten.unsqueeze.default(arg649_1, -1);  arg649_1 = None
        unsqueeze_2495 = torch.ops.aten.unsqueeze.default(unsqueeze_2494, -1);  unsqueeze_2494 = None
        add_730 = torch.ops.aten.add.Tensor(mul_935, unsqueeze_2495);  mul_935 = unsqueeze_2495 = None
        relu_309 = torch.ops.aten.relu.default(add_730);  add_730 = None
        convolution_578 = torch.ops.aten.convolution.default(relu_309, arg650_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_309 = arg650_1 = None
        convolution_579 = torch.ops.aten.convolution.default(convolution_578, arg651_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_578 = arg651_1 = None
        add_731 = torch.ops.aten.add.Tensor(arg653_1, 0.001);  arg653_1 = None
        sqrt_312 = torch.ops.aten.sqrt.default(add_731);  add_731 = None
        reciprocal_312 = torch.ops.aten.reciprocal.default(sqrt_312);  sqrt_312 = None
        mul_936 = torch.ops.aten.mul.Tensor(reciprocal_312, 1);  reciprocal_312 = None
        unsqueeze_2496 = torch.ops.aten.unsqueeze.default(arg652_1, -1);  arg652_1 = None
        unsqueeze_2497 = torch.ops.aten.unsqueeze.default(unsqueeze_2496, -1);  unsqueeze_2496 = None
        unsqueeze_2498 = torch.ops.aten.unsqueeze.default(mul_936, -1);  mul_936 = None
        unsqueeze_2499 = torch.ops.aten.unsqueeze.default(unsqueeze_2498, -1);  unsqueeze_2498 = None
        sub_312 = torch.ops.aten.sub.Tensor(convolution_579, unsqueeze_2497);  convolution_579 = unsqueeze_2497 = None
        mul_937 = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_2499);  sub_312 = unsqueeze_2499 = None
        unsqueeze_2500 = torch.ops.aten.unsqueeze.default(arg654_1, -1);  arg654_1 = None
        unsqueeze_2501 = torch.ops.aten.unsqueeze.default(unsqueeze_2500, -1);  unsqueeze_2500 = None
        mul_938 = torch.ops.aten.mul.Tensor(mul_937, unsqueeze_2501);  mul_937 = unsqueeze_2501 = None
        unsqueeze_2502 = torch.ops.aten.unsqueeze.default(arg655_1, -1);  arg655_1 = None
        unsqueeze_2503 = torch.ops.aten.unsqueeze.default(unsqueeze_2502, -1);  unsqueeze_2502 = None
        add_732 = torch.ops.aten.add.Tensor(mul_938, unsqueeze_2503);  mul_938 = unsqueeze_2503 = None
        add_733 = torch.ops.aten.add.Tensor(add_728, add_732);  add_728 = add_732 = None
        relu_310 = torch.ops.aten.relu.default(add_733)
        convolution_580 = torch.ops.aten.convolution.default(relu_310, arg656_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_310 = arg656_1 = None
        convolution_581 = torch.ops.aten.convolution.default(convolution_580, arg657_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_580 = arg657_1 = None
        add_734 = torch.ops.aten.add.Tensor(arg659_1, 0.001);  arg659_1 = None
        sqrt_313 = torch.ops.aten.sqrt.default(add_734);  add_734 = None
        reciprocal_313 = torch.ops.aten.reciprocal.default(sqrt_313);  sqrt_313 = None
        mul_939 = torch.ops.aten.mul.Tensor(reciprocal_313, 1);  reciprocal_313 = None
        unsqueeze_2504 = torch.ops.aten.unsqueeze.default(arg658_1, -1);  arg658_1 = None
        unsqueeze_2505 = torch.ops.aten.unsqueeze.default(unsqueeze_2504, -1);  unsqueeze_2504 = None
        unsqueeze_2506 = torch.ops.aten.unsqueeze.default(mul_939, -1);  mul_939 = None
        unsqueeze_2507 = torch.ops.aten.unsqueeze.default(unsqueeze_2506, -1);  unsqueeze_2506 = None
        sub_313 = torch.ops.aten.sub.Tensor(convolution_581, unsqueeze_2505);  convolution_581 = unsqueeze_2505 = None
        mul_940 = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_2507);  sub_313 = unsqueeze_2507 = None
        unsqueeze_2508 = torch.ops.aten.unsqueeze.default(arg660_1, -1);  arg660_1 = None
        unsqueeze_2509 = torch.ops.aten.unsqueeze.default(unsqueeze_2508, -1);  unsqueeze_2508 = None
        mul_941 = torch.ops.aten.mul.Tensor(mul_940, unsqueeze_2509);  mul_940 = unsqueeze_2509 = None
        unsqueeze_2510 = torch.ops.aten.unsqueeze.default(arg661_1, -1);  arg661_1 = None
        unsqueeze_2511 = torch.ops.aten.unsqueeze.default(unsqueeze_2510, -1);  unsqueeze_2510 = None
        add_735 = torch.ops.aten.add.Tensor(mul_941, unsqueeze_2511);  mul_941 = unsqueeze_2511 = None
        relu_311 = torch.ops.aten.relu.default(add_735);  add_735 = None
        convolution_582 = torch.ops.aten.convolution.default(relu_311, arg662_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_311 = arg662_1 = None
        convolution_583 = torch.ops.aten.convolution.default(convolution_582, arg663_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_582 = arg663_1 = None
        add_736 = torch.ops.aten.add.Tensor(arg665_1, 0.001);  arg665_1 = None
        sqrt_314 = torch.ops.aten.sqrt.default(add_736);  add_736 = None
        reciprocal_314 = torch.ops.aten.reciprocal.default(sqrt_314);  sqrt_314 = None
        mul_942 = torch.ops.aten.mul.Tensor(reciprocal_314, 1);  reciprocal_314 = None
        unsqueeze_2512 = torch.ops.aten.unsqueeze.default(arg664_1, -1);  arg664_1 = None
        unsqueeze_2513 = torch.ops.aten.unsqueeze.default(unsqueeze_2512, -1);  unsqueeze_2512 = None
        unsqueeze_2514 = torch.ops.aten.unsqueeze.default(mul_942, -1);  mul_942 = None
        unsqueeze_2515 = torch.ops.aten.unsqueeze.default(unsqueeze_2514, -1);  unsqueeze_2514 = None
        sub_314 = torch.ops.aten.sub.Tensor(convolution_583, unsqueeze_2513);  convolution_583 = unsqueeze_2513 = None
        mul_943 = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_2515);  sub_314 = unsqueeze_2515 = None
        unsqueeze_2516 = torch.ops.aten.unsqueeze.default(arg666_1, -1);  arg666_1 = None
        unsqueeze_2517 = torch.ops.aten.unsqueeze.default(unsqueeze_2516, -1);  unsqueeze_2516 = None
        mul_944 = torch.ops.aten.mul.Tensor(mul_943, unsqueeze_2517);  mul_943 = unsqueeze_2517 = None
        unsqueeze_2518 = torch.ops.aten.unsqueeze.default(arg667_1, -1);  arg667_1 = None
        unsqueeze_2519 = torch.ops.aten.unsqueeze.default(unsqueeze_2518, -1);  unsqueeze_2518 = None
        add_737 = torch.ops.aten.add.Tensor(mul_944, unsqueeze_2519);  mul_944 = unsqueeze_2519 = None
        _low_memory_max_pool2d_with_offsets_65 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_714, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_130 = _low_memory_max_pool2d_with_offsets_65[0];  _low_memory_max_pool2d_with_offsets_65 = None
        add_738 = torch.ops.aten.add.Tensor(add_737, getitem_130);  add_737 = getitem_130 = None
        relu_312 = torch.ops.aten.relu.default(add_712);  add_712 = None
        convolution_584 = torch.ops.aten.convolution.default(relu_312, arg668_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_312 = arg668_1 = None
        convolution_585 = torch.ops.aten.convolution.default(convolution_584, arg669_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_584 = arg669_1 = None
        add_739 = torch.ops.aten.add.Tensor(arg671_1, 0.001);  arg671_1 = None
        sqrt_315 = torch.ops.aten.sqrt.default(add_739);  add_739 = None
        reciprocal_315 = torch.ops.aten.reciprocal.default(sqrt_315);  sqrt_315 = None
        mul_945 = torch.ops.aten.mul.Tensor(reciprocal_315, 1);  reciprocal_315 = None
        unsqueeze_2520 = torch.ops.aten.unsqueeze.default(arg670_1, -1);  arg670_1 = None
        unsqueeze_2521 = torch.ops.aten.unsqueeze.default(unsqueeze_2520, -1);  unsqueeze_2520 = None
        unsqueeze_2522 = torch.ops.aten.unsqueeze.default(mul_945, -1);  mul_945 = None
        unsqueeze_2523 = torch.ops.aten.unsqueeze.default(unsqueeze_2522, -1);  unsqueeze_2522 = None
        sub_315 = torch.ops.aten.sub.Tensor(convolution_585, unsqueeze_2521);  convolution_585 = unsqueeze_2521 = None
        mul_946 = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_2523);  sub_315 = unsqueeze_2523 = None
        unsqueeze_2524 = torch.ops.aten.unsqueeze.default(arg672_1, -1);  arg672_1 = None
        unsqueeze_2525 = torch.ops.aten.unsqueeze.default(unsqueeze_2524, -1);  unsqueeze_2524 = None
        mul_947 = torch.ops.aten.mul.Tensor(mul_946, unsqueeze_2525);  mul_946 = unsqueeze_2525 = None
        unsqueeze_2526 = torch.ops.aten.unsqueeze.default(arg673_1, -1);  arg673_1 = None
        unsqueeze_2527 = torch.ops.aten.unsqueeze.default(unsqueeze_2526, -1);  unsqueeze_2526 = None
        add_740 = torch.ops.aten.add.Tensor(mul_947, unsqueeze_2527);  mul_947 = unsqueeze_2527 = None
        relu_313 = torch.ops.aten.relu.default(add_740);  add_740 = None
        convolution_586 = torch.ops.aten.convolution.default(relu_313, arg674_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_313 = arg674_1 = None
        convolution_587 = torch.ops.aten.convolution.default(convolution_586, arg675_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_586 = arg675_1 = None
        add_741 = torch.ops.aten.add.Tensor(arg677_1, 0.001);  arg677_1 = None
        sqrt_316 = torch.ops.aten.sqrt.default(add_741);  add_741 = None
        reciprocal_316 = torch.ops.aten.reciprocal.default(sqrt_316);  sqrt_316 = None
        mul_948 = torch.ops.aten.mul.Tensor(reciprocal_316, 1);  reciprocal_316 = None
        unsqueeze_2528 = torch.ops.aten.unsqueeze.default(arg676_1, -1);  arg676_1 = None
        unsqueeze_2529 = torch.ops.aten.unsqueeze.default(unsqueeze_2528, -1);  unsqueeze_2528 = None
        unsqueeze_2530 = torch.ops.aten.unsqueeze.default(mul_948, -1);  mul_948 = None
        unsqueeze_2531 = torch.ops.aten.unsqueeze.default(unsqueeze_2530, -1);  unsqueeze_2530 = None
        sub_316 = torch.ops.aten.sub.Tensor(convolution_587, unsqueeze_2529);  convolution_587 = unsqueeze_2529 = None
        mul_949 = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_2531);  sub_316 = unsqueeze_2531 = None
        unsqueeze_2532 = torch.ops.aten.unsqueeze.default(arg678_1, -1);  arg678_1 = None
        unsqueeze_2533 = torch.ops.aten.unsqueeze.default(unsqueeze_2532, -1);  unsqueeze_2532 = None
        mul_950 = torch.ops.aten.mul.Tensor(mul_949, unsqueeze_2533);  mul_949 = unsqueeze_2533 = None
        unsqueeze_2534 = torch.ops.aten.unsqueeze.default(arg679_1, -1);  arg679_1 = None
        unsqueeze_2535 = torch.ops.aten.unsqueeze.default(unsqueeze_2534, -1);  unsqueeze_2534 = None
        add_742 = torch.ops.aten.add.Tensor(mul_950, unsqueeze_2535);  mul_950 = unsqueeze_2535 = None
        add_743 = torch.ops.aten.add.Tensor(add_742, add_714);  add_742 = add_714 = None
        cat_28 = torch.ops.aten.cat.default([add_719, add_724, add_733, add_738, add_743], 1);  add_719 = add_724 = add_733 = add_738 = add_743 = None
        relu_314 = torch.ops.aten.relu.default(cat_26);  cat_26 = None
        convolution_588 = torch.ops.aten.convolution.default(relu_314, arg680_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_314 = arg680_1 = None
        add_744 = torch.ops.aten.add.Tensor(arg682_1, 0.001);  arg682_1 = None
        sqrt_317 = torch.ops.aten.sqrt.default(add_744);  add_744 = None
        reciprocal_317 = torch.ops.aten.reciprocal.default(sqrt_317);  sqrt_317 = None
        mul_951 = torch.ops.aten.mul.Tensor(reciprocal_317, 1);  reciprocal_317 = None
        unsqueeze_2536 = torch.ops.aten.unsqueeze.default(arg681_1, -1);  arg681_1 = None
        unsqueeze_2537 = torch.ops.aten.unsqueeze.default(unsqueeze_2536, -1);  unsqueeze_2536 = None
        unsqueeze_2538 = torch.ops.aten.unsqueeze.default(mul_951, -1);  mul_951 = None
        unsqueeze_2539 = torch.ops.aten.unsqueeze.default(unsqueeze_2538, -1);  unsqueeze_2538 = None
        sub_317 = torch.ops.aten.sub.Tensor(convolution_588, unsqueeze_2537);  convolution_588 = unsqueeze_2537 = None
        mul_952 = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_2539);  sub_317 = unsqueeze_2539 = None
        unsqueeze_2540 = torch.ops.aten.unsqueeze.default(arg683_1, -1);  arg683_1 = None
        unsqueeze_2541 = torch.ops.aten.unsqueeze.default(unsqueeze_2540, -1);  unsqueeze_2540 = None
        mul_953 = torch.ops.aten.mul.Tensor(mul_952, unsqueeze_2541);  mul_952 = unsqueeze_2541 = None
        unsqueeze_2542 = torch.ops.aten.unsqueeze.default(arg684_1, -1);  arg684_1 = None
        unsqueeze_2543 = torch.ops.aten.unsqueeze.default(unsqueeze_2542, -1);  unsqueeze_2542 = None
        add_745 = torch.ops.aten.add.Tensor(mul_953, unsqueeze_2543);  mul_953 = unsqueeze_2543 = None
        relu_315 = torch.ops.aten.relu.default(cat_28)
        convolution_589 = torch.ops.aten.convolution.default(relu_315, arg685_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_315 = arg685_1 = None
        add_746 = torch.ops.aten.add.Tensor(arg687_1, 0.001);  arg687_1 = None
        sqrt_318 = torch.ops.aten.sqrt.default(add_746);  add_746 = None
        reciprocal_318 = torch.ops.aten.reciprocal.default(sqrt_318);  sqrt_318 = None
        mul_954 = torch.ops.aten.mul.Tensor(reciprocal_318, 1);  reciprocal_318 = None
        unsqueeze_2544 = torch.ops.aten.unsqueeze.default(arg686_1, -1);  arg686_1 = None
        unsqueeze_2545 = torch.ops.aten.unsqueeze.default(unsqueeze_2544, -1);  unsqueeze_2544 = None
        unsqueeze_2546 = torch.ops.aten.unsqueeze.default(mul_954, -1);  mul_954 = None
        unsqueeze_2547 = torch.ops.aten.unsqueeze.default(unsqueeze_2546, -1);  unsqueeze_2546 = None
        sub_318 = torch.ops.aten.sub.Tensor(convolution_589, unsqueeze_2545);  convolution_589 = unsqueeze_2545 = None
        mul_955 = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_2547);  sub_318 = unsqueeze_2547 = None
        unsqueeze_2548 = torch.ops.aten.unsqueeze.default(arg688_1, -1);  arg688_1 = None
        unsqueeze_2549 = torch.ops.aten.unsqueeze.default(unsqueeze_2548, -1);  unsqueeze_2548 = None
        mul_956 = torch.ops.aten.mul.Tensor(mul_955, unsqueeze_2549);  mul_955 = unsqueeze_2549 = None
        unsqueeze_2550 = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
        unsqueeze_2551 = torch.ops.aten.unsqueeze.default(unsqueeze_2550, -1);  unsqueeze_2550 = None
        add_747 = torch.ops.aten.add.Tensor(mul_956, unsqueeze_2551);  mul_956 = unsqueeze_2551 = None
        relu_316 = torch.ops.aten.relu.default(add_745)
        convolution_590 = torch.ops.aten.convolution.default(relu_316, arg690_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_316 = arg690_1 = None
        convolution_591 = torch.ops.aten.convolution.default(convolution_590, arg691_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_590 = arg691_1 = None
        add_748 = torch.ops.aten.add.Tensor(arg693_1, 0.001);  arg693_1 = None
        sqrt_319 = torch.ops.aten.sqrt.default(add_748);  add_748 = None
        reciprocal_319 = torch.ops.aten.reciprocal.default(sqrt_319);  sqrt_319 = None
        mul_957 = torch.ops.aten.mul.Tensor(reciprocal_319, 1);  reciprocal_319 = None
        unsqueeze_2552 = torch.ops.aten.unsqueeze.default(arg692_1, -1);  arg692_1 = None
        unsqueeze_2553 = torch.ops.aten.unsqueeze.default(unsqueeze_2552, -1);  unsqueeze_2552 = None
        unsqueeze_2554 = torch.ops.aten.unsqueeze.default(mul_957, -1);  mul_957 = None
        unsqueeze_2555 = torch.ops.aten.unsqueeze.default(unsqueeze_2554, -1);  unsqueeze_2554 = None
        sub_319 = torch.ops.aten.sub.Tensor(convolution_591, unsqueeze_2553);  convolution_591 = unsqueeze_2553 = None
        mul_958 = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_2555);  sub_319 = unsqueeze_2555 = None
        unsqueeze_2556 = torch.ops.aten.unsqueeze.default(arg694_1, -1);  arg694_1 = None
        unsqueeze_2557 = torch.ops.aten.unsqueeze.default(unsqueeze_2556, -1);  unsqueeze_2556 = None
        mul_959 = torch.ops.aten.mul.Tensor(mul_958, unsqueeze_2557);  mul_958 = unsqueeze_2557 = None
        unsqueeze_2558 = torch.ops.aten.unsqueeze.default(arg695_1, -1);  arg695_1 = None
        unsqueeze_2559 = torch.ops.aten.unsqueeze.default(unsqueeze_2558, -1);  unsqueeze_2558 = None
        add_749 = torch.ops.aten.add.Tensor(mul_959, unsqueeze_2559);  mul_959 = unsqueeze_2559 = None
        relu_317 = torch.ops.aten.relu.default(add_749);  add_749 = None
        convolution_592 = torch.ops.aten.convolution.default(relu_317, arg696_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_317 = arg696_1 = None
        convolution_593 = torch.ops.aten.convolution.default(convolution_592, arg697_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_592 = arg697_1 = None
        add_750 = torch.ops.aten.add.Tensor(arg699_1, 0.001);  arg699_1 = None
        sqrt_320 = torch.ops.aten.sqrt.default(add_750);  add_750 = None
        reciprocal_320 = torch.ops.aten.reciprocal.default(sqrt_320);  sqrt_320 = None
        mul_960 = torch.ops.aten.mul.Tensor(reciprocal_320, 1);  reciprocal_320 = None
        unsqueeze_2560 = torch.ops.aten.unsqueeze.default(arg698_1, -1);  arg698_1 = None
        unsqueeze_2561 = torch.ops.aten.unsqueeze.default(unsqueeze_2560, -1);  unsqueeze_2560 = None
        unsqueeze_2562 = torch.ops.aten.unsqueeze.default(mul_960, -1);  mul_960 = None
        unsqueeze_2563 = torch.ops.aten.unsqueeze.default(unsqueeze_2562, -1);  unsqueeze_2562 = None
        sub_320 = torch.ops.aten.sub.Tensor(convolution_593, unsqueeze_2561);  convolution_593 = unsqueeze_2561 = None
        mul_961 = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_2563);  sub_320 = unsqueeze_2563 = None
        unsqueeze_2564 = torch.ops.aten.unsqueeze.default(arg700_1, -1);  arg700_1 = None
        unsqueeze_2565 = torch.ops.aten.unsqueeze.default(unsqueeze_2564, -1);  unsqueeze_2564 = None
        mul_962 = torch.ops.aten.mul.Tensor(mul_961, unsqueeze_2565);  mul_961 = unsqueeze_2565 = None
        unsqueeze_2566 = torch.ops.aten.unsqueeze.default(arg701_1, -1);  arg701_1 = None
        unsqueeze_2567 = torch.ops.aten.unsqueeze.default(unsqueeze_2566, -1);  unsqueeze_2566 = None
        add_751 = torch.ops.aten.add.Tensor(mul_962, unsqueeze_2567);  mul_962 = unsqueeze_2567 = None
        _low_memory_max_pool2d_with_offsets_66 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_745, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_132 = _low_memory_max_pool2d_with_offsets_66[0];  _low_memory_max_pool2d_with_offsets_66 = None
        add_752 = torch.ops.aten.add.Tensor(add_751, getitem_132);  add_751 = getitem_132 = None
        relu_318 = torch.ops.aten.relu.default(add_747)
        convolution_594 = torch.ops.aten.convolution.default(relu_318, arg702_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432);  relu_318 = arg702_1 = None
        convolution_595 = torch.ops.aten.convolution.default(convolution_594, arg703_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_594 = arg703_1 = None
        add_753 = torch.ops.aten.add.Tensor(arg705_1, 0.001);  arg705_1 = None
        sqrt_321 = torch.ops.aten.sqrt.default(add_753);  add_753 = None
        reciprocal_321 = torch.ops.aten.reciprocal.default(sqrt_321);  sqrt_321 = None
        mul_963 = torch.ops.aten.mul.Tensor(reciprocal_321, 1);  reciprocal_321 = None
        unsqueeze_2568 = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
        unsqueeze_2569 = torch.ops.aten.unsqueeze.default(unsqueeze_2568, -1);  unsqueeze_2568 = None
        unsqueeze_2570 = torch.ops.aten.unsqueeze.default(mul_963, -1);  mul_963 = None
        unsqueeze_2571 = torch.ops.aten.unsqueeze.default(unsqueeze_2570, -1);  unsqueeze_2570 = None
        sub_321 = torch.ops.aten.sub.Tensor(convolution_595, unsqueeze_2569);  convolution_595 = unsqueeze_2569 = None
        mul_964 = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_2571);  sub_321 = unsqueeze_2571 = None
        unsqueeze_2572 = torch.ops.aten.unsqueeze.default(arg706_1, -1);  arg706_1 = None
        unsqueeze_2573 = torch.ops.aten.unsqueeze.default(unsqueeze_2572, -1);  unsqueeze_2572 = None
        mul_965 = torch.ops.aten.mul.Tensor(mul_964, unsqueeze_2573);  mul_964 = unsqueeze_2573 = None
        unsqueeze_2574 = torch.ops.aten.unsqueeze.default(arg707_1, -1);  arg707_1 = None
        unsqueeze_2575 = torch.ops.aten.unsqueeze.default(unsqueeze_2574, -1);  unsqueeze_2574 = None
        add_754 = torch.ops.aten.add.Tensor(mul_965, unsqueeze_2575);  mul_965 = unsqueeze_2575 = None
        relu_319 = torch.ops.aten.relu.default(add_754);  add_754 = None
        convolution_596 = torch.ops.aten.convolution.default(relu_319, arg708_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432);  relu_319 = arg708_1 = None
        convolution_597 = torch.ops.aten.convolution.default(convolution_596, arg709_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_596 = arg709_1 = None
        add_755 = torch.ops.aten.add.Tensor(arg711_1, 0.001);  arg711_1 = None
        sqrt_322 = torch.ops.aten.sqrt.default(add_755);  add_755 = None
        reciprocal_322 = torch.ops.aten.reciprocal.default(sqrt_322);  sqrt_322 = None
        mul_966 = torch.ops.aten.mul.Tensor(reciprocal_322, 1);  reciprocal_322 = None
        unsqueeze_2576 = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
        unsqueeze_2577 = torch.ops.aten.unsqueeze.default(unsqueeze_2576, -1);  unsqueeze_2576 = None
        unsqueeze_2578 = torch.ops.aten.unsqueeze.default(mul_966, -1);  mul_966 = None
        unsqueeze_2579 = torch.ops.aten.unsqueeze.default(unsqueeze_2578, -1);  unsqueeze_2578 = None
        sub_322 = torch.ops.aten.sub.Tensor(convolution_597, unsqueeze_2577);  convolution_597 = unsqueeze_2577 = None
        mul_967 = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_2579);  sub_322 = unsqueeze_2579 = None
        unsqueeze_2580 = torch.ops.aten.unsqueeze.default(arg712_1, -1);  arg712_1 = None
        unsqueeze_2581 = torch.ops.aten.unsqueeze.default(unsqueeze_2580, -1);  unsqueeze_2580 = None
        mul_968 = torch.ops.aten.mul.Tensor(mul_967, unsqueeze_2581);  mul_967 = unsqueeze_2581 = None
        unsqueeze_2582 = torch.ops.aten.unsqueeze.default(arg713_1, -1);  arg713_1 = None
        unsqueeze_2583 = torch.ops.aten.unsqueeze.default(unsqueeze_2582, -1);  unsqueeze_2582 = None
        add_756 = torch.ops.aten.add.Tensor(mul_968, unsqueeze_2583);  mul_968 = unsqueeze_2583 = None
        _low_memory_max_pool2d_with_offsets_67 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_747, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_134 = _low_memory_max_pool2d_with_offsets_67[0];  _low_memory_max_pool2d_with_offsets_67 = None
        add_757 = torch.ops.aten.add.Tensor(add_756, getitem_134);  add_756 = getitem_134 = None
        relu_320 = torch.ops.aten.relu.default(add_747)
        convolution_598 = torch.ops.aten.convolution.default(relu_320, arg714_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_320 = arg714_1 = None
        convolution_599 = torch.ops.aten.convolution.default(convolution_598, arg715_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_598 = arg715_1 = None
        add_758 = torch.ops.aten.add.Tensor(arg717_1, 0.001);  arg717_1 = None
        sqrt_323 = torch.ops.aten.sqrt.default(add_758);  add_758 = None
        reciprocal_323 = torch.ops.aten.reciprocal.default(sqrt_323);  sqrt_323 = None
        mul_969 = torch.ops.aten.mul.Tensor(reciprocal_323, 1);  reciprocal_323 = None
        unsqueeze_2584 = torch.ops.aten.unsqueeze.default(arg716_1, -1);  arg716_1 = None
        unsqueeze_2585 = torch.ops.aten.unsqueeze.default(unsqueeze_2584, -1);  unsqueeze_2584 = None
        unsqueeze_2586 = torch.ops.aten.unsqueeze.default(mul_969, -1);  mul_969 = None
        unsqueeze_2587 = torch.ops.aten.unsqueeze.default(unsqueeze_2586, -1);  unsqueeze_2586 = None
        sub_323 = torch.ops.aten.sub.Tensor(convolution_599, unsqueeze_2585);  convolution_599 = unsqueeze_2585 = None
        mul_970 = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_2587);  sub_323 = unsqueeze_2587 = None
        unsqueeze_2588 = torch.ops.aten.unsqueeze.default(arg718_1, -1);  arg718_1 = None
        unsqueeze_2589 = torch.ops.aten.unsqueeze.default(unsqueeze_2588, -1);  unsqueeze_2588 = None
        mul_971 = torch.ops.aten.mul.Tensor(mul_970, unsqueeze_2589);  mul_970 = unsqueeze_2589 = None
        unsqueeze_2590 = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_2591 = torch.ops.aten.unsqueeze.default(unsqueeze_2590, -1);  unsqueeze_2590 = None
        add_759 = torch.ops.aten.add.Tensor(mul_971, unsqueeze_2591);  mul_971 = unsqueeze_2591 = None
        relu_321 = torch.ops.aten.relu.default(add_759);  add_759 = None
        convolution_600 = torch.ops.aten.convolution.default(relu_321, arg720_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_321 = arg720_1 = None
        convolution_601 = torch.ops.aten.convolution.default(convolution_600, arg721_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_600 = arg721_1 = None
        add_760 = torch.ops.aten.add.Tensor(arg723_1, 0.001);  arg723_1 = None
        sqrt_324 = torch.ops.aten.sqrt.default(add_760);  add_760 = None
        reciprocal_324 = torch.ops.aten.reciprocal.default(sqrt_324);  sqrt_324 = None
        mul_972 = torch.ops.aten.mul.Tensor(reciprocal_324, 1);  reciprocal_324 = None
        unsqueeze_2592 = torch.ops.aten.unsqueeze.default(arg722_1, -1);  arg722_1 = None
        unsqueeze_2593 = torch.ops.aten.unsqueeze.default(unsqueeze_2592, -1);  unsqueeze_2592 = None
        unsqueeze_2594 = torch.ops.aten.unsqueeze.default(mul_972, -1);  mul_972 = None
        unsqueeze_2595 = torch.ops.aten.unsqueeze.default(unsqueeze_2594, -1);  unsqueeze_2594 = None
        sub_324 = torch.ops.aten.sub.Tensor(convolution_601, unsqueeze_2593);  convolution_601 = unsqueeze_2593 = None
        mul_973 = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_2595);  sub_324 = unsqueeze_2595 = None
        unsqueeze_2596 = torch.ops.aten.unsqueeze.default(arg724_1, -1);  arg724_1 = None
        unsqueeze_2597 = torch.ops.aten.unsqueeze.default(unsqueeze_2596, -1);  unsqueeze_2596 = None
        mul_974 = torch.ops.aten.mul.Tensor(mul_973, unsqueeze_2597);  mul_973 = unsqueeze_2597 = None
        unsqueeze_2598 = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_2599 = torch.ops.aten.unsqueeze.default(unsqueeze_2598, -1);  unsqueeze_2598 = None
        add_761 = torch.ops.aten.add.Tensor(mul_974, unsqueeze_2599);  mul_974 = unsqueeze_2599 = None
        relu_322 = torch.ops.aten.relu.default(add_747)
        convolution_602 = torch.ops.aten.convolution.default(relu_322, arg726_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_322 = arg726_1 = None
        convolution_603 = torch.ops.aten.convolution.default(convolution_602, arg727_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_602 = arg727_1 = None
        add_762 = torch.ops.aten.add.Tensor(arg729_1, 0.001);  arg729_1 = None
        sqrt_325 = torch.ops.aten.sqrt.default(add_762);  add_762 = None
        reciprocal_325 = torch.ops.aten.reciprocal.default(sqrt_325);  sqrt_325 = None
        mul_975 = torch.ops.aten.mul.Tensor(reciprocal_325, 1);  reciprocal_325 = None
        unsqueeze_2600 = torch.ops.aten.unsqueeze.default(arg728_1, -1);  arg728_1 = None
        unsqueeze_2601 = torch.ops.aten.unsqueeze.default(unsqueeze_2600, -1);  unsqueeze_2600 = None
        unsqueeze_2602 = torch.ops.aten.unsqueeze.default(mul_975, -1);  mul_975 = None
        unsqueeze_2603 = torch.ops.aten.unsqueeze.default(unsqueeze_2602, -1);  unsqueeze_2602 = None
        sub_325 = torch.ops.aten.sub.Tensor(convolution_603, unsqueeze_2601);  convolution_603 = unsqueeze_2601 = None
        mul_976 = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_2603);  sub_325 = unsqueeze_2603 = None
        unsqueeze_2604 = torch.ops.aten.unsqueeze.default(arg730_1, -1);  arg730_1 = None
        unsqueeze_2605 = torch.ops.aten.unsqueeze.default(unsqueeze_2604, -1);  unsqueeze_2604 = None
        mul_977 = torch.ops.aten.mul.Tensor(mul_976, unsqueeze_2605);  mul_976 = unsqueeze_2605 = None
        unsqueeze_2606 = torch.ops.aten.unsqueeze.default(arg731_1, -1);  arg731_1 = None
        unsqueeze_2607 = torch.ops.aten.unsqueeze.default(unsqueeze_2606, -1);  unsqueeze_2606 = None
        add_763 = torch.ops.aten.add.Tensor(mul_977, unsqueeze_2607);  mul_977 = unsqueeze_2607 = None
        relu_323 = torch.ops.aten.relu.default(add_763);  add_763 = None
        convolution_604 = torch.ops.aten.convolution.default(relu_323, arg732_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_323 = arg732_1 = None
        convolution_605 = torch.ops.aten.convolution.default(convolution_604, arg733_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_604 = arg733_1 = None
        add_764 = torch.ops.aten.add.Tensor(arg735_1, 0.001);  arg735_1 = None
        sqrt_326 = torch.ops.aten.sqrt.default(add_764);  add_764 = None
        reciprocal_326 = torch.ops.aten.reciprocal.default(sqrt_326);  sqrt_326 = None
        mul_978 = torch.ops.aten.mul.Tensor(reciprocal_326, 1);  reciprocal_326 = None
        unsqueeze_2608 = torch.ops.aten.unsqueeze.default(arg734_1, -1);  arg734_1 = None
        unsqueeze_2609 = torch.ops.aten.unsqueeze.default(unsqueeze_2608, -1);  unsqueeze_2608 = None
        unsqueeze_2610 = torch.ops.aten.unsqueeze.default(mul_978, -1);  mul_978 = None
        unsqueeze_2611 = torch.ops.aten.unsqueeze.default(unsqueeze_2610, -1);  unsqueeze_2610 = None
        sub_326 = torch.ops.aten.sub.Tensor(convolution_605, unsqueeze_2609);  convolution_605 = unsqueeze_2609 = None
        mul_979 = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_2611);  sub_326 = unsqueeze_2611 = None
        unsqueeze_2612 = torch.ops.aten.unsqueeze.default(arg736_1, -1);  arg736_1 = None
        unsqueeze_2613 = torch.ops.aten.unsqueeze.default(unsqueeze_2612, -1);  unsqueeze_2612 = None
        mul_980 = torch.ops.aten.mul.Tensor(mul_979, unsqueeze_2613);  mul_979 = unsqueeze_2613 = None
        unsqueeze_2614 = torch.ops.aten.unsqueeze.default(arg737_1, -1);  arg737_1 = None
        unsqueeze_2615 = torch.ops.aten.unsqueeze.default(unsqueeze_2614, -1);  unsqueeze_2614 = None
        add_765 = torch.ops.aten.add.Tensor(mul_980, unsqueeze_2615);  mul_980 = unsqueeze_2615 = None
        add_766 = torch.ops.aten.add.Tensor(add_761, add_765);  add_761 = add_765 = None
        relu_324 = torch.ops.aten.relu.default(add_766)
        convolution_606 = torch.ops.aten.convolution.default(relu_324, arg738_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_324 = arg738_1 = None
        convolution_607 = torch.ops.aten.convolution.default(convolution_606, arg739_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_606 = arg739_1 = None
        add_767 = torch.ops.aten.add.Tensor(arg741_1, 0.001);  arg741_1 = None
        sqrt_327 = torch.ops.aten.sqrt.default(add_767);  add_767 = None
        reciprocal_327 = torch.ops.aten.reciprocal.default(sqrt_327);  sqrt_327 = None
        mul_981 = torch.ops.aten.mul.Tensor(reciprocal_327, 1);  reciprocal_327 = None
        unsqueeze_2616 = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
        unsqueeze_2617 = torch.ops.aten.unsqueeze.default(unsqueeze_2616, -1);  unsqueeze_2616 = None
        unsqueeze_2618 = torch.ops.aten.unsqueeze.default(mul_981, -1);  mul_981 = None
        unsqueeze_2619 = torch.ops.aten.unsqueeze.default(unsqueeze_2618, -1);  unsqueeze_2618 = None
        sub_327 = torch.ops.aten.sub.Tensor(convolution_607, unsqueeze_2617);  convolution_607 = unsqueeze_2617 = None
        mul_982 = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_2619);  sub_327 = unsqueeze_2619 = None
        unsqueeze_2620 = torch.ops.aten.unsqueeze.default(arg742_1, -1);  arg742_1 = None
        unsqueeze_2621 = torch.ops.aten.unsqueeze.default(unsqueeze_2620, -1);  unsqueeze_2620 = None
        mul_983 = torch.ops.aten.mul.Tensor(mul_982, unsqueeze_2621);  mul_982 = unsqueeze_2621 = None
        unsqueeze_2622 = torch.ops.aten.unsqueeze.default(arg743_1, -1);  arg743_1 = None
        unsqueeze_2623 = torch.ops.aten.unsqueeze.default(unsqueeze_2622, -1);  unsqueeze_2622 = None
        add_768 = torch.ops.aten.add.Tensor(mul_983, unsqueeze_2623);  mul_983 = unsqueeze_2623 = None
        relu_325 = torch.ops.aten.relu.default(add_768);  add_768 = None
        convolution_608 = torch.ops.aten.convolution.default(relu_325, arg744_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_325 = arg744_1 = None
        convolution_609 = torch.ops.aten.convolution.default(convolution_608, arg745_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_608 = arg745_1 = None
        add_769 = torch.ops.aten.add.Tensor(arg747_1, 0.001);  arg747_1 = None
        sqrt_328 = torch.ops.aten.sqrt.default(add_769);  add_769 = None
        reciprocal_328 = torch.ops.aten.reciprocal.default(sqrt_328);  sqrt_328 = None
        mul_984 = torch.ops.aten.mul.Tensor(reciprocal_328, 1);  reciprocal_328 = None
        unsqueeze_2624 = torch.ops.aten.unsqueeze.default(arg746_1, -1);  arg746_1 = None
        unsqueeze_2625 = torch.ops.aten.unsqueeze.default(unsqueeze_2624, -1);  unsqueeze_2624 = None
        unsqueeze_2626 = torch.ops.aten.unsqueeze.default(mul_984, -1);  mul_984 = None
        unsqueeze_2627 = torch.ops.aten.unsqueeze.default(unsqueeze_2626, -1);  unsqueeze_2626 = None
        sub_328 = torch.ops.aten.sub.Tensor(convolution_609, unsqueeze_2625);  convolution_609 = unsqueeze_2625 = None
        mul_985 = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_2627);  sub_328 = unsqueeze_2627 = None
        unsqueeze_2628 = torch.ops.aten.unsqueeze.default(arg748_1, -1);  arg748_1 = None
        unsqueeze_2629 = torch.ops.aten.unsqueeze.default(unsqueeze_2628, -1);  unsqueeze_2628 = None
        mul_986 = torch.ops.aten.mul.Tensor(mul_985, unsqueeze_2629);  mul_985 = unsqueeze_2629 = None
        unsqueeze_2630 = torch.ops.aten.unsqueeze.default(arg749_1, -1);  arg749_1 = None
        unsqueeze_2631 = torch.ops.aten.unsqueeze.default(unsqueeze_2630, -1);  unsqueeze_2630 = None
        add_770 = torch.ops.aten.add.Tensor(mul_986, unsqueeze_2631);  mul_986 = unsqueeze_2631 = None
        _low_memory_max_pool2d_with_offsets_68 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_747, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_136 = _low_memory_max_pool2d_with_offsets_68[0];  _low_memory_max_pool2d_with_offsets_68 = None
        add_771 = torch.ops.aten.add.Tensor(add_770, getitem_136);  add_770 = getitem_136 = None
        relu_326 = torch.ops.aten.relu.default(add_745);  add_745 = None
        convolution_610 = torch.ops.aten.convolution.default(relu_326, arg750_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_326 = arg750_1 = None
        convolution_611 = torch.ops.aten.convolution.default(convolution_610, arg751_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_610 = arg751_1 = None
        add_772 = torch.ops.aten.add.Tensor(arg753_1, 0.001);  arg753_1 = None
        sqrt_329 = torch.ops.aten.sqrt.default(add_772);  add_772 = None
        reciprocal_329 = torch.ops.aten.reciprocal.default(sqrt_329);  sqrt_329 = None
        mul_987 = torch.ops.aten.mul.Tensor(reciprocal_329, 1);  reciprocal_329 = None
        unsqueeze_2632 = torch.ops.aten.unsqueeze.default(arg752_1, -1);  arg752_1 = None
        unsqueeze_2633 = torch.ops.aten.unsqueeze.default(unsqueeze_2632, -1);  unsqueeze_2632 = None
        unsqueeze_2634 = torch.ops.aten.unsqueeze.default(mul_987, -1);  mul_987 = None
        unsqueeze_2635 = torch.ops.aten.unsqueeze.default(unsqueeze_2634, -1);  unsqueeze_2634 = None
        sub_329 = torch.ops.aten.sub.Tensor(convolution_611, unsqueeze_2633);  convolution_611 = unsqueeze_2633 = None
        mul_988 = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_2635);  sub_329 = unsqueeze_2635 = None
        unsqueeze_2636 = torch.ops.aten.unsqueeze.default(arg754_1, -1);  arg754_1 = None
        unsqueeze_2637 = torch.ops.aten.unsqueeze.default(unsqueeze_2636, -1);  unsqueeze_2636 = None
        mul_989 = torch.ops.aten.mul.Tensor(mul_988, unsqueeze_2637);  mul_988 = unsqueeze_2637 = None
        unsqueeze_2638 = torch.ops.aten.unsqueeze.default(arg755_1, -1);  arg755_1 = None
        unsqueeze_2639 = torch.ops.aten.unsqueeze.default(unsqueeze_2638, -1);  unsqueeze_2638 = None
        add_773 = torch.ops.aten.add.Tensor(mul_989, unsqueeze_2639);  mul_989 = unsqueeze_2639 = None
        relu_327 = torch.ops.aten.relu.default(add_773);  add_773 = None
        convolution_612 = torch.ops.aten.convolution.default(relu_327, arg756_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_327 = arg756_1 = None
        convolution_613 = torch.ops.aten.convolution.default(convolution_612, arg757_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_612 = arg757_1 = None
        add_774 = torch.ops.aten.add.Tensor(arg759_1, 0.001);  arg759_1 = None
        sqrt_330 = torch.ops.aten.sqrt.default(add_774);  add_774 = None
        reciprocal_330 = torch.ops.aten.reciprocal.default(sqrt_330);  sqrt_330 = None
        mul_990 = torch.ops.aten.mul.Tensor(reciprocal_330, 1);  reciprocal_330 = None
        unsqueeze_2640 = torch.ops.aten.unsqueeze.default(arg758_1, -1);  arg758_1 = None
        unsqueeze_2641 = torch.ops.aten.unsqueeze.default(unsqueeze_2640, -1);  unsqueeze_2640 = None
        unsqueeze_2642 = torch.ops.aten.unsqueeze.default(mul_990, -1);  mul_990 = None
        unsqueeze_2643 = torch.ops.aten.unsqueeze.default(unsqueeze_2642, -1);  unsqueeze_2642 = None
        sub_330 = torch.ops.aten.sub.Tensor(convolution_613, unsqueeze_2641);  convolution_613 = unsqueeze_2641 = None
        mul_991 = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_2643);  sub_330 = unsqueeze_2643 = None
        unsqueeze_2644 = torch.ops.aten.unsqueeze.default(arg760_1, -1);  arg760_1 = None
        unsqueeze_2645 = torch.ops.aten.unsqueeze.default(unsqueeze_2644, -1);  unsqueeze_2644 = None
        mul_992 = torch.ops.aten.mul.Tensor(mul_991, unsqueeze_2645);  mul_991 = unsqueeze_2645 = None
        unsqueeze_2646 = torch.ops.aten.unsqueeze.default(arg761_1, -1);  arg761_1 = None
        unsqueeze_2647 = torch.ops.aten.unsqueeze.default(unsqueeze_2646, -1);  unsqueeze_2646 = None
        add_775 = torch.ops.aten.add.Tensor(mul_992, unsqueeze_2647);  mul_992 = unsqueeze_2647 = None
        add_776 = torch.ops.aten.add.Tensor(add_775, add_747);  add_775 = add_747 = None
        cat_29 = torch.ops.aten.cat.default([add_752, add_757, add_766, add_771, add_776], 1);  add_752 = add_757 = add_766 = add_771 = add_776 = None
        relu_328 = torch.ops.aten.relu.default(cat_28);  cat_28 = None
        convolution_614 = torch.ops.aten.convolution.default(relu_328, arg762_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_328 = arg762_1 = None
        add_777 = torch.ops.aten.add.Tensor(arg764_1, 0.001);  arg764_1 = None
        sqrt_331 = torch.ops.aten.sqrt.default(add_777);  add_777 = None
        reciprocal_331 = torch.ops.aten.reciprocal.default(sqrt_331);  sqrt_331 = None
        mul_993 = torch.ops.aten.mul.Tensor(reciprocal_331, 1);  reciprocal_331 = None
        unsqueeze_2648 = torch.ops.aten.unsqueeze.default(arg763_1, -1);  arg763_1 = None
        unsqueeze_2649 = torch.ops.aten.unsqueeze.default(unsqueeze_2648, -1);  unsqueeze_2648 = None
        unsqueeze_2650 = torch.ops.aten.unsqueeze.default(mul_993, -1);  mul_993 = None
        unsqueeze_2651 = torch.ops.aten.unsqueeze.default(unsqueeze_2650, -1);  unsqueeze_2650 = None
        sub_331 = torch.ops.aten.sub.Tensor(convolution_614, unsqueeze_2649);  convolution_614 = unsqueeze_2649 = None
        mul_994 = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_2651);  sub_331 = unsqueeze_2651 = None
        unsqueeze_2652 = torch.ops.aten.unsqueeze.default(arg765_1, -1);  arg765_1 = None
        unsqueeze_2653 = torch.ops.aten.unsqueeze.default(unsqueeze_2652, -1);  unsqueeze_2652 = None
        mul_995 = torch.ops.aten.mul.Tensor(mul_994, unsqueeze_2653);  mul_994 = unsqueeze_2653 = None
        unsqueeze_2654 = torch.ops.aten.unsqueeze.default(arg766_1, -1);  arg766_1 = None
        unsqueeze_2655 = torch.ops.aten.unsqueeze.default(unsqueeze_2654, -1);  unsqueeze_2654 = None
        add_778 = torch.ops.aten.add.Tensor(mul_995, unsqueeze_2655);  mul_995 = unsqueeze_2655 = None
        relu_329 = torch.ops.aten.relu.default(cat_29)
        convolution_615 = torch.ops.aten.convolution.default(relu_329, arg767_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_329 = arg767_1 = None
        add_779 = torch.ops.aten.add.Tensor(arg769_1, 0.001);  arg769_1 = None
        sqrt_332 = torch.ops.aten.sqrt.default(add_779);  add_779 = None
        reciprocal_332 = torch.ops.aten.reciprocal.default(sqrt_332);  sqrt_332 = None
        mul_996 = torch.ops.aten.mul.Tensor(reciprocal_332, 1);  reciprocal_332 = None
        unsqueeze_2656 = torch.ops.aten.unsqueeze.default(arg768_1, -1);  arg768_1 = None
        unsqueeze_2657 = torch.ops.aten.unsqueeze.default(unsqueeze_2656, -1);  unsqueeze_2656 = None
        unsqueeze_2658 = torch.ops.aten.unsqueeze.default(mul_996, -1);  mul_996 = None
        unsqueeze_2659 = torch.ops.aten.unsqueeze.default(unsqueeze_2658, -1);  unsqueeze_2658 = None
        sub_332 = torch.ops.aten.sub.Tensor(convolution_615, unsqueeze_2657);  convolution_615 = unsqueeze_2657 = None
        mul_997 = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_2659);  sub_332 = unsqueeze_2659 = None
        unsqueeze_2660 = torch.ops.aten.unsqueeze.default(arg770_1, -1);  arg770_1 = None
        unsqueeze_2661 = torch.ops.aten.unsqueeze.default(unsqueeze_2660, -1);  unsqueeze_2660 = None
        mul_998 = torch.ops.aten.mul.Tensor(mul_997, unsqueeze_2661);  mul_997 = unsqueeze_2661 = None
        unsqueeze_2662 = torch.ops.aten.unsqueeze.default(arg771_1, -1);  arg771_1 = None
        unsqueeze_2663 = torch.ops.aten.unsqueeze.default(unsqueeze_2662, -1);  unsqueeze_2662 = None
        add_780 = torch.ops.aten.add.Tensor(mul_998, unsqueeze_2663);  mul_998 = unsqueeze_2663 = None
        relu_330 = torch.ops.aten.relu.default(add_778)
        convolution_616 = torch.ops.aten.convolution.default(relu_330, arg772_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_330 = arg772_1 = None
        convolution_617 = torch.ops.aten.convolution.default(convolution_616, arg773_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_616 = arg773_1 = None
        add_781 = torch.ops.aten.add.Tensor(arg775_1, 0.001);  arg775_1 = None
        sqrt_333 = torch.ops.aten.sqrt.default(add_781);  add_781 = None
        reciprocal_333 = torch.ops.aten.reciprocal.default(sqrt_333);  sqrt_333 = None
        mul_999 = torch.ops.aten.mul.Tensor(reciprocal_333, 1);  reciprocal_333 = None
        unsqueeze_2664 = torch.ops.aten.unsqueeze.default(arg774_1, -1);  arg774_1 = None
        unsqueeze_2665 = torch.ops.aten.unsqueeze.default(unsqueeze_2664, -1);  unsqueeze_2664 = None
        unsqueeze_2666 = torch.ops.aten.unsqueeze.default(mul_999, -1);  mul_999 = None
        unsqueeze_2667 = torch.ops.aten.unsqueeze.default(unsqueeze_2666, -1);  unsqueeze_2666 = None
        sub_333 = torch.ops.aten.sub.Tensor(convolution_617, unsqueeze_2665);  convolution_617 = unsqueeze_2665 = None
        mul_1000 = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_2667);  sub_333 = unsqueeze_2667 = None
        unsqueeze_2668 = torch.ops.aten.unsqueeze.default(arg776_1, -1);  arg776_1 = None
        unsqueeze_2669 = torch.ops.aten.unsqueeze.default(unsqueeze_2668, -1);  unsqueeze_2668 = None
        mul_1001 = torch.ops.aten.mul.Tensor(mul_1000, unsqueeze_2669);  mul_1000 = unsqueeze_2669 = None
        unsqueeze_2670 = torch.ops.aten.unsqueeze.default(arg777_1, -1);  arg777_1 = None
        unsqueeze_2671 = torch.ops.aten.unsqueeze.default(unsqueeze_2670, -1);  unsqueeze_2670 = None
        add_782 = torch.ops.aten.add.Tensor(mul_1001, unsqueeze_2671);  mul_1001 = unsqueeze_2671 = None
        relu_331 = torch.ops.aten.relu.default(add_782);  add_782 = None
        convolution_618 = torch.ops.aten.convolution.default(relu_331, arg778_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_331 = arg778_1 = None
        convolution_619 = torch.ops.aten.convolution.default(convolution_618, arg779_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_618 = arg779_1 = None
        add_783 = torch.ops.aten.add.Tensor(arg781_1, 0.001);  arg781_1 = None
        sqrt_334 = torch.ops.aten.sqrt.default(add_783);  add_783 = None
        reciprocal_334 = torch.ops.aten.reciprocal.default(sqrt_334);  sqrt_334 = None
        mul_1002 = torch.ops.aten.mul.Tensor(reciprocal_334, 1);  reciprocal_334 = None
        unsqueeze_2672 = torch.ops.aten.unsqueeze.default(arg780_1, -1);  arg780_1 = None
        unsqueeze_2673 = torch.ops.aten.unsqueeze.default(unsqueeze_2672, -1);  unsqueeze_2672 = None
        unsqueeze_2674 = torch.ops.aten.unsqueeze.default(mul_1002, -1);  mul_1002 = None
        unsqueeze_2675 = torch.ops.aten.unsqueeze.default(unsqueeze_2674, -1);  unsqueeze_2674 = None
        sub_334 = torch.ops.aten.sub.Tensor(convolution_619, unsqueeze_2673);  convolution_619 = unsqueeze_2673 = None
        mul_1003 = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_2675);  sub_334 = unsqueeze_2675 = None
        unsqueeze_2676 = torch.ops.aten.unsqueeze.default(arg782_1, -1);  arg782_1 = None
        unsqueeze_2677 = torch.ops.aten.unsqueeze.default(unsqueeze_2676, -1);  unsqueeze_2676 = None
        mul_1004 = torch.ops.aten.mul.Tensor(mul_1003, unsqueeze_2677);  mul_1003 = unsqueeze_2677 = None
        unsqueeze_2678 = torch.ops.aten.unsqueeze.default(arg783_1, -1);  arg783_1 = None
        unsqueeze_2679 = torch.ops.aten.unsqueeze.default(unsqueeze_2678, -1);  unsqueeze_2678 = None
        add_784 = torch.ops.aten.add.Tensor(mul_1004, unsqueeze_2679);  mul_1004 = unsqueeze_2679 = None
        _low_memory_max_pool2d_with_offsets_69 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_778, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_138 = _low_memory_max_pool2d_with_offsets_69[0];  _low_memory_max_pool2d_with_offsets_69 = None
        add_785 = torch.ops.aten.add.Tensor(add_784, getitem_138);  add_784 = getitem_138 = None
        relu_332 = torch.ops.aten.relu.default(add_780)
        convolution_620 = torch.ops.aten.convolution.default(relu_332, arg784_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432);  relu_332 = arg784_1 = None
        convolution_621 = torch.ops.aten.convolution.default(convolution_620, arg785_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_620 = arg785_1 = None
        add_786 = torch.ops.aten.add.Tensor(arg787_1, 0.001);  arg787_1 = None
        sqrt_335 = torch.ops.aten.sqrt.default(add_786);  add_786 = None
        reciprocal_335 = torch.ops.aten.reciprocal.default(sqrt_335);  sqrt_335 = None
        mul_1005 = torch.ops.aten.mul.Tensor(reciprocal_335, 1);  reciprocal_335 = None
        unsqueeze_2680 = torch.ops.aten.unsqueeze.default(arg786_1, -1);  arg786_1 = None
        unsqueeze_2681 = torch.ops.aten.unsqueeze.default(unsqueeze_2680, -1);  unsqueeze_2680 = None
        unsqueeze_2682 = torch.ops.aten.unsqueeze.default(mul_1005, -1);  mul_1005 = None
        unsqueeze_2683 = torch.ops.aten.unsqueeze.default(unsqueeze_2682, -1);  unsqueeze_2682 = None
        sub_335 = torch.ops.aten.sub.Tensor(convolution_621, unsqueeze_2681);  convolution_621 = unsqueeze_2681 = None
        mul_1006 = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_2683);  sub_335 = unsqueeze_2683 = None
        unsqueeze_2684 = torch.ops.aten.unsqueeze.default(arg788_1, -1);  arg788_1 = None
        unsqueeze_2685 = torch.ops.aten.unsqueeze.default(unsqueeze_2684, -1);  unsqueeze_2684 = None
        mul_1007 = torch.ops.aten.mul.Tensor(mul_1006, unsqueeze_2685);  mul_1006 = unsqueeze_2685 = None
        unsqueeze_2686 = torch.ops.aten.unsqueeze.default(arg789_1, -1);  arg789_1 = None
        unsqueeze_2687 = torch.ops.aten.unsqueeze.default(unsqueeze_2686, -1);  unsqueeze_2686 = None
        add_787 = torch.ops.aten.add.Tensor(mul_1007, unsqueeze_2687);  mul_1007 = unsqueeze_2687 = None
        relu_333 = torch.ops.aten.relu.default(add_787);  add_787 = None
        convolution_622 = torch.ops.aten.convolution.default(relu_333, arg790_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432);  relu_333 = arg790_1 = None
        convolution_623 = torch.ops.aten.convolution.default(convolution_622, arg791_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_622 = arg791_1 = None
        add_788 = torch.ops.aten.add.Tensor(arg793_1, 0.001);  arg793_1 = None
        sqrt_336 = torch.ops.aten.sqrt.default(add_788);  add_788 = None
        reciprocal_336 = torch.ops.aten.reciprocal.default(sqrt_336);  sqrt_336 = None
        mul_1008 = torch.ops.aten.mul.Tensor(reciprocal_336, 1);  reciprocal_336 = None
        unsqueeze_2688 = torch.ops.aten.unsqueeze.default(arg792_1, -1);  arg792_1 = None
        unsqueeze_2689 = torch.ops.aten.unsqueeze.default(unsqueeze_2688, -1);  unsqueeze_2688 = None
        unsqueeze_2690 = torch.ops.aten.unsqueeze.default(mul_1008, -1);  mul_1008 = None
        unsqueeze_2691 = torch.ops.aten.unsqueeze.default(unsqueeze_2690, -1);  unsqueeze_2690 = None
        sub_336 = torch.ops.aten.sub.Tensor(convolution_623, unsqueeze_2689);  convolution_623 = unsqueeze_2689 = None
        mul_1009 = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_2691);  sub_336 = unsqueeze_2691 = None
        unsqueeze_2692 = torch.ops.aten.unsqueeze.default(arg794_1, -1);  arg794_1 = None
        unsqueeze_2693 = torch.ops.aten.unsqueeze.default(unsqueeze_2692, -1);  unsqueeze_2692 = None
        mul_1010 = torch.ops.aten.mul.Tensor(mul_1009, unsqueeze_2693);  mul_1009 = unsqueeze_2693 = None
        unsqueeze_2694 = torch.ops.aten.unsqueeze.default(arg795_1, -1);  arg795_1 = None
        unsqueeze_2695 = torch.ops.aten.unsqueeze.default(unsqueeze_2694, -1);  unsqueeze_2694 = None
        add_789 = torch.ops.aten.add.Tensor(mul_1010, unsqueeze_2695);  mul_1010 = unsqueeze_2695 = None
        _low_memory_max_pool2d_with_offsets_70 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_780, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_140 = _low_memory_max_pool2d_with_offsets_70[0];  _low_memory_max_pool2d_with_offsets_70 = None
        add_790 = torch.ops.aten.add.Tensor(add_789, getitem_140);  add_789 = getitem_140 = None
        relu_334 = torch.ops.aten.relu.default(add_780)
        convolution_624 = torch.ops.aten.convolution.default(relu_334, arg796_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_334 = arg796_1 = None
        convolution_625 = torch.ops.aten.convolution.default(convolution_624, arg797_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_624 = arg797_1 = None
        add_791 = torch.ops.aten.add.Tensor(arg799_1, 0.001);  arg799_1 = None
        sqrt_337 = torch.ops.aten.sqrt.default(add_791);  add_791 = None
        reciprocal_337 = torch.ops.aten.reciprocal.default(sqrt_337);  sqrt_337 = None
        mul_1011 = torch.ops.aten.mul.Tensor(reciprocal_337, 1);  reciprocal_337 = None
        unsqueeze_2696 = torch.ops.aten.unsqueeze.default(arg798_1, -1);  arg798_1 = None
        unsqueeze_2697 = torch.ops.aten.unsqueeze.default(unsqueeze_2696, -1);  unsqueeze_2696 = None
        unsqueeze_2698 = torch.ops.aten.unsqueeze.default(mul_1011, -1);  mul_1011 = None
        unsqueeze_2699 = torch.ops.aten.unsqueeze.default(unsqueeze_2698, -1);  unsqueeze_2698 = None
        sub_337 = torch.ops.aten.sub.Tensor(convolution_625, unsqueeze_2697);  convolution_625 = unsqueeze_2697 = None
        mul_1012 = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_2699);  sub_337 = unsqueeze_2699 = None
        unsqueeze_2700 = torch.ops.aten.unsqueeze.default(arg800_1, -1);  arg800_1 = None
        unsqueeze_2701 = torch.ops.aten.unsqueeze.default(unsqueeze_2700, -1);  unsqueeze_2700 = None
        mul_1013 = torch.ops.aten.mul.Tensor(mul_1012, unsqueeze_2701);  mul_1012 = unsqueeze_2701 = None
        unsqueeze_2702 = torch.ops.aten.unsqueeze.default(arg801_1, -1);  arg801_1 = None
        unsqueeze_2703 = torch.ops.aten.unsqueeze.default(unsqueeze_2702, -1);  unsqueeze_2702 = None
        add_792 = torch.ops.aten.add.Tensor(mul_1013, unsqueeze_2703);  mul_1013 = unsqueeze_2703 = None
        relu_335 = torch.ops.aten.relu.default(add_792);  add_792 = None
        convolution_626 = torch.ops.aten.convolution.default(relu_335, arg802_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432);  relu_335 = arg802_1 = None
        convolution_627 = torch.ops.aten.convolution.default(convolution_626, arg803_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_626 = arg803_1 = None
        add_793 = torch.ops.aten.add.Tensor(arg805_1, 0.001);  arg805_1 = None
        sqrt_338 = torch.ops.aten.sqrt.default(add_793);  add_793 = None
        reciprocal_338 = torch.ops.aten.reciprocal.default(sqrt_338);  sqrt_338 = None
        mul_1014 = torch.ops.aten.mul.Tensor(reciprocal_338, 1);  reciprocal_338 = None
        unsqueeze_2704 = torch.ops.aten.unsqueeze.default(arg804_1, -1);  arg804_1 = None
        unsqueeze_2705 = torch.ops.aten.unsqueeze.default(unsqueeze_2704, -1);  unsqueeze_2704 = None
        unsqueeze_2706 = torch.ops.aten.unsqueeze.default(mul_1014, -1);  mul_1014 = None
        unsqueeze_2707 = torch.ops.aten.unsqueeze.default(unsqueeze_2706, -1);  unsqueeze_2706 = None
        sub_338 = torch.ops.aten.sub.Tensor(convolution_627, unsqueeze_2705);  convolution_627 = unsqueeze_2705 = None
        mul_1015 = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_2707);  sub_338 = unsqueeze_2707 = None
        unsqueeze_2708 = torch.ops.aten.unsqueeze.default(arg806_1, -1);  arg806_1 = None
        unsqueeze_2709 = torch.ops.aten.unsqueeze.default(unsqueeze_2708, -1);  unsqueeze_2708 = None
        mul_1016 = torch.ops.aten.mul.Tensor(mul_1015, unsqueeze_2709);  mul_1015 = unsqueeze_2709 = None
        unsqueeze_2710 = torch.ops.aten.unsqueeze.default(arg807_1, -1);  arg807_1 = None
        unsqueeze_2711 = torch.ops.aten.unsqueeze.default(unsqueeze_2710, -1);  unsqueeze_2710 = None
        add_794 = torch.ops.aten.add.Tensor(mul_1016, unsqueeze_2711);  mul_1016 = unsqueeze_2711 = None
        relu_336 = torch.ops.aten.relu.default(add_780)
        convolution_628 = torch.ops.aten.convolution.default(relu_336, arg808_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_336 = arg808_1 = None
        convolution_629 = torch.ops.aten.convolution.default(convolution_628, arg809_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_628 = arg809_1 = None
        add_795 = torch.ops.aten.add.Tensor(arg811_1, 0.001);  arg811_1 = None
        sqrt_339 = torch.ops.aten.sqrt.default(add_795);  add_795 = None
        reciprocal_339 = torch.ops.aten.reciprocal.default(sqrt_339);  sqrt_339 = None
        mul_1017 = torch.ops.aten.mul.Tensor(reciprocal_339, 1);  reciprocal_339 = None
        unsqueeze_2712 = torch.ops.aten.unsqueeze.default(arg810_1, -1);  arg810_1 = None
        unsqueeze_2713 = torch.ops.aten.unsqueeze.default(unsqueeze_2712, -1);  unsqueeze_2712 = None
        unsqueeze_2714 = torch.ops.aten.unsqueeze.default(mul_1017, -1);  mul_1017 = None
        unsqueeze_2715 = torch.ops.aten.unsqueeze.default(unsqueeze_2714, -1);  unsqueeze_2714 = None
        sub_339 = torch.ops.aten.sub.Tensor(convolution_629, unsqueeze_2713);  convolution_629 = unsqueeze_2713 = None
        mul_1018 = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_2715);  sub_339 = unsqueeze_2715 = None
        unsqueeze_2716 = torch.ops.aten.unsqueeze.default(arg812_1, -1);  arg812_1 = None
        unsqueeze_2717 = torch.ops.aten.unsqueeze.default(unsqueeze_2716, -1);  unsqueeze_2716 = None
        mul_1019 = torch.ops.aten.mul.Tensor(mul_1018, unsqueeze_2717);  mul_1018 = unsqueeze_2717 = None
        unsqueeze_2718 = torch.ops.aten.unsqueeze.default(arg813_1, -1);  arg813_1 = None
        unsqueeze_2719 = torch.ops.aten.unsqueeze.default(unsqueeze_2718, -1);  unsqueeze_2718 = None
        add_796 = torch.ops.aten.add.Tensor(mul_1019, unsqueeze_2719);  mul_1019 = unsqueeze_2719 = None
        relu_337 = torch.ops.aten.relu.default(add_796);  add_796 = None
        convolution_630 = torch.ops.aten.convolution.default(relu_337, arg814_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_337 = arg814_1 = None
        convolution_631 = torch.ops.aten.convolution.default(convolution_630, arg815_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_630 = arg815_1 = None
        add_797 = torch.ops.aten.add.Tensor(arg817_1, 0.001);  arg817_1 = None
        sqrt_340 = torch.ops.aten.sqrt.default(add_797);  add_797 = None
        reciprocal_340 = torch.ops.aten.reciprocal.default(sqrt_340);  sqrt_340 = None
        mul_1020 = torch.ops.aten.mul.Tensor(reciprocal_340, 1);  reciprocal_340 = None
        unsqueeze_2720 = torch.ops.aten.unsqueeze.default(arg816_1, -1);  arg816_1 = None
        unsqueeze_2721 = torch.ops.aten.unsqueeze.default(unsqueeze_2720, -1);  unsqueeze_2720 = None
        unsqueeze_2722 = torch.ops.aten.unsqueeze.default(mul_1020, -1);  mul_1020 = None
        unsqueeze_2723 = torch.ops.aten.unsqueeze.default(unsqueeze_2722, -1);  unsqueeze_2722 = None
        sub_340 = torch.ops.aten.sub.Tensor(convolution_631, unsqueeze_2721);  convolution_631 = unsqueeze_2721 = None
        mul_1021 = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_2723);  sub_340 = unsqueeze_2723 = None
        unsqueeze_2724 = torch.ops.aten.unsqueeze.default(arg818_1, -1);  arg818_1 = None
        unsqueeze_2725 = torch.ops.aten.unsqueeze.default(unsqueeze_2724, -1);  unsqueeze_2724 = None
        mul_1022 = torch.ops.aten.mul.Tensor(mul_1021, unsqueeze_2725);  mul_1021 = unsqueeze_2725 = None
        unsqueeze_2726 = torch.ops.aten.unsqueeze.default(arg819_1, -1);  arg819_1 = None
        unsqueeze_2727 = torch.ops.aten.unsqueeze.default(unsqueeze_2726, -1);  unsqueeze_2726 = None
        add_798 = torch.ops.aten.add.Tensor(mul_1022, unsqueeze_2727);  mul_1022 = unsqueeze_2727 = None
        add_799 = torch.ops.aten.add.Tensor(add_794, add_798);  add_794 = add_798 = None
        relu_338 = torch.ops.aten.relu.default(add_799)
        convolution_632 = torch.ops.aten.convolution.default(relu_338, arg820_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_338 = arg820_1 = None
        convolution_633 = torch.ops.aten.convolution.default(convolution_632, arg821_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_632 = arg821_1 = None
        add_800 = torch.ops.aten.add.Tensor(arg823_1, 0.001);  arg823_1 = None
        sqrt_341 = torch.ops.aten.sqrt.default(add_800);  add_800 = None
        reciprocal_341 = torch.ops.aten.reciprocal.default(sqrt_341);  sqrt_341 = None
        mul_1023 = torch.ops.aten.mul.Tensor(reciprocal_341, 1);  reciprocal_341 = None
        unsqueeze_2728 = torch.ops.aten.unsqueeze.default(arg822_1, -1);  arg822_1 = None
        unsqueeze_2729 = torch.ops.aten.unsqueeze.default(unsqueeze_2728, -1);  unsqueeze_2728 = None
        unsqueeze_2730 = torch.ops.aten.unsqueeze.default(mul_1023, -1);  mul_1023 = None
        unsqueeze_2731 = torch.ops.aten.unsqueeze.default(unsqueeze_2730, -1);  unsqueeze_2730 = None
        sub_341 = torch.ops.aten.sub.Tensor(convolution_633, unsqueeze_2729);  convolution_633 = unsqueeze_2729 = None
        mul_1024 = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_2731);  sub_341 = unsqueeze_2731 = None
        unsqueeze_2732 = torch.ops.aten.unsqueeze.default(arg824_1, -1);  arg824_1 = None
        unsqueeze_2733 = torch.ops.aten.unsqueeze.default(unsqueeze_2732, -1);  unsqueeze_2732 = None
        mul_1025 = torch.ops.aten.mul.Tensor(mul_1024, unsqueeze_2733);  mul_1024 = unsqueeze_2733 = None
        unsqueeze_2734 = torch.ops.aten.unsqueeze.default(arg825_1, -1);  arg825_1 = None
        unsqueeze_2735 = torch.ops.aten.unsqueeze.default(unsqueeze_2734, -1);  unsqueeze_2734 = None
        add_801 = torch.ops.aten.add.Tensor(mul_1025, unsqueeze_2735);  mul_1025 = unsqueeze_2735 = None
        relu_339 = torch.ops.aten.relu.default(add_801);  add_801 = None
        convolution_634 = torch.ops.aten.convolution.default(relu_339, arg826_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_339 = arg826_1 = None
        convolution_635 = torch.ops.aten.convolution.default(convolution_634, arg827_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_634 = arg827_1 = None
        add_802 = torch.ops.aten.add.Tensor(arg829_1, 0.001);  arg829_1 = None
        sqrt_342 = torch.ops.aten.sqrt.default(add_802);  add_802 = None
        reciprocal_342 = torch.ops.aten.reciprocal.default(sqrt_342);  sqrt_342 = None
        mul_1026 = torch.ops.aten.mul.Tensor(reciprocal_342, 1);  reciprocal_342 = None
        unsqueeze_2736 = torch.ops.aten.unsqueeze.default(arg828_1, -1);  arg828_1 = None
        unsqueeze_2737 = torch.ops.aten.unsqueeze.default(unsqueeze_2736, -1);  unsqueeze_2736 = None
        unsqueeze_2738 = torch.ops.aten.unsqueeze.default(mul_1026, -1);  mul_1026 = None
        unsqueeze_2739 = torch.ops.aten.unsqueeze.default(unsqueeze_2738, -1);  unsqueeze_2738 = None
        sub_342 = torch.ops.aten.sub.Tensor(convolution_635, unsqueeze_2737);  convolution_635 = unsqueeze_2737 = None
        mul_1027 = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_2739);  sub_342 = unsqueeze_2739 = None
        unsqueeze_2740 = torch.ops.aten.unsqueeze.default(arg830_1, -1);  arg830_1 = None
        unsqueeze_2741 = torch.ops.aten.unsqueeze.default(unsqueeze_2740, -1);  unsqueeze_2740 = None
        mul_1028 = torch.ops.aten.mul.Tensor(mul_1027, unsqueeze_2741);  mul_1027 = unsqueeze_2741 = None
        unsqueeze_2742 = torch.ops.aten.unsqueeze.default(arg831_1, -1);  arg831_1 = None
        unsqueeze_2743 = torch.ops.aten.unsqueeze.default(unsqueeze_2742, -1);  unsqueeze_2742 = None
        add_803 = torch.ops.aten.add.Tensor(mul_1028, unsqueeze_2743);  mul_1028 = unsqueeze_2743 = None
        _low_memory_max_pool2d_with_offsets_71 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_780, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_142 = _low_memory_max_pool2d_with_offsets_71[0];  _low_memory_max_pool2d_with_offsets_71 = None
        add_804 = torch.ops.aten.add.Tensor(add_803, getitem_142);  add_803 = getitem_142 = None
        relu_340 = torch.ops.aten.relu.default(add_778);  add_778 = None
        convolution_636 = torch.ops.aten.convolution.default(relu_340, arg832_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_340 = arg832_1 = None
        convolution_637 = torch.ops.aten.convolution.default(convolution_636, arg833_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_636 = arg833_1 = None
        add_805 = torch.ops.aten.add.Tensor(arg835_1, 0.001);  arg835_1 = None
        sqrt_343 = torch.ops.aten.sqrt.default(add_805);  add_805 = None
        reciprocal_343 = torch.ops.aten.reciprocal.default(sqrt_343);  sqrt_343 = None
        mul_1029 = torch.ops.aten.mul.Tensor(reciprocal_343, 1);  reciprocal_343 = None
        unsqueeze_2744 = torch.ops.aten.unsqueeze.default(arg834_1, -1);  arg834_1 = None
        unsqueeze_2745 = torch.ops.aten.unsqueeze.default(unsqueeze_2744, -1);  unsqueeze_2744 = None
        unsqueeze_2746 = torch.ops.aten.unsqueeze.default(mul_1029, -1);  mul_1029 = None
        unsqueeze_2747 = torch.ops.aten.unsqueeze.default(unsqueeze_2746, -1);  unsqueeze_2746 = None
        sub_343 = torch.ops.aten.sub.Tensor(convolution_637, unsqueeze_2745);  convolution_637 = unsqueeze_2745 = None
        mul_1030 = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_2747);  sub_343 = unsqueeze_2747 = None
        unsqueeze_2748 = torch.ops.aten.unsqueeze.default(arg836_1, -1);  arg836_1 = None
        unsqueeze_2749 = torch.ops.aten.unsqueeze.default(unsqueeze_2748, -1);  unsqueeze_2748 = None
        mul_1031 = torch.ops.aten.mul.Tensor(mul_1030, unsqueeze_2749);  mul_1030 = unsqueeze_2749 = None
        unsqueeze_2750 = torch.ops.aten.unsqueeze.default(arg837_1, -1);  arg837_1 = None
        unsqueeze_2751 = torch.ops.aten.unsqueeze.default(unsqueeze_2750, -1);  unsqueeze_2750 = None
        add_806 = torch.ops.aten.add.Tensor(mul_1031, unsqueeze_2751);  mul_1031 = unsqueeze_2751 = None
        relu_341 = torch.ops.aten.relu.default(add_806);  add_806 = None
        convolution_638 = torch.ops.aten.convolution.default(relu_341, arg838_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  relu_341 = arg838_1 = None
        convolution_639 = torch.ops.aten.convolution.default(convolution_638, arg839_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_638 = arg839_1 = None
        add_807 = torch.ops.aten.add.Tensor(arg841_1, 0.001);  arg841_1 = None
        sqrt_344 = torch.ops.aten.sqrt.default(add_807);  add_807 = None
        reciprocal_344 = torch.ops.aten.reciprocal.default(sqrt_344);  sqrt_344 = None
        mul_1032 = torch.ops.aten.mul.Tensor(reciprocal_344, 1);  reciprocal_344 = None
        unsqueeze_2752 = torch.ops.aten.unsqueeze.default(arg840_1, -1);  arg840_1 = None
        unsqueeze_2753 = torch.ops.aten.unsqueeze.default(unsqueeze_2752, -1);  unsqueeze_2752 = None
        unsqueeze_2754 = torch.ops.aten.unsqueeze.default(mul_1032, -1);  mul_1032 = None
        unsqueeze_2755 = torch.ops.aten.unsqueeze.default(unsqueeze_2754, -1);  unsqueeze_2754 = None
        sub_344 = torch.ops.aten.sub.Tensor(convolution_639, unsqueeze_2753);  convolution_639 = unsqueeze_2753 = None
        mul_1033 = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_2755);  sub_344 = unsqueeze_2755 = None
        unsqueeze_2756 = torch.ops.aten.unsqueeze.default(arg842_1, -1);  arg842_1 = None
        unsqueeze_2757 = torch.ops.aten.unsqueeze.default(unsqueeze_2756, -1);  unsqueeze_2756 = None
        mul_1034 = torch.ops.aten.mul.Tensor(mul_1033, unsqueeze_2757);  mul_1033 = unsqueeze_2757 = None
        unsqueeze_2758 = torch.ops.aten.unsqueeze.default(arg843_1, -1);  arg843_1 = None
        unsqueeze_2759 = torch.ops.aten.unsqueeze.default(unsqueeze_2758, -1);  unsqueeze_2758 = None
        add_808 = torch.ops.aten.add.Tensor(mul_1034, unsqueeze_2759);  mul_1034 = unsqueeze_2759 = None
        add_809 = torch.ops.aten.add.Tensor(add_808, add_780);  add_808 = add_780 = None
        cat_30 = torch.ops.aten.cat.default([add_785, add_790, add_799, add_804, add_809], 1);  add_785 = add_790 = add_799 = add_804 = add_809 = None
        relu_342 = torch.ops.aten.relu.default(cat_29);  cat_29 = None
        convolution_640 = torch.ops.aten.convolution.default(relu_342, arg844_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_342 = arg844_1 = None
        add_810 = torch.ops.aten.add.Tensor(arg846_1, 0.001);  arg846_1 = None
        sqrt_345 = torch.ops.aten.sqrt.default(add_810);  add_810 = None
        reciprocal_345 = torch.ops.aten.reciprocal.default(sqrt_345);  sqrt_345 = None
        mul_1035 = torch.ops.aten.mul.Tensor(reciprocal_345, 1);  reciprocal_345 = None
        unsqueeze_2760 = torch.ops.aten.unsqueeze.default(arg845_1, -1);  arg845_1 = None
        unsqueeze_2761 = torch.ops.aten.unsqueeze.default(unsqueeze_2760, -1);  unsqueeze_2760 = None
        unsqueeze_2762 = torch.ops.aten.unsqueeze.default(mul_1035, -1);  mul_1035 = None
        unsqueeze_2763 = torch.ops.aten.unsqueeze.default(unsqueeze_2762, -1);  unsqueeze_2762 = None
        sub_345 = torch.ops.aten.sub.Tensor(convolution_640, unsqueeze_2761);  convolution_640 = unsqueeze_2761 = None
        mul_1036 = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_2763);  sub_345 = unsqueeze_2763 = None
        unsqueeze_2764 = torch.ops.aten.unsqueeze.default(arg847_1, -1);  arg847_1 = None
        unsqueeze_2765 = torch.ops.aten.unsqueeze.default(unsqueeze_2764, -1);  unsqueeze_2764 = None
        mul_1037 = torch.ops.aten.mul.Tensor(mul_1036, unsqueeze_2765);  mul_1036 = unsqueeze_2765 = None
        unsqueeze_2766 = torch.ops.aten.unsqueeze.default(arg848_1, -1);  arg848_1 = None
        unsqueeze_2767 = torch.ops.aten.unsqueeze.default(unsqueeze_2766, -1);  unsqueeze_2766 = None
        add_811 = torch.ops.aten.add.Tensor(mul_1037, unsqueeze_2767);  mul_1037 = unsqueeze_2767 = None
        relu_343 = torch.ops.aten.relu.default(cat_30)
        convolution_641 = torch.ops.aten.convolution.default(relu_343, arg849_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_343 = arg849_1 = None
        add_812 = torch.ops.aten.add.Tensor(arg851_1, 0.001);  arg851_1 = None
        sqrt_346 = torch.ops.aten.sqrt.default(add_812);  add_812 = None
        reciprocal_346 = torch.ops.aten.reciprocal.default(sqrt_346);  sqrt_346 = None
        mul_1038 = torch.ops.aten.mul.Tensor(reciprocal_346, 1);  reciprocal_346 = None
        unsqueeze_2768 = torch.ops.aten.unsqueeze.default(arg850_1, -1);  arg850_1 = None
        unsqueeze_2769 = torch.ops.aten.unsqueeze.default(unsqueeze_2768, -1);  unsqueeze_2768 = None
        unsqueeze_2770 = torch.ops.aten.unsqueeze.default(mul_1038, -1);  mul_1038 = None
        unsqueeze_2771 = torch.ops.aten.unsqueeze.default(unsqueeze_2770, -1);  unsqueeze_2770 = None
        sub_346 = torch.ops.aten.sub.Tensor(convolution_641, unsqueeze_2769);  convolution_641 = unsqueeze_2769 = None
        mul_1039 = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_2771);  sub_346 = unsqueeze_2771 = None
        unsqueeze_2772 = torch.ops.aten.unsqueeze.default(arg852_1, -1);  arg852_1 = None
        unsqueeze_2773 = torch.ops.aten.unsqueeze.default(unsqueeze_2772, -1);  unsqueeze_2772 = None
        mul_1040 = torch.ops.aten.mul.Tensor(mul_1039, unsqueeze_2773);  mul_1039 = unsqueeze_2773 = None
        unsqueeze_2774 = torch.ops.aten.unsqueeze.default(arg853_1, -1);  arg853_1 = None
        unsqueeze_2775 = torch.ops.aten.unsqueeze.default(unsqueeze_2774, -1);  unsqueeze_2774 = None
        add_813 = torch.ops.aten.add.Tensor(mul_1040, unsqueeze_2775);  mul_1040 = unsqueeze_2775 = None
        relu_344 = torch.ops.aten.relu.default(add_811)
        constant_pad_nd_70 = torch.ops.aten.constant_pad_nd.default(relu_344, [2, 2, 2, 2], 0.0);  relu_344 = None
        convolution_642 = torch.ops.aten.convolution.default(constant_pad_nd_70, arg854_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864);  constant_pad_nd_70 = arg854_1 = None
        convolution_643 = torch.ops.aten.convolution.default(convolution_642, arg855_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_642 = arg855_1 = None
        add_814 = torch.ops.aten.add.Tensor(arg857_1, 0.001);  arg857_1 = None
        sqrt_347 = torch.ops.aten.sqrt.default(add_814);  add_814 = None
        reciprocal_347 = torch.ops.aten.reciprocal.default(sqrt_347);  sqrt_347 = None
        mul_1041 = torch.ops.aten.mul.Tensor(reciprocal_347, 1);  reciprocal_347 = None
        unsqueeze_2776 = torch.ops.aten.unsqueeze.default(arg856_1, -1);  arg856_1 = None
        unsqueeze_2777 = torch.ops.aten.unsqueeze.default(unsqueeze_2776, -1);  unsqueeze_2776 = None
        unsqueeze_2778 = torch.ops.aten.unsqueeze.default(mul_1041, -1);  mul_1041 = None
        unsqueeze_2779 = torch.ops.aten.unsqueeze.default(unsqueeze_2778, -1);  unsqueeze_2778 = None
        sub_347 = torch.ops.aten.sub.Tensor(convolution_643, unsqueeze_2777);  convolution_643 = unsqueeze_2777 = None
        mul_1042 = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_2779);  sub_347 = unsqueeze_2779 = None
        unsqueeze_2780 = torch.ops.aten.unsqueeze.default(arg858_1, -1);  arg858_1 = None
        unsqueeze_2781 = torch.ops.aten.unsqueeze.default(unsqueeze_2780, -1);  unsqueeze_2780 = None
        mul_1043 = torch.ops.aten.mul.Tensor(mul_1042, unsqueeze_2781);  mul_1042 = unsqueeze_2781 = None
        unsqueeze_2782 = torch.ops.aten.unsqueeze.default(arg859_1, -1);  arg859_1 = None
        unsqueeze_2783 = torch.ops.aten.unsqueeze.default(unsqueeze_2782, -1);  unsqueeze_2782 = None
        add_815 = torch.ops.aten.add.Tensor(mul_1043, unsqueeze_2783);  mul_1043 = unsqueeze_2783 = None
        relu_345 = torch.ops.aten.relu.default(add_815);  add_815 = None
        convolution_644 = torch.ops.aten.convolution.default(relu_345, arg860_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_345 = arg860_1 = None
        convolution_645 = torch.ops.aten.convolution.default(convolution_644, arg861_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_644 = arg861_1 = None
        add_816 = torch.ops.aten.add.Tensor(arg863_1, 0.001);  arg863_1 = None
        sqrt_348 = torch.ops.aten.sqrt.default(add_816);  add_816 = None
        reciprocal_348 = torch.ops.aten.reciprocal.default(sqrt_348);  sqrt_348 = None
        mul_1044 = torch.ops.aten.mul.Tensor(reciprocal_348, 1);  reciprocal_348 = None
        unsqueeze_2784 = torch.ops.aten.unsqueeze.default(arg862_1, -1);  arg862_1 = None
        unsqueeze_2785 = torch.ops.aten.unsqueeze.default(unsqueeze_2784, -1);  unsqueeze_2784 = None
        unsqueeze_2786 = torch.ops.aten.unsqueeze.default(mul_1044, -1);  mul_1044 = None
        unsqueeze_2787 = torch.ops.aten.unsqueeze.default(unsqueeze_2786, -1);  unsqueeze_2786 = None
        sub_348 = torch.ops.aten.sub.Tensor(convolution_645, unsqueeze_2785);  convolution_645 = unsqueeze_2785 = None
        mul_1045 = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_2787);  sub_348 = unsqueeze_2787 = None
        unsqueeze_2788 = torch.ops.aten.unsqueeze.default(arg864_1, -1);  arg864_1 = None
        unsqueeze_2789 = torch.ops.aten.unsqueeze.default(unsqueeze_2788, -1);  unsqueeze_2788 = None
        mul_1046 = torch.ops.aten.mul.Tensor(mul_1045, unsqueeze_2789);  mul_1045 = unsqueeze_2789 = None
        unsqueeze_2790 = torch.ops.aten.unsqueeze.default(arg865_1, -1);  arg865_1 = None
        unsqueeze_2791 = torch.ops.aten.unsqueeze.default(unsqueeze_2790, -1);  unsqueeze_2790 = None
        add_817 = torch.ops.aten.add.Tensor(mul_1046, unsqueeze_2791);  mul_1046 = unsqueeze_2791 = None
        constant_pad_nd_71 = torch.ops.aten.constant_pad_nd.default(add_811, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_72 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_71, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_71 = None
        getitem_144 = _low_memory_max_pool2d_with_offsets_72[0];  _low_memory_max_pool2d_with_offsets_72 = None
        add_818 = torch.ops.aten.add.Tensor(add_817, getitem_144);  add_817 = getitem_144 = None
        relu_346 = torch.ops.aten.relu.default(add_813)
        constant_pad_nd_72 = torch.ops.aten.constant_pad_nd.default(relu_346, [3, 3, 3, 3], 0.0);  relu_346 = None
        convolution_646 = torch.ops.aten.convolution.default(constant_pad_nd_72, arg866_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864);  constant_pad_nd_72 = arg866_1 = None
        convolution_647 = torch.ops.aten.convolution.default(convolution_646, arg867_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_646 = arg867_1 = None
        add_819 = torch.ops.aten.add.Tensor(arg869_1, 0.001);  arg869_1 = None
        sqrt_349 = torch.ops.aten.sqrt.default(add_819);  add_819 = None
        reciprocal_349 = torch.ops.aten.reciprocal.default(sqrt_349);  sqrt_349 = None
        mul_1047 = torch.ops.aten.mul.Tensor(reciprocal_349, 1);  reciprocal_349 = None
        unsqueeze_2792 = torch.ops.aten.unsqueeze.default(arg868_1, -1);  arg868_1 = None
        unsqueeze_2793 = torch.ops.aten.unsqueeze.default(unsqueeze_2792, -1);  unsqueeze_2792 = None
        unsqueeze_2794 = torch.ops.aten.unsqueeze.default(mul_1047, -1);  mul_1047 = None
        unsqueeze_2795 = torch.ops.aten.unsqueeze.default(unsqueeze_2794, -1);  unsqueeze_2794 = None
        sub_349 = torch.ops.aten.sub.Tensor(convolution_647, unsqueeze_2793);  convolution_647 = unsqueeze_2793 = None
        mul_1048 = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_2795);  sub_349 = unsqueeze_2795 = None
        unsqueeze_2796 = torch.ops.aten.unsqueeze.default(arg870_1, -1);  arg870_1 = None
        unsqueeze_2797 = torch.ops.aten.unsqueeze.default(unsqueeze_2796, -1);  unsqueeze_2796 = None
        mul_1049 = torch.ops.aten.mul.Tensor(mul_1048, unsqueeze_2797);  mul_1048 = unsqueeze_2797 = None
        unsqueeze_2798 = torch.ops.aten.unsqueeze.default(arg871_1, -1);  arg871_1 = None
        unsqueeze_2799 = torch.ops.aten.unsqueeze.default(unsqueeze_2798, -1);  unsqueeze_2798 = None
        add_820 = torch.ops.aten.add.Tensor(mul_1049, unsqueeze_2799);  mul_1049 = unsqueeze_2799 = None
        relu_347 = torch.ops.aten.relu.default(add_820);  add_820 = None
        convolution_648 = torch.ops.aten.convolution.default(relu_347, arg872_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864);  relu_347 = arg872_1 = None
        convolution_649 = torch.ops.aten.convolution.default(convolution_648, arg873_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_648 = arg873_1 = None
        add_821 = torch.ops.aten.add.Tensor(arg875_1, 0.001);  arg875_1 = None
        sqrt_350 = torch.ops.aten.sqrt.default(add_821);  add_821 = None
        reciprocal_350 = torch.ops.aten.reciprocal.default(sqrt_350);  sqrt_350 = None
        mul_1050 = torch.ops.aten.mul.Tensor(reciprocal_350, 1);  reciprocal_350 = None
        unsqueeze_2800 = torch.ops.aten.unsqueeze.default(arg874_1, -1);  arg874_1 = None
        unsqueeze_2801 = torch.ops.aten.unsqueeze.default(unsqueeze_2800, -1);  unsqueeze_2800 = None
        unsqueeze_2802 = torch.ops.aten.unsqueeze.default(mul_1050, -1);  mul_1050 = None
        unsqueeze_2803 = torch.ops.aten.unsqueeze.default(unsqueeze_2802, -1);  unsqueeze_2802 = None
        sub_350 = torch.ops.aten.sub.Tensor(convolution_649, unsqueeze_2801);  convolution_649 = unsqueeze_2801 = None
        mul_1051 = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_2803);  sub_350 = unsqueeze_2803 = None
        unsqueeze_2804 = torch.ops.aten.unsqueeze.default(arg876_1, -1);  arg876_1 = None
        unsqueeze_2805 = torch.ops.aten.unsqueeze.default(unsqueeze_2804, -1);  unsqueeze_2804 = None
        mul_1052 = torch.ops.aten.mul.Tensor(mul_1051, unsqueeze_2805);  mul_1051 = unsqueeze_2805 = None
        unsqueeze_2806 = torch.ops.aten.unsqueeze.default(arg877_1, -1);  arg877_1 = None
        unsqueeze_2807 = torch.ops.aten.unsqueeze.default(unsqueeze_2806, -1);  unsqueeze_2806 = None
        add_822 = torch.ops.aten.add.Tensor(mul_1052, unsqueeze_2807);  mul_1052 = unsqueeze_2807 = None
        constant_pad_nd_73 = torch.ops.aten.constant_pad_nd.default(add_813, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_73 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_73, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_73 = None
        getitem_146 = _low_memory_max_pool2d_with_offsets_73[0];  _low_memory_max_pool2d_with_offsets_73 = None
        add_823 = torch.ops.aten.add.Tensor(add_822, getitem_146);  add_822 = getitem_146 = None
        relu_348 = torch.ops.aten.relu.default(add_813)
        constant_pad_nd_74 = torch.ops.aten.constant_pad_nd.default(relu_348, [2, 2, 2, 2], 0.0);  relu_348 = None
        convolution_650 = torch.ops.aten.convolution.default(constant_pad_nd_74, arg878_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864);  constant_pad_nd_74 = arg878_1 = None
        convolution_651 = torch.ops.aten.convolution.default(convolution_650, arg879_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_650 = arg879_1 = None
        add_824 = torch.ops.aten.add.Tensor(arg881_1, 0.001);  arg881_1 = None
        sqrt_351 = torch.ops.aten.sqrt.default(add_824);  add_824 = None
        reciprocal_351 = torch.ops.aten.reciprocal.default(sqrt_351);  sqrt_351 = None
        mul_1053 = torch.ops.aten.mul.Tensor(reciprocal_351, 1);  reciprocal_351 = None
        unsqueeze_2808 = torch.ops.aten.unsqueeze.default(arg880_1, -1);  arg880_1 = None
        unsqueeze_2809 = torch.ops.aten.unsqueeze.default(unsqueeze_2808, -1);  unsqueeze_2808 = None
        unsqueeze_2810 = torch.ops.aten.unsqueeze.default(mul_1053, -1);  mul_1053 = None
        unsqueeze_2811 = torch.ops.aten.unsqueeze.default(unsqueeze_2810, -1);  unsqueeze_2810 = None
        sub_351 = torch.ops.aten.sub.Tensor(convolution_651, unsqueeze_2809);  convolution_651 = unsqueeze_2809 = None
        mul_1054 = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_2811);  sub_351 = unsqueeze_2811 = None
        unsqueeze_2812 = torch.ops.aten.unsqueeze.default(arg882_1, -1);  arg882_1 = None
        unsqueeze_2813 = torch.ops.aten.unsqueeze.default(unsqueeze_2812, -1);  unsqueeze_2812 = None
        mul_1055 = torch.ops.aten.mul.Tensor(mul_1054, unsqueeze_2813);  mul_1054 = unsqueeze_2813 = None
        unsqueeze_2814 = torch.ops.aten.unsqueeze.default(arg883_1, -1);  arg883_1 = None
        unsqueeze_2815 = torch.ops.aten.unsqueeze.default(unsqueeze_2814, -1);  unsqueeze_2814 = None
        add_825 = torch.ops.aten.add.Tensor(mul_1055, unsqueeze_2815);  mul_1055 = unsqueeze_2815 = None
        relu_349 = torch.ops.aten.relu.default(add_825);  add_825 = None
        convolution_652 = torch.ops.aten.convolution.default(relu_349, arg884_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_349 = arg884_1 = None
        convolution_653 = torch.ops.aten.convolution.default(convolution_652, arg885_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_652 = arg885_1 = None
        add_826 = torch.ops.aten.add.Tensor(arg887_1, 0.001);  arg887_1 = None
        sqrt_352 = torch.ops.aten.sqrt.default(add_826);  add_826 = None
        reciprocal_352 = torch.ops.aten.reciprocal.default(sqrt_352);  sqrt_352 = None
        mul_1056 = torch.ops.aten.mul.Tensor(reciprocal_352, 1);  reciprocal_352 = None
        unsqueeze_2816 = torch.ops.aten.unsqueeze.default(arg886_1, -1);  arg886_1 = None
        unsqueeze_2817 = torch.ops.aten.unsqueeze.default(unsqueeze_2816, -1);  unsqueeze_2816 = None
        unsqueeze_2818 = torch.ops.aten.unsqueeze.default(mul_1056, -1);  mul_1056 = None
        unsqueeze_2819 = torch.ops.aten.unsqueeze.default(unsqueeze_2818, -1);  unsqueeze_2818 = None
        sub_352 = torch.ops.aten.sub.Tensor(convolution_653, unsqueeze_2817);  convolution_653 = unsqueeze_2817 = None
        mul_1057 = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_2819);  sub_352 = unsqueeze_2819 = None
        unsqueeze_2820 = torch.ops.aten.unsqueeze.default(arg888_1, -1);  arg888_1 = None
        unsqueeze_2821 = torch.ops.aten.unsqueeze.default(unsqueeze_2820, -1);  unsqueeze_2820 = None
        mul_1058 = torch.ops.aten.mul.Tensor(mul_1057, unsqueeze_2821);  mul_1057 = unsqueeze_2821 = None
        unsqueeze_2822 = torch.ops.aten.unsqueeze.default(arg889_1, -1);  arg889_1 = None
        unsqueeze_2823 = torch.ops.aten.unsqueeze.default(unsqueeze_2822, -1);  unsqueeze_2822 = None
        add_827 = torch.ops.aten.add.Tensor(mul_1058, unsqueeze_2823);  mul_1058 = unsqueeze_2823 = None
        relu_350 = torch.ops.aten.relu.default(add_813)
        constant_pad_nd_75 = torch.ops.aten.constant_pad_nd.default(relu_350, [1, 1, 1, 1], 0.0);  relu_350 = None
        convolution_654 = torch.ops.aten.convolution.default(constant_pad_nd_75, arg890_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864);  constant_pad_nd_75 = arg890_1 = None
        convolution_655 = torch.ops.aten.convolution.default(convolution_654, arg891_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_654 = arg891_1 = None
        add_828 = torch.ops.aten.add.Tensor(arg893_1, 0.001);  arg893_1 = None
        sqrt_353 = torch.ops.aten.sqrt.default(add_828);  add_828 = None
        reciprocal_353 = torch.ops.aten.reciprocal.default(sqrt_353);  sqrt_353 = None
        mul_1059 = torch.ops.aten.mul.Tensor(reciprocal_353, 1);  reciprocal_353 = None
        unsqueeze_2824 = torch.ops.aten.unsqueeze.default(arg892_1, -1);  arg892_1 = None
        unsqueeze_2825 = torch.ops.aten.unsqueeze.default(unsqueeze_2824, -1);  unsqueeze_2824 = None
        unsqueeze_2826 = torch.ops.aten.unsqueeze.default(mul_1059, -1);  mul_1059 = None
        unsqueeze_2827 = torch.ops.aten.unsqueeze.default(unsqueeze_2826, -1);  unsqueeze_2826 = None
        sub_353 = torch.ops.aten.sub.Tensor(convolution_655, unsqueeze_2825);  convolution_655 = unsqueeze_2825 = None
        mul_1060 = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_2827);  sub_353 = unsqueeze_2827 = None
        unsqueeze_2828 = torch.ops.aten.unsqueeze.default(arg894_1, -1);  arg894_1 = None
        unsqueeze_2829 = torch.ops.aten.unsqueeze.default(unsqueeze_2828, -1);  unsqueeze_2828 = None
        mul_1061 = torch.ops.aten.mul.Tensor(mul_1060, unsqueeze_2829);  mul_1060 = unsqueeze_2829 = None
        unsqueeze_2830 = torch.ops.aten.unsqueeze.default(arg895_1, -1);  arg895_1 = None
        unsqueeze_2831 = torch.ops.aten.unsqueeze.default(unsqueeze_2830, -1);  unsqueeze_2830 = None
        add_829 = torch.ops.aten.add.Tensor(mul_1061, unsqueeze_2831);  mul_1061 = unsqueeze_2831 = None
        relu_351 = torch.ops.aten.relu.default(add_829);  add_829 = None
        convolution_656 = torch.ops.aten.convolution.default(relu_351, arg896_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_351 = arg896_1 = None
        convolution_657 = torch.ops.aten.convolution.default(convolution_656, arg897_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_656 = arg897_1 = None
        add_830 = torch.ops.aten.add.Tensor(arg899_1, 0.001);  arg899_1 = None
        sqrt_354 = torch.ops.aten.sqrt.default(add_830);  add_830 = None
        reciprocal_354 = torch.ops.aten.reciprocal.default(sqrt_354);  sqrt_354 = None
        mul_1062 = torch.ops.aten.mul.Tensor(reciprocal_354, 1);  reciprocal_354 = None
        unsqueeze_2832 = torch.ops.aten.unsqueeze.default(arg898_1, -1);  arg898_1 = None
        unsqueeze_2833 = torch.ops.aten.unsqueeze.default(unsqueeze_2832, -1);  unsqueeze_2832 = None
        unsqueeze_2834 = torch.ops.aten.unsqueeze.default(mul_1062, -1);  mul_1062 = None
        unsqueeze_2835 = torch.ops.aten.unsqueeze.default(unsqueeze_2834, -1);  unsqueeze_2834 = None
        sub_354 = torch.ops.aten.sub.Tensor(convolution_657, unsqueeze_2833);  convolution_657 = unsqueeze_2833 = None
        mul_1063 = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_2835);  sub_354 = unsqueeze_2835 = None
        unsqueeze_2836 = torch.ops.aten.unsqueeze.default(arg900_1, -1);  arg900_1 = None
        unsqueeze_2837 = torch.ops.aten.unsqueeze.default(unsqueeze_2836, -1);  unsqueeze_2836 = None
        mul_1064 = torch.ops.aten.mul.Tensor(mul_1063, unsqueeze_2837);  mul_1063 = unsqueeze_2837 = None
        unsqueeze_2838 = torch.ops.aten.unsqueeze.default(arg901_1, -1);  arg901_1 = None
        unsqueeze_2839 = torch.ops.aten.unsqueeze.default(unsqueeze_2838, -1);  unsqueeze_2838 = None
        add_831 = torch.ops.aten.add.Tensor(mul_1064, unsqueeze_2839);  mul_1064 = unsqueeze_2839 = None
        add_832 = torch.ops.aten.add.Tensor(add_827, add_831);  add_827 = add_831 = None
        relu_352 = torch.ops.aten.relu.default(add_832)
        convolution_658 = torch.ops.aten.convolution.default(relu_352, arg902_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_352 = arg902_1 = None
        convolution_659 = torch.ops.aten.convolution.default(convolution_658, arg903_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_658 = arg903_1 = None
        add_833 = torch.ops.aten.add.Tensor(arg905_1, 0.001);  arg905_1 = None
        sqrt_355 = torch.ops.aten.sqrt.default(add_833);  add_833 = None
        reciprocal_355 = torch.ops.aten.reciprocal.default(sqrt_355);  sqrt_355 = None
        mul_1065 = torch.ops.aten.mul.Tensor(reciprocal_355, 1);  reciprocal_355 = None
        unsqueeze_2840 = torch.ops.aten.unsqueeze.default(arg904_1, -1);  arg904_1 = None
        unsqueeze_2841 = torch.ops.aten.unsqueeze.default(unsqueeze_2840, -1);  unsqueeze_2840 = None
        unsqueeze_2842 = torch.ops.aten.unsqueeze.default(mul_1065, -1);  mul_1065 = None
        unsqueeze_2843 = torch.ops.aten.unsqueeze.default(unsqueeze_2842, -1);  unsqueeze_2842 = None
        sub_355 = torch.ops.aten.sub.Tensor(convolution_659, unsqueeze_2841);  convolution_659 = unsqueeze_2841 = None
        mul_1066 = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_2843);  sub_355 = unsqueeze_2843 = None
        unsqueeze_2844 = torch.ops.aten.unsqueeze.default(arg906_1, -1);  arg906_1 = None
        unsqueeze_2845 = torch.ops.aten.unsqueeze.default(unsqueeze_2844, -1);  unsqueeze_2844 = None
        mul_1067 = torch.ops.aten.mul.Tensor(mul_1066, unsqueeze_2845);  mul_1066 = unsqueeze_2845 = None
        unsqueeze_2846 = torch.ops.aten.unsqueeze.default(arg907_1, -1);  arg907_1 = None
        unsqueeze_2847 = torch.ops.aten.unsqueeze.default(unsqueeze_2846, -1);  unsqueeze_2846 = None
        add_834 = torch.ops.aten.add.Tensor(mul_1067, unsqueeze_2847);  mul_1067 = unsqueeze_2847 = None
        relu_353 = torch.ops.aten.relu.default(add_834);  add_834 = None
        convolution_660 = torch.ops.aten.convolution.default(relu_353, arg908_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_353 = arg908_1 = None
        convolution_661 = torch.ops.aten.convolution.default(convolution_660, arg909_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_660 = arg909_1 = None
        add_835 = torch.ops.aten.add.Tensor(arg911_1, 0.001);  arg911_1 = None
        sqrt_356 = torch.ops.aten.sqrt.default(add_835);  add_835 = None
        reciprocal_356 = torch.ops.aten.reciprocal.default(sqrt_356);  sqrt_356 = None
        mul_1068 = torch.ops.aten.mul.Tensor(reciprocal_356, 1);  reciprocal_356 = None
        unsqueeze_2848 = torch.ops.aten.unsqueeze.default(arg910_1, -1);  arg910_1 = None
        unsqueeze_2849 = torch.ops.aten.unsqueeze.default(unsqueeze_2848, -1);  unsqueeze_2848 = None
        unsqueeze_2850 = torch.ops.aten.unsqueeze.default(mul_1068, -1);  mul_1068 = None
        unsqueeze_2851 = torch.ops.aten.unsqueeze.default(unsqueeze_2850, -1);  unsqueeze_2850 = None
        sub_356 = torch.ops.aten.sub.Tensor(convolution_661, unsqueeze_2849);  convolution_661 = unsqueeze_2849 = None
        mul_1069 = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_2851);  sub_356 = unsqueeze_2851 = None
        unsqueeze_2852 = torch.ops.aten.unsqueeze.default(arg912_1, -1);  arg912_1 = None
        unsqueeze_2853 = torch.ops.aten.unsqueeze.default(unsqueeze_2852, -1);  unsqueeze_2852 = None
        mul_1070 = torch.ops.aten.mul.Tensor(mul_1069, unsqueeze_2853);  mul_1069 = unsqueeze_2853 = None
        unsqueeze_2854 = torch.ops.aten.unsqueeze.default(arg913_1, -1);  arg913_1 = None
        unsqueeze_2855 = torch.ops.aten.unsqueeze.default(unsqueeze_2854, -1);  unsqueeze_2854 = None
        add_836 = torch.ops.aten.add.Tensor(mul_1070, unsqueeze_2855);  mul_1070 = unsqueeze_2855 = None
        constant_pad_nd_76 = torch.ops.aten.constant_pad_nd.default(add_813, [1, 1, 1, 1], -inf)
        _low_memory_max_pool2d_with_offsets_74 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_76, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_76 = None
        getitem_148 = _low_memory_max_pool2d_with_offsets_74[0];  _low_memory_max_pool2d_with_offsets_74 = None
        add_837 = torch.ops.aten.add.Tensor(add_836, getitem_148);  add_836 = getitem_148 = None
        relu_354 = torch.ops.aten.relu.default(add_811);  add_811 = None
        constant_pad_nd_77 = torch.ops.aten.constant_pad_nd.default(relu_354, [1, 1, 1, 1], 0.0);  relu_354 = None
        convolution_662 = torch.ops.aten.convolution.default(constant_pad_nd_77, arg914_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864);  constant_pad_nd_77 = arg914_1 = None
        convolution_663 = torch.ops.aten.convolution.default(convolution_662, arg915_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_662 = arg915_1 = None
        add_838 = torch.ops.aten.add.Tensor(arg917_1, 0.001);  arg917_1 = None
        sqrt_357 = torch.ops.aten.sqrt.default(add_838);  add_838 = None
        reciprocal_357 = torch.ops.aten.reciprocal.default(sqrt_357);  sqrt_357 = None
        mul_1071 = torch.ops.aten.mul.Tensor(reciprocal_357, 1);  reciprocal_357 = None
        unsqueeze_2856 = torch.ops.aten.unsqueeze.default(arg916_1, -1);  arg916_1 = None
        unsqueeze_2857 = torch.ops.aten.unsqueeze.default(unsqueeze_2856, -1);  unsqueeze_2856 = None
        unsqueeze_2858 = torch.ops.aten.unsqueeze.default(mul_1071, -1);  mul_1071 = None
        unsqueeze_2859 = torch.ops.aten.unsqueeze.default(unsqueeze_2858, -1);  unsqueeze_2858 = None
        sub_357 = torch.ops.aten.sub.Tensor(convolution_663, unsqueeze_2857);  convolution_663 = unsqueeze_2857 = None
        mul_1072 = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_2859);  sub_357 = unsqueeze_2859 = None
        unsqueeze_2860 = torch.ops.aten.unsqueeze.default(arg918_1, -1);  arg918_1 = None
        unsqueeze_2861 = torch.ops.aten.unsqueeze.default(unsqueeze_2860, -1);  unsqueeze_2860 = None
        mul_1073 = torch.ops.aten.mul.Tensor(mul_1072, unsqueeze_2861);  mul_1072 = unsqueeze_2861 = None
        unsqueeze_2862 = torch.ops.aten.unsqueeze.default(arg919_1, -1);  arg919_1 = None
        unsqueeze_2863 = torch.ops.aten.unsqueeze.default(unsqueeze_2862, -1);  unsqueeze_2862 = None
        add_839 = torch.ops.aten.add.Tensor(mul_1073, unsqueeze_2863);  mul_1073 = unsqueeze_2863 = None
        relu_355 = torch.ops.aten.relu.default(add_839);  add_839 = None
        convolution_664 = torch.ops.aten.convolution.default(relu_355, arg920_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_355 = arg920_1 = None
        convolution_665 = torch.ops.aten.convolution.default(convolution_664, arg921_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_664 = arg921_1 = None
        add_840 = torch.ops.aten.add.Tensor(arg923_1, 0.001);  arg923_1 = None
        sqrt_358 = torch.ops.aten.sqrt.default(add_840);  add_840 = None
        reciprocal_358 = torch.ops.aten.reciprocal.default(sqrt_358);  sqrt_358 = None
        mul_1074 = torch.ops.aten.mul.Tensor(reciprocal_358, 1);  reciprocal_358 = None
        unsqueeze_2864 = torch.ops.aten.unsqueeze.default(arg922_1, -1);  arg922_1 = None
        unsqueeze_2865 = torch.ops.aten.unsqueeze.default(unsqueeze_2864, -1);  unsqueeze_2864 = None
        unsqueeze_2866 = torch.ops.aten.unsqueeze.default(mul_1074, -1);  mul_1074 = None
        unsqueeze_2867 = torch.ops.aten.unsqueeze.default(unsqueeze_2866, -1);  unsqueeze_2866 = None
        sub_358 = torch.ops.aten.sub.Tensor(convolution_665, unsqueeze_2865);  convolution_665 = unsqueeze_2865 = None
        mul_1075 = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_2867);  sub_358 = unsqueeze_2867 = None
        unsqueeze_2868 = torch.ops.aten.unsqueeze.default(arg924_1, -1);  arg924_1 = None
        unsqueeze_2869 = torch.ops.aten.unsqueeze.default(unsqueeze_2868, -1);  unsqueeze_2868 = None
        mul_1076 = torch.ops.aten.mul.Tensor(mul_1075, unsqueeze_2869);  mul_1075 = unsqueeze_2869 = None
        unsqueeze_2870 = torch.ops.aten.unsqueeze.default(arg925_1, -1);  arg925_1 = None
        unsqueeze_2871 = torch.ops.aten.unsqueeze.default(unsqueeze_2870, -1);  unsqueeze_2870 = None
        add_841 = torch.ops.aten.add.Tensor(mul_1076, unsqueeze_2871);  mul_1076 = unsqueeze_2871 = None
        relu_356 = torch.ops.aten.relu.default(add_813);  add_813 = None
        convolution_666 = torch.ops.aten.convolution.default(relu_356, arg926_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_356 = arg926_1 = None
        add_842 = torch.ops.aten.add.Tensor(arg928_1, 0.001);  arg928_1 = None
        sqrt_359 = torch.ops.aten.sqrt.default(add_842);  add_842 = None
        reciprocal_359 = torch.ops.aten.reciprocal.default(sqrt_359);  sqrt_359 = None
        mul_1077 = torch.ops.aten.mul.Tensor(reciprocal_359, 1);  reciprocal_359 = None
        unsqueeze_2872 = torch.ops.aten.unsqueeze.default(arg927_1, -1);  arg927_1 = None
        unsqueeze_2873 = torch.ops.aten.unsqueeze.default(unsqueeze_2872, -1);  unsqueeze_2872 = None
        unsqueeze_2874 = torch.ops.aten.unsqueeze.default(mul_1077, -1);  mul_1077 = None
        unsqueeze_2875 = torch.ops.aten.unsqueeze.default(unsqueeze_2874, -1);  unsqueeze_2874 = None
        sub_359 = torch.ops.aten.sub.Tensor(convolution_666, unsqueeze_2873);  convolution_666 = unsqueeze_2873 = None
        mul_1078 = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_2875);  sub_359 = unsqueeze_2875 = None
        unsqueeze_2876 = torch.ops.aten.unsqueeze.default(arg929_1, -1);  arg929_1 = None
        unsqueeze_2877 = torch.ops.aten.unsqueeze.default(unsqueeze_2876, -1);  unsqueeze_2876 = None
        mul_1079 = torch.ops.aten.mul.Tensor(mul_1078, unsqueeze_2877);  mul_1078 = unsqueeze_2877 = None
        unsqueeze_2878 = torch.ops.aten.unsqueeze.default(arg930_1, -1);  arg930_1 = None
        unsqueeze_2879 = torch.ops.aten.unsqueeze.default(unsqueeze_2878, -1);  unsqueeze_2878 = None
        add_843 = torch.ops.aten.add.Tensor(mul_1079, unsqueeze_2879);  mul_1079 = unsqueeze_2879 = None
        add_844 = torch.ops.aten.add.Tensor(add_841, add_843);  add_841 = add_843 = None
        cat_31 = torch.ops.aten.cat.default([add_818, add_823, add_832, add_837, add_844], 1);  add_818 = add_823 = add_832 = add_837 = add_844 = None
        relu_357 = torch.ops.aten.relu.default(cat_30);  cat_30 = None
        avg_pool2d_14 = torch.ops.aten.avg_pool2d.default(relu_357, [1, 1], [2, 2], [0, 0], False, False)
        convolution_667 = torch.ops.aten.convolution.default(avg_pool2d_14, arg931_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_14 = arg931_1 = None
        constant_pad_nd_79 = torch.ops.aten.constant_pad_nd.default(relu_357, [-1, 1, -1, 1], 0.0);  relu_357 = None
        avg_pool2d_15 = torch.ops.aten.avg_pool2d.default(constant_pad_nd_79, [1, 1], [2, 2], [0, 0], False, False);  constant_pad_nd_79 = None
        convolution_668 = torch.ops.aten.convolution.default(avg_pool2d_15, arg932_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_15 = arg932_1 = None
        cat_32 = torch.ops.aten.cat.default([convolution_667, convolution_668], 1);  convolution_667 = convolution_668 = None
        add_845 = torch.ops.aten.add.Tensor(arg934_1, 0.001);  arg934_1 = None
        sqrt_360 = torch.ops.aten.sqrt.default(add_845);  add_845 = None
        reciprocal_360 = torch.ops.aten.reciprocal.default(sqrt_360);  sqrt_360 = None
        mul_1080 = torch.ops.aten.mul.Tensor(reciprocal_360, 1);  reciprocal_360 = None
        unsqueeze_2880 = torch.ops.aten.unsqueeze.default(arg933_1, -1);  arg933_1 = None
        unsqueeze_2881 = torch.ops.aten.unsqueeze.default(unsqueeze_2880, -1);  unsqueeze_2880 = None
        unsqueeze_2882 = torch.ops.aten.unsqueeze.default(mul_1080, -1);  mul_1080 = None
        unsqueeze_2883 = torch.ops.aten.unsqueeze.default(unsqueeze_2882, -1);  unsqueeze_2882 = None
        sub_360 = torch.ops.aten.sub.Tensor(cat_32, unsqueeze_2881);  cat_32 = unsqueeze_2881 = None
        mul_1081 = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_2883);  sub_360 = unsqueeze_2883 = None
        unsqueeze_2884 = torch.ops.aten.unsqueeze.default(arg935_1, -1);  arg935_1 = None
        unsqueeze_2885 = torch.ops.aten.unsqueeze.default(unsqueeze_2884, -1);  unsqueeze_2884 = None
        mul_1082 = torch.ops.aten.mul.Tensor(mul_1081, unsqueeze_2885);  mul_1081 = unsqueeze_2885 = None
        unsqueeze_2886 = torch.ops.aten.unsqueeze.default(arg936_1, -1);  arg936_1 = None
        unsqueeze_2887 = torch.ops.aten.unsqueeze.default(unsqueeze_2886, -1);  unsqueeze_2886 = None
        add_846 = torch.ops.aten.add.Tensor(mul_1082, unsqueeze_2887);  mul_1082 = unsqueeze_2887 = None
        relu_358 = torch.ops.aten.relu.default(cat_31)
        convolution_669 = torch.ops.aten.convolution.default(relu_358, arg937_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_358 = arg937_1 = None
        add_847 = torch.ops.aten.add.Tensor(arg939_1, 0.001);  arg939_1 = None
        sqrt_361 = torch.ops.aten.sqrt.default(add_847);  add_847 = None
        reciprocal_361 = torch.ops.aten.reciprocal.default(sqrt_361);  sqrt_361 = None
        mul_1083 = torch.ops.aten.mul.Tensor(reciprocal_361, 1);  reciprocal_361 = None
        unsqueeze_2888 = torch.ops.aten.unsqueeze.default(arg938_1, -1);  arg938_1 = None
        unsqueeze_2889 = torch.ops.aten.unsqueeze.default(unsqueeze_2888, -1);  unsqueeze_2888 = None
        unsqueeze_2890 = torch.ops.aten.unsqueeze.default(mul_1083, -1);  mul_1083 = None
        unsqueeze_2891 = torch.ops.aten.unsqueeze.default(unsqueeze_2890, -1);  unsqueeze_2890 = None
        sub_361 = torch.ops.aten.sub.Tensor(convolution_669, unsqueeze_2889);  convolution_669 = unsqueeze_2889 = None
        mul_1084 = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_2891);  sub_361 = unsqueeze_2891 = None
        unsqueeze_2892 = torch.ops.aten.unsqueeze.default(arg940_1, -1);  arg940_1 = None
        unsqueeze_2893 = torch.ops.aten.unsqueeze.default(unsqueeze_2892, -1);  unsqueeze_2892 = None
        mul_1085 = torch.ops.aten.mul.Tensor(mul_1084, unsqueeze_2893);  mul_1084 = unsqueeze_2893 = None
        unsqueeze_2894 = torch.ops.aten.unsqueeze.default(arg941_1, -1);  arg941_1 = None
        unsqueeze_2895 = torch.ops.aten.unsqueeze.default(unsqueeze_2894, -1);  unsqueeze_2894 = None
        add_848 = torch.ops.aten.add.Tensor(mul_1085, unsqueeze_2895);  mul_1085 = unsqueeze_2895 = None
        relu_359 = torch.ops.aten.relu.default(add_846)
        convolution_670 = torch.ops.aten.convolution.default(relu_359, arg942_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_359 = arg942_1 = None
        convolution_671 = torch.ops.aten.convolution.default(convolution_670, arg943_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_670 = arg943_1 = None
        add_849 = torch.ops.aten.add.Tensor(arg945_1, 0.001);  arg945_1 = None
        sqrt_362 = torch.ops.aten.sqrt.default(add_849);  add_849 = None
        reciprocal_362 = torch.ops.aten.reciprocal.default(sqrt_362);  sqrt_362 = None
        mul_1086 = torch.ops.aten.mul.Tensor(reciprocal_362, 1);  reciprocal_362 = None
        unsqueeze_2896 = torch.ops.aten.unsqueeze.default(arg944_1, -1);  arg944_1 = None
        unsqueeze_2897 = torch.ops.aten.unsqueeze.default(unsqueeze_2896, -1);  unsqueeze_2896 = None
        unsqueeze_2898 = torch.ops.aten.unsqueeze.default(mul_1086, -1);  mul_1086 = None
        unsqueeze_2899 = torch.ops.aten.unsqueeze.default(unsqueeze_2898, -1);  unsqueeze_2898 = None
        sub_362 = torch.ops.aten.sub.Tensor(convolution_671, unsqueeze_2897);  convolution_671 = unsqueeze_2897 = None
        mul_1087 = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_2899);  sub_362 = unsqueeze_2899 = None
        unsqueeze_2900 = torch.ops.aten.unsqueeze.default(arg946_1, -1);  arg946_1 = None
        unsqueeze_2901 = torch.ops.aten.unsqueeze.default(unsqueeze_2900, -1);  unsqueeze_2900 = None
        mul_1088 = torch.ops.aten.mul.Tensor(mul_1087, unsqueeze_2901);  mul_1087 = unsqueeze_2901 = None
        unsqueeze_2902 = torch.ops.aten.unsqueeze.default(arg947_1, -1);  arg947_1 = None
        unsqueeze_2903 = torch.ops.aten.unsqueeze.default(unsqueeze_2902, -1);  unsqueeze_2902 = None
        add_850 = torch.ops.aten.add.Tensor(mul_1088, unsqueeze_2903);  mul_1088 = unsqueeze_2903 = None
        relu_360 = torch.ops.aten.relu.default(add_850);  add_850 = None
        convolution_672 = torch.ops.aten.convolution.default(relu_360, arg948_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_360 = arg948_1 = None
        convolution_673 = torch.ops.aten.convolution.default(convolution_672, arg949_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_672 = arg949_1 = None
        add_851 = torch.ops.aten.add.Tensor(arg951_1, 0.001);  arg951_1 = None
        sqrt_363 = torch.ops.aten.sqrt.default(add_851);  add_851 = None
        reciprocal_363 = torch.ops.aten.reciprocal.default(sqrt_363);  sqrt_363 = None
        mul_1089 = torch.ops.aten.mul.Tensor(reciprocal_363, 1);  reciprocal_363 = None
        unsqueeze_2904 = torch.ops.aten.unsqueeze.default(arg950_1, -1);  arg950_1 = None
        unsqueeze_2905 = torch.ops.aten.unsqueeze.default(unsqueeze_2904, -1);  unsqueeze_2904 = None
        unsqueeze_2906 = torch.ops.aten.unsqueeze.default(mul_1089, -1);  mul_1089 = None
        unsqueeze_2907 = torch.ops.aten.unsqueeze.default(unsqueeze_2906, -1);  unsqueeze_2906 = None
        sub_363 = torch.ops.aten.sub.Tensor(convolution_673, unsqueeze_2905);  convolution_673 = unsqueeze_2905 = None
        mul_1090 = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_2907);  sub_363 = unsqueeze_2907 = None
        unsqueeze_2908 = torch.ops.aten.unsqueeze.default(arg952_1, -1);  arg952_1 = None
        unsqueeze_2909 = torch.ops.aten.unsqueeze.default(unsqueeze_2908, -1);  unsqueeze_2908 = None
        mul_1091 = torch.ops.aten.mul.Tensor(mul_1090, unsqueeze_2909);  mul_1090 = unsqueeze_2909 = None
        unsqueeze_2910 = torch.ops.aten.unsqueeze.default(arg953_1, -1);  arg953_1 = None
        unsqueeze_2911 = torch.ops.aten.unsqueeze.default(unsqueeze_2910, -1);  unsqueeze_2910 = None
        add_852 = torch.ops.aten.add.Tensor(mul_1091, unsqueeze_2911);  mul_1091 = unsqueeze_2911 = None
        _low_memory_max_pool2d_with_offsets_75 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_846, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_150 = _low_memory_max_pool2d_with_offsets_75[0];  _low_memory_max_pool2d_with_offsets_75 = None
        add_853 = torch.ops.aten.add.Tensor(add_852, getitem_150);  add_852 = getitem_150 = None
        relu_361 = torch.ops.aten.relu.default(add_848)
        convolution_674 = torch.ops.aten.convolution.default(relu_361, arg954_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864);  relu_361 = arg954_1 = None
        convolution_675 = torch.ops.aten.convolution.default(convolution_674, arg955_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_674 = arg955_1 = None
        add_854 = torch.ops.aten.add.Tensor(arg957_1, 0.001);  arg957_1 = None
        sqrt_364 = torch.ops.aten.sqrt.default(add_854);  add_854 = None
        reciprocal_364 = torch.ops.aten.reciprocal.default(sqrt_364);  sqrt_364 = None
        mul_1092 = torch.ops.aten.mul.Tensor(reciprocal_364, 1);  reciprocal_364 = None
        unsqueeze_2912 = torch.ops.aten.unsqueeze.default(arg956_1, -1);  arg956_1 = None
        unsqueeze_2913 = torch.ops.aten.unsqueeze.default(unsqueeze_2912, -1);  unsqueeze_2912 = None
        unsqueeze_2914 = torch.ops.aten.unsqueeze.default(mul_1092, -1);  mul_1092 = None
        unsqueeze_2915 = torch.ops.aten.unsqueeze.default(unsqueeze_2914, -1);  unsqueeze_2914 = None
        sub_364 = torch.ops.aten.sub.Tensor(convolution_675, unsqueeze_2913);  convolution_675 = unsqueeze_2913 = None
        mul_1093 = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_2915);  sub_364 = unsqueeze_2915 = None
        unsqueeze_2916 = torch.ops.aten.unsqueeze.default(arg958_1, -1);  arg958_1 = None
        unsqueeze_2917 = torch.ops.aten.unsqueeze.default(unsqueeze_2916, -1);  unsqueeze_2916 = None
        mul_1094 = torch.ops.aten.mul.Tensor(mul_1093, unsqueeze_2917);  mul_1093 = unsqueeze_2917 = None
        unsqueeze_2918 = torch.ops.aten.unsqueeze.default(arg959_1, -1);  arg959_1 = None
        unsqueeze_2919 = torch.ops.aten.unsqueeze.default(unsqueeze_2918, -1);  unsqueeze_2918 = None
        add_855 = torch.ops.aten.add.Tensor(mul_1094, unsqueeze_2919);  mul_1094 = unsqueeze_2919 = None
        relu_362 = torch.ops.aten.relu.default(add_855);  add_855 = None
        convolution_676 = torch.ops.aten.convolution.default(relu_362, arg960_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864);  relu_362 = arg960_1 = None
        convolution_677 = torch.ops.aten.convolution.default(convolution_676, arg961_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_676 = arg961_1 = None
        add_856 = torch.ops.aten.add.Tensor(arg963_1, 0.001);  arg963_1 = None
        sqrt_365 = torch.ops.aten.sqrt.default(add_856);  add_856 = None
        reciprocal_365 = torch.ops.aten.reciprocal.default(sqrt_365);  sqrt_365 = None
        mul_1095 = torch.ops.aten.mul.Tensor(reciprocal_365, 1);  reciprocal_365 = None
        unsqueeze_2920 = torch.ops.aten.unsqueeze.default(arg962_1, -1);  arg962_1 = None
        unsqueeze_2921 = torch.ops.aten.unsqueeze.default(unsqueeze_2920, -1);  unsqueeze_2920 = None
        unsqueeze_2922 = torch.ops.aten.unsqueeze.default(mul_1095, -1);  mul_1095 = None
        unsqueeze_2923 = torch.ops.aten.unsqueeze.default(unsqueeze_2922, -1);  unsqueeze_2922 = None
        sub_365 = torch.ops.aten.sub.Tensor(convolution_677, unsqueeze_2921);  convolution_677 = unsqueeze_2921 = None
        mul_1096 = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_2923);  sub_365 = unsqueeze_2923 = None
        unsqueeze_2924 = torch.ops.aten.unsqueeze.default(arg964_1, -1);  arg964_1 = None
        unsqueeze_2925 = torch.ops.aten.unsqueeze.default(unsqueeze_2924, -1);  unsqueeze_2924 = None
        mul_1097 = torch.ops.aten.mul.Tensor(mul_1096, unsqueeze_2925);  mul_1096 = unsqueeze_2925 = None
        unsqueeze_2926 = torch.ops.aten.unsqueeze.default(arg965_1, -1);  arg965_1 = None
        unsqueeze_2927 = torch.ops.aten.unsqueeze.default(unsqueeze_2926, -1);  unsqueeze_2926 = None
        add_857 = torch.ops.aten.add.Tensor(mul_1097, unsqueeze_2927);  mul_1097 = unsqueeze_2927 = None
        _low_memory_max_pool2d_with_offsets_76 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_848, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_152 = _low_memory_max_pool2d_with_offsets_76[0];  _low_memory_max_pool2d_with_offsets_76 = None
        add_858 = torch.ops.aten.add.Tensor(add_857, getitem_152);  add_857 = getitem_152 = None
        relu_363 = torch.ops.aten.relu.default(add_848)
        convolution_678 = torch.ops.aten.convolution.default(relu_363, arg966_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_363 = arg966_1 = None
        convolution_679 = torch.ops.aten.convolution.default(convolution_678, arg967_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_678 = arg967_1 = None
        add_859 = torch.ops.aten.add.Tensor(arg969_1, 0.001);  arg969_1 = None
        sqrt_366 = torch.ops.aten.sqrt.default(add_859);  add_859 = None
        reciprocal_366 = torch.ops.aten.reciprocal.default(sqrt_366);  sqrt_366 = None
        mul_1098 = torch.ops.aten.mul.Tensor(reciprocal_366, 1);  reciprocal_366 = None
        unsqueeze_2928 = torch.ops.aten.unsqueeze.default(arg968_1, -1);  arg968_1 = None
        unsqueeze_2929 = torch.ops.aten.unsqueeze.default(unsqueeze_2928, -1);  unsqueeze_2928 = None
        unsqueeze_2930 = torch.ops.aten.unsqueeze.default(mul_1098, -1);  mul_1098 = None
        unsqueeze_2931 = torch.ops.aten.unsqueeze.default(unsqueeze_2930, -1);  unsqueeze_2930 = None
        sub_366 = torch.ops.aten.sub.Tensor(convolution_679, unsqueeze_2929);  convolution_679 = unsqueeze_2929 = None
        mul_1099 = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_2931);  sub_366 = unsqueeze_2931 = None
        unsqueeze_2932 = torch.ops.aten.unsqueeze.default(arg970_1, -1);  arg970_1 = None
        unsqueeze_2933 = torch.ops.aten.unsqueeze.default(unsqueeze_2932, -1);  unsqueeze_2932 = None
        mul_1100 = torch.ops.aten.mul.Tensor(mul_1099, unsqueeze_2933);  mul_1099 = unsqueeze_2933 = None
        unsqueeze_2934 = torch.ops.aten.unsqueeze.default(arg971_1, -1);  arg971_1 = None
        unsqueeze_2935 = torch.ops.aten.unsqueeze.default(unsqueeze_2934, -1);  unsqueeze_2934 = None
        add_860 = torch.ops.aten.add.Tensor(mul_1100, unsqueeze_2935);  mul_1100 = unsqueeze_2935 = None
        relu_364 = torch.ops.aten.relu.default(add_860);  add_860 = None
        convolution_680 = torch.ops.aten.convolution.default(relu_364, arg972_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_364 = arg972_1 = None
        convolution_681 = torch.ops.aten.convolution.default(convolution_680, arg973_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_680 = arg973_1 = None
        add_861 = torch.ops.aten.add.Tensor(arg975_1, 0.001);  arg975_1 = None
        sqrt_367 = torch.ops.aten.sqrt.default(add_861);  add_861 = None
        reciprocal_367 = torch.ops.aten.reciprocal.default(sqrt_367);  sqrt_367 = None
        mul_1101 = torch.ops.aten.mul.Tensor(reciprocal_367, 1);  reciprocal_367 = None
        unsqueeze_2936 = torch.ops.aten.unsqueeze.default(arg974_1, -1);  arg974_1 = None
        unsqueeze_2937 = torch.ops.aten.unsqueeze.default(unsqueeze_2936, -1);  unsqueeze_2936 = None
        unsqueeze_2938 = torch.ops.aten.unsqueeze.default(mul_1101, -1);  mul_1101 = None
        unsqueeze_2939 = torch.ops.aten.unsqueeze.default(unsqueeze_2938, -1);  unsqueeze_2938 = None
        sub_367 = torch.ops.aten.sub.Tensor(convolution_681, unsqueeze_2937);  convolution_681 = unsqueeze_2937 = None
        mul_1102 = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_2939);  sub_367 = unsqueeze_2939 = None
        unsqueeze_2940 = torch.ops.aten.unsqueeze.default(arg976_1, -1);  arg976_1 = None
        unsqueeze_2941 = torch.ops.aten.unsqueeze.default(unsqueeze_2940, -1);  unsqueeze_2940 = None
        mul_1103 = torch.ops.aten.mul.Tensor(mul_1102, unsqueeze_2941);  mul_1102 = unsqueeze_2941 = None
        unsqueeze_2942 = torch.ops.aten.unsqueeze.default(arg977_1, -1);  arg977_1 = None
        unsqueeze_2943 = torch.ops.aten.unsqueeze.default(unsqueeze_2942, -1);  unsqueeze_2942 = None
        add_862 = torch.ops.aten.add.Tensor(mul_1103, unsqueeze_2943);  mul_1103 = unsqueeze_2943 = None
        relu_365 = torch.ops.aten.relu.default(add_848)
        convolution_682 = torch.ops.aten.convolution.default(relu_365, arg978_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_365 = arg978_1 = None
        convolution_683 = torch.ops.aten.convolution.default(convolution_682, arg979_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_682 = arg979_1 = None
        add_863 = torch.ops.aten.add.Tensor(arg981_1, 0.001);  arg981_1 = None
        sqrt_368 = torch.ops.aten.sqrt.default(add_863);  add_863 = None
        reciprocal_368 = torch.ops.aten.reciprocal.default(sqrt_368);  sqrt_368 = None
        mul_1104 = torch.ops.aten.mul.Tensor(reciprocal_368, 1);  reciprocal_368 = None
        unsqueeze_2944 = torch.ops.aten.unsqueeze.default(arg980_1, -1);  arg980_1 = None
        unsqueeze_2945 = torch.ops.aten.unsqueeze.default(unsqueeze_2944, -1);  unsqueeze_2944 = None
        unsqueeze_2946 = torch.ops.aten.unsqueeze.default(mul_1104, -1);  mul_1104 = None
        unsqueeze_2947 = torch.ops.aten.unsqueeze.default(unsqueeze_2946, -1);  unsqueeze_2946 = None
        sub_368 = torch.ops.aten.sub.Tensor(convolution_683, unsqueeze_2945);  convolution_683 = unsqueeze_2945 = None
        mul_1105 = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_2947);  sub_368 = unsqueeze_2947 = None
        unsqueeze_2948 = torch.ops.aten.unsqueeze.default(arg982_1, -1);  arg982_1 = None
        unsqueeze_2949 = torch.ops.aten.unsqueeze.default(unsqueeze_2948, -1);  unsqueeze_2948 = None
        mul_1106 = torch.ops.aten.mul.Tensor(mul_1105, unsqueeze_2949);  mul_1105 = unsqueeze_2949 = None
        unsqueeze_2950 = torch.ops.aten.unsqueeze.default(arg983_1, -1);  arg983_1 = None
        unsqueeze_2951 = torch.ops.aten.unsqueeze.default(unsqueeze_2950, -1);  unsqueeze_2950 = None
        add_864 = torch.ops.aten.add.Tensor(mul_1106, unsqueeze_2951);  mul_1106 = unsqueeze_2951 = None
        relu_366 = torch.ops.aten.relu.default(add_864);  add_864 = None
        convolution_684 = torch.ops.aten.convolution.default(relu_366, arg984_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_366 = arg984_1 = None
        convolution_685 = torch.ops.aten.convolution.default(convolution_684, arg985_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_684 = arg985_1 = None
        add_865 = torch.ops.aten.add.Tensor(arg987_1, 0.001);  arg987_1 = None
        sqrt_369 = torch.ops.aten.sqrt.default(add_865);  add_865 = None
        reciprocal_369 = torch.ops.aten.reciprocal.default(sqrt_369);  sqrt_369 = None
        mul_1107 = torch.ops.aten.mul.Tensor(reciprocal_369, 1);  reciprocal_369 = None
        unsqueeze_2952 = torch.ops.aten.unsqueeze.default(arg986_1, -1);  arg986_1 = None
        unsqueeze_2953 = torch.ops.aten.unsqueeze.default(unsqueeze_2952, -1);  unsqueeze_2952 = None
        unsqueeze_2954 = torch.ops.aten.unsqueeze.default(mul_1107, -1);  mul_1107 = None
        unsqueeze_2955 = torch.ops.aten.unsqueeze.default(unsqueeze_2954, -1);  unsqueeze_2954 = None
        sub_369 = torch.ops.aten.sub.Tensor(convolution_685, unsqueeze_2953);  convolution_685 = unsqueeze_2953 = None
        mul_1108 = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_2955);  sub_369 = unsqueeze_2955 = None
        unsqueeze_2956 = torch.ops.aten.unsqueeze.default(arg988_1, -1);  arg988_1 = None
        unsqueeze_2957 = torch.ops.aten.unsqueeze.default(unsqueeze_2956, -1);  unsqueeze_2956 = None
        mul_1109 = torch.ops.aten.mul.Tensor(mul_1108, unsqueeze_2957);  mul_1108 = unsqueeze_2957 = None
        unsqueeze_2958 = torch.ops.aten.unsqueeze.default(arg989_1, -1);  arg989_1 = None
        unsqueeze_2959 = torch.ops.aten.unsqueeze.default(unsqueeze_2958, -1);  unsqueeze_2958 = None
        add_866 = torch.ops.aten.add.Tensor(mul_1109, unsqueeze_2959);  mul_1109 = unsqueeze_2959 = None
        add_867 = torch.ops.aten.add.Tensor(add_862, add_866);  add_862 = add_866 = None
        relu_367 = torch.ops.aten.relu.default(add_867)
        convolution_686 = torch.ops.aten.convolution.default(relu_367, arg990_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_367 = arg990_1 = None
        convolution_687 = torch.ops.aten.convolution.default(convolution_686, arg991_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_686 = arg991_1 = None
        add_868 = torch.ops.aten.add.Tensor(arg993_1, 0.001);  arg993_1 = None
        sqrt_370 = torch.ops.aten.sqrt.default(add_868);  add_868 = None
        reciprocal_370 = torch.ops.aten.reciprocal.default(sqrt_370);  sqrt_370 = None
        mul_1110 = torch.ops.aten.mul.Tensor(reciprocal_370, 1);  reciprocal_370 = None
        unsqueeze_2960 = torch.ops.aten.unsqueeze.default(arg992_1, -1);  arg992_1 = None
        unsqueeze_2961 = torch.ops.aten.unsqueeze.default(unsqueeze_2960, -1);  unsqueeze_2960 = None
        unsqueeze_2962 = torch.ops.aten.unsqueeze.default(mul_1110, -1);  mul_1110 = None
        unsqueeze_2963 = torch.ops.aten.unsqueeze.default(unsqueeze_2962, -1);  unsqueeze_2962 = None
        sub_370 = torch.ops.aten.sub.Tensor(convolution_687, unsqueeze_2961);  convolution_687 = unsqueeze_2961 = None
        mul_1111 = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_2963);  sub_370 = unsqueeze_2963 = None
        unsqueeze_2964 = torch.ops.aten.unsqueeze.default(arg994_1, -1);  arg994_1 = None
        unsqueeze_2965 = torch.ops.aten.unsqueeze.default(unsqueeze_2964, -1);  unsqueeze_2964 = None
        mul_1112 = torch.ops.aten.mul.Tensor(mul_1111, unsqueeze_2965);  mul_1111 = unsqueeze_2965 = None
        unsqueeze_2966 = torch.ops.aten.unsqueeze.default(arg995_1, -1);  arg995_1 = None
        unsqueeze_2967 = torch.ops.aten.unsqueeze.default(unsqueeze_2966, -1);  unsqueeze_2966 = None
        add_869 = torch.ops.aten.add.Tensor(mul_1112, unsqueeze_2967);  mul_1112 = unsqueeze_2967 = None
        relu_368 = torch.ops.aten.relu.default(add_869);  add_869 = None
        convolution_688 = torch.ops.aten.convolution.default(relu_368, arg996_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_368 = arg996_1 = None
        convolution_689 = torch.ops.aten.convolution.default(convolution_688, arg997_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_688 = arg997_1 = None
        add_870 = torch.ops.aten.add.Tensor(arg999_1, 0.001);  arg999_1 = None
        sqrt_371 = torch.ops.aten.sqrt.default(add_870);  add_870 = None
        reciprocal_371 = torch.ops.aten.reciprocal.default(sqrt_371);  sqrt_371 = None
        mul_1113 = torch.ops.aten.mul.Tensor(reciprocal_371, 1);  reciprocal_371 = None
        unsqueeze_2968 = torch.ops.aten.unsqueeze.default(arg998_1, -1);  arg998_1 = None
        unsqueeze_2969 = torch.ops.aten.unsqueeze.default(unsqueeze_2968, -1);  unsqueeze_2968 = None
        unsqueeze_2970 = torch.ops.aten.unsqueeze.default(mul_1113, -1);  mul_1113 = None
        unsqueeze_2971 = torch.ops.aten.unsqueeze.default(unsqueeze_2970, -1);  unsqueeze_2970 = None
        sub_371 = torch.ops.aten.sub.Tensor(convolution_689, unsqueeze_2969);  convolution_689 = unsqueeze_2969 = None
        mul_1114 = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_2971);  sub_371 = unsqueeze_2971 = None
        unsqueeze_2972 = torch.ops.aten.unsqueeze.default(arg1000_1, -1);  arg1000_1 = None
        unsqueeze_2973 = torch.ops.aten.unsqueeze.default(unsqueeze_2972, -1);  unsqueeze_2972 = None
        mul_1115 = torch.ops.aten.mul.Tensor(mul_1114, unsqueeze_2973);  mul_1114 = unsqueeze_2973 = None
        unsqueeze_2974 = torch.ops.aten.unsqueeze.default(arg1001_1, -1);  arg1001_1 = None
        unsqueeze_2975 = torch.ops.aten.unsqueeze.default(unsqueeze_2974, -1);  unsqueeze_2974 = None
        add_871 = torch.ops.aten.add.Tensor(mul_1115, unsqueeze_2975);  mul_1115 = unsqueeze_2975 = None
        _low_memory_max_pool2d_with_offsets_77 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_848, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_154 = _low_memory_max_pool2d_with_offsets_77[0];  _low_memory_max_pool2d_with_offsets_77 = None
        add_872 = torch.ops.aten.add.Tensor(add_871, getitem_154);  add_871 = getitem_154 = None
        relu_369 = torch.ops.aten.relu.default(add_846);  add_846 = None
        convolution_690 = torch.ops.aten.convolution.default(relu_369, arg1002_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_369 = arg1002_1 = None
        convolution_691 = torch.ops.aten.convolution.default(convolution_690, arg1003_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_690 = arg1003_1 = None
        add_873 = torch.ops.aten.add.Tensor(arg1005_1, 0.001);  arg1005_1 = None
        sqrt_372 = torch.ops.aten.sqrt.default(add_873);  add_873 = None
        reciprocal_372 = torch.ops.aten.reciprocal.default(sqrt_372);  sqrt_372 = None
        mul_1116 = torch.ops.aten.mul.Tensor(reciprocal_372, 1);  reciprocal_372 = None
        unsqueeze_2976 = torch.ops.aten.unsqueeze.default(arg1004_1, -1);  arg1004_1 = None
        unsqueeze_2977 = torch.ops.aten.unsqueeze.default(unsqueeze_2976, -1);  unsqueeze_2976 = None
        unsqueeze_2978 = torch.ops.aten.unsqueeze.default(mul_1116, -1);  mul_1116 = None
        unsqueeze_2979 = torch.ops.aten.unsqueeze.default(unsqueeze_2978, -1);  unsqueeze_2978 = None
        sub_372 = torch.ops.aten.sub.Tensor(convolution_691, unsqueeze_2977);  convolution_691 = unsqueeze_2977 = None
        mul_1117 = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_2979);  sub_372 = unsqueeze_2979 = None
        unsqueeze_2980 = torch.ops.aten.unsqueeze.default(arg1006_1, -1);  arg1006_1 = None
        unsqueeze_2981 = torch.ops.aten.unsqueeze.default(unsqueeze_2980, -1);  unsqueeze_2980 = None
        mul_1118 = torch.ops.aten.mul.Tensor(mul_1117, unsqueeze_2981);  mul_1117 = unsqueeze_2981 = None
        unsqueeze_2982 = torch.ops.aten.unsqueeze.default(arg1007_1, -1);  arg1007_1 = None
        unsqueeze_2983 = torch.ops.aten.unsqueeze.default(unsqueeze_2982, -1);  unsqueeze_2982 = None
        add_874 = torch.ops.aten.add.Tensor(mul_1118, unsqueeze_2983);  mul_1118 = unsqueeze_2983 = None
        relu_370 = torch.ops.aten.relu.default(add_874);  add_874 = None
        convolution_692 = torch.ops.aten.convolution.default(relu_370, arg1008_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_370 = arg1008_1 = None
        convolution_693 = torch.ops.aten.convolution.default(convolution_692, arg1009_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_692 = arg1009_1 = None
        add_875 = torch.ops.aten.add.Tensor(arg1011_1, 0.001);  arg1011_1 = None
        sqrt_373 = torch.ops.aten.sqrt.default(add_875);  add_875 = None
        reciprocal_373 = torch.ops.aten.reciprocal.default(sqrt_373);  sqrt_373 = None
        mul_1119 = torch.ops.aten.mul.Tensor(reciprocal_373, 1);  reciprocal_373 = None
        unsqueeze_2984 = torch.ops.aten.unsqueeze.default(arg1010_1, -1);  arg1010_1 = None
        unsqueeze_2985 = torch.ops.aten.unsqueeze.default(unsqueeze_2984, -1);  unsqueeze_2984 = None
        unsqueeze_2986 = torch.ops.aten.unsqueeze.default(mul_1119, -1);  mul_1119 = None
        unsqueeze_2987 = torch.ops.aten.unsqueeze.default(unsqueeze_2986, -1);  unsqueeze_2986 = None
        sub_373 = torch.ops.aten.sub.Tensor(convolution_693, unsqueeze_2985);  convolution_693 = unsqueeze_2985 = None
        mul_1120 = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_2987);  sub_373 = unsqueeze_2987 = None
        unsqueeze_2988 = torch.ops.aten.unsqueeze.default(arg1012_1, -1);  arg1012_1 = None
        unsqueeze_2989 = torch.ops.aten.unsqueeze.default(unsqueeze_2988, -1);  unsqueeze_2988 = None
        mul_1121 = torch.ops.aten.mul.Tensor(mul_1120, unsqueeze_2989);  mul_1120 = unsqueeze_2989 = None
        unsqueeze_2990 = torch.ops.aten.unsqueeze.default(arg1013_1, -1);  arg1013_1 = None
        unsqueeze_2991 = torch.ops.aten.unsqueeze.default(unsqueeze_2990, -1);  unsqueeze_2990 = None
        add_876 = torch.ops.aten.add.Tensor(mul_1121, unsqueeze_2991);  mul_1121 = unsqueeze_2991 = None
        add_877 = torch.ops.aten.add.Tensor(add_876, add_848);  add_876 = add_848 = None
        cat_33 = torch.ops.aten.cat.default([add_853, add_858, add_867, add_872, add_877], 1);  add_853 = add_858 = add_867 = add_872 = add_877 = None
        relu_371 = torch.ops.aten.relu.default(cat_31);  cat_31 = None
        convolution_694 = torch.ops.aten.convolution.default(relu_371, arg1014_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_371 = arg1014_1 = None
        add_878 = torch.ops.aten.add.Tensor(arg1016_1, 0.001);  arg1016_1 = None
        sqrt_374 = torch.ops.aten.sqrt.default(add_878);  add_878 = None
        reciprocal_374 = torch.ops.aten.reciprocal.default(sqrt_374);  sqrt_374 = None
        mul_1122 = torch.ops.aten.mul.Tensor(reciprocal_374, 1);  reciprocal_374 = None
        unsqueeze_2992 = torch.ops.aten.unsqueeze.default(arg1015_1, -1);  arg1015_1 = None
        unsqueeze_2993 = torch.ops.aten.unsqueeze.default(unsqueeze_2992, -1);  unsqueeze_2992 = None
        unsqueeze_2994 = torch.ops.aten.unsqueeze.default(mul_1122, -1);  mul_1122 = None
        unsqueeze_2995 = torch.ops.aten.unsqueeze.default(unsqueeze_2994, -1);  unsqueeze_2994 = None
        sub_374 = torch.ops.aten.sub.Tensor(convolution_694, unsqueeze_2993);  convolution_694 = unsqueeze_2993 = None
        mul_1123 = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_2995);  sub_374 = unsqueeze_2995 = None
        unsqueeze_2996 = torch.ops.aten.unsqueeze.default(arg1017_1, -1);  arg1017_1 = None
        unsqueeze_2997 = torch.ops.aten.unsqueeze.default(unsqueeze_2996, -1);  unsqueeze_2996 = None
        mul_1124 = torch.ops.aten.mul.Tensor(mul_1123, unsqueeze_2997);  mul_1123 = unsqueeze_2997 = None
        unsqueeze_2998 = torch.ops.aten.unsqueeze.default(arg1018_1, -1);  arg1018_1 = None
        unsqueeze_2999 = torch.ops.aten.unsqueeze.default(unsqueeze_2998, -1);  unsqueeze_2998 = None
        add_879 = torch.ops.aten.add.Tensor(mul_1124, unsqueeze_2999);  mul_1124 = unsqueeze_2999 = None
        relu_372 = torch.ops.aten.relu.default(cat_33)
        convolution_695 = torch.ops.aten.convolution.default(relu_372, arg1019_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_372 = arg1019_1 = None
        add_880 = torch.ops.aten.add.Tensor(arg1021_1, 0.001);  arg1021_1 = None
        sqrt_375 = torch.ops.aten.sqrt.default(add_880);  add_880 = None
        reciprocal_375 = torch.ops.aten.reciprocal.default(sqrt_375);  sqrt_375 = None
        mul_1125 = torch.ops.aten.mul.Tensor(reciprocal_375, 1);  reciprocal_375 = None
        unsqueeze_3000 = torch.ops.aten.unsqueeze.default(arg1020_1, -1);  arg1020_1 = None
        unsqueeze_3001 = torch.ops.aten.unsqueeze.default(unsqueeze_3000, -1);  unsqueeze_3000 = None
        unsqueeze_3002 = torch.ops.aten.unsqueeze.default(mul_1125, -1);  mul_1125 = None
        unsqueeze_3003 = torch.ops.aten.unsqueeze.default(unsqueeze_3002, -1);  unsqueeze_3002 = None
        sub_375 = torch.ops.aten.sub.Tensor(convolution_695, unsqueeze_3001);  convolution_695 = unsqueeze_3001 = None
        mul_1126 = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_3003);  sub_375 = unsqueeze_3003 = None
        unsqueeze_3004 = torch.ops.aten.unsqueeze.default(arg1022_1, -1);  arg1022_1 = None
        unsqueeze_3005 = torch.ops.aten.unsqueeze.default(unsqueeze_3004, -1);  unsqueeze_3004 = None
        mul_1127 = torch.ops.aten.mul.Tensor(mul_1126, unsqueeze_3005);  mul_1126 = unsqueeze_3005 = None
        unsqueeze_3006 = torch.ops.aten.unsqueeze.default(arg1023_1, -1);  arg1023_1 = None
        unsqueeze_3007 = torch.ops.aten.unsqueeze.default(unsqueeze_3006, -1);  unsqueeze_3006 = None
        add_881 = torch.ops.aten.add.Tensor(mul_1127, unsqueeze_3007);  mul_1127 = unsqueeze_3007 = None
        relu_373 = torch.ops.aten.relu.default(add_879)
        convolution_696 = torch.ops.aten.convolution.default(relu_373, arg1024_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_373 = arg1024_1 = None
        convolution_697 = torch.ops.aten.convolution.default(convolution_696, arg1025_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_696 = arg1025_1 = None
        add_882 = torch.ops.aten.add.Tensor(arg1027_1, 0.001);  arg1027_1 = None
        sqrt_376 = torch.ops.aten.sqrt.default(add_882);  add_882 = None
        reciprocal_376 = torch.ops.aten.reciprocal.default(sqrt_376);  sqrt_376 = None
        mul_1128 = torch.ops.aten.mul.Tensor(reciprocal_376, 1);  reciprocal_376 = None
        unsqueeze_3008 = torch.ops.aten.unsqueeze.default(arg1026_1, -1);  arg1026_1 = None
        unsqueeze_3009 = torch.ops.aten.unsqueeze.default(unsqueeze_3008, -1);  unsqueeze_3008 = None
        unsqueeze_3010 = torch.ops.aten.unsqueeze.default(mul_1128, -1);  mul_1128 = None
        unsqueeze_3011 = torch.ops.aten.unsqueeze.default(unsqueeze_3010, -1);  unsqueeze_3010 = None
        sub_376 = torch.ops.aten.sub.Tensor(convolution_697, unsqueeze_3009);  convolution_697 = unsqueeze_3009 = None
        mul_1129 = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_3011);  sub_376 = unsqueeze_3011 = None
        unsqueeze_3012 = torch.ops.aten.unsqueeze.default(arg1028_1, -1);  arg1028_1 = None
        unsqueeze_3013 = torch.ops.aten.unsqueeze.default(unsqueeze_3012, -1);  unsqueeze_3012 = None
        mul_1130 = torch.ops.aten.mul.Tensor(mul_1129, unsqueeze_3013);  mul_1129 = unsqueeze_3013 = None
        unsqueeze_3014 = torch.ops.aten.unsqueeze.default(arg1029_1, -1);  arg1029_1 = None
        unsqueeze_3015 = torch.ops.aten.unsqueeze.default(unsqueeze_3014, -1);  unsqueeze_3014 = None
        add_883 = torch.ops.aten.add.Tensor(mul_1130, unsqueeze_3015);  mul_1130 = unsqueeze_3015 = None
        relu_374 = torch.ops.aten.relu.default(add_883);  add_883 = None
        convolution_698 = torch.ops.aten.convolution.default(relu_374, arg1030_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_374 = arg1030_1 = None
        convolution_699 = torch.ops.aten.convolution.default(convolution_698, arg1031_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_698 = arg1031_1 = None
        add_884 = torch.ops.aten.add.Tensor(arg1033_1, 0.001);  arg1033_1 = None
        sqrt_377 = torch.ops.aten.sqrt.default(add_884);  add_884 = None
        reciprocal_377 = torch.ops.aten.reciprocal.default(sqrt_377);  sqrt_377 = None
        mul_1131 = torch.ops.aten.mul.Tensor(reciprocal_377, 1);  reciprocal_377 = None
        unsqueeze_3016 = torch.ops.aten.unsqueeze.default(arg1032_1, -1);  arg1032_1 = None
        unsqueeze_3017 = torch.ops.aten.unsqueeze.default(unsqueeze_3016, -1);  unsqueeze_3016 = None
        unsqueeze_3018 = torch.ops.aten.unsqueeze.default(mul_1131, -1);  mul_1131 = None
        unsqueeze_3019 = torch.ops.aten.unsqueeze.default(unsqueeze_3018, -1);  unsqueeze_3018 = None
        sub_377 = torch.ops.aten.sub.Tensor(convolution_699, unsqueeze_3017);  convolution_699 = unsqueeze_3017 = None
        mul_1132 = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_3019);  sub_377 = unsqueeze_3019 = None
        unsqueeze_3020 = torch.ops.aten.unsqueeze.default(arg1034_1, -1);  arg1034_1 = None
        unsqueeze_3021 = torch.ops.aten.unsqueeze.default(unsqueeze_3020, -1);  unsqueeze_3020 = None
        mul_1133 = torch.ops.aten.mul.Tensor(mul_1132, unsqueeze_3021);  mul_1132 = unsqueeze_3021 = None
        unsqueeze_3022 = torch.ops.aten.unsqueeze.default(arg1035_1, -1);  arg1035_1 = None
        unsqueeze_3023 = torch.ops.aten.unsqueeze.default(unsqueeze_3022, -1);  unsqueeze_3022 = None
        add_885 = torch.ops.aten.add.Tensor(mul_1133, unsqueeze_3023);  mul_1133 = unsqueeze_3023 = None
        _low_memory_max_pool2d_with_offsets_78 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_879, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_156 = _low_memory_max_pool2d_with_offsets_78[0];  _low_memory_max_pool2d_with_offsets_78 = None
        add_886 = torch.ops.aten.add.Tensor(add_885, getitem_156);  add_885 = getitem_156 = None
        relu_375 = torch.ops.aten.relu.default(add_881)
        convolution_700 = torch.ops.aten.convolution.default(relu_375, arg1036_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864);  relu_375 = arg1036_1 = None
        convolution_701 = torch.ops.aten.convolution.default(convolution_700, arg1037_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_700 = arg1037_1 = None
        add_887 = torch.ops.aten.add.Tensor(arg1039_1, 0.001);  arg1039_1 = None
        sqrt_378 = torch.ops.aten.sqrt.default(add_887);  add_887 = None
        reciprocal_378 = torch.ops.aten.reciprocal.default(sqrt_378);  sqrt_378 = None
        mul_1134 = torch.ops.aten.mul.Tensor(reciprocal_378, 1);  reciprocal_378 = None
        unsqueeze_3024 = torch.ops.aten.unsqueeze.default(arg1038_1, -1);  arg1038_1 = None
        unsqueeze_3025 = torch.ops.aten.unsqueeze.default(unsqueeze_3024, -1);  unsqueeze_3024 = None
        unsqueeze_3026 = torch.ops.aten.unsqueeze.default(mul_1134, -1);  mul_1134 = None
        unsqueeze_3027 = torch.ops.aten.unsqueeze.default(unsqueeze_3026, -1);  unsqueeze_3026 = None
        sub_378 = torch.ops.aten.sub.Tensor(convolution_701, unsqueeze_3025);  convolution_701 = unsqueeze_3025 = None
        mul_1135 = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_3027);  sub_378 = unsqueeze_3027 = None
        unsqueeze_3028 = torch.ops.aten.unsqueeze.default(arg1040_1, -1);  arg1040_1 = None
        unsqueeze_3029 = torch.ops.aten.unsqueeze.default(unsqueeze_3028, -1);  unsqueeze_3028 = None
        mul_1136 = torch.ops.aten.mul.Tensor(mul_1135, unsqueeze_3029);  mul_1135 = unsqueeze_3029 = None
        unsqueeze_3030 = torch.ops.aten.unsqueeze.default(arg1041_1, -1);  arg1041_1 = None
        unsqueeze_3031 = torch.ops.aten.unsqueeze.default(unsqueeze_3030, -1);  unsqueeze_3030 = None
        add_888 = torch.ops.aten.add.Tensor(mul_1136, unsqueeze_3031);  mul_1136 = unsqueeze_3031 = None
        relu_376 = torch.ops.aten.relu.default(add_888);  add_888 = None
        convolution_702 = torch.ops.aten.convolution.default(relu_376, arg1042_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864);  relu_376 = arg1042_1 = None
        convolution_703 = torch.ops.aten.convolution.default(convolution_702, arg1043_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_702 = arg1043_1 = None
        add_889 = torch.ops.aten.add.Tensor(arg1045_1, 0.001);  arg1045_1 = None
        sqrt_379 = torch.ops.aten.sqrt.default(add_889);  add_889 = None
        reciprocal_379 = torch.ops.aten.reciprocal.default(sqrt_379);  sqrt_379 = None
        mul_1137 = torch.ops.aten.mul.Tensor(reciprocal_379, 1);  reciprocal_379 = None
        unsqueeze_3032 = torch.ops.aten.unsqueeze.default(arg1044_1, -1);  arg1044_1 = None
        unsqueeze_3033 = torch.ops.aten.unsqueeze.default(unsqueeze_3032, -1);  unsqueeze_3032 = None
        unsqueeze_3034 = torch.ops.aten.unsqueeze.default(mul_1137, -1);  mul_1137 = None
        unsqueeze_3035 = torch.ops.aten.unsqueeze.default(unsqueeze_3034, -1);  unsqueeze_3034 = None
        sub_379 = torch.ops.aten.sub.Tensor(convolution_703, unsqueeze_3033);  convolution_703 = unsqueeze_3033 = None
        mul_1138 = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_3035);  sub_379 = unsqueeze_3035 = None
        unsqueeze_3036 = torch.ops.aten.unsqueeze.default(arg1046_1, -1);  arg1046_1 = None
        unsqueeze_3037 = torch.ops.aten.unsqueeze.default(unsqueeze_3036, -1);  unsqueeze_3036 = None
        mul_1139 = torch.ops.aten.mul.Tensor(mul_1138, unsqueeze_3037);  mul_1138 = unsqueeze_3037 = None
        unsqueeze_3038 = torch.ops.aten.unsqueeze.default(arg1047_1, -1);  arg1047_1 = None
        unsqueeze_3039 = torch.ops.aten.unsqueeze.default(unsqueeze_3038, -1);  unsqueeze_3038 = None
        add_890 = torch.ops.aten.add.Tensor(mul_1139, unsqueeze_3039);  mul_1139 = unsqueeze_3039 = None
        _low_memory_max_pool2d_with_offsets_79 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_881, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_158 = _low_memory_max_pool2d_with_offsets_79[0];  _low_memory_max_pool2d_with_offsets_79 = None
        add_891 = torch.ops.aten.add.Tensor(add_890, getitem_158);  add_890 = getitem_158 = None
        relu_377 = torch.ops.aten.relu.default(add_881)
        convolution_704 = torch.ops.aten.convolution.default(relu_377, arg1048_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_377 = arg1048_1 = None
        convolution_705 = torch.ops.aten.convolution.default(convolution_704, arg1049_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_704 = arg1049_1 = None
        add_892 = torch.ops.aten.add.Tensor(arg1051_1, 0.001);  arg1051_1 = None
        sqrt_380 = torch.ops.aten.sqrt.default(add_892);  add_892 = None
        reciprocal_380 = torch.ops.aten.reciprocal.default(sqrt_380);  sqrt_380 = None
        mul_1140 = torch.ops.aten.mul.Tensor(reciprocal_380, 1);  reciprocal_380 = None
        unsqueeze_3040 = torch.ops.aten.unsqueeze.default(arg1050_1, -1);  arg1050_1 = None
        unsqueeze_3041 = torch.ops.aten.unsqueeze.default(unsqueeze_3040, -1);  unsqueeze_3040 = None
        unsqueeze_3042 = torch.ops.aten.unsqueeze.default(mul_1140, -1);  mul_1140 = None
        unsqueeze_3043 = torch.ops.aten.unsqueeze.default(unsqueeze_3042, -1);  unsqueeze_3042 = None
        sub_380 = torch.ops.aten.sub.Tensor(convolution_705, unsqueeze_3041);  convolution_705 = unsqueeze_3041 = None
        mul_1141 = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_3043);  sub_380 = unsqueeze_3043 = None
        unsqueeze_3044 = torch.ops.aten.unsqueeze.default(arg1052_1, -1);  arg1052_1 = None
        unsqueeze_3045 = torch.ops.aten.unsqueeze.default(unsqueeze_3044, -1);  unsqueeze_3044 = None
        mul_1142 = torch.ops.aten.mul.Tensor(mul_1141, unsqueeze_3045);  mul_1141 = unsqueeze_3045 = None
        unsqueeze_3046 = torch.ops.aten.unsqueeze.default(arg1053_1, -1);  arg1053_1 = None
        unsqueeze_3047 = torch.ops.aten.unsqueeze.default(unsqueeze_3046, -1);  unsqueeze_3046 = None
        add_893 = torch.ops.aten.add.Tensor(mul_1142, unsqueeze_3047);  mul_1142 = unsqueeze_3047 = None
        relu_378 = torch.ops.aten.relu.default(add_893);  add_893 = None
        convolution_706 = torch.ops.aten.convolution.default(relu_378, arg1054_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_378 = arg1054_1 = None
        convolution_707 = torch.ops.aten.convolution.default(convolution_706, arg1055_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_706 = arg1055_1 = None
        add_894 = torch.ops.aten.add.Tensor(arg1057_1, 0.001);  arg1057_1 = None
        sqrt_381 = torch.ops.aten.sqrt.default(add_894);  add_894 = None
        reciprocal_381 = torch.ops.aten.reciprocal.default(sqrt_381);  sqrt_381 = None
        mul_1143 = torch.ops.aten.mul.Tensor(reciprocal_381, 1);  reciprocal_381 = None
        unsqueeze_3048 = torch.ops.aten.unsqueeze.default(arg1056_1, -1);  arg1056_1 = None
        unsqueeze_3049 = torch.ops.aten.unsqueeze.default(unsqueeze_3048, -1);  unsqueeze_3048 = None
        unsqueeze_3050 = torch.ops.aten.unsqueeze.default(mul_1143, -1);  mul_1143 = None
        unsqueeze_3051 = torch.ops.aten.unsqueeze.default(unsqueeze_3050, -1);  unsqueeze_3050 = None
        sub_381 = torch.ops.aten.sub.Tensor(convolution_707, unsqueeze_3049);  convolution_707 = unsqueeze_3049 = None
        mul_1144 = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_3051);  sub_381 = unsqueeze_3051 = None
        unsqueeze_3052 = torch.ops.aten.unsqueeze.default(arg1058_1, -1);  arg1058_1 = None
        unsqueeze_3053 = torch.ops.aten.unsqueeze.default(unsqueeze_3052, -1);  unsqueeze_3052 = None
        mul_1145 = torch.ops.aten.mul.Tensor(mul_1144, unsqueeze_3053);  mul_1144 = unsqueeze_3053 = None
        unsqueeze_3054 = torch.ops.aten.unsqueeze.default(arg1059_1, -1);  arg1059_1 = None
        unsqueeze_3055 = torch.ops.aten.unsqueeze.default(unsqueeze_3054, -1);  unsqueeze_3054 = None
        add_895 = torch.ops.aten.add.Tensor(mul_1145, unsqueeze_3055);  mul_1145 = unsqueeze_3055 = None
        relu_379 = torch.ops.aten.relu.default(add_881)
        convolution_708 = torch.ops.aten.convolution.default(relu_379, arg1060_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_379 = arg1060_1 = None
        convolution_709 = torch.ops.aten.convolution.default(convolution_708, arg1061_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_708 = arg1061_1 = None
        add_896 = torch.ops.aten.add.Tensor(arg1063_1, 0.001);  arg1063_1 = None
        sqrt_382 = torch.ops.aten.sqrt.default(add_896);  add_896 = None
        reciprocal_382 = torch.ops.aten.reciprocal.default(sqrt_382);  sqrt_382 = None
        mul_1146 = torch.ops.aten.mul.Tensor(reciprocal_382, 1);  reciprocal_382 = None
        unsqueeze_3056 = torch.ops.aten.unsqueeze.default(arg1062_1, -1);  arg1062_1 = None
        unsqueeze_3057 = torch.ops.aten.unsqueeze.default(unsqueeze_3056, -1);  unsqueeze_3056 = None
        unsqueeze_3058 = torch.ops.aten.unsqueeze.default(mul_1146, -1);  mul_1146 = None
        unsqueeze_3059 = torch.ops.aten.unsqueeze.default(unsqueeze_3058, -1);  unsqueeze_3058 = None
        sub_382 = torch.ops.aten.sub.Tensor(convolution_709, unsqueeze_3057);  convolution_709 = unsqueeze_3057 = None
        mul_1147 = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_3059);  sub_382 = unsqueeze_3059 = None
        unsqueeze_3060 = torch.ops.aten.unsqueeze.default(arg1064_1, -1);  arg1064_1 = None
        unsqueeze_3061 = torch.ops.aten.unsqueeze.default(unsqueeze_3060, -1);  unsqueeze_3060 = None
        mul_1148 = torch.ops.aten.mul.Tensor(mul_1147, unsqueeze_3061);  mul_1147 = unsqueeze_3061 = None
        unsqueeze_3062 = torch.ops.aten.unsqueeze.default(arg1065_1, -1);  arg1065_1 = None
        unsqueeze_3063 = torch.ops.aten.unsqueeze.default(unsqueeze_3062, -1);  unsqueeze_3062 = None
        add_897 = torch.ops.aten.add.Tensor(mul_1148, unsqueeze_3063);  mul_1148 = unsqueeze_3063 = None
        relu_380 = torch.ops.aten.relu.default(add_897);  add_897 = None
        convolution_710 = torch.ops.aten.convolution.default(relu_380, arg1066_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_380 = arg1066_1 = None
        convolution_711 = torch.ops.aten.convolution.default(convolution_710, arg1067_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_710 = arg1067_1 = None
        add_898 = torch.ops.aten.add.Tensor(arg1069_1, 0.001);  arg1069_1 = None
        sqrt_383 = torch.ops.aten.sqrt.default(add_898);  add_898 = None
        reciprocal_383 = torch.ops.aten.reciprocal.default(sqrt_383);  sqrt_383 = None
        mul_1149 = torch.ops.aten.mul.Tensor(reciprocal_383, 1);  reciprocal_383 = None
        unsqueeze_3064 = torch.ops.aten.unsqueeze.default(arg1068_1, -1);  arg1068_1 = None
        unsqueeze_3065 = torch.ops.aten.unsqueeze.default(unsqueeze_3064, -1);  unsqueeze_3064 = None
        unsqueeze_3066 = torch.ops.aten.unsqueeze.default(mul_1149, -1);  mul_1149 = None
        unsqueeze_3067 = torch.ops.aten.unsqueeze.default(unsqueeze_3066, -1);  unsqueeze_3066 = None
        sub_383 = torch.ops.aten.sub.Tensor(convolution_711, unsqueeze_3065);  convolution_711 = unsqueeze_3065 = None
        mul_1150 = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_3067);  sub_383 = unsqueeze_3067 = None
        unsqueeze_3068 = torch.ops.aten.unsqueeze.default(arg1070_1, -1);  arg1070_1 = None
        unsqueeze_3069 = torch.ops.aten.unsqueeze.default(unsqueeze_3068, -1);  unsqueeze_3068 = None
        mul_1151 = torch.ops.aten.mul.Tensor(mul_1150, unsqueeze_3069);  mul_1150 = unsqueeze_3069 = None
        unsqueeze_3070 = torch.ops.aten.unsqueeze.default(arg1071_1, -1);  arg1071_1 = None
        unsqueeze_3071 = torch.ops.aten.unsqueeze.default(unsqueeze_3070, -1);  unsqueeze_3070 = None
        add_899 = torch.ops.aten.add.Tensor(mul_1151, unsqueeze_3071);  mul_1151 = unsqueeze_3071 = None
        add_900 = torch.ops.aten.add.Tensor(add_895, add_899);  add_895 = add_899 = None
        relu_381 = torch.ops.aten.relu.default(add_900)
        convolution_712 = torch.ops.aten.convolution.default(relu_381, arg1072_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_381 = arg1072_1 = None
        convolution_713 = torch.ops.aten.convolution.default(convolution_712, arg1073_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_712 = arg1073_1 = None
        add_901 = torch.ops.aten.add.Tensor(arg1075_1, 0.001);  arg1075_1 = None
        sqrt_384 = torch.ops.aten.sqrt.default(add_901);  add_901 = None
        reciprocal_384 = torch.ops.aten.reciprocal.default(sqrt_384);  sqrt_384 = None
        mul_1152 = torch.ops.aten.mul.Tensor(reciprocal_384, 1);  reciprocal_384 = None
        unsqueeze_3072 = torch.ops.aten.unsqueeze.default(arg1074_1, -1);  arg1074_1 = None
        unsqueeze_3073 = torch.ops.aten.unsqueeze.default(unsqueeze_3072, -1);  unsqueeze_3072 = None
        unsqueeze_3074 = torch.ops.aten.unsqueeze.default(mul_1152, -1);  mul_1152 = None
        unsqueeze_3075 = torch.ops.aten.unsqueeze.default(unsqueeze_3074, -1);  unsqueeze_3074 = None
        sub_384 = torch.ops.aten.sub.Tensor(convolution_713, unsqueeze_3073);  convolution_713 = unsqueeze_3073 = None
        mul_1153 = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_3075);  sub_384 = unsqueeze_3075 = None
        unsqueeze_3076 = torch.ops.aten.unsqueeze.default(arg1076_1, -1);  arg1076_1 = None
        unsqueeze_3077 = torch.ops.aten.unsqueeze.default(unsqueeze_3076, -1);  unsqueeze_3076 = None
        mul_1154 = torch.ops.aten.mul.Tensor(mul_1153, unsqueeze_3077);  mul_1153 = unsqueeze_3077 = None
        unsqueeze_3078 = torch.ops.aten.unsqueeze.default(arg1077_1, -1);  arg1077_1 = None
        unsqueeze_3079 = torch.ops.aten.unsqueeze.default(unsqueeze_3078, -1);  unsqueeze_3078 = None
        add_902 = torch.ops.aten.add.Tensor(mul_1154, unsqueeze_3079);  mul_1154 = unsqueeze_3079 = None
        relu_382 = torch.ops.aten.relu.default(add_902);  add_902 = None
        convolution_714 = torch.ops.aten.convolution.default(relu_382, arg1078_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_382 = arg1078_1 = None
        convolution_715 = torch.ops.aten.convolution.default(convolution_714, arg1079_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_714 = arg1079_1 = None
        add_903 = torch.ops.aten.add.Tensor(arg1081_1, 0.001);  arg1081_1 = None
        sqrt_385 = torch.ops.aten.sqrt.default(add_903);  add_903 = None
        reciprocal_385 = torch.ops.aten.reciprocal.default(sqrt_385);  sqrt_385 = None
        mul_1155 = torch.ops.aten.mul.Tensor(reciprocal_385, 1);  reciprocal_385 = None
        unsqueeze_3080 = torch.ops.aten.unsqueeze.default(arg1080_1, -1);  arg1080_1 = None
        unsqueeze_3081 = torch.ops.aten.unsqueeze.default(unsqueeze_3080, -1);  unsqueeze_3080 = None
        unsqueeze_3082 = torch.ops.aten.unsqueeze.default(mul_1155, -1);  mul_1155 = None
        unsqueeze_3083 = torch.ops.aten.unsqueeze.default(unsqueeze_3082, -1);  unsqueeze_3082 = None
        sub_385 = torch.ops.aten.sub.Tensor(convolution_715, unsqueeze_3081);  convolution_715 = unsqueeze_3081 = None
        mul_1156 = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_3083);  sub_385 = unsqueeze_3083 = None
        unsqueeze_3084 = torch.ops.aten.unsqueeze.default(arg1082_1, -1);  arg1082_1 = None
        unsqueeze_3085 = torch.ops.aten.unsqueeze.default(unsqueeze_3084, -1);  unsqueeze_3084 = None
        mul_1157 = torch.ops.aten.mul.Tensor(mul_1156, unsqueeze_3085);  mul_1156 = unsqueeze_3085 = None
        unsqueeze_3086 = torch.ops.aten.unsqueeze.default(arg1083_1, -1);  arg1083_1 = None
        unsqueeze_3087 = torch.ops.aten.unsqueeze.default(unsqueeze_3086, -1);  unsqueeze_3086 = None
        add_904 = torch.ops.aten.add.Tensor(mul_1157, unsqueeze_3087);  mul_1157 = unsqueeze_3087 = None
        _low_memory_max_pool2d_with_offsets_80 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_881, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_160 = _low_memory_max_pool2d_with_offsets_80[0];  _low_memory_max_pool2d_with_offsets_80 = None
        add_905 = torch.ops.aten.add.Tensor(add_904, getitem_160);  add_904 = getitem_160 = None
        relu_383 = torch.ops.aten.relu.default(add_879);  add_879 = None
        convolution_716 = torch.ops.aten.convolution.default(relu_383, arg1084_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_383 = arg1084_1 = None
        convolution_717 = torch.ops.aten.convolution.default(convolution_716, arg1085_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_716 = arg1085_1 = None
        add_906 = torch.ops.aten.add.Tensor(arg1087_1, 0.001);  arg1087_1 = None
        sqrt_386 = torch.ops.aten.sqrt.default(add_906);  add_906 = None
        reciprocal_386 = torch.ops.aten.reciprocal.default(sqrt_386);  sqrt_386 = None
        mul_1158 = torch.ops.aten.mul.Tensor(reciprocal_386, 1);  reciprocal_386 = None
        unsqueeze_3088 = torch.ops.aten.unsqueeze.default(arg1086_1, -1);  arg1086_1 = None
        unsqueeze_3089 = torch.ops.aten.unsqueeze.default(unsqueeze_3088, -1);  unsqueeze_3088 = None
        unsqueeze_3090 = torch.ops.aten.unsqueeze.default(mul_1158, -1);  mul_1158 = None
        unsqueeze_3091 = torch.ops.aten.unsqueeze.default(unsqueeze_3090, -1);  unsqueeze_3090 = None
        sub_386 = torch.ops.aten.sub.Tensor(convolution_717, unsqueeze_3089);  convolution_717 = unsqueeze_3089 = None
        mul_1159 = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_3091);  sub_386 = unsqueeze_3091 = None
        unsqueeze_3092 = torch.ops.aten.unsqueeze.default(arg1088_1, -1);  arg1088_1 = None
        unsqueeze_3093 = torch.ops.aten.unsqueeze.default(unsqueeze_3092, -1);  unsqueeze_3092 = None
        mul_1160 = torch.ops.aten.mul.Tensor(mul_1159, unsqueeze_3093);  mul_1159 = unsqueeze_3093 = None
        unsqueeze_3094 = torch.ops.aten.unsqueeze.default(arg1089_1, -1);  arg1089_1 = None
        unsqueeze_3095 = torch.ops.aten.unsqueeze.default(unsqueeze_3094, -1);  unsqueeze_3094 = None
        add_907 = torch.ops.aten.add.Tensor(mul_1160, unsqueeze_3095);  mul_1160 = unsqueeze_3095 = None
        relu_384 = torch.ops.aten.relu.default(add_907);  add_907 = None
        convolution_718 = torch.ops.aten.convolution.default(relu_384, arg1090_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_384 = arg1090_1 = None
        convolution_719 = torch.ops.aten.convolution.default(convolution_718, arg1091_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_718 = arg1091_1 = None
        add_908 = torch.ops.aten.add.Tensor(arg1093_1, 0.001);  arg1093_1 = None
        sqrt_387 = torch.ops.aten.sqrt.default(add_908);  add_908 = None
        reciprocal_387 = torch.ops.aten.reciprocal.default(sqrt_387);  sqrt_387 = None
        mul_1161 = torch.ops.aten.mul.Tensor(reciprocal_387, 1);  reciprocal_387 = None
        unsqueeze_3096 = torch.ops.aten.unsqueeze.default(arg1092_1, -1);  arg1092_1 = None
        unsqueeze_3097 = torch.ops.aten.unsqueeze.default(unsqueeze_3096, -1);  unsqueeze_3096 = None
        unsqueeze_3098 = torch.ops.aten.unsqueeze.default(mul_1161, -1);  mul_1161 = None
        unsqueeze_3099 = torch.ops.aten.unsqueeze.default(unsqueeze_3098, -1);  unsqueeze_3098 = None
        sub_387 = torch.ops.aten.sub.Tensor(convolution_719, unsqueeze_3097);  convolution_719 = unsqueeze_3097 = None
        mul_1162 = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_3099);  sub_387 = unsqueeze_3099 = None
        unsqueeze_3100 = torch.ops.aten.unsqueeze.default(arg1094_1, -1);  arg1094_1 = None
        unsqueeze_3101 = torch.ops.aten.unsqueeze.default(unsqueeze_3100, -1);  unsqueeze_3100 = None
        mul_1163 = torch.ops.aten.mul.Tensor(mul_1162, unsqueeze_3101);  mul_1162 = unsqueeze_3101 = None
        unsqueeze_3102 = torch.ops.aten.unsqueeze.default(arg1095_1, -1);  arg1095_1 = None
        unsqueeze_3103 = torch.ops.aten.unsqueeze.default(unsqueeze_3102, -1);  unsqueeze_3102 = None
        add_909 = torch.ops.aten.add.Tensor(mul_1163, unsqueeze_3103);  mul_1163 = unsqueeze_3103 = None
        add_910 = torch.ops.aten.add.Tensor(add_909, add_881);  add_909 = add_881 = None
        cat_34 = torch.ops.aten.cat.default([add_886, add_891, add_900, add_905, add_910], 1);  add_886 = add_891 = add_900 = add_905 = add_910 = None
        relu_385 = torch.ops.aten.relu.default(cat_33);  cat_33 = None
        convolution_720 = torch.ops.aten.convolution.default(relu_385, arg1096_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_385 = arg1096_1 = None
        add_911 = torch.ops.aten.add.Tensor(arg1098_1, 0.001);  arg1098_1 = None
        sqrt_388 = torch.ops.aten.sqrt.default(add_911);  add_911 = None
        reciprocal_388 = torch.ops.aten.reciprocal.default(sqrt_388);  sqrt_388 = None
        mul_1164 = torch.ops.aten.mul.Tensor(reciprocal_388, 1);  reciprocal_388 = None
        unsqueeze_3104 = torch.ops.aten.unsqueeze.default(arg1097_1, -1);  arg1097_1 = None
        unsqueeze_3105 = torch.ops.aten.unsqueeze.default(unsqueeze_3104, -1);  unsqueeze_3104 = None
        unsqueeze_3106 = torch.ops.aten.unsqueeze.default(mul_1164, -1);  mul_1164 = None
        unsqueeze_3107 = torch.ops.aten.unsqueeze.default(unsqueeze_3106, -1);  unsqueeze_3106 = None
        sub_388 = torch.ops.aten.sub.Tensor(convolution_720, unsqueeze_3105);  convolution_720 = unsqueeze_3105 = None
        mul_1165 = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_3107);  sub_388 = unsqueeze_3107 = None
        unsqueeze_3108 = torch.ops.aten.unsqueeze.default(arg1099_1, -1);  arg1099_1 = None
        unsqueeze_3109 = torch.ops.aten.unsqueeze.default(unsqueeze_3108, -1);  unsqueeze_3108 = None
        mul_1166 = torch.ops.aten.mul.Tensor(mul_1165, unsqueeze_3109);  mul_1165 = unsqueeze_3109 = None
        unsqueeze_3110 = torch.ops.aten.unsqueeze.default(arg1100_1, -1);  arg1100_1 = None
        unsqueeze_3111 = torch.ops.aten.unsqueeze.default(unsqueeze_3110, -1);  unsqueeze_3110 = None
        add_912 = torch.ops.aten.add.Tensor(mul_1166, unsqueeze_3111);  mul_1166 = unsqueeze_3111 = None
        relu_386 = torch.ops.aten.relu.default(cat_34);  cat_34 = None
        convolution_721 = torch.ops.aten.convolution.default(relu_386, arg1101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_386 = arg1101_1 = None
        add_913 = torch.ops.aten.add.Tensor(arg1103_1, 0.001);  arg1103_1 = None
        sqrt_389 = torch.ops.aten.sqrt.default(add_913);  add_913 = None
        reciprocal_389 = torch.ops.aten.reciprocal.default(sqrt_389);  sqrt_389 = None
        mul_1167 = torch.ops.aten.mul.Tensor(reciprocal_389, 1);  reciprocal_389 = None
        unsqueeze_3112 = torch.ops.aten.unsqueeze.default(arg1102_1, -1);  arg1102_1 = None
        unsqueeze_3113 = torch.ops.aten.unsqueeze.default(unsqueeze_3112, -1);  unsqueeze_3112 = None
        unsqueeze_3114 = torch.ops.aten.unsqueeze.default(mul_1167, -1);  mul_1167 = None
        unsqueeze_3115 = torch.ops.aten.unsqueeze.default(unsqueeze_3114, -1);  unsqueeze_3114 = None
        sub_389 = torch.ops.aten.sub.Tensor(convolution_721, unsqueeze_3113);  convolution_721 = unsqueeze_3113 = None
        mul_1168 = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_3115);  sub_389 = unsqueeze_3115 = None
        unsqueeze_3116 = torch.ops.aten.unsqueeze.default(arg1104_1, -1);  arg1104_1 = None
        unsqueeze_3117 = torch.ops.aten.unsqueeze.default(unsqueeze_3116, -1);  unsqueeze_3116 = None
        mul_1169 = torch.ops.aten.mul.Tensor(mul_1168, unsqueeze_3117);  mul_1168 = unsqueeze_3117 = None
        unsqueeze_3118 = torch.ops.aten.unsqueeze.default(arg1105_1, -1);  arg1105_1 = None
        unsqueeze_3119 = torch.ops.aten.unsqueeze.default(unsqueeze_3118, -1);  unsqueeze_3118 = None
        add_914 = torch.ops.aten.add.Tensor(mul_1169, unsqueeze_3119);  mul_1169 = unsqueeze_3119 = None
        relu_387 = torch.ops.aten.relu.default(add_912)
        convolution_722 = torch.ops.aten.convolution.default(relu_387, arg1106_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_387 = arg1106_1 = None
        convolution_723 = torch.ops.aten.convolution.default(convolution_722, arg1107_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_722 = arg1107_1 = None
        add_915 = torch.ops.aten.add.Tensor(arg1109_1, 0.001);  arg1109_1 = None
        sqrt_390 = torch.ops.aten.sqrt.default(add_915);  add_915 = None
        reciprocal_390 = torch.ops.aten.reciprocal.default(sqrt_390);  sqrt_390 = None
        mul_1170 = torch.ops.aten.mul.Tensor(reciprocal_390, 1);  reciprocal_390 = None
        unsqueeze_3120 = torch.ops.aten.unsqueeze.default(arg1108_1, -1);  arg1108_1 = None
        unsqueeze_3121 = torch.ops.aten.unsqueeze.default(unsqueeze_3120, -1);  unsqueeze_3120 = None
        unsqueeze_3122 = torch.ops.aten.unsqueeze.default(mul_1170, -1);  mul_1170 = None
        unsqueeze_3123 = torch.ops.aten.unsqueeze.default(unsqueeze_3122, -1);  unsqueeze_3122 = None
        sub_390 = torch.ops.aten.sub.Tensor(convolution_723, unsqueeze_3121);  convolution_723 = unsqueeze_3121 = None
        mul_1171 = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_3123);  sub_390 = unsqueeze_3123 = None
        unsqueeze_3124 = torch.ops.aten.unsqueeze.default(arg1110_1, -1);  arg1110_1 = None
        unsqueeze_3125 = torch.ops.aten.unsqueeze.default(unsqueeze_3124, -1);  unsqueeze_3124 = None
        mul_1172 = torch.ops.aten.mul.Tensor(mul_1171, unsqueeze_3125);  mul_1171 = unsqueeze_3125 = None
        unsqueeze_3126 = torch.ops.aten.unsqueeze.default(arg1111_1, -1);  arg1111_1 = None
        unsqueeze_3127 = torch.ops.aten.unsqueeze.default(unsqueeze_3126, -1);  unsqueeze_3126 = None
        add_916 = torch.ops.aten.add.Tensor(mul_1172, unsqueeze_3127);  mul_1172 = unsqueeze_3127 = None
        relu_388 = torch.ops.aten.relu.default(add_916);  add_916 = None
        convolution_724 = torch.ops.aten.convolution.default(relu_388, arg1112_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_388 = arg1112_1 = None
        convolution_725 = torch.ops.aten.convolution.default(convolution_724, arg1113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_724 = arg1113_1 = None
        add_917 = torch.ops.aten.add.Tensor(arg1115_1, 0.001);  arg1115_1 = None
        sqrt_391 = torch.ops.aten.sqrt.default(add_917);  add_917 = None
        reciprocal_391 = torch.ops.aten.reciprocal.default(sqrt_391);  sqrt_391 = None
        mul_1173 = torch.ops.aten.mul.Tensor(reciprocal_391, 1);  reciprocal_391 = None
        unsqueeze_3128 = torch.ops.aten.unsqueeze.default(arg1114_1, -1);  arg1114_1 = None
        unsqueeze_3129 = torch.ops.aten.unsqueeze.default(unsqueeze_3128, -1);  unsqueeze_3128 = None
        unsqueeze_3130 = torch.ops.aten.unsqueeze.default(mul_1173, -1);  mul_1173 = None
        unsqueeze_3131 = torch.ops.aten.unsqueeze.default(unsqueeze_3130, -1);  unsqueeze_3130 = None
        sub_391 = torch.ops.aten.sub.Tensor(convolution_725, unsqueeze_3129);  convolution_725 = unsqueeze_3129 = None
        mul_1174 = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_3131);  sub_391 = unsqueeze_3131 = None
        unsqueeze_3132 = torch.ops.aten.unsqueeze.default(arg1116_1, -1);  arg1116_1 = None
        unsqueeze_3133 = torch.ops.aten.unsqueeze.default(unsqueeze_3132, -1);  unsqueeze_3132 = None
        mul_1175 = torch.ops.aten.mul.Tensor(mul_1174, unsqueeze_3133);  mul_1174 = unsqueeze_3133 = None
        unsqueeze_3134 = torch.ops.aten.unsqueeze.default(arg1117_1, -1);  arg1117_1 = None
        unsqueeze_3135 = torch.ops.aten.unsqueeze.default(unsqueeze_3134, -1);  unsqueeze_3134 = None
        add_918 = torch.ops.aten.add.Tensor(mul_1175, unsqueeze_3135);  mul_1175 = unsqueeze_3135 = None
        _low_memory_max_pool2d_with_offsets_81 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_912, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_162 = _low_memory_max_pool2d_with_offsets_81[0];  _low_memory_max_pool2d_with_offsets_81 = None
        add_919 = torch.ops.aten.add.Tensor(add_918, getitem_162);  add_918 = getitem_162 = None
        relu_389 = torch.ops.aten.relu.default(add_914)
        convolution_726 = torch.ops.aten.convolution.default(relu_389, arg1118_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864);  relu_389 = arg1118_1 = None
        convolution_727 = torch.ops.aten.convolution.default(convolution_726, arg1119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_726 = arg1119_1 = None
        add_920 = torch.ops.aten.add.Tensor(arg1121_1, 0.001);  arg1121_1 = None
        sqrt_392 = torch.ops.aten.sqrt.default(add_920);  add_920 = None
        reciprocal_392 = torch.ops.aten.reciprocal.default(sqrt_392);  sqrt_392 = None
        mul_1176 = torch.ops.aten.mul.Tensor(reciprocal_392, 1);  reciprocal_392 = None
        unsqueeze_3136 = torch.ops.aten.unsqueeze.default(arg1120_1, -1);  arg1120_1 = None
        unsqueeze_3137 = torch.ops.aten.unsqueeze.default(unsqueeze_3136, -1);  unsqueeze_3136 = None
        unsqueeze_3138 = torch.ops.aten.unsqueeze.default(mul_1176, -1);  mul_1176 = None
        unsqueeze_3139 = torch.ops.aten.unsqueeze.default(unsqueeze_3138, -1);  unsqueeze_3138 = None
        sub_392 = torch.ops.aten.sub.Tensor(convolution_727, unsqueeze_3137);  convolution_727 = unsqueeze_3137 = None
        mul_1177 = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_3139);  sub_392 = unsqueeze_3139 = None
        unsqueeze_3140 = torch.ops.aten.unsqueeze.default(arg1122_1, -1);  arg1122_1 = None
        unsqueeze_3141 = torch.ops.aten.unsqueeze.default(unsqueeze_3140, -1);  unsqueeze_3140 = None
        mul_1178 = torch.ops.aten.mul.Tensor(mul_1177, unsqueeze_3141);  mul_1177 = unsqueeze_3141 = None
        unsqueeze_3142 = torch.ops.aten.unsqueeze.default(arg1123_1, -1);  arg1123_1 = None
        unsqueeze_3143 = torch.ops.aten.unsqueeze.default(unsqueeze_3142, -1);  unsqueeze_3142 = None
        add_921 = torch.ops.aten.add.Tensor(mul_1178, unsqueeze_3143);  mul_1178 = unsqueeze_3143 = None
        relu_390 = torch.ops.aten.relu.default(add_921);  add_921 = None
        convolution_728 = torch.ops.aten.convolution.default(relu_390, arg1124_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864);  relu_390 = arg1124_1 = None
        convolution_729 = torch.ops.aten.convolution.default(convolution_728, arg1125_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_728 = arg1125_1 = None
        add_922 = torch.ops.aten.add.Tensor(arg1127_1, 0.001);  arg1127_1 = None
        sqrt_393 = torch.ops.aten.sqrt.default(add_922);  add_922 = None
        reciprocal_393 = torch.ops.aten.reciprocal.default(sqrt_393);  sqrt_393 = None
        mul_1179 = torch.ops.aten.mul.Tensor(reciprocal_393, 1);  reciprocal_393 = None
        unsqueeze_3144 = torch.ops.aten.unsqueeze.default(arg1126_1, -1);  arg1126_1 = None
        unsqueeze_3145 = torch.ops.aten.unsqueeze.default(unsqueeze_3144, -1);  unsqueeze_3144 = None
        unsqueeze_3146 = torch.ops.aten.unsqueeze.default(mul_1179, -1);  mul_1179 = None
        unsqueeze_3147 = torch.ops.aten.unsqueeze.default(unsqueeze_3146, -1);  unsqueeze_3146 = None
        sub_393 = torch.ops.aten.sub.Tensor(convolution_729, unsqueeze_3145);  convolution_729 = unsqueeze_3145 = None
        mul_1180 = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_3147);  sub_393 = unsqueeze_3147 = None
        unsqueeze_3148 = torch.ops.aten.unsqueeze.default(arg1128_1, -1);  arg1128_1 = None
        unsqueeze_3149 = torch.ops.aten.unsqueeze.default(unsqueeze_3148, -1);  unsqueeze_3148 = None
        mul_1181 = torch.ops.aten.mul.Tensor(mul_1180, unsqueeze_3149);  mul_1180 = unsqueeze_3149 = None
        unsqueeze_3150 = torch.ops.aten.unsqueeze.default(arg1129_1, -1);  arg1129_1 = None
        unsqueeze_3151 = torch.ops.aten.unsqueeze.default(unsqueeze_3150, -1);  unsqueeze_3150 = None
        add_923 = torch.ops.aten.add.Tensor(mul_1181, unsqueeze_3151);  mul_1181 = unsqueeze_3151 = None
        _low_memory_max_pool2d_with_offsets_82 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_914, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_164 = _low_memory_max_pool2d_with_offsets_82[0];  _low_memory_max_pool2d_with_offsets_82 = None
        add_924 = torch.ops.aten.add.Tensor(add_923, getitem_164);  add_923 = getitem_164 = None
        relu_391 = torch.ops.aten.relu.default(add_914)
        convolution_730 = torch.ops.aten.convolution.default(relu_391, arg1130_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_391 = arg1130_1 = None
        convolution_731 = torch.ops.aten.convolution.default(convolution_730, arg1131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_730 = arg1131_1 = None
        add_925 = torch.ops.aten.add.Tensor(arg1133_1, 0.001);  arg1133_1 = None
        sqrt_394 = torch.ops.aten.sqrt.default(add_925);  add_925 = None
        reciprocal_394 = torch.ops.aten.reciprocal.default(sqrt_394);  sqrt_394 = None
        mul_1182 = torch.ops.aten.mul.Tensor(reciprocal_394, 1);  reciprocal_394 = None
        unsqueeze_3152 = torch.ops.aten.unsqueeze.default(arg1132_1, -1);  arg1132_1 = None
        unsqueeze_3153 = torch.ops.aten.unsqueeze.default(unsqueeze_3152, -1);  unsqueeze_3152 = None
        unsqueeze_3154 = torch.ops.aten.unsqueeze.default(mul_1182, -1);  mul_1182 = None
        unsqueeze_3155 = torch.ops.aten.unsqueeze.default(unsqueeze_3154, -1);  unsqueeze_3154 = None
        sub_394 = torch.ops.aten.sub.Tensor(convolution_731, unsqueeze_3153);  convolution_731 = unsqueeze_3153 = None
        mul_1183 = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_3155);  sub_394 = unsqueeze_3155 = None
        unsqueeze_3156 = torch.ops.aten.unsqueeze.default(arg1134_1, -1);  arg1134_1 = None
        unsqueeze_3157 = torch.ops.aten.unsqueeze.default(unsqueeze_3156, -1);  unsqueeze_3156 = None
        mul_1184 = torch.ops.aten.mul.Tensor(mul_1183, unsqueeze_3157);  mul_1183 = unsqueeze_3157 = None
        unsqueeze_3158 = torch.ops.aten.unsqueeze.default(arg1135_1, -1);  arg1135_1 = None
        unsqueeze_3159 = torch.ops.aten.unsqueeze.default(unsqueeze_3158, -1);  unsqueeze_3158 = None
        add_926 = torch.ops.aten.add.Tensor(mul_1184, unsqueeze_3159);  mul_1184 = unsqueeze_3159 = None
        relu_392 = torch.ops.aten.relu.default(add_926);  add_926 = None
        convolution_732 = torch.ops.aten.convolution.default(relu_392, arg1136_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864);  relu_392 = arg1136_1 = None
        convolution_733 = torch.ops.aten.convolution.default(convolution_732, arg1137_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_732 = arg1137_1 = None
        add_927 = torch.ops.aten.add.Tensor(arg1139_1, 0.001);  arg1139_1 = None
        sqrt_395 = torch.ops.aten.sqrt.default(add_927);  add_927 = None
        reciprocal_395 = torch.ops.aten.reciprocal.default(sqrt_395);  sqrt_395 = None
        mul_1185 = torch.ops.aten.mul.Tensor(reciprocal_395, 1);  reciprocal_395 = None
        unsqueeze_3160 = torch.ops.aten.unsqueeze.default(arg1138_1, -1);  arg1138_1 = None
        unsqueeze_3161 = torch.ops.aten.unsqueeze.default(unsqueeze_3160, -1);  unsqueeze_3160 = None
        unsqueeze_3162 = torch.ops.aten.unsqueeze.default(mul_1185, -1);  mul_1185 = None
        unsqueeze_3163 = torch.ops.aten.unsqueeze.default(unsqueeze_3162, -1);  unsqueeze_3162 = None
        sub_395 = torch.ops.aten.sub.Tensor(convolution_733, unsqueeze_3161);  convolution_733 = unsqueeze_3161 = None
        mul_1186 = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_3163);  sub_395 = unsqueeze_3163 = None
        unsqueeze_3164 = torch.ops.aten.unsqueeze.default(arg1140_1, -1);  arg1140_1 = None
        unsqueeze_3165 = torch.ops.aten.unsqueeze.default(unsqueeze_3164, -1);  unsqueeze_3164 = None
        mul_1187 = torch.ops.aten.mul.Tensor(mul_1186, unsqueeze_3165);  mul_1186 = unsqueeze_3165 = None
        unsqueeze_3166 = torch.ops.aten.unsqueeze.default(arg1141_1, -1);  arg1141_1 = None
        unsqueeze_3167 = torch.ops.aten.unsqueeze.default(unsqueeze_3166, -1);  unsqueeze_3166 = None
        add_928 = torch.ops.aten.add.Tensor(mul_1187, unsqueeze_3167);  mul_1187 = unsqueeze_3167 = None
        relu_393 = torch.ops.aten.relu.default(add_914)
        convolution_734 = torch.ops.aten.convolution.default(relu_393, arg1142_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_393 = arg1142_1 = None
        convolution_735 = torch.ops.aten.convolution.default(convolution_734, arg1143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_734 = arg1143_1 = None
        add_929 = torch.ops.aten.add.Tensor(arg1145_1, 0.001);  arg1145_1 = None
        sqrt_396 = torch.ops.aten.sqrt.default(add_929);  add_929 = None
        reciprocal_396 = torch.ops.aten.reciprocal.default(sqrt_396);  sqrt_396 = None
        mul_1188 = torch.ops.aten.mul.Tensor(reciprocal_396, 1);  reciprocal_396 = None
        unsqueeze_3168 = torch.ops.aten.unsqueeze.default(arg1144_1, -1);  arg1144_1 = None
        unsqueeze_3169 = torch.ops.aten.unsqueeze.default(unsqueeze_3168, -1);  unsqueeze_3168 = None
        unsqueeze_3170 = torch.ops.aten.unsqueeze.default(mul_1188, -1);  mul_1188 = None
        unsqueeze_3171 = torch.ops.aten.unsqueeze.default(unsqueeze_3170, -1);  unsqueeze_3170 = None
        sub_396 = torch.ops.aten.sub.Tensor(convolution_735, unsqueeze_3169);  convolution_735 = unsqueeze_3169 = None
        mul_1189 = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_3171);  sub_396 = unsqueeze_3171 = None
        unsqueeze_3172 = torch.ops.aten.unsqueeze.default(arg1146_1, -1);  arg1146_1 = None
        unsqueeze_3173 = torch.ops.aten.unsqueeze.default(unsqueeze_3172, -1);  unsqueeze_3172 = None
        mul_1190 = torch.ops.aten.mul.Tensor(mul_1189, unsqueeze_3173);  mul_1189 = unsqueeze_3173 = None
        unsqueeze_3174 = torch.ops.aten.unsqueeze.default(arg1147_1, -1);  arg1147_1 = None
        unsqueeze_3175 = torch.ops.aten.unsqueeze.default(unsqueeze_3174, -1);  unsqueeze_3174 = None
        add_930 = torch.ops.aten.add.Tensor(mul_1190, unsqueeze_3175);  mul_1190 = unsqueeze_3175 = None
        relu_394 = torch.ops.aten.relu.default(add_930);  add_930 = None
        convolution_736 = torch.ops.aten.convolution.default(relu_394, arg1148_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_394 = arg1148_1 = None
        convolution_737 = torch.ops.aten.convolution.default(convolution_736, arg1149_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_736 = arg1149_1 = None
        add_931 = torch.ops.aten.add.Tensor(arg1151_1, 0.001);  arg1151_1 = None
        sqrt_397 = torch.ops.aten.sqrt.default(add_931);  add_931 = None
        reciprocal_397 = torch.ops.aten.reciprocal.default(sqrt_397);  sqrt_397 = None
        mul_1191 = torch.ops.aten.mul.Tensor(reciprocal_397, 1);  reciprocal_397 = None
        unsqueeze_3176 = torch.ops.aten.unsqueeze.default(arg1150_1, -1);  arg1150_1 = None
        unsqueeze_3177 = torch.ops.aten.unsqueeze.default(unsqueeze_3176, -1);  unsqueeze_3176 = None
        unsqueeze_3178 = torch.ops.aten.unsqueeze.default(mul_1191, -1);  mul_1191 = None
        unsqueeze_3179 = torch.ops.aten.unsqueeze.default(unsqueeze_3178, -1);  unsqueeze_3178 = None
        sub_397 = torch.ops.aten.sub.Tensor(convolution_737, unsqueeze_3177);  convolution_737 = unsqueeze_3177 = None
        mul_1192 = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_3179);  sub_397 = unsqueeze_3179 = None
        unsqueeze_3180 = torch.ops.aten.unsqueeze.default(arg1152_1, -1);  arg1152_1 = None
        unsqueeze_3181 = torch.ops.aten.unsqueeze.default(unsqueeze_3180, -1);  unsqueeze_3180 = None
        mul_1193 = torch.ops.aten.mul.Tensor(mul_1192, unsqueeze_3181);  mul_1192 = unsqueeze_3181 = None
        unsqueeze_3182 = torch.ops.aten.unsqueeze.default(arg1153_1, -1);  arg1153_1 = None
        unsqueeze_3183 = torch.ops.aten.unsqueeze.default(unsqueeze_3182, -1);  unsqueeze_3182 = None
        add_932 = torch.ops.aten.add.Tensor(mul_1193, unsqueeze_3183);  mul_1193 = unsqueeze_3183 = None
        add_933 = torch.ops.aten.add.Tensor(add_928, add_932);  add_928 = add_932 = None
        relu_395 = torch.ops.aten.relu.default(add_933)
        convolution_738 = torch.ops.aten.convolution.default(relu_395, arg1154_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_395 = arg1154_1 = None
        convolution_739 = torch.ops.aten.convolution.default(convolution_738, arg1155_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_738 = arg1155_1 = None
        add_934 = torch.ops.aten.add.Tensor(arg1157_1, 0.001);  arg1157_1 = None
        sqrt_398 = torch.ops.aten.sqrt.default(add_934);  add_934 = None
        reciprocal_398 = torch.ops.aten.reciprocal.default(sqrt_398);  sqrt_398 = None
        mul_1194 = torch.ops.aten.mul.Tensor(reciprocal_398, 1);  reciprocal_398 = None
        unsqueeze_3184 = torch.ops.aten.unsqueeze.default(arg1156_1, -1);  arg1156_1 = None
        unsqueeze_3185 = torch.ops.aten.unsqueeze.default(unsqueeze_3184, -1);  unsqueeze_3184 = None
        unsqueeze_3186 = torch.ops.aten.unsqueeze.default(mul_1194, -1);  mul_1194 = None
        unsqueeze_3187 = torch.ops.aten.unsqueeze.default(unsqueeze_3186, -1);  unsqueeze_3186 = None
        sub_398 = torch.ops.aten.sub.Tensor(convolution_739, unsqueeze_3185);  convolution_739 = unsqueeze_3185 = None
        mul_1195 = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_3187);  sub_398 = unsqueeze_3187 = None
        unsqueeze_3188 = torch.ops.aten.unsqueeze.default(arg1158_1, -1);  arg1158_1 = None
        unsqueeze_3189 = torch.ops.aten.unsqueeze.default(unsqueeze_3188, -1);  unsqueeze_3188 = None
        mul_1196 = torch.ops.aten.mul.Tensor(mul_1195, unsqueeze_3189);  mul_1195 = unsqueeze_3189 = None
        unsqueeze_3190 = torch.ops.aten.unsqueeze.default(arg1159_1, -1);  arg1159_1 = None
        unsqueeze_3191 = torch.ops.aten.unsqueeze.default(unsqueeze_3190, -1);  unsqueeze_3190 = None
        add_935 = torch.ops.aten.add.Tensor(mul_1196, unsqueeze_3191);  mul_1196 = unsqueeze_3191 = None
        relu_396 = torch.ops.aten.relu.default(add_935);  add_935 = None
        convolution_740 = torch.ops.aten.convolution.default(relu_396, arg1160_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_396 = arg1160_1 = None
        convolution_741 = torch.ops.aten.convolution.default(convolution_740, arg1161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_740 = arg1161_1 = None
        add_936 = torch.ops.aten.add.Tensor(arg1163_1, 0.001);  arg1163_1 = None
        sqrt_399 = torch.ops.aten.sqrt.default(add_936);  add_936 = None
        reciprocal_399 = torch.ops.aten.reciprocal.default(sqrt_399);  sqrt_399 = None
        mul_1197 = torch.ops.aten.mul.Tensor(reciprocal_399, 1);  reciprocal_399 = None
        unsqueeze_3192 = torch.ops.aten.unsqueeze.default(arg1162_1, -1);  arg1162_1 = None
        unsqueeze_3193 = torch.ops.aten.unsqueeze.default(unsqueeze_3192, -1);  unsqueeze_3192 = None
        unsqueeze_3194 = torch.ops.aten.unsqueeze.default(mul_1197, -1);  mul_1197 = None
        unsqueeze_3195 = torch.ops.aten.unsqueeze.default(unsqueeze_3194, -1);  unsqueeze_3194 = None
        sub_399 = torch.ops.aten.sub.Tensor(convolution_741, unsqueeze_3193);  convolution_741 = unsqueeze_3193 = None
        mul_1198 = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_3195);  sub_399 = unsqueeze_3195 = None
        unsqueeze_3196 = torch.ops.aten.unsqueeze.default(arg1164_1, -1);  arg1164_1 = None
        unsqueeze_3197 = torch.ops.aten.unsqueeze.default(unsqueeze_3196, -1);  unsqueeze_3196 = None
        mul_1199 = torch.ops.aten.mul.Tensor(mul_1198, unsqueeze_3197);  mul_1198 = unsqueeze_3197 = None
        unsqueeze_3198 = torch.ops.aten.unsqueeze.default(arg1165_1, -1);  arg1165_1 = None
        unsqueeze_3199 = torch.ops.aten.unsqueeze.default(unsqueeze_3198, -1);  unsqueeze_3198 = None
        add_937 = torch.ops.aten.add.Tensor(mul_1199, unsqueeze_3199);  mul_1199 = unsqueeze_3199 = None
        _low_memory_max_pool2d_with_offsets_83 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(add_914, [3, 3], [1, 1], [1, 1], [1, 1], False)
        getitem_166 = _low_memory_max_pool2d_with_offsets_83[0];  _low_memory_max_pool2d_with_offsets_83 = None
        add_938 = torch.ops.aten.add.Tensor(add_937, getitem_166);  add_937 = getitem_166 = None
        relu_397 = torch.ops.aten.relu.default(add_912);  add_912 = None
        convolution_742 = torch.ops.aten.convolution.default(relu_397, arg1166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_397 = arg1166_1 = None
        convolution_743 = torch.ops.aten.convolution.default(convolution_742, arg1167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_742 = arg1167_1 = None
        add_939 = torch.ops.aten.add.Tensor(arg1169_1, 0.001);  arg1169_1 = None
        sqrt_400 = torch.ops.aten.sqrt.default(add_939);  add_939 = None
        reciprocal_400 = torch.ops.aten.reciprocal.default(sqrt_400);  sqrt_400 = None
        mul_1200 = torch.ops.aten.mul.Tensor(reciprocal_400, 1);  reciprocal_400 = None
        unsqueeze_3200 = torch.ops.aten.unsqueeze.default(arg1168_1, -1);  arg1168_1 = None
        unsqueeze_3201 = torch.ops.aten.unsqueeze.default(unsqueeze_3200, -1);  unsqueeze_3200 = None
        unsqueeze_3202 = torch.ops.aten.unsqueeze.default(mul_1200, -1);  mul_1200 = None
        unsqueeze_3203 = torch.ops.aten.unsqueeze.default(unsqueeze_3202, -1);  unsqueeze_3202 = None
        sub_400 = torch.ops.aten.sub.Tensor(convolution_743, unsqueeze_3201);  convolution_743 = unsqueeze_3201 = None
        mul_1201 = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_3203);  sub_400 = unsqueeze_3203 = None
        unsqueeze_3204 = torch.ops.aten.unsqueeze.default(arg1170_1, -1);  arg1170_1 = None
        unsqueeze_3205 = torch.ops.aten.unsqueeze.default(unsqueeze_3204, -1);  unsqueeze_3204 = None
        mul_1202 = torch.ops.aten.mul.Tensor(mul_1201, unsqueeze_3205);  mul_1201 = unsqueeze_3205 = None
        unsqueeze_3206 = torch.ops.aten.unsqueeze.default(arg1171_1, -1);  arg1171_1 = None
        unsqueeze_3207 = torch.ops.aten.unsqueeze.default(unsqueeze_3206, -1);  unsqueeze_3206 = None
        add_940 = torch.ops.aten.add.Tensor(mul_1202, unsqueeze_3207);  mul_1202 = unsqueeze_3207 = None
        relu_398 = torch.ops.aten.relu.default(add_940);  add_940 = None
        convolution_744 = torch.ops.aten.convolution.default(relu_398, arg1172_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864);  relu_398 = arg1172_1 = None
        convolution_745 = torch.ops.aten.convolution.default(convolution_744, arg1173_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_744 = arg1173_1 = None
        add_941 = torch.ops.aten.add.Tensor(arg1175_1, 0.001);  arg1175_1 = None
        sqrt_401 = torch.ops.aten.sqrt.default(add_941);  add_941 = None
        reciprocal_401 = torch.ops.aten.reciprocal.default(sqrt_401);  sqrt_401 = None
        mul_1203 = torch.ops.aten.mul.Tensor(reciprocal_401, 1);  reciprocal_401 = None
        unsqueeze_3208 = torch.ops.aten.unsqueeze.default(arg1174_1, -1);  arg1174_1 = None
        unsqueeze_3209 = torch.ops.aten.unsqueeze.default(unsqueeze_3208, -1);  unsqueeze_3208 = None
        unsqueeze_3210 = torch.ops.aten.unsqueeze.default(mul_1203, -1);  mul_1203 = None
        unsqueeze_3211 = torch.ops.aten.unsqueeze.default(unsqueeze_3210, -1);  unsqueeze_3210 = None
        sub_401 = torch.ops.aten.sub.Tensor(convolution_745, unsqueeze_3209);  convolution_745 = unsqueeze_3209 = None
        mul_1204 = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_3211);  sub_401 = unsqueeze_3211 = None
        unsqueeze_3212 = torch.ops.aten.unsqueeze.default(arg1176_1, -1);  arg1176_1 = None
        unsqueeze_3213 = torch.ops.aten.unsqueeze.default(unsqueeze_3212, -1);  unsqueeze_3212 = None
        mul_1205 = torch.ops.aten.mul.Tensor(mul_1204, unsqueeze_3213);  mul_1204 = unsqueeze_3213 = None
        unsqueeze_3214 = torch.ops.aten.unsqueeze.default(arg1177_1, -1);  arg1177_1 = None
        unsqueeze_3215 = torch.ops.aten.unsqueeze.default(unsqueeze_3214, -1);  unsqueeze_3214 = None
        add_942 = torch.ops.aten.add.Tensor(mul_1205, unsqueeze_3215);  mul_1205 = unsqueeze_3215 = None
        add_943 = torch.ops.aten.add.Tensor(add_942, add_914);  add_942 = add_914 = None
        cat_35 = torch.ops.aten.cat.default([add_919, add_924, add_933, add_938, add_943], 1);  add_919 = add_924 = add_933 = add_938 = add_943 = None
        relu_399 = torch.ops.aten.relu.default(cat_35);  cat_35 = None
        mean_1 = torch.ops.aten.mean.dim(relu_399, [-1, -2], True);  relu_399 = None
        view_1 = torch.ops.aten.view.default(mean_1, [8, 4320]);  mean_1 = None
        permute_1 = torch.ops.aten.permute.default(arg1178_1, [1, 0]);  arg1178_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg1179_1, view_1, permute_1);  arg1179_1 = view_1 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf0, (96, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 10517856, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 331, 331), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf2, (96,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf3, (96,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf4, (96,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf5, (96,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf6, (54, 96, 1, 1), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf7, (54,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf8, (54,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf9, (54,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf10, (54,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 9600, device=device(type='cuda', index=0))
    reader.tensor(buf11, (96, 1, 5, 5), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf12, (54, 96, 1, 1), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf13, (54,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf14, (54,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf15, (54,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf16, (54,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 5400, device=device(type='cuda', index=0))
    reader.tensor(buf17, (54, 1, 5, 5), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf18, (54, 54, 1, 1), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf19, (54,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf20, (54,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf21, (54,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf22, (54,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf23, (54, 96, 1, 1), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf24, (54,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf25, (54,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf26, (54,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf27, (54,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 10584, device=device(type='cuda', index=0))
    reader.tensor(buf28, (54, 1, 7, 7), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf29, (54, 54, 1, 1), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf30, (54,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf31, (54,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf32, (54,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf33, (54,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 10584, device=device(type='cuda', index=0))
    reader.tensor(buf34, (54, 1, 7, 7), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf35, (54, 54, 1, 1), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf36, (54,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf37, (54,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf38, (54,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf39, (54,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 5400, device=device(type='cuda', index=0))
    reader.tensor(buf40, (54, 1, 5, 5), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf41, (54, 54, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf42, (54,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf43, (54,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf44, (54,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf45, (54,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 5400, device=device(type='cuda', index=0))
    reader.tensor(buf46, (54, 1, 5, 5), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf47, (54, 54, 1, 1), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf48, (54,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf49, (54,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf50, (54,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf51, (54,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf52, (54, 1, 3, 3), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf53, (54, 54, 1, 1), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf54, (54,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf55, (54,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf56, (54,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf57, (54,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf58, (54, 1, 3, 3), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf59, (54, 54, 1, 1), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf60, (54,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf61, (54,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf62, (54,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf63, (54,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf64, (54, 1, 3, 3), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf65, (54, 54, 1, 1), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf66, (54,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf67, (54,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf68, (54,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf69, (54,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf70, (54, 1, 3, 3), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf71, (54, 54, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf72, (54,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf73, (54,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf74, (54,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf75, (54,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf76, (96, 1, 3, 3), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf77, (54, 96, 1, 1), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf78, (54,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf79, (54,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf80, (54,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf81, (54,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf82, (54, 1, 3, 3), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf83, (54, 54, 1, 1), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf84, (54,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf85, (54,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf86, (54,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf87, (54,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf88, (54, 54, 1, 1), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf89, (54,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf90, (54,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf91, (54,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf92, (54,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf93, (54, 96, 1, 1), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf94, (54, 96, 1, 1), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf95, (108,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf96, (108,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf97, (108,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf98, (108,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 116640, device=device(type='cuda', index=0))
    reader.tensor(buf99, (108, 270, 1, 1), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf100, (108,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf101, (108,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf102, (108,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf103, (108,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 10800, device=device(type='cuda', index=0))
    reader.tensor(buf104, (108, 1, 5, 5), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf105, (108, 108, 1, 1), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf106, (108,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf107, (108,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf108, (108,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf109, (108,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 10800, device=device(type='cuda', index=0))
    reader.tensor(buf110, (108, 1, 5, 5), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf111, (108, 108, 1, 1), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf112, (108,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf113, (108,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf114, (108,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf115, (108,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 21168, device=device(type='cuda', index=0))
    reader.tensor(buf116, (108, 1, 7, 7), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf117, (108, 108, 1, 1), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf118, (108,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf119, (108,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf120, (108,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf121, (108,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 21168, device=device(type='cuda', index=0))
    reader.tensor(buf122, (108, 1, 7, 7), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf123, (108, 108, 1, 1), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf124, (108,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf125, (108,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf126, (108,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf127, (108,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 10800, device=device(type='cuda', index=0))
    reader.tensor(buf128, (108, 1, 5, 5), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf129, (108, 108, 1, 1), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf130, (108,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf131, (108,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf132, (108,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf133, (108,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 10800, device=device(type='cuda', index=0))
    reader.tensor(buf134, (108, 1, 5, 5), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf135, (108, 108, 1, 1), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf136, (108,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf137, (108,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf138, (108,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf139, (108,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf140, (108, 1, 3, 3), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf141, (108, 108, 1, 1), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf142, (108,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf143, (108,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf144, (108,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf145, (108,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf146, (108, 1, 3, 3), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf147, (108, 108, 1, 1), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf148, (108,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf149, (108,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf150, (108,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf151, (108,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf152, (108, 1, 3, 3), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf153, (108, 108, 1, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf154, (108,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf155, (108,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf156, (108,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf157, (108,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf158, (108, 1, 3, 3), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf159, (108, 108, 1, 1), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf160, (108,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf161, (108,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf162, (108,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf163, (108,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf164, (108, 1, 3, 3), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf165, (108, 108, 1, 1), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf166, (108,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf167, (108,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf168, (108,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf169, (108,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf170, (108, 1, 3, 3), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf171, (108, 108, 1, 1), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf172, (108,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf173, (108,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf174, (108,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf175, (108,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf176, (108, 108, 1, 1), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf177, (108,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf178, (108,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf179, (108,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf180, (108,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 116640, device=device(type='cuda', index=0))
    reader.tensor(buf181, (108, 270, 1, 1), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 116640, device=device(type='cuda', index=0))
    reader.tensor(buf182, (108, 270, 1, 1), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf183, (216,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf184, (216,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf185, (216,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf186, (216,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 466560, device=device(type='cuda', index=0))
    reader.tensor(buf187, (216, 540, 1, 1), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf188, (216,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf189, (216,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf190, (216,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf191, (216,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf192, (216, 1, 5, 5), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf193, (216, 216, 1, 1), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf194, (216,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf195, (216,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf196, (216,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf197, (216,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf198, (216, 1, 5, 5), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf199, (216, 216, 1, 1), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf200, (216,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf201, (216,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf202, (216,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf203, (216,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf204, (216, 1, 7, 7), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf205, (216, 216, 1, 1), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf206, (216,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf207, (216,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf208, (216,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf209, (216,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf210, (216, 1, 7, 7), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf211, (216, 216, 1, 1), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf212, (216,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf213, (216,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf214, (216,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf215, (216,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf216, (216, 1, 5, 5), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf217, (216, 216, 1, 1), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf218, (216,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf219, (216,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf220, (216,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf221, (216,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf222, (216, 1, 5, 5), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf223, (216, 216, 1, 1), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf224, (216,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf225, (216,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf226, (216,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf227, (216,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf228, (216, 1, 3, 3), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf229, (216, 216, 1, 1), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf230, (216,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf231, (216,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf232, (216,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf233, (216,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf234, (216, 1, 3, 3), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf235, (216, 216, 1, 1), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf236, (216,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf237, (216,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf238, (216,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf239, (216,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf240, (216, 1, 3, 3), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf241, (216, 216, 1, 1), is_leaf=True)  # arg241_1
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
    buf247 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf247, (216, 216, 1, 1), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf248, (216,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf249, (216,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf250, (216,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf251, (216,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf252, (216, 1, 3, 3), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf253, (216, 216, 1, 1), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf254, (216,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf255, (216,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf256, (216,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf257, (216,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf258, (216, 1, 3, 3), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf259, (216, 216, 1, 1), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf260, (216,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf261, (216,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf262, (216,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf263, (216,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 466560, device=device(type='cuda', index=0))
    reader.tensor(buf264, (216, 540, 1, 1), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf265, (216,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf266, (216,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf267, (216,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf268, (216,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 933120, device=device(type='cuda', index=0))
    reader.tensor(buf269, (216, 1080, 1, 1), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf270, (216,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf271, (216,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf272, (216,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf273, (216,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf274, (216, 1, 5, 5), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf275, (216, 216, 1, 1), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf276, (216,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf277, (216,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf278, (216,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf279, (216,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf280, (216, 1, 5, 5), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf281, (216, 216, 1, 1), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf282, (216,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf283, (216,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf284, (216,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf285, (216,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf286, (216, 1, 7, 7), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf287, (216, 216, 1, 1), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf288, (216,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf289, (216,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf290, (216,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf291, (216,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf292, (216, 1, 7, 7), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf293, (216, 216, 1, 1), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf294, (216,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf295, (216,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf296, (216,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf297, (216,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf298, (216, 1, 5, 5), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf299, (216, 216, 1, 1), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf300, (216,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf301, (216,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf302, (216,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf303, (216,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf304, (216, 1, 5, 5), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf305, (216, 216, 1, 1), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf306, (216,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf307, (216,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf308, (216,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf309, (216,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf310, (216, 1, 3, 3), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf311, (216, 216, 1, 1), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf312, (216,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf313, (216,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf314, (216,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf315, (216,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf316, (216, 1, 3, 3), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf317, (216, 216, 1, 1), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf318, (216,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf319, (216,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf320, (216,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf321, (216,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf322, (216, 1, 3, 3), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf323, (216, 216, 1, 1), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf324, (216,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf325, (216,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf326, (216,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf327, (216,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf328, (216, 1, 3, 3), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf329, (216, 216, 1, 1), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf330, (216,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf331, (216,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf332, (216,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf333, (216,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf334, (216, 1, 3, 3), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf335, (216, 216, 1, 1), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf336, (216,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf337, (216,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf338, (216,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf339, (216,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf340, (216, 1, 3, 3), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf341, (216, 216, 1, 1), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf342, (216,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf343, (216,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf344, (216,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf345, (216,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 933120, device=device(type='cuda', index=0))
    reader.tensor(buf346, (216, 1080, 1, 1), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf347, (216,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf348, (216,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf349, (216,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf350, (216,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 933120, device=device(type='cuda', index=0))
    reader.tensor(buf351, (216, 1080, 1, 1), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf352, (216,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf353, (216,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf354, (216,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf355, (216,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf356, (216, 1, 5, 5), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf357, (216, 216, 1, 1), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf358, (216,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf359, (216,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf360, (216,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf361, (216,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf362, (216, 1, 5, 5), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf363, (216, 216, 1, 1), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf364, (216,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf365, (216,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf366, (216,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf367, (216,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf368, (216, 1, 7, 7), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf369, (216, 216, 1, 1), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf370, (216,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf371, (216,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf372, (216,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf373, (216,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf374, (216, 1, 7, 7), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf375, (216, 216, 1, 1), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf376, (216,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf377, (216,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf378, (216,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf379, (216,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf380, (216, 1, 5, 5), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf381, (216, 216, 1, 1), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf382, (216,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf383, (216,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf384, (216,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf385, (216,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf386, (216, 1, 5, 5), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf387, (216, 216, 1, 1), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf388, (216,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf389, (216,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf390, (216,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf391, (216,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf392, (216, 1, 3, 3), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf393, (216, 216, 1, 1), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf394, (216,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf395, (216,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf396, (216,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf397, (216,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf398, (216, 1, 3, 3), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf399, (216, 216, 1, 1), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf400, (216,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf401, (216,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf402, (216,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf403, (216,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf404, (216, 1, 3, 3), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf405, (216, 216, 1, 1), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf406, (216,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf407, (216,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf408, (216,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf409, (216,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf410, (216, 1, 3, 3), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf411, (216, 216, 1, 1), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf412, (216,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf413, (216,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf414, (216,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf415, (216,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf416, (216, 1, 3, 3), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf417, (216, 216, 1, 1), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf418, (216,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf419, (216,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf420, (216,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf421, (216,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf422, (216, 1, 3, 3), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf423, (216, 216, 1, 1), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf424, (216,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf425, (216,), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf426, (216,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf427, (216,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 933120, device=device(type='cuda', index=0))
    reader.tensor(buf428, (216, 1080, 1, 1), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf429, (216,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf430, (216,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf431, (216,), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf432, (216,), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 933120, device=device(type='cuda', index=0))
    reader.tensor(buf433, (216, 1080, 1, 1), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf434, (216,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf435, (216,), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf436, (216,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf437, (216,), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf438, (216, 1, 5, 5), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf439, (216, 216, 1, 1), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf440, (216,), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf441, (216,), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf442, (216,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf443, (216,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf444, (216, 1, 5, 5), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf445, (216, 216, 1, 1), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf446, (216,), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf447, (216,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf448, (216,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf449, (216,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf450, (216, 1, 7, 7), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf451, (216, 216, 1, 1), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf452, (216,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf453, (216,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf454, (216,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf455, (216,), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 42336, device=device(type='cuda', index=0))
    reader.tensor(buf456, (216, 1, 7, 7), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf457, (216, 216, 1, 1), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf458, (216,), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf459, (216,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf460, (216,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf461, (216,), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf462, (216, 1, 5, 5), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf463, (216, 216, 1, 1), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf464, (216,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf465, (216,), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf466, (216,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf467, (216,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 21600, device=device(type='cuda', index=0))
    reader.tensor(buf468, (216, 1, 5, 5), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf469, (216, 216, 1, 1), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf470, (216,), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf471, (216,), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf472, (216,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf473, (216,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf474, (216, 1, 3, 3), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf475, (216, 216, 1, 1), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf476, (216,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf477, (216,), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf478, (216,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf479, (216,), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf480, (216, 1, 3, 3), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf481, (216, 216, 1, 1), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf482, (216,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf483, (216,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf484, (216,), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf485, (216,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf486, (216, 1, 3, 3), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf487, (216, 216, 1, 1), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf488, (216,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf489, (216,), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf490, (216,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf491, (216,), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf492, (216, 1, 3, 3), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf493, (216, 216, 1, 1), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf494, (216,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf495, (216,), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf496, (216,), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf497, (216,), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf498, (216, 1, 3, 3), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf499, (216, 216, 1, 1), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf500, (216,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf501, (216,), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf502, (216,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf503, (216,), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 7776, device=device(type='cuda', index=0))
    reader.tensor(buf504, (216, 1, 3, 3), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf505, (216, 216, 1, 1), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf506, (216,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf507, (216,), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf508, (216,), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf509, (216,), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 1866240, device=device(type='cuda', index=0))
    reader.tensor(buf510, (432, 1080, 1, 1), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf511, (432,), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf512, (432,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf513, (432,), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf514, (432,), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 1866240, device=device(type='cuda', index=0))
    reader.tensor(buf515, (432, 1080, 1, 1), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf516, (432,), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf517, (432,), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf518, (432,), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf519, (432,), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf520, (432, 1, 5, 5), is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf521, (432, 432, 1, 1), is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf522, (432,), is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf523, (432,), is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf524, (432,), is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf525, (432,), is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf526, (432, 1, 5, 5), is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf527, (432, 432, 1, 1), is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf528, (432,), is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf529, (432,), is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf530, (432,), is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf531, (432,), is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf532, (432, 1, 7, 7), is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf533, (432, 432, 1, 1), is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf534, (432,), is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf535, (432,), is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf536, (432,), is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf537, (432,), is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf538, (432, 1, 7, 7), is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf539, (432, 432, 1, 1), is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf540, (432,), is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf541, (432,), is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf542, (432,), is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf543, (432,), is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf544, (432, 1, 5, 5), is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf545, (432, 432, 1, 1), is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf546, (432,), is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf547, (432,), is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf548, (432,), is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf549, (432,), is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf550, (432, 1, 5, 5), is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf551, (432, 432, 1, 1), is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf552, (432,), is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf553, (432,), is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf554, (432,), is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf555, (432,), is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf556, (432, 1, 3, 3), is_leaf=True)  # arg556_1
    buf557 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf557, (432, 432, 1, 1), is_leaf=True)  # arg557_1
    buf558 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf558, (432,), is_leaf=True)  # arg558_1
    buf559 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf559, (432,), is_leaf=True)  # arg559_1
    buf560 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf560, (432,), is_leaf=True)  # arg560_1
    buf561 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf561, (432,), is_leaf=True)  # arg561_1
    buf562 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf562, (432, 1, 3, 3), is_leaf=True)  # arg562_1
    buf563 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf563, (432, 432, 1, 1), is_leaf=True)  # arg563_1
    buf564 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf564, (432,), is_leaf=True)  # arg564_1
    buf565 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf565, (432,), is_leaf=True)  # arg565_1
    buf566 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf566, (432,), is_leaf=True)  # arg566_1
    buf567 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf567, (432,), is_leaf=True)  # arg567_1
    buf568 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf568, (432, 1, 3, 3), is_leaf=True)  # arg568_1
    buf569 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf569, (432, 432, 1, 1), is_leaf=True)  # arg569_1
    buf570 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf570, (432,), is_leaf=True)  # arg570_1
    buf571 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf571, (432,), is_leaf=True)  # arg571_1
    buf572 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf572, (432,), is_leaf=True)  # arg572_1
    buf573 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf573, (432,), is_leaf=True)  # arg573_1
    buf574 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf574, (432, 1, 3, 3), is_leaf=True)  # arg574_1
    buf575 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf575, (432, 432, 1, 1), is_leaf=True)  # arg575_1
    buf576 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf576, (432,), is_leaf=True)  # arg576_1
    buf577 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf577, (432,), is_leaf=True)  # arg577_1
    buf578 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf578, (432,), is_leaf=True)  # arg578_1
    buf579 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf579, (432,), is_leaf=True)  # arg579_1
    buf580 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf580, (432, 1, 3, 3), is_leaf=True)  # arg580_1
    buf581 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf581, (432, 432, 1, 1), is_leaf=True)  # arg581_1
    buf582 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf582, (432,), is_leaf=True)  # arg582_1
    buf583 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf583, (432,), is_leaf=True)  # arg583_1
    buf584 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf584, (432,), is_leaf=True)  # arg584_1
    buf585 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf585, (432,), is_leaf=True)  # arg585_1
    buf586 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf586, (432, 1, 3, 3), is_leaf=True)  # arg586_1
    buf587 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf587, (432, 432, 1, 1), is_leaf=True)  # arg587_1
    buf588 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf588, (432,), is_leaf=True)  # arg588_1
    buf589 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf589, (432,), is_leaf=True)  # arg589_1
    buf590 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf590, (432,), is_leaf=True)  # arg590_1
    buf591 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf591, (432,), is_leaf=True)  # arg591_1
    buf592 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf592, (432, 432, 1, 1), is_leaf=True)  # arg592_1
    buf593 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf593, (432,), is_leaf=True)  # arg593_1
    buf594 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf594, (432,), is_leaf=True)  # arg594_1
    buf595 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf595, (432,), is_leaf=True)  # arg595_1
    buf596 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf596, (432,), is_leaf=True)  # arg596_1
    buf597 = reader.storage(None, 933120, device=device(type='cuda', index=0))
    reader.tensor(buf597, (216, 1080, 1, 1), is_leaf=True)  # arg597_1
    buf598 = reader.storage(None, 933120, device=device(type='cuda', index=0))
    reader.tensor(buf598, (216, 1080, 1, 1), is_leaf=True)  # arg598_1
    buf599 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf599, (432,), is_leaf=True)  # arg599_1
    buf600 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf600, (432,), is_leaf=True)  # arg600_1
    buf601 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf601, (432,), is_leaf=True)  # arg601_1
    buf602 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf602, (432,), is_leaf=True)  # arg602_1
    buf603 = reader.storage(None, 3732480, device=device(type='cuda', index=0))
    reader.tensor(buf603, (432, 2160, 1, 1), is_leaf=True)  # arg603_1
    buf604 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf604, (432,), is_leaf=True)  # arg604_1
    buf605 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf605, (432,), is_leaf=True)  # arg605_1
    buf606 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf606, (432,), is_leaf=True)  # arg606_1
    buf607 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf607, (432,), is_leaf=True)  # arg607_1
    buf608 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf608, (432, 1, 5, 5), is_leaf=True)  # arg608_1
    buf609 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf609, (432, 432, 1, 1), is_leaf=True)  # arg609_1
    buf610 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf610, (432,), is_leaf=True)  # arg610_1
    buf611 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf611, (432,), is_leaf=True)  # arg611_1
    buf612 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf612, (432,), is_leaf=True)  # arg612_1
    buf613 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf613, (432,), is_leaf=True)  # arg613_1
    buf614 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf614, (432, 1, 5, 5), is_leaf=True)  # arg614_1
    buf615 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf615, (432, 432, 1, 1), is_leaf=True)  # arg615_1
    buf616 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf616, (432,), is_leaf=True)  # arg616_1
    buf617 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf617, (432,), is_leaf=True)  # arg617_1
    buf618 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf618, (432,), is_leaf=True)  # arg618_1
    buf619 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf619, (432,), is_leaf=True)  # arg619_1
    buf620 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf620, (432, 1, 7, 7), is_leaf=True)  # arg620_1
    buf621 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf621, (432, 432, 1, 1), is_leaf=True)  # arg621_1
    buf622 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf622, (432,), is_leaf=True)  # arg622_1
    buf623 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf623, (432,), is_leaf=True)  # arg623_1
    buf624 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf624, (432,), is_leaf=True)  # arg624_1
    buf625 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf625, (432,), is_leaf=True)  # arg625_1
    buf626 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf626, (432, 1, 7, 7), is_leaf=True)  # arg626_1
    buf627 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf627, (432, 432, 1, 1), is_leaf=True)  # arg627_1
    buf628 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf628, (432,), is_leaf=True)  # arg628_1
    buf629 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf629, (432,), is_leaf=True)  # arg629_1
    buf630 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf630, (432,), is_leaf=True)  # arg630_1
    buf631 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf631, (432,), is_leaf=True)  # arg631_1
    buf632 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf632, (432, 1, 5, 5), is_leaf=True)  # arg632_1
    buf633 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf633, (432, 432, 1, 1), is_leaf=True)  # arg633_1
    buf634 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf634, (432,), is_leaf=True)  # arg634_1
    buf635 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf635, (432,), is_leaf=True)  # arg635_1
    buf636 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf636, (432,), is_leaf=True)  # arg636_1
    buf637 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf637, (432,), is_leaf=True)  # arg637_1
    buf638 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf638, (432, 1, 5, 5), is_leaf=True)  # arg638_1
    buf639 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf639, (432, 432, 1, 1), is_leaf=True)  # arg639_1
    buf640 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf640, (432,), is_leaf=True)  # arg640_1
    buf641 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf641, (432,), is_leaf=True)  # arg641_1
    buf642 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf642, (432,), is_leaf=True)  # arg642_1
    buf643 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf643, (432,), is_leaf=True)  # arg643_1
    buf644 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf644, (432, 1, 3, 3), is_leaf=True)  # arg644_1
    buf645 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf645, (432, 432, 1, 1), is_leaf=True)  # arg645_1
    buf646 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf646, (432,), is_leaf=True)  # arg646_1
    buf647 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf647, (432,), is_leaf=True)  # arg647_1
    buf648 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf648, (432,), is_leaf=True)  # arg648_1
    buf649 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf649, (432,), is_leaf=True)  # arg649_1
    buf650 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf650, (432, 1, 3, 3), is_leaf=True)  # arg650_1
    buf651 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf651, (432, 432, 1, 1), is_leaf=True)  # arg651_1
    buf652 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf652, (432,), is_leaf=True)  # arg652_1
    buf653 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf653, (432,), is_leaf=True)  # arg653_1
    buf654 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf654, (432,), is_leaf=True)  # arg654_1
    buf655 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf655, (432,), is_leaf=True)  # arg655_1
    buf656 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf656, (432, 1, 3, 3), is_leaf=True)  # arg656_1
    buf657 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf657, (432, 432, 1, 1), is_leaf=True)  # arg657_1
    buf658 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf658, (432,), is_leaf=True)  # arg658_1
    buf659 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf659, (432,), is_leaf=True)  # arg659_1
    buf660 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf660, (432,), is_leaf=True)  # arg660_1
    buf661 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf661, (432,), is_leaf=True)  # arg661_1
    buf662 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf662, (432, 1, 3, 3), is_leaf=True)  # arg662_1
    buf663 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf663, (432, 432, 1, 1), is_leaf=True)  # arg663_1
    buf664 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf664, (432,), is_leaf=True)  # arg664_1
    buf665 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf665, (432,), is_leaf=True)  # arg665_1
    buf666 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf666, (432,), is_leaf=True)  # arg666_1
    buf667 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf667, (432,), is_leaf=True)  # arg667_1
    buf668 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf668, (432, 1, 3, 3), is_leaf=True)  # arg668_1
    buf669 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf669, (432, 432, 1, 1), is_leaf=True)  # arg669_1
    buf670 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf670, (432,), is_leaf=True)  # arg670_1
    buf671 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf671, (432,), is_leaf=True)  # arg671_1
    buf672 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf672, (432,), is_leaf=True)  # arg672_1
    buf673 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf673, (432,), is_leaf=True)  # arg673_1
    buf674 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf674, (432, 1, 3, 3), is_leaf=True)  # arg674_1
    buf675 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf675, (432, 432, 1, 1), is_leaf=True)  # arg675_1
    buf676 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf676, (432,), is_leaf=True)  # arg676_1
    buf677 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf677, (432,), is_leaf=True)  # arg677_1
    buf678 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf678, (432,), is_leaf=True)  # arg678_1
    buf679 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf679, (432,), is_leaf=True)  # arg679_1
    buf680 = reader.storage(None, 3732480, device=device(type='cuda', index=0))
    reader.tensor(buf680, (432, 2160, 1, 1), is_leaf=True)  # arg680_1
    buf681 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf681, (432,), is_leaf=True)  # arg681_1
    buf682 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf682, (432,), is_leaf=True)  # arg682_1
    buf683 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf683, (432,), is_leaf=True)  # arg683_1
    buf684 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf684, (432,), is_leaf=True)  # arg684_1
    buf685 = reader.storage(None, 3732480, device=device(type='cuda', index=0))
    reader.tensor(buf685, (432, 2160, 1, 1), is_leaf=True)  # arg685_1
    buf686 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf686, (432,), is_leaf=True)  # arg686_1
    buf687 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf687, (432,), is_leaf=True)  # arg687_1
    buf688 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf688, (432,), is_leaf=True)  # arg688_1
    buf689 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf689, (432,), is_leaf=True)  # arg689_1
    buf690 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf690, (432, 1, 5, 5), is_leaf=True)  # arg690_1
    buf691 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf691, (432, 432, 1, 1), is_leaf=True)  # arg691_1
    buf692 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf692, (432,), is_leaf=True)  # arg692_1
    buf693 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf693, (432,), is_leaf=True)  # arg693_1
    buf694 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf694, (432,), is_leaf=True)  # arg694_1
    buf695 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf695, (432,), is_leaf=True)  # arg695_1
    buf696 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf696, (432, 1, 5, 5), is_leaf=True)  # arg696_1
    buf697 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf697, (432, 432, 1, 1), is_leaf=True)  # arg697_1
    buf698 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf698, (432,), is_leaf=True)  # arg698_1
    buf699 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf699, (432,), is_leaf=True)  # arg699_1
    buf700 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf700, (432,), is_leaf=True)  # arg700_1
    buf701 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf701, (432,), is_leaf=True)  # arg701_1
    buf702 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf702, (432, 1, 7, 7), is_leaf=True)  # arg702_1
    buf703 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf703, (432, 432, 1, 1), is_leaf=True)  # arg703_1
    buf704 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf704, (432,), is_leaf=True)  # arg704_1
    buf705 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf705, (432,), is_leaf=True)  # arg705_1
    buf706 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf706, (432,), is_leaf=True)  # arg706_1
    buf707 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf707, (432,), is_leaf=True)  # arg707_1
    buf708 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf708, (432, 1, 7, 7), is_leaf=True)  # arg708_1
    buf709 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf709, (432, 432, 1, 1), is_leaf=True)  # arg709_1
    buf710 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf710, (432,), is_leaf=True)  # arg710_1
    buf711 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf711, (432,), is_leaf=True)  # arg711_1
    buf712 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf712, (432,), is_leaf=True)  # arg712_1
    buf713 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf713, (432,), is_leaf=True)  # arg713_1
    buf714 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf714, (432, 1, 5, 5), is_leaf=True)  # arg714_1
    buf715 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf715, (432, 432, 1, 1), is_leaf=True)  # arg715_1
    buf716 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf716, (432,), is_leaf=True)  # arg716_1
    buf717 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf717, (432,), is_leaf=True)  # arg717_1
    buf718 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf718, (432,), is_leaf=True)  # arg718_1
    buf719 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf719, (432,), is_leaf=True)  # arg719_1
    buf720 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf720, (432, 1, 5, 5), is_leaf=True)  # arg720_1
    buf721 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf721, (432, 432, 1, 1), is_leaf=True)  # arg721_1
    buf722 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf722, (432,), is_leaf=True)  # arg722_1
    buf723 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf723, (432,), is_leaf=True)  # arg723_1
    buf724 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf724, (432,), is_leaf=True)  # arg724_1
    buf725 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf725, (432,), is_leaf=True)  # arg725_1
    buf726 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf726, (432, 1, 3, 3), is_leaf=True)  # arg726_1
    buf727 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf727, (432, 432, 1, 1), is_leaf=True)  # arg727_1
    buf728 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf728, (432,), is_leaf=True)  # arg728_1
    buf729 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf729, (432,), is_leaf=True)  # arg729_1
    buf730 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf730, (432,), is_leaf=True)  # arg730_1
    buf731 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf731, (432,), is_leaf=True)  # arg731_1
    buf732 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf732, (432, 1, 3, 3), is_leaf=True)  # arg732_1
    buf733 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf733, (432, 432, 1, 1), is_leaf=True)  # arg733_1
    buf734 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf734, (432,), is_leaf=True)  # arg734_1
    buf735 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf735, (432,), is_leaf=True)  # arg735_1
    buf736 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf736, (432,), is_leaf=True)  # arg736_1
    buf737 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf737, (432,), is_leaf=True)  # arg737_1
    buf738 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf738, (432, 1, 3, 3), is_leaf=True)  # arg738_1
    buf739 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf739, (432, 432, 1, 1), is_leaf=True)  # arg739_1
    buf740 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf740, (432,), is_leaf=True)  # arg740_1
    buf741 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf741, (432,), is_leaf=True)  # arg741_1
    buf742 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf742, (432,), is_leaf=True)  # arg742_1
    buf743 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf743, (432,), is_leaf=True)  # arg743_1
    buf744 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf744, (432, 1, 3, 3), is_leaf=True)  # arg744_1
    buf745 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf745, (432, 432, 1, 1), is_leaf=True)  # arg745_1
    buf746 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf746, (432,), is_leaf=True)  # arg746_1
    buf747 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf747, (432,), is_leaf=True)  # arg747_1
    buf748 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf748, (432,), is_leaf=True)  # arg748_1
    buf749 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf749, (432,), is_leaf=True)  # arg749_1
    buf750 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf750, (432, 1, 3, 3), is_leaf=True)  # arg750_1
    buf751 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf751, (432, 432, 1, 1), is_leaf=True)  # arg751_1
    buf752 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf752, (432,), is_leaf=True)  # arg752_1
    buf753 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf753, (432,), is_leaf=True)  # arg753_1
    buf754 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf754, (432,), is_leaf=True)  # arg754_1
    buf755 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf755, (432,), is_leaf=True)  # arg755_1
    buf756 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf756, (432, 1, 3, 3), is_leaf=True)  # arg756_1
    buf757 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf757, (432, 432, 1, 1), is_leaf=True)  # arg757_1
    buf758 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf758, (432,), is_leaf=True)  # arg758_1
    buf759 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf759, (432,), is_leaf=True)  # arg759_1
    buf760 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf760, (432,), is_leaf=True)  # arg760_1
    buf761 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf761, (432,), is_leaf=True)  # arg761_1
    buf762 = reader.storage(None, 3732480, device=device(type='cuda', index=0))
    reader.tensor(buf762, (432, 2160, 1, 1), is_leaf=True)  # arg762_1
    buf763 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf763, (432,), is_leaf=True)  # arg763_1
    buf764 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf764, (432,), is_leaf=True)  # arg764_1
    buf765 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf765, (432,), is_leaf=True)  # arg765_1
    buf766 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf766, (432,), is_leaf=True)  # arg766_1
    buf767 = reader.storage(None, 3732480, device=device(type='cuda', index=0))
    reader.tensor(buf767, (432, 2160, 1, 1), is_leaf=True)  # arg767_1
    buf768 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf768, (432,), is_leaf=True)  # arg768_1
    buf769 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf769, (432,), is_leaf=True)  # arg769_1
    buf770 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf770, (432,), is_leaf=True)  # arg770_1
    buf771 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf771, (432,), is_leaf=True)  # arg771_1
    buf772 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf772, (432, 1, 5, 5), is_leaf=True)  # arg772_1
    buf773 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf773, (432, 432, 1, 1), is_leaf=True)  # arg773_1
    buf774 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf774, (432,), is_leaf=True)  # arg774_1
    buf775 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf775, (432,), is_leaf=True)  # arg775_1
    buf776 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf776, (432,), is_leaf=True)  # arg776_1
    buf777 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf777, (432,), is_leaf=True)  # arg777_1
    buf778 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf778, (432, 1, 5, 5), is_leaf=True)  # arg778_1
    buf779 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf779, (432, 432, 1, 1), is_leaf=True)  # arg779_1
    buf780 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf780, (432,), is_leaf=True)  # arg780_1
    buf781 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf781, (432,), is_leaf=True)  # arg781_1
    buf782 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf782, (432,), is_leaf=True)  # arg782_1
    buf783 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf783, (432,), is_leaf=True)  # arg783_1
    buf784 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf784, (432, 1, 7, 7), is_leaf=True)  # arg784_1
    buf785 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf785, (432, 432, 1, 1), is_leaf=True)  # arg785_1
    buf786 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf786, (432,), is_leaf=True)  # arg786_1
    buf787 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf787, (432,), is_leaf=True)  # arg787_1
    buf788 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf788, (432,), is_leaf=True)  # arg788_1
    buf789 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf789, (432,), is_leaf=True)  # arg789_1
    buf790 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf790, (432, 1, 7, 7), is_leaf=True)  # arg790_1
    buf791 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf791, (432, 432, 1, 1), is_leaf=True)  # arg791_1
    buf792 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf792, (432,), is_leaf=True)  # arg792_1
    buf793 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf793, (432,), is_leaf=True)  # arg793_1
    buf794 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf794, (432,), is_leaf=True)  # arg794_1
    buf795 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf795, (432,), is_leaf=True)  # arg795_1
    buf796 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf796, (432, 1, 5, 5), is_leaf=True)  # arg796_1
    buf797 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf797, (432, 432, 1, 1), is_leaf=True)  # arg797_1
    buf798 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf798, (432,), is_leaf=True)  # arg798_1
    buf799 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf799, (432,), is_leaf=True)  # arg799_1
    buf800 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf800, (432,), is_leaf=True)  # arg800_1
    buf801 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf801, (432,), is_leaf=True)  # arg801_1
    buf802 = reader.storage(None, 43200, device=device(type='cuda', index=0))
    reader.tensor(buf802, (432, 1, 5, 5), is_leaf=True)  # arg802_1
    buf803 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf803, (432, 432, 1, 1), is_leaf=True)  # arg803_1
    buf804 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf804, (432,), is_leaf=True)  # arg804_1
    buf805 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf805, (432,), is_leaf=True)  # arg805_1
    buf806 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf806, (432,), is_leaf=True)  # arg806_1
    buf807 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf807, (432,), is_leaf=True)  # arg807_1
    buf808 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf808, (432, 1, 3, 3), is_leaf=True)  # arg808_1
    buf809 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf809, (432, 432, 1, 1), is_leaf=True)  # arg809_1
    buf810 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf810, (432,), is_leaf=True)  # arg810_1
    buf811 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf811, (432,), is_leaf=True)  # arg811_1
    buf812 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf812, (432,), is_leaf=True)  # arg812_1
    buf813 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf813, (432,), is_leaf=True)  # arg813_1
    buf814 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf814, (432, 1, 3, 3), is_leaf=True)  # arg814_1
    buf815 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf815, (432, 432, 1, 1), is_leaf=True)  # arg815_1
    buf816 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf816, (432,), is_leaf=True)  # arg816_1
    buf817 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf817, (432,), is_leaf=True)  # arg817_1
    buf818 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf818, (432,), is_leaf=True)  # arg818_1
    buf819 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf819, (432,), is_leaf=True)  # arg819_1
    buf820 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf820, (432, 1, 3, 3), is_leaf=True)  # arg820_1
    buf821 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf821, (432, 432, 1, 1), is_leaf=True)  # arg821_1
    buf822 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf822, (432,), is_leaf=True)  # arg822_1
    buf823 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf823, (432,), is_leaf=True)  # arg823_1
    buf824 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf824, (432,), is_leaf=True)  # arg824_1
    buf825 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf825, (432,), is_leaf=True)  # arg825_1
    buf826 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf826, (432, 1, 3, 3), is_leaf=True)  # arg826_1
    buf827 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf827, (432, 432, 1, 1), is_leaf=True)  # arg827_1
    buf828 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf828, (432,), is_leaf=True)  # arg828_1
    buf829 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf829, (432,), is_leaf=True)  # arg829_1
    buf830 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf830, (432,), is_leaf=True)  # arg830_1
    buf831 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf831, (432,), is_leaf=True)  # arg831_1
    buf832 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf832, (432, 1, 3, 3), is_leaf=True)  # arg832_1
    buf833 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf833, (432, 432, 1, 1), is_leaf=True)  # arg833_1
    buf834 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf834, (432,), is_leaf=True)  # arg834_1
    buf835 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf835, (432,), is_leaf=True)  # arg835_1
    buf836 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf836, (432,), is_leaf=True)  # arg836_1
    buf837 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf837, (432,), is_leaf=True)  # arg837_1
    buf838 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf838, (432, 1, 3, 3), is_leaf=True)  # arg838_1
    buf839 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf839, (432, 432, 1, 1), is_leaf=True)  # arg839_1
    buf840 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf840, (432,), is_leaf=True)  # arg840_1
    buf841 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf841, (432,), is_leaf=True)  # arg841_1
    buf842 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf842, (432,), is_leaf=True)  # arg842_1
    buf843 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf843, (432,), is_leaf=True)  # arg843_1
    buf844 = reader.storage(None, 7464960, device=device(type='cuda', index=0))
    reader.tensor(buf844, (864, 2160, 1, 1), is_leaf=True)  # arg844_1
    buf845 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf845, (864,), is_leaf=True)  # arg845_1
    buf846 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf846, (864,), is_leaf=True)  # arg846_1
    buf847 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf847, (864,), is_leaf=True)  # arg847_1
    buf848 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf848, (864,), is_leaf=True)  # arg848_1
    buf849 = reader.storage(None, 7464960, device=device(type='cuda', index=0))
    reader.tensor(buf849, (864, 2160, 1, 1), is_leaf=True)  # arg849_1
    buf850 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf850, (864,), is_leaf=True)  # arg850_1
    buf851 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf851, (864,), is_leaf=True)  # arg851_1
    buf852 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf852, (864,), is_leaf=True)  # arg852_1
    buf853 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf853, (864,), is_leaf=True)  # arg853_1
    buf854 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf854, (864, 1, 5, 5), is_leaf=True)  # arg854_1
    buf855 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf855, (864, 864, 1, 1), is_leaf=True)  # arg855_1
    buf856 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf856, (864,), is_leaf=True)  # arg856_1
    buf857 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf857, (864,), is_leaf=True)  # arg857_1
    buf858 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf858, (864,), is_leaf=True)  # arg858_1
    buf859 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf859, (864,), is_leaf=True)  # arg859_1
    buf860 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf860, (864, 1, 5, 5), is_leaf=True)  # arg860_1
    buf861 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf861, (864, 864, 1, 1), is_leaf=True)  # arg861_1
    buf862 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf862, (864,), is_leaf=True)  # arg862_1
    buf863 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf863, (864,), is_leaf=True)  # arg863_1
    buf864 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf864, (864,), is_leaf=True)  # arg864_1
    buf865 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf865, (864,), is_leaf=True)  # arg865_1
    buf866 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf866, (864, 1, 7, 7), is_leaf=True)  # arg866_1
    buf867 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf867, (864, 864, 1, 1), is_leaf=True)  # arg867_1
    buf868 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf868, (864,), is_leaf=True)  # arg868_1
    buf869 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf869, (864,), is_leaf=True)  # arg869_1
    buf870 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf870, (864,), is_leaf=True)  # arg870_1
    buf871 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf871, (864,), is_leaf=True)  # arg871_1
    buf872 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf872, (864, 1, 7, 7), is_leaf=True)  # arg872_1
    buf873 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf873, (864, 864, 1, 1), is_leaf=True)  # arg873_1
    buf874 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf874, (864,), is_leaf=True)  # arg874_1
    buf875 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf875, (864,), is_leaf=True)  # arg875_1
    buf876 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf876, (864,), is_leaf=True)  # arg876_1
    buf877 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf877, (864,), is_leaf=True)  # arg877_1
    buf878 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf878, (864, 1, 5, 5), is_leaf=True)  # arg878_1
    buf879 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf879, (864, 864, 1, 1), is_leaf=True)  # arg879_1
    buf880 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf880, (864,), is_leaf=True)  # arg880_1
    buf881 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf881, (864,), is_leaf=True)  # arg881_1
    buf882 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf882, (864,), is_leaf=True)  # arg882_1
    buf883 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf883, (864,), is_leaf=True)  # arg883_1
    buf884 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf884, (864, 1, 5, 5), is_leaf=True)  # arg884_1
    buf885 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf885, (864, 864, 1, 1), is_leaf=True)  # arg885_1
    buf886 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf886, (864,), is_leaf=True)  # arg886_1
    buf887 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf887, (864,), is_leaf=True)  # arg887_1
    buf888 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf888, (864,), is_leaf=True)  # arg888_1
    buf889 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf889, (864,), is_leaf=True)  # arg889_1
    buf890 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf890, (864, 1, 3, 3), is_leaf=True)  # arg890_1
    buf891 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf891, (864, 864, 1, 1), is_leaf=True)  # arg891_1
    buf892 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf892, (864,), is_leaf=True)  # arg892_1
    buf893 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf893, (864,), is_leaf=True)  # arg893_1
    buf894 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf894, (864,), is_leaf=True)  # arg894_1
    buf895 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf895, (864,), is_leaf=True)  # arg895_1
    buf896 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf896, (864, 1, 3, 3), is_leaf=True)  # arg896_1
    buf897 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf897, (864, 864, 1, 1), is_leaf=True)  # arg897_1
    buf898 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf898, (864,), is_leaf=True)  # arg898_1
    buf899 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf899, (864,), is_leaf=True)  # arg899_1
    buf900 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf900, (864,), is_leaf=True)  # arg900_1
    buf901 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf901, (864,), is_leaf=True)  # arg901_1
    buf902 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf902, (864, 1, 3, 3), is_leaf=True)  # arg902_1
    buf903 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf903, (864, 864, 1, 1), is_leaf=True)  # arg903_1
    buf904 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf904, (864,), is_leaf=True)  # arg904_1
    buf905 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf905, (864,), is_leaf=True)  # arg905_1
    buf906 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf906, (864,), is_leaf=True)  # arg906_1
    buf907 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf907, (864,), is_leaf=True)  # arg907_1
    buf908 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf908, (864, 1, 3, 3), is_leaf=True)  # arg908_1
    buf909 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf909, (864, 864, 1, 1), is_leaf=True)  # arg909_1
    buf910 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf910, (864,), is_leaf=True)  # arg910_1
    buf911 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf911, (864,), is_leaf=True)  # arg911_1
    buf912 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf912, (864,), is_leaf=True)  # arg912_1
    buf913 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf913, (864,), is_leaf=True)  # arg913_1
    buf914 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf914, (864, 1, 3, 3), is_leaf=True)  # arg914_1
    buf915 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf915, (864, 864, 1, 1), is_leaf=True)  # arg915_1
    buf916 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf916, (864,), is_leaf=True)  # arg916_1
    buf917 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf917, (864,), is_leaf=True)  # arg917_1
    buf918 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf918, (864,), is_leaf=True)  # arg918_1
    buf919 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf919, (864,), is_leaf=True)  # arg919_1
    buf920 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf920, (864, 1, 3, 3), is_leaf=True)  # arg920_1
    buf921 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf921, (864, 864, 1, 1), is_leaf=True)  # arg921_1
    buf922 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf922, (864,), is_leaf=True)  # arg922_1
    buf923 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf923, (864,), is_leaf=True)  # arg923_1
    buf924 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf924, (864,), is_leaf=True)  # arg924_1
    buf925 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf925, (864,), is_leaf=True)  # arg925_1
    buf926 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf926, (864, 864, 1, 1), is_leaf=True)  # arg926_1
    buf927 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf927, (864,), is_leaf=True)  # arg927_1
    buf928 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf928, (864,), is_leaf=True)  # arg928_1
    buf929 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf929, (864,), is_leaf=True)  # arg929_1
    buf930 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf930, (864,), is_leaf=True)  # arg930_1
    buf931 = reader.storage(None, 3732480, device=device(type='cuda', index=0))
    reader.tensor(buf931, (432, 2160, 1, 1), is_leaf=True)  # arg931_1
    buf932 = reader.storage(None, 3732480, device=device(type='cuda', index=0))
    reader.tensor(buf932, (432, 2160, 1, 1), is_leaf=True)  # arg932_1
    buf933 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf933, (864,), is_leaf=True)  # arg933_1
    buf934 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf934, (864,), is_leaf=True)  # arg934_1
    buf935 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf935, (864,), is_leaf=True)  # arg935_1
    buf936 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf936, (864,), is_leaf=True)  # arg936_1
    buf937 = reader.storage(None, 14929920, device=device(type='cuda', index=0))
    reader.tensor(buf937, (864, 4320, 1, 1), is_leaf=True)  # arg937_1
    buf938 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf938, (864,), is_leaf=True)  # arg938_1
    buf939 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf939, (864,), is_leaf=True)  # arg939_1
    buf940 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf940, (864,), is_leaf=True)  # arg940_1
    buf941 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf941, (864,), is_leaf=True)  # arg941_1
    buf942 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf942, (864, 1, 5, 5), is_leaf=True)  # arg942_1
    buf943 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf943, (864, 864, 1, 1), is_leaf=True)  # arg943_1
    buf944 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf944, (864,), is_leaf=True)  # arg944_1
    buf945 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf945, (864,), is_leaf=True)  # arg945_1
    buf946 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf946, (864,), is_leaf=True)  # arg946_1
    buf947 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf947, (864,), is_leaf=True)  # arg947_1
    buf948 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf948, (864, 1, 5, 5), is_leaf=True)  # arg948_1
    buf949 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf949, (864, 864, 1, 1), is_leaf=True)  # arg949_1
    buf950 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf950, (864,), is_leaf=True)  # arg950_1
    buf951 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf951, (864,), is_leaf=True)  # arg951_1
    buf952 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf952, (864,), is_leaf=True)  # arg952_1
    buf953 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf953, (864,), is_leaf=True)  # arg953_1
    buf954 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf954, (864, 1, 7, 7), is_leaf=True)  # arg954_1
    buf955 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf955, (864, 864, 1, 1), is_leaf=True)  # arg955_1
    buf956 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf956, (864,), is_leaf=True)  # arg956_1
    buf957 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf957, (864,), is_leaf=True)  # arg957_1
    buf958 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf958, (864,), is_leaf=True)  # arg958_1
    buf959 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf959, (864,), is_leaf=True)  # arg959_1
    buf960 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf960, (864, 1, 7, 7), is_leaf=True)  # arg960_1
    buf961 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf961, (864, 864, 1, 1), is_leaf=True)  # arg961_1
    buf962 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf962, (864,), is_leaf=True)  # arg962_1
    buf963 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf963, (864,), is_leaf=True)  # arg963_1
    buf964 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf964, (864,), is_leaf=True)  # arg964_1
    buf965 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf965, (864,), is_leaf=True)  # arg965_1
    buf966 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf966, (864, 1, 5, 5), is_leaf=True)  # arg966_1
    buf967 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf967, (864, 864, 1, 1), is_leaf=True)  # arg967_1
    buf968 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf968, (864,), is_leaf=True)  # arg968_1
    buf969 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf969, (864,), is_leaf=True)  # arg969_1
    buf970 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf970, (864,), is_leaf=True)  # arg970_1
    buf971 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf971, (864,), is_leaf=True)  # arg971_1
    buf972 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf972, (864, 1, 5, 5), is_leaf=True)  # arg972_1
    buf973 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf973, (864, 864, 1, 1), is_leaf=True)  # arg973_1
    buf974 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf974, (864,), is_leaf=True)  # arg974_1
    buf975 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf975, (864,), is_leaf=True)  # arg975_1
    buf976 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf976, (864,), is_leaf=True)  # arg976_1
    buf977 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf977, (864,), is_leaf=True)  # arg977_1
    buf978 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf978, (864, 1, 3, 3), is_leaf=True)  # arg978_1
    buf979 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf979, (864, 864, 1, 1), is_leaf=True)  # arg979_1
    buf980 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf980, (864,), is_leaf=True)  # arg980_1
    buf981 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf981, (864,), is_leaf=True)  # arg981_1
    buf982 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf982, (864,), is_leaf=True)  # arg982_1
    buf983 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf983, (864,), is_leaf=True)  # arg983_1
    buf984 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf984, (864, 1, 3, 3), is_leaf=True)  # arg984_1
    buf985 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf985, (864, 864, 1, 1), is_leaf=True)  # arg985_1
    buf986 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf986, (864,), is_leaf=True)  # arg986_1
    buf987 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf987, (864,), is_leaf=True)  # arg987_1
    buf988 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf988, (864,), is_leaf=True)  # arg988_1
    buf989 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf989, (864,), is_leaf=True)  # arg989_1
    buf990 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf990, (864, 1, 3, 3), is_leaf=True)  # arg990_1
    buf991 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf991, (864, 864, 1, 1), is_leaf=True)  # arg991_1
    buf992 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf992, (864,), is_leaf=True)  # arg992_1
    buf993 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf993, (864,), is_leaf=True)  # arg993_1
    buf994 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf994, (864,), is_leaf=True)  # arg994_1
    buf995 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf995, (864,), is_leaf=True)  # arg995_1
    buf996 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf996, (864, 1, 3, 3), is_leaf=True)  # arg996_1
    buf997 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf997, (864, 864, 1, 1), is_leaf=True)  # arg997_1
    buf998 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf998, (864,), is_leaf=True)  # arg998_1
    buf999 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf999, (864,), is_leaf=True)  # arg999_1
    buf1000 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1000, (864,), is_leaf=True)  # arg1000_1
    buf1001 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1001, (864,), is_leaf=True)  # arg1001_1
    buf1002 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1002, (864, 1, 3, 3), is_leaf=True)  # arg1002_1
    buf1003 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1003, (864, 864, 1, 1), is_leaf=True)  # arg1003_1
    buf1004 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1004, (864,), is_leaf=True)  # arg1004_1
    buf1005 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1005, (864,), is_leaf=True)  # arg1005_1
    buf1006 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1006, (864,), is_leaf=True)  # arg1006_1
    buf1007 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1007, (864,), is_leaf=True)  # arg1007_1
    buf1008 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1008, (864, 1, 3, 3), is_leaf=True)  # arg1008_1
    buf1009 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1009, (864, 864, 1, 1), is_leaf=True)  # arg1009_1
    buf1010 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1010, (864,), is_leaf=True)  # arg1010_1
    buf1011 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1011, (864,), is_leaf=True)  # arg1011_1
    buf1012 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1012, (864,), is_leaf=True)  # arg1012_1
    buf1013 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1013, (864,), is_leaf=True)  # arg1013_1
    buf1014 = reader.storage(None, 14929920, device=device(type='cuda', index=0))
    reader.tensor(buf1014, (864, 4320, 1, 1), is_leaf=True)  # arg1014_1
    buf1015 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1015, (864,), is_leaf=True)  # arg1015_1
    buf1016 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1016, (864,), is_leaf=True)  # arg1016_1
    buf1017 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1017, (864,), is_leaf=True)  # arg1017_1
    buf1018 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1018, (864,), is_leaf=True)  # arg1018_1
    buf1019 = reader.storage(None, 14929920, device=device(type='cuda', index=0))
    reader.tensor(buf1019, (864, 4320, 1, 1), is_leaf=True)  # arg1019_1
    buf1020 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1020, (864,), is_leaf=True)  # arg1020_1
    buf1021 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1021, (864,), is_leaf=True)  # arg1021_1
    buf1022 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1022, (864,), is_leaf=True)  # arg1022_1
    buf1023 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1023, (864,), is_leaf=True)  # arg1023_1
    buf1024 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1024, (864, 1, 5, 5), is_leaf=True)  # arg1024_1
    buf1025 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1025, (864, 864, 1, 1), is_leaf=True)  # arg1025_1
    buf1026 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1026, (864,), is_leaf=True)  # arg1026_1
    buf1027 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1027, (864,), is_leaf=True)  # arg1027_1
    buf1028 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1028, (864,), is_leaf=True)  # arg1028_1
    buf1029 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1029, (864,), is_leaf=True)  # arg1029_1
    buf1030 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1030, (864, 1, 5, 5), is_leaf=True)  # arg1030_1
    buf1031 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1031, (864, 864, 1, 1), is_leaf=True)  # arg1031_1
    buf1032 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1032, (864,), is_leaf=True)  # arg1032_1
    buf1033 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1033, (864,), is_leaf=True)  # arg1033_1
    buf1034 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1034, (864,), is_leaf=True)  # arg1034_1
    buf1035 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1035, (864,), is_leaf=True)  # arg1035_1
    buf1036 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf1036, (864, 1, 7, 7), is_leaf=True)  # arg1036_1
    buf1037 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1037, (864, 864, 1, 1), is_leaf=True)  # arg1037_1
    buf1038 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1038, (864,), is_leaf=True)  # arg1038_1
    buf1039 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1039, (864,), is_leaf=True)  # arg1039_1
    buf1040 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1040, (864,), is_leaf=True)  # arg1040_1
    buf1041 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1041, (864,), is_leaf=True)  # arg1041_1
    buf1042 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf1042, (864, 1, 7, 7), is_leaf=True)  # arg1042_1
    buf1043 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1043, (864, 864, 1, 1), is_leaf=True)  # arg1043_1
    buf1044 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1044, (864,), is_leaf=True)  # arg1044_1
    buf1045 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1045, (864,), is_leaf=True)  # arg1045_1
    buf1046 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1046, (864,), is_leaf=True)  # arg1046_1
    buf1047 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1047, (864,), is_leaf=True)  # arg1047_1
    buf1048 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1048, (864, 1, 5, 5), is_leaf=True)  # arg1048_1
    buf1049 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1049, (864, 864, 1, 1), is_leaf=True)  # arg1049_1
    buf1050 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1050, (864,), is_leaf=True)  # arg1050_1
    buf1051 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1051, (864,), is_leaf=True)  # arg1051_1
    buf1052 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1052, (864,), is_leaf=True)  # arg1052_1
    buf1053 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1053, (864,), is_leaf=True)  # arg1053_1
    buf1054 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1054, (864, 1, 5, 5), is_leaf=True)  # arg1054_1
    buf1055 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1055, (864, 864, 1, 1), is_leaf=True)  # arg1055_1
    buf1056 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1056, (864,), is_leaf=True)  # arg1056_1
    buf1057 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1057, (864,), is_leaf=True)  # arg1057_1
    buf1058 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1058, (864,), is_leaf=True)  # arg1058_1
    buf1059 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1059, (864,), is_leaf=True)  # arg1059_1
    buf1060 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1060, (864, 1, 3, 3), is_leaf=True)  # arg1060_1
    buf1061 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1061, (864, 864, 1, 1), is_leaf=True)  # arg1061_1
    buf1062 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1062, (864,), is_leaf=True)  # arg1062_1
    buf1063 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1063, (864,), is_leaf=True)  # arg1063_1
    buf1064 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1064, (864,), is_leaf=True)  # arg1064_1
    buf1065 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1065, (864,), is_leaf=True)  # arg1065_1
    buf1066 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1066, (864, 1, 3, 3), is_leaf=True)  # arg1066_1
    buf1067 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1067, (864, 864, 1, 1), is_leaf=True)  # arg1067_1
    buf1068 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1068, (864,), is_leaf=True)  # arg1068_1
    buf1069 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1069, (864,), is_leaf=True)  # arg1069_1
    buf1070 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1070, (864,), is_leaf=True)  # arg1070_1
    buf1071 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1071, (864,), is_leaf=True)  # arg1071_1
    buf1072 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1072, (864, 1, 3, 3), is_leaf=True)  # arg1072_1
    buf1073 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1073, (864, 864, 1, 1), is_leaf=True)  # arg1073_1
    buf1074 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1074, (864,), is_leaf=True)  # arg1074_1
    buf1075 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1075, (864,), is_leaf=True)  # arg1075_1
    buf1076 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1076, (864,), is_leaf=True)  # arg1076_1
    buf1077 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1077, (864,), is_leaf=True)  # arg1077_1
    buf1078 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1078, (864, 1, 3, 3), is_leaf=True)  # arg1078_1
    buf1079 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1079, (864, 864, 1, 1), is_leaf=True)  # arg1079_1
    buf1080 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1080, (864,), is_leaf=True)  # arg1080_1
    buf1081 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1081, (864,), is_leaf=True)  # arg1081_1
    buf1082 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1082, (864,), is_leaf=True)  # arg1082_1
    buf1083 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1083, (864,), is_leaf=True)  # arg1083_1
    buf1084 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1084, (864, 1, 3, 3), is_leaf=True)  # arg1084_1
    buf1085 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1085, (864, 864, 1, 1), is_leaf=True)  # arg1085_1
    buf1086 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1086, (864,), is_leaf=True)  # arg1086_1
    buf1087 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1087, (864,), is_leaf=True)  # arg1087_1
    buf1088 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1088, (864,), is_leaf=True)  # arg1088_1
    buf1089 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1089, (864,), is_leaf=True)  # arg1089_1
    buf1090 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1090, (864, 1, 3, 3), is_leaf=True)  # arg1090_1
    buf1091 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1091, (864, 864, 1, 1), is_leaf=True)  # arg1091_1
    buf1092 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1092, (864,), is_leaf=True)  # arg1092_1
    buf1093 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1093, (864,), is_leaf=True)  # arg1093_1
    buf1094 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1094, (864,), is_leaf=True)  # arg1094_1
    buf1095 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1095, (864,), is_leaf=True)  # arg1095_1
    buf1096 = reader.storage(None, 14929920, device=device(type='cuda', index=0))
    reader.tensor(buf1096, (864, 4320, 1, 1), is_leaf=True)  # arg1096_1
    buf1097 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1097, (864,), is_leaf=True)  # arg1097_1
    buf1098 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1098, (864,), is_leaf=True)  # arg1098_1
    buf1099 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1099, (864,), is_leaf=True)  # arg1099_1
    buf1100 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1100, (864,), is_leaf=True)  # arg1100_1
    buf1101 = reader.storage(None, 14929920, device=device(type='cuda', index=0))
    reader.tensor(buf1101, (864, 4320, 1, 1), is_leaf=True)  # arg1101_1
    buf1102 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1102, (864,), is_leaf=True)  # arg1102_1
    buf1103 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1103, (864,), is_leaf=True)  # arg1103_1
    buf1104 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1104, (864,), is_leaf=True)  # arg1104_1
    buf1105 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1105, (864,), is_leaf=True)  # arg1105_1
    buf1106 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1106, (864, 1, 5, 5), is_leaf=True)  # arg1106_1
    buf1107 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1107, (864, 864, 1, 1), is_leaf=True)  # arg1107_1
    buf1108 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1108, (864,), is_leaf=True)  # arg1108_1
    buf1109 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1109, (864,), is_leaf=True)  # arg1109_1
    buf1110 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1110, (864,), is_leaf=True)  # arg1110_1
    buf1111 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1111, (864,), is_leaf=True)  # arg1111_1
    buf1112 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1112, (864, 1, 5, 5), is_leaf=True)  # arg1112_1
    buf1113 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1113, (864, 864, 1, 1), is_leaf=True)  # arg1113_1
    buf1114 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1114, (864,), is_leaf=True)  # arg1114_1
    buf1115 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1115, (864,), is_leaf=True)  # arg1115_1
    buf1116 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1116, (864,), is_leaf=True)  # arg1116_1
    buf1117 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1117, (864,), is_leaf=True)  # arg1117_1
    buf1118 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf1118, (864, 1, 7, 7), is_leaf=True)  # arg1118_1
    buf1119 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1119, (864, 864, 1, 1), is_leaf=True)  # arg1119_1
    buf1120 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1120, (864,), is_leaf=True)  # arg1120_1
    buf1121 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1121, (864,), is_leaf=True)  # arg1121_1
    buf1122 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1122, (864,), is_leaf=True)  # arg1122_1
    buf1123 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1123, (864,), is_leaf=True)  # arg1123_1
    buf1124 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf1124, (864, 1, 7, 7), is_leaf=True)  # arg1124_1
    buf1125 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1125, (864, 864, 1, 1), is_leaf=True)  # arg1125_1
    buf1126 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1126, (864,), is_leaf=True)  # arg1126_1
    buf1127 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1127, (864,), is_leaf=True)  # arg1127_1
    buf1128 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1128, (864,), is_leaf=True)  # arg1128_1
    buf1129 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1129, (864,), is_leaf=True)  # arg1129_1
    buf1130 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1130, (864, 1, 5, 5), is_leaf=True)  # arg1130_1
    buf1131 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1131, (864, 864, 1, 1), is_leaf=True)  # arg1131_1
    buf1132 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1132, (864,), is_leaf=True)  # arg1132_1
    buf1133 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1133, (864,), is_leaf=True)  # arg1133_1
    buf1134 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1134, (864,), is_leaf=True)  # arg1134_1
    buf1135 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1135, (864,), is_leaf=True)  # arg1135_1
    buf1136 = reader.storage(None, 86400, device=device(type='cuda', index=0))
    reader.tensor(buf1136, (864, 1, 5, 5), is_leaf=True)  # arg1136_1
    buf1137 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1137, (864, 864, 1, 1), is_leaf=True)  # arg1137_1
    buf1138 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1138, (864,), is_leaf=True)  # arg1138_1
    buf1139 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1139, (864,), is_leaf=True)  # arg1139_1
    buf1140 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1140, (864,), is_leaf=True)  # arg1140_1
    buf1141 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1141, (864,), is_leaf=True)  # arg1141_1
    buf1142 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1142, (864, 1, 3, 3), is_leaf=True)  # arg1142_1
    buf1143 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1143, (864, 864, 1, 1), is_leaf=True)  # arg1143_1
    buf1144 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1144, (864,), is_leaf=True)  # arg1144_1
    buf1145 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1145, (864,), is_leaf=True)  # arg1145_1
    buf1146 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1146, (864,), is_leaf=True)  # arg1146_1
    buf1147 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1147, (864,), is_leaf=True)  # arg1147_1
    buf1148 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1148, (864, 1, 3, 3), is_leaf=True)  # arg1148_1
    buf1149 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1149, (864, 864, 1, 1), is_leaf=True)  # arg1149_1
    buf1150 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1150, (864,), is_leaf=True)  # arg1150_1
    buf1151 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1151, (864,), is_leaf=True)  # arg1151_1
    buf1152 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1152, (864,), is_leaf=True)  # arg1152_1
    buf1153 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1153, (864,), is_leaf=True)  # arg1153_1
    buf1154 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1154, (864, 1, 3, 3), is_leaf=True)  # arg1154_1
    buf1155 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1155, (864, 864, 1, 1), is_leaf=True)  # arg1155_1
    buf1156 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1156, (864,), is_leaf=True)  # arg1156_1
    buf1157 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1157, (864,), is_leaf=True)  # arg1157_1
    buf1158 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1158, (864,), is_leaf=True)  # arg1158_1
    buf1159 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1159, (864,), is_leaf=True)  # arg1159_1
    buf1160 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1160, (864, 1, 3, 3), is_leaf=True)  # arg1160_1
    buf1161 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1161, (864, 864, 1, 1), is_leaf=True)  # arg1161_1
    buf1162 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1162, (864,), is_leaf=True)  # arg1162_1
    buf1163 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1163, (864,), is_leaf=True)  # arg1163_1
    buf1164 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1164, (864,), is_leaf=True)  # arg1164_1
    buf1165 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1165, (864,), is_leaf=True)  # arg1165_1
    buf1166 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1166, (864, 1, 3, 3), is_leaf=True)  # arg1166_1
    buf1167 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1167, (864, 864, 1, 1), is_leaf=True)  # arg1167_1
    buf1168 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1168, (864,), is_leaf=True)  # arg1168_1
    buf1169 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1169, (864,), is_leaf=True)  # arg1169_1
    buf1170 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1170, (864,), is_leaf=True)  # arg1170_1
    buf1171 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1171, (864,), is_leaf=True)  # arg1171_1
    buf1172 = reader.storage(None, 31104, device=device(type='cuda', index=0))
    reader.tensor(buf1172, (864, 1, 3, 3), is_leaf=True)  # arg1172_1
    buf1173 = reader.storage(None, 2985984, device=device(type='cuda', index=0))
    reader.tensor(buf1173, (864, 864, 1, 1), is_leaf=True)  # arg1173_1
    buf1174 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1174, (864,), is_leaf=True)  # arg1174_1
    buf1175 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1175, (864,), is_leaf=True)  # arg1175_1
    buf1176 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1176, (864,), is_leaf=True)  # arg1176_1
    buf1177 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf1177, (864,), is_leaf=True)  # arg1177_1
    buf1178 = reader.storage(None, 17280000, device=device(type='cuda', index=0))
    reader.tensor(buf1178, (1000, 4320), is_leaf=True)  # arg1178_1
    buf1179 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf1179, (1000,), is_leaf=True)  # arg1179_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)