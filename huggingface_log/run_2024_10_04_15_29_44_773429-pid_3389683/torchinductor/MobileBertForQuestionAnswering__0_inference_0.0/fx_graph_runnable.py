
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

torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.0a0+git5380214
# torch cuda version: 12.1
# torch git version: 5380214107813f63c7c59f477487d5447085b45a


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA H100 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor(1000))
        self.register_buffer('_tensor_constant1', tensor(1000))
        self.register_buffer('_tensor_constant2', tensor(1000))
        self.register_buffer('_tensor_constant3', tensor(1000))
        self.register_buffer('_tensor_constant4', tensor(1000))
        self.register_buffer('_tensor_constant5', tensor(1000))
        self.register_buffer('_tensor_constant6', tensor(1000))
        self.register_buffer('_tensor_constant7', tensor(1000))
        self.register_buffer('_tensor_constant8', tensor(1000))
        self.register_buffer('_tensor_constant9', tensor(1000))
        self.register_buffer('_tensor_constant10', tensor(1000))
        self.register_buffer('_tensor_constant11', tensor(1000))
        self.register_buffer('_tensor_constant12', tensor(1000))
        self.register_buffer('_tensor_constant13', tensor(1000))
        self.register_buffer('_tensor_constant14', tensor(1000))
        self.register_buffer('_tensor_constant15', tensor(1000))
        self.register_buffer('_tensor_constant16', tensor(1000))
        self.register_buffer('_tensor_constant17', tensor(1000))
        self.register_buffer('_tensor_constant18', tensor(1000))
        self.register_buffer('_tensor_constant19', tensor(1000))
        self.register_buffer('_tensor_constant20', tensor(1000))
        self.register_buffer('_tensor_constant21', tensor(1000))
        self.register_buffer('_tensor_constant22', tensor(1000))
        self.register_buffer('_tensor_constant23', tensor(1000))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1):
        full = torch.ops.aten.full.default([128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default_1 = torch.ops.aten.full.default([128, 1, 1, 128], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_1 = None
        slice_4 = torch.ops.aten.slice.Tensor(arg1112_1, 1, 0, 128);  arg1112_1 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        slice_6 = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(slice_6, [0, 0, 0, 1, 0, 0], 0.0);  slice_6 = None
        slice_8 = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)
        constant_pad_nd_1 = torch.ops.aten.constant_pad_nd.default(slice_8, [0, 0, 1, 0, 0, 0], 0.0);  slice_8 = None
        cat = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
        view = torch.ops.aten.view.default(cat, [16384, 384]);  cat = None
        permute = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm = torch.ops.aten.addmm.default(arg5_1, view, permute);  arg5_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [128, 128, 512]);  addmm = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, slice_4);  arg2_1 = slice_4 = None
        embedding_2 = torch.ops.aten.embedding.default(arg3_1, full_default);  arg3_1 = full_default = None
        add = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_1, arg7_1);  add_1 = arg7_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        view_2 = torch.ops.aten.view.default(add_2, [16384, 512])
        permute_1 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg29_1, view_2, permute_1);  arg29_1 = view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [128, 128, 128]);  addmm_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_3, arg31_1);  view_3 = arg31_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, arg30_1);  mul_2 = arg30_1 = None
        view_4 = torch.ops.aten.view.default(add_2, [16384, 512])
        permute_2 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg33_1, view_4, permute_2);  arg33_1 = view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(addmm_2, [128, 128, 128]);  addmm_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(view_5, arg35_1);  view_5 = arg35_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_3, arg34_1);  mul_3 = arg34_1 = None
        view_6 = torch.ops.aten.view.default(add_4, [16384, 128])
        permute_3 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg9_1, view_6, permute_3);  arg9_1 = view_6 = permute_3 = None
        view_7 = torch.ops.aten.view.default(addmm_3, [128, 128, 128]);  addmm_3 = None
        view_8 = torch.ops.aten.view.default(add_4, [16384, 128]);  add_4 = None
        permute_4 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg11_1, view_8, permute_4);  arg11_1 = view_8 = permute_4 = None
        view_9 = torch.ops.aten.view.default(addmm_4, [128, 128, 128]);  addmm_4 = None
        view_10 = torch.ops.aten.view.default(add_2, [16384, 512])
        permute_5 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg13_1, view_10, permute_5);  arg13_1 = view_10 = permute_5 = None
        view_11 = torch.ops.aten.view.default(addmm_5, [128, 128, 128]);  addmm_5 = None
        view_12 = torch.ops.aten.view.default(view_7, [128, 128, 4, 32]);  view_7 = None
        view_13 = torch.ops.aten.view.default(view_9, [128, 128, 4, 32]);  view_9 = None
        view_14 = torch.ops.aten.view.default(view_11, [128, 128, 4, 32]);  view_11 = None
        permute_default_69 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        permute_default_70 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        permute_default_71 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_69, permute_default_70, permute_default_71, None, False, scale = 0.17677669529663687);  permute_default_69 = permute_default_70 = permute_default_71 = None
        getitem_25 = _scaled_dot_product_efficient_attention_default_23[0];  _scaled_dot_product_efficient_attention_default_23 = None
        permute_10 = torch.ops.aten.permute.default(getitem_25, [0, 2, 1, 3]);  getitem_25 = None
        clone_5 = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_21 = torch.ops.aten.view.default(clone_5, [128, 128, 128]);  clone_5 = None
        view_22 = torch.ops.aten.view.default(view_21, [16384, 128]);  view_21 = None
        permute_11 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg15_1, view_22, permute_11);  arg15_1 = view_22 = permute_11 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [128, 128, 128]);  addmm_6 = None
        add_6 = torch.ops.aten.add.Tensor(view_23, add_3);  view_23 = add_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(add_6, arg17_1);  add_6 = arg17_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
        view_24 = torch.ops.aten.view.default(add_7, [16384, 128])
        permute_12 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg37_1, view_24, permute_12);  arg37_1 = view_24 = permute_12 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [128, 128, 512]);  addmm_7 = None
        relu = torch.ops.aten.relu.default(view_25);  view_25 = None
        view_26 = torch.ops.aten.view.default(relu, [16384, 512]);  relu = None
        permute_13 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg39_1, view_26, permute_13);  arg39_1 = view_26 = permute_13 = None
        view_27 = torch.ops.aten.view.default(addmm_8, [128, 128, 128]);  addmm_8 = None
        add_8 = torch.ops.aten.add.Tensor(view_27, add_7);  view_27 = add_7 = None
        mul_5 = torch.ops.aten.mul.Tensor(add_8, arg41_1);  add_8 = arg41_1 = None
        add_9 = torch.ops.aten.add.Tensor(mul_5, arg40_1);  mul_5 = arg40_1 = None
        view_28 = torch.ops.aten.view.default(add_9, [16384, 128])
        permute_14 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg43_1, view_28, permute_14);  arg43_1 = view_28 = permute_14 = None
        view_29 = torch.ops.aten.view.default(addmm_9, [128, 128, 512]);  addmm_9 = None
        relu_1 = torch.ops.aten.relu.default(view_29);  view_29 = None
        view_30 = torch.ops.aten.view.default(relu_1, [16384, 512]);  relu_1 = None
        permute_15 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg45_1, view_30, permute_15);  arg45_1 = view_30 = permute_15 = None
        view_31 = torch.ops.aten.view.default(addmm_10, [128, 128, 128]);  addmm_10 = None
        add_10 = torch.ops.aten.add.Tensor(view_31, add_9);  view_31 = add_9 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_10, arg47_1);  add_10 = arg47_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_6, arg46_1);  mul_6 = arg46_1 = None
        view_32 = torch.ops.aten.view.default(add_11, [16384, 128])
        permute_16 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg49_1, view_32, permute_16);  arg49_1 = view_32 = permute_16 = None
        view_33 = torch.ops.aten.view.default(addmm_11, [128, 128, 512]);  addmm_11 = None
        relu_2 = torch.ops.aten.relu.default(view_33);  view_33 = None
        view_34 = torch.ops.aten.view.default(relu_2, [16384, 512]);  relu_2 = None
        permute_17 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg51_1, view_34, permute_17);  arg51_1 = view_34 = permute_17 = None
        view_35 = torch.ops.aten.view.default(addmm_12, [128, 128, 128]);  addmm_12 = None
        add_12 = torch.ops.aten.add.Tensor(view_35, add_11);  view_35 = add_11 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_12, arg53_1);  add_12 = arg53_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_7, arg52_1);  mul_7 = arg52_1 = None
        view_36 = torch.ops.aten.view.default(add_13, [16384, 128])
        permute_18 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg19_1, view_36, permute_18);  arg19_1 = view_36 = permute_18 = None
        view_37 = torch.ops.aten.view.default(addmm_13, [128, 128, 512]);  addmm_13 = None
        relu_3 = torch.ops.aten.relu.default(view_37);  view_37 = None
        view_38 = torch.ops.aten.view.default(relu_3, [16384, 512]);  relu_3 = None
        permute_19 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg21_1, view_38, permute_19);  arg21_1 = view_38 = permute_19 = None
        view_39 = torch.ops.aten.view.default(addmm_14, [128, 128, 128]);  addmm_14 = None
        add_14 = torch.ops.aten.add.Tensor(view_39, add_13);  view_39 = add_13 = None
        mul_8 = torch.ops.aten.mul.Tensor(add_14, arg23_1);  add_14 = arg23_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_8, arg22_1);  mul_8 = arg22_1 = None
        view_40 = torch.ops.aten.view.default(add_15, [16384, 128]);  add_15 = None
        permute_20 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg25_1, view_40, permute_20);  arg25_1 = view_40 = permute_20 = None
        view_41 = torch.ops.aten.view.default(addmm_15, [128, 128, 512]);  addmm_15 = None
        add_16 = torch.ops.aten.add.Tensor(view_41, add_2);  view_41 = add_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_16, arg27_1);  add_16 = arg27_1 = None
        add_17 = torch.ops.aten.add.Tensor(mul_9, arg26_1);  mul_9 = arg26_1 = None
        view_42 = torch.ops.aten.view.default(add_17, [16384, 512])
        permute_21 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg75_1, view_42, permute_21);  arg75_1 = view_42 = permute_21 = None
        view_43 = torch.ops.aten.view.default(addmm_16, [128, 128, 128]);  addmm_16 = None
        mul_10 = torch.ops.aten.mul.Tensor(view_43, arg77_1);  view_43 = arg77_1 = None
        add_18 = torch.ops.aten.add.Tensor(mul_10, arg76_1);  mul_10 = arg76_1 = None
        view_44 = torch.ops.aten.view.default(add_17, [16384, 512])
        permute_22 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg79_1, view_44, permute_22);  arg79_1 = view_44 = permute_22 = None
        view_45 = torch.ops.aten.view.default(addmm_17, [128, 128, 128]);  addmm_17 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_45, arg81_1);  view_45 = arg81_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_11, arg80_1);  mul_11 = arg80_1 = None
        view_46 = torch.ops.aten.view.default(add_19, [16384, 128])
        permute_23 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg55_1, view_46, permute_23);  arg55_1 = view_46 = permute_23 = None
        view_47 = torch.ops.aten.view.default(addmm_18, [128, 128, 128]);  addmm_18 = None
        view_48 = torch.ops.aten.view.default(add_19, [16384, 128]);  add_19 = None
        permute_24 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg57_1, view_48, permute_24);  arg57_1 = view_48 = permute_24 = None
        view_49 = torch.ops.aten.view.default(addmm_19, [128, 128, 128]);  addmm_19 = None
        view_50 = torch.ops.aten.view.default(add_17, [16384, 512])
        permute_25 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg59_1, view_50, permute_25);  arg59_1 = view_50 = permute_25 = None
        view_51 = torch.ops.aten.view.default(addmm_20, [128, 128, 128]);  addmm_20 = None
        view_52 = torch.ops.aten.view.default(view_47, [128, 128, 4, 32]);  view_47 = None
        view_53 = torch.ops.aten.view.default(view_49, [128, 128, 4, 32]);  view_49 = None
        view_54 = torch.ops.aten.view.default(view_51, [128, 128, 4, 32]);  view_51 = None
        permute_default_66 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        permute_default_67 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        permute_default_68 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_66, permute_default_67, permute_default_68, None, False, scale = 0.17677669529663687);  permute_default_66 = permute_default_67 = permute_default_68 = None
        getitem_24 = _scaled_dot_product_efficient_attention_default_22[0];  _scaled_dot_product_efficient_attention_default_22 = None
        permute_30 = torch.ops.aten.permute.default(getitem_24, [0, 2, 1, 3]);  getitem_24 = None
        clone_11 = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_61 = torch.ops.aten.view.default(clone_11, [128, 128, 128]);  clone_11 = None
        view_62 = torch.ops.aten.view.default(view_61, [16384, 128]);  view_61 = None
        permute_31 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg61_1, view_62, permute_31);  arg61_1 = view_62 = permute_31 = None
        view_63 = torch.ops.aten.view.default(addmm_21, [128, 128, 128]);  addmm_21 = None
        add_21 = torch.ops.aten.add.Tensor(view_63, add_18);  view_63 = add_18 = None
        mul_12 = torch.ops.aten.mul.Tensor(add_21, arg63_1);  add_21 = arg63_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_12, arg62_1);  mul_12 = arg62_1 = None
        view_64 = torch.ops.aten.view.default(add_22, [16384, 128])
        permute_32 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg83_1, view_64, permute_32);  arg83_1 = view_64 = permute_32 = None
        view_65 = torch.ops.aten.view.default(addmm_22, [128, 128, 512]);  addmm_22 = None
        relu_4 = torch.ops.aten.relu.default(view_65);  view_65 = None
        view_66 = torch.ops.aten.view.default(relu_4, [16384, 512]);  relu_4 = None
        permute_33 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg85_1, view_66, permute_33);  arg85_1 = view_66 = permute_33 = None
        view_67 = torch.ops.aten.view.default(addmm_23, [128, 128, 128]);  addmm_23 = None
        add_23 = torch.ops.aten.add.Tensor(view_67, add_22);  view_67 = add_22 = None
        mul_13 = torch.ops.aten.mul.Tensor(add_23, arg87_1);  add_23 = arg87_1 = None
        add_24 = torch.ops.aten.add.Tensor(mul_13, arg86_1);  mul_13 = arg86_1 = None
        view_68 = torch.ops.aten.view.default(add_24, [16384, 128])
        permute_34 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg89_1, view_68, permute_34);  arg89_1 = view_68 = permute_34 = None
        view_69 = torch.ops.aten.view.default(addmm_24, [128, 128, 512]);  addmm_24 = None
        relu_5 = torch.ops.aten.relu.default(view_69);  view_69 = None
        view_70 = torch.ops.aten.view.default(relu_5, [16384, 512]);  relu_5 = None
        permute_35 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg91_1, view_70, permute_35);  arg91_1 = view_70 = permute_35 = None
        view_71 = torch.ops.aten.view.default(addmm_25, [128, 128, 128]);  addmm_25 = None
        add_25 = torch.ops.aten.add.Tensor(view_71, add_24);  view_71 = add_24 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_25, arg93_1);  add_25 = arg93_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_14, arg92_1);  mul_14 = arg92_1 = None
        view_72 = torch.ops.aten.view.default(add_26, [16384, 128])
        permute_36 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg95_1, view_72, permute_36);  arg95_1 = view_72 = permute_36 = None
        view_73 = torch.ops.aten.view.default(addmm_26, [128, 128, 512]);  addmm_26 = None
        relu_6 = torch.ops.aten.relu.default(view_73);  view_73 = None
        view_74 = torch.ops.aten.view.default(relu_6, [16384, 512]);  relu_6 = None
        permute_37 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg97_1, view_74, permute_37);  arg97_1 = view_74 = permute_37 = None
        view_75 = torch.ops.aten.view.default(addmm_27, [128, 128, 128]);  addmm_27 = None
        add_27 = torch.ops.aten.add.Tensor(view_75, add_26);  view_75 = add_26 = None
        mul_15 = torch.ops.aten.mul.Tensor(add_27, arg99_1);  add_27 = arg99_1 = None
        add_28 = torch.ops.aten.add.Tensor(mul_15, arg98_1);  mul_15 = arg98_1 = None
        view_76 = torch.ops.aten.view.default(add_28, [16384, 128])
        permute_38 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg65_1, view_76, permute_38);  arg65_1 = view_76 = permute_38 = None
        view_77 = torch.ops.aten.view.default(addmm_28, [128, 128, 512]);  addmm_28 = None
        relu_7 = torch.ops.aten.relu.default(view_77);  view_77 = None
        view_78 = torch.ops.aten.view.default(relu_7, [16384, 512]);  relu_7 = None
        permute_39 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg67_1, view_78, permute_39);  arg67_1 = view_78 = permute_39 = None
        view_79 = torch.ops.aten.view.default(addmm_29, [128, 128, 128]);  addmm_29 = None
        add_29 = torch.ops.aten.add.Tensor(view_79, add_28);  view_79 = add_28 = None
        mul_16 = torch.ops.aten.mul.Tensor(add_29, arg69_1);  add_29 = arg69_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_16, arg68_1);  mul_16 = arg68_1 = None
        view_80 = torch.ops.aten.view.default(add_30, [16384, 128]);  add_30 = None
        permute_40 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg71_1, view_80, permute_40);  arg71_1 = view_80 = permute_40 = None
        view_81 = torch.ops.aten.view.default(addmm_30, [128, 128, 512]);  addmm_30 = None
        add_31 = torch.ops.aten.add.Tensor(view_81, add_17);  view_81 = add_17 = None
        mul_17 = torch.ops.aten.mul.Tensor(add_31, arg73_1);  add_31 = arg73_1 = None
        add_32 = torch.ops.aten.add.Tensor(mul_17, arg72_1);  mul_17 = arg72_1 = None
        view_82 = torch.ops.aten.view.default(add_32, [16384, 512])
        permute_41 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg121_1, view_82, permute_41);  arg121_1 = view_82 = permute_41 = None
        view_83 = torch.ops.aten.view.default(addmm_31, [128, 128, 128]);  addmm_31 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_83, arg123_1);  view_83 = arg123_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_18, arg122_1);  mul_18 = arg122_1 = None
        view_84 = torch.ops.aten.view.default(add_32, [16384, 512])
        permute_42 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg125_1, view_84, permute_42);  arg125_1 = view_84 = permute_42 = None
        view_85 = torch.ops.aten.view.default(addmm_32, [128, 128, 128]);  addmm_32 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_85, arg127_1);  view_85 = arg127_1 = None
        add_34 = torch.ops.aten.add.Tensor(mul_19, arg126_1);  mul_19 = arg126_1 = None
        view_86 = torch.ops.aten.view.default(add_34, [16384, 128])
        permute_43 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg101_1, view_86, permute_43);  arg101_1 = view_86 = permute_43 = None
        view_87 = torch.ops.aten.view.default(addmm_33, [128, 128, 128]);  addmm_33 = None
        view_88 = torch.ops.aten.view.default(add_34, [16384, 128]);  add_34 = None
        permute_44 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg103_1, view_88, permute_44);  arg103_1 = view_88 = permute_44 = None
        view_89 = torch.ops.aten.view.default(addmm_34, [128, 128, 128]);  addmm_34 = None
        view_90 = torch.ops.aten.view.default(add_32, [16384, 512])
        permute_45 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg105_1, view_90, permute_45);  arg105_1 = view_90 = permute_45 = None
        view_91 = torch.ops.aten.view.default(addmm_35, [128, 128, 128]);  addmm_35 = None
        view_92 = torch.ops.aten.view.default(view_87, [128, 128, 4, 32]);  view_87 = None
        view_93 = torch.ops.aten.view.default(view_89, [128, 128, 4, 32]);  view_89 = None
        view_94 = torch.ops.aten.view.default(view_91, [128, 128, 4, 32]);  view_91 = None
        permute_default_63 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        permute_default_64 = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        permute_default_65 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_63, permute_default_64, permute_default_65, None, False, scale = 0.17677669529663687);  permute_default_63 = permute_default_64 = permute_default_65 = None
        getitem_23 = _scaled_dot_product_efficient_attention_default_21[0];  _scaled_dot_product_efficient_attention_default_21 = None
        permute_50 = torch.ops.aten.permute.default(getitem_23, [0, 2, 1, 3]);  getitem_23 = None
        clone_17 = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_101 = torch.ops.aten.view.default(clone_17, [128, 128, 128]);  clone_17 = None
        view_102 = torch.ops.aten.view.default(view_101, [16384, 128]);  view_101 = None
        permute_51 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg107_1, view_102, permute_51);  arg107_1 = view_102 = permute_51 = None
        view_103 = torch.ops.aten.view.default(addmm_36, [128, 128, 128]);  addmm_36 = None
        add_36 = torch.ops.aten.add.Tensor(view_103, add_33);  view_103 = add_33 = None
        mul_20 = torch.ops.aten.mul.Tensor(add_36, arg109_1);  add_36 = arg109_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_20, arg108_1);  mul_20 = arg108_1 = None
        view_104 = torch.ops.aten.view.default(add_37, [16384, 128])
        permute_52 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg129_1, view_104, permute_52);  arg129_1 = view_104 = permute_52 = None
        view_105 = torch.ops.aten.view.default(addmm_37, [128, 128, 512]);  addmm_37 = None
        relu_8 = torch.ops.aten.relu.default(view_105);  view_105 = None
        view_106 = torch.ops.aten.view.default(relu_8, [16384, 512]);  relu_8 = None
        permute_53 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg131_1, view_106, permute_53);  arg131_1 = view_106 = permute_53 = None
        view_107 = torch.ops.aten.view.default(addmm_38, [128, 128, 128]);  addmm_38 = None
        add_38 = torch.ops.aten.add.Tensor(view_107, add_37);  view_107 = add_37 = None
        mul_21 = torch.ops.aten.mul.Tensor(add_38, arg133_1);  add_38 = arg133_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_21, arg132_1);  mul_21 = arg132_1 = None
        view_108 = torch.ops.aten.view.default(add_39, [16384, 128])
        permute_54 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg135_1, view_108, permute_54);  arg135_1 = view_108 = permute_54 = None
        view_109 = torch.ops.aten.view.default(addmm_39, [128, 128, 512]);  addmm_39 = None
        relu_9 = torch.ops.aten.relu.default(view_109);  view_109 = None
        view_110 = torch.ops.aten.view.default(relu_9, [16384, 512]);  relu_9 = None
        permute_55 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg137_1, view_110, permute_55);  arg137_1 = view_110 = permute_55 = None
        view_111 = torch.ops.aten.view.default(addmm_40, [128, 128, 128]);  addmm_40 = None
        add_40 = torch.ops.aten.add.Tensor(view_111, add_39);  view_111 = add_39 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_40, arg139_1);  add_40 = arg139_1 = None
        add_41 = torch.ops.aten.add.Tensor(mul_22, arg138_1);  mul_22 = arg138_1 = None
        view_112 = torch.ops.aten.view.default(add_41, [16384, 128])
        permute_56 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg141_1, view_112, permute_56);  arg141_1 = view_112 = permute_56 = None
        view_113 = torch.ops.aten.view.default(addmm_41, [128, 128, 512]);  addmm_41 = None
        relu_10 = torch.ops.aten.relu.default(view_113);  view_113 = None
        view_114 = torch.ops.aten.view.default(relu_10, [16384, 512]);  relu_10 = None
        permute_57 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg143_1, view_114, permute_57);  arg143_1 = view_114 = permute_57 = None
        view_115 = torch.ops.aten.view.default(addmm_42, [128, 128, 128]);  addmm_42 = None
        add_42 = torch.ops.aten.add.Tensor(view_115, add_41);  view_115 = add_41 = None
        mul_23 = torch.ops.aten.mul.Tensor(add_42, arg145_1);  add_42 = arg145_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_23, arg144_1);  mul_23 = arg144_1 = None
        view_116 = torch.ops.aten.view.default(add_43, [16384, 128])
        permute_58 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg111_1, view_116, permute_58);  arg111_1 = view_116 = permute_58 = None
        view_117 = torch.ops.aten.view.default(addmm_43, [128, 128, 512]);  addmm_43 = None
        relu_11 = torch.ops.aten.relu.default(view_117);  view_117 = None
        view_118 = torch.ops.aten.view.default(relu_11, [16384, 512]);  relu_11 = None
        permute_59 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg113_1, view_118, permute_59);  arg113_1 = view_118 = permute_59 = None
        view_119 = torch.ops.aten.view.default(addmm_44, [128, 128, 128]);  addmm_44 = None
        add_44 = torch.ops.aten.add.Tensor(view_119, add_43);  view_119 = add_43 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_44, arg115_1);  add_44 = arg115_1 = None
        add_45 = torch.ops.aten.add.Tensor(mul_24, arg114_1);  mul_24 = arg114_1 = None
        view_120 = torch.ops.aten.view.default(add_45, [16384, 128]);  add_45 = None
        permute_60 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg117_1, view_120, permute_60);  arg117_1 = view_120 = permute_60 = None
        view_121 = torch.ops.aten.view.default(addmm_45, [128, 128, 512]);  addmm_45 = None
        add_46 = torch.ops.aten.add.Tensor(view_121, add_32);  view_121 = add_32 = None
        mul_25 = torch.ops.aten.mul.Tensor(add_46, arg119_1);  add_46 = arg119_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_25, arg118_1);  mul_25 = arg118_1 = None
        view_122 = torch.ops.aten.view.default(add_47, [16384, 512])
        permute_61 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg167_1, view_122, permute_61);  arg167_1 = view_122 = permute_61 = None
        view_123 = torch.ops.aten.view.default(addmm_46, [128, 128, 128]);  addmm_46 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_123, arg169_1);  view_123 = arg169_1 = None
        add_48 = torch.ops.aten.add.Tensor(mul_26, arg168_1);  mul_26 = arg168_1 = None
        view_124 = torch.ops.aten.view.default(add_47, [16384, 512])
        permute_62 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg171_1, view_124, permute_62);  arg171_1 = view_124 = permute_62 = None
        view_125 = torch.ops.aten.view.default(addmm_47, [128, 128, 128]);  addmm_47 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_125, arg173_1);  view_125 = arg173_1 = None
        add_49 = torch.ops.aten.add.Tensor(mul_27, arg172_1);  mul_27 = arg172_1 = None
        view_126 = torch.ops.aten.view.default(add_49, [16384, 128])
        permute_63 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg147_1, view_126, permute_63);  arg147_1 = view_126 = permute_63 = None
        view_127 = torch.ops.aten.view.default(addmm_48, [128, 128, 128]);  addmm_48 = None
        view_128 = torch.ops.aten.view.default(add_49, [16384, 128]);  add_49 = None
        permute_64 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg149_1, view_128, permute_64);  arg149_1 = view_128 = permute_64 = None
        view_129 = torch.ops.aten.view.default(addmm_49, [128, 128, 128]);  addmm_49 = None
        view_130 = torch.ops.aten.view.default(add_47, [16384, 512])
        permute_65 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg151_1, view_130, permute_65);  arg151_1 = view_130 = permute_65 = None
        view_131 = torch.ops.aten.view.default(addmm_50, [128, 128, 128]);  addmm_50 = None
        view_132 = torch.ops.aten.view.default(view_127, [128, 128, 4, 32]);  view_127 = None
        view_133 = torch.ops.aten.view.default(view_129, [128, 128, 4, 32]);  view_129 = None
        view_134 = torch.ops.aten.view.default(view_131, [128, 128, 4, 32]);  view_131 = None
        permute_default_60 = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        permute_default_61 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        permute_default_62 = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_60, permute_default_61, permute_default_62, None, False, scale = 0.17677669529663687);  permute_default_60 = permute_default_61 = permute_default_62 = None
        getitem_22 = _scaled_dot_product_efficient_attention_default_20[0];  _scaled_dot_product_efficient_attention_default_20 = None
        permute_70 = torch.ops.aten.permute.default(getitem_22, [0, 2, 1, 3]);  getitem_22 = None
        clone_23 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_141 = torch.ops.aten.view.default(clone_23, [128, 128, 128]);  clone_23 = None
        view_142 = torch.ops.aten.view.default(view_141, [16384, 128]);  view_141 = None
        permute_71 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg153_1, view_142, permute_71);  arg153_1 = view_142 = permute_71 = None
        view_143 = torch.ops.aten.view.default(addmm_51, [128, 128, 128]);  addmm_51 = None
        add_51 = torch.ops.aten.add.Tensor(view_143, add_48);  view_143 = add_48 = None
        mul_28 = torch.ops.aten.mul.Tensor(add_51, arg155_1);  add_51 = arg155_1 = None
        add_52 = torch.ops.aten.add.Tensor(mul_28, arg154_1);  mul_28 = arg154_1 = None
        view_144 = torch.ops.aten.view.default(add_52, [16384, 128])
        permute_72 = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg175_1, view_144, permute_72);  arg175_1 = view_144 = permute_72 = None
        view_145 = torch.ops.aten.view.default(addmm_52, [128, 128, 512]);  addmm_52 = None
        relu_12 = torch.ops.aten.relu.default(view_145);  view_145 = None
        view_146 = torch.ops.aten.view.default(relu_12, [16384, 512]);  relu_12 = None
        permute_73 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg177_1, view_146, permute_73);  arg177_1 = view_146 = permute_73 = None
        view_147 = torch.ops.aten.view.default(addmm_53, [128, 128, 128]);  addmm_53 = None
        add_53 = torch.ops.aten.add.Tensor(view_147, add_52);  view_147 = add_52 = None
        mul_29 = torch.ops.aten.mul.Tensor(add_53, arg179_1);  add_53 = arg179_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_29, arg178_1);  mul_29 = arg178_1 = None
        view_148 = torch.ops.aten.view.default(add_54, [16384, 128])
        permute_74 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg181_1, view_148, permute_74);  arg181_1 = view_148 = permute_74 = None
        view_149 = torch.ops.aten.view.default(addmm_54, [128, 128, 512]);  addmm_54 = None
        relu_13 = torch.ops.aten.relu.default(view_149);  view_149 = None
        view_150 = torch.ops.aten.view.default(relu_13, [16384, 512]);  relu_13 = None
        permute_75 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg183_1, view_150, permute_75);  arg183_1 = view_150 = permute_75 = None
        view_151 = torch.ops.aten.view.default(addmm_55, [128, 128, 128]);  addmm_55 = None
        add_55 = torch.ops.aten.add.Tensor(view_151, add_54);  view_151 = add_54 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_55, arg185_1);  add_55 = arg185_1 = None
        add_56 = torch.ops.aten.add.Tensor(mul_30, arg184_1);  mul_30 = arg184_1 = None
        view_152 = torch.ops.aten.view.default(add_56, [16384, 128])
        permute_76 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg187_1, view_152, permute_76);  arg187_1 = view_152 = permute_76 = None
        view_153 = torch.ops.aten.view.default(addmm_56, [128, 128, 512]);  addmm_56 = None
        relu_14 = torch.ops.aten.relu.default(view_153);  view_153 = None
        view_154 = torch.ops.aten.view.default(relu_14, [16384, 512]);  relu_14 = None
        permute_77 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg189_1, view_154, permute_77);  arg189_1 = view_154 = permute_77 = None
        view_155 = torch.ops.aten.view.default(addmm_57, [128, 128, 128]);  addmm_57 = None
        add_57 = torch.ops.aten.add.Tensor(view_155, add_56);  view_155 = add_56 = None
        mul_31 = torch.ops.aten.mul.Tensor(add_57, arg191_1);  add_57 = arg191_1 = None
        add_58 = torch.ops.aten.add.Tensor(mul_31, arg190_1);  mul_31 = arg190_1 = None
        view_156 = torch.ops.aten.view.default(add_58, [16384, 128])
        permute_78 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg157_1, view_156, permute_78);  arg157_1 = view_156 = permute_78 = None
        view_157 = torch.ops.aten.view.default(addmm_58, [128, 128, 512]);  addmm_58 = None
        relu_15 = torch.ops.aten.relu.default(view_157);  view_157 = None
        view_158 = torch.ops.aten.view.default(relu_15, [16384, 512]);  relu_15 = None
        permute_79 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg159_1, view_158, permute_79);  arg159_1 = view_158 = permute_79 = None
        view_159 = torch.ops.aten.view.default(addmm_59, [128, 128, 128]);  addmm_59 = None
        add_59 = torch.ops.aten.add.Tensor(view_159, add_58);  view_159 = add_58 = None
        mul_32 = torch.ops.aten.mul.Tensor(add_59, arg161_1);  add_59 = arg161_1 = None
        add_60 = torch.ops.aten.add.Tensor(mul_32, arg160_1);  mul_32 = arg160_1 = None
        view_160 = torch.ops.aten.view.default(add_60, [16384, 128]);  add_60 = None
        permute_80 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg163_1, view_160, permute_80);  arg163_1 = view_160 = permute_80 = None
        view_161 = torch.ops.aten.view.default(addmm_60, [128, 128, 512]);  addmm_60 = None
        add_61 = torch.ops.aten.add.Tensor(view_161, add_47);  view_161 = add_47 = None
        mul_33 = torch.ops.aten.mul.Tensor(add_61, arg165_1);  add_61 = arg165_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_33, arg164_1);  mul_33 = arg164_1 = None
        view_162 = torch.ops.aten.view.default(add_62, [16384, 512])
        permute_81 = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg213_1, view_162, permute_81);  arg213_1 = view_162 = permute_81 = None
        view_163 = torch.ops.aten.view.default(addmm_61, [128, 128, 128]);  addmm_61 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_163, arg215_1);  view_163 = arg215_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_34, arg214_1);  mul_34 = arg214_1 = None
        view_164 = torch.ops.aten.view.default(add_62, [16384, 512])
        permute_82 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg217_1, view_164, permute_82);  arg217_1 = view_164 = permute_82 = None
        view_165 = torch.ops.aten.view.default(addmm_62, [128, 128, 128]);  addmm_62 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_165, arg219_1);  view_165 = arg219_1 = None
        add_64 = torch.ops.aten.add.Tensor(mul_35, arg218_1);  mul_35 = arg218_1 = None
        view_166 = torch.ops.aten.view.default(add_64, [16384, 128])
        permute_83 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg193_1, view_166, permute_83);  arg193_1 = view_166 = permute_83 = None
        view_167 = torch.ops.aten.view.default(addmm_63, [128, 128, 128]);  addmm_63 = None
        view_168 = torch.ops.aten.view.default(add_64, [16384, 128]);  add_64 = None
        permute_84 = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg195_1, view_168, permute_84);  arg195_1 = view_168 = permute_84 = None
        view_169 = torch.ops.aten.view.default(addmm_64, [128, 128, 128]);  addmm_64 = None
        view_170 = torch.ops.aten.view.default(add_62, [16384, 512])
        permute_85 = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg197_1, view_170, permute_85);  arg197_1 = view_170 = permute_85 = None
        view_171 = torch.ops.aten.view.default(addmm_65, [128, 128, 128]);  addmm_65 = None
        view_172 = torch.ops.aten.view.default(view_167, [128, 128, 4, 32]);  view_167 = None
        view_173 = torch.ops.aten.view.default(view_169, [128, 128, 4, 32]);  view_169 = None
        view_174 = torch.ops.aten.view.default(view_171, [128, 128, 4, 32]);  view_171 = None
        permute_default_57 = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        permute_default_58 = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        permute_default_59 = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_57, permute_default_58, permute_default_59, None, False, scale = 0.17677669529663687);  permute_default_57 = permute_default_58 = permute_default_59 = None
        getitem_21 = _scaled_dot_product_efficient_attention_default_19[0];  _scaled_dot_product_efficient_attention_default_19 = None
        permute_90 = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
        clone_29 = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_181 = torch.ops.aten.view.default(clone_29, [128, 128, 128]);  clone_29 = None
        view_182 = torch.ops.aten.view.default(view_181, [16384, 128]);  view_181 = None
        permute_91 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg199_1, view_182, permute_91);  arg199_1 = view_182 = permute_91 = None
        view_183 = torch.ops.aten.view.default(addmm_66, [128, 128, 128]);  addmm_66 = None
        add_66 = torch.ops.aten.add.Tensor(view_183, add_63);  view_183 = add_63 = None
        mul_36 = torch.ops.aten.mul.Tensor(add_66, arg201_1);  add_66 = arg201_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_36, arg200_1);  mul_36 = arg200_1 = None
        view_184 = torch.ops.aten.view.default(add_67, [16384, 128])
        permute_92 = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg221_1, view_184, permute_92);  arg221_1 = view_184 = permute_92 = None
        view_185 = torch.ops.aten.view.default(addmm_67, [128, 128, 512]);  addmm_67 = None
        relu_16 = torch.ops.aten.relu.default(view_185);  view_185 = None
        view_186 = torch.ops.aten.view.default(relu_16, [16384, 512]);  relu_16 = None
        permute_93 = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg223_1, view_186, permute_93);  arg223_1 = view_186 = permute_93 = None
        view_187 = torch.ops.aten.view.default(addmm_68, [128, 128, 128]);  addmm_68 = None
        add_68 = torch.ops.aten.add.Tensor(view_187, add_67);  view_187 = add_67 = None
        mul_37 = torch.ops.aten.mul.Tensor(add_68, arg225_1);  add_68 = arg225_1 = None
        add_69 = torch.ops.aten.add.Tensor(mul_37, arg224_1);  mul_37 = arg224_1 = None
        view_188 = torch.ops.aten.view.default(add_69, [16384, 128])
        permute_94 = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg227_1, view_188, permute_94);  arg227_1 = view_188 = permute_94 = None
        view_189 = torch.ops.aten.view.default(addmm_69, [128, 128, 512]);  addmm_69 = None
        relu_17 = torch.ops.aten.relu.default(view_189);  view_189 = None
        view_190 = torch.ops.aten.view.default(relu_17, [16384, 512]);  relu_17 = None
        permute_95 = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg229_1, view_190, permute_95);  arg229_1 = view_190 = permute_95 = None
        view_191 = torch.ops.aten.view.default(addmm_70, [128, 128, 128]);  addmm_70 = None
        add_70 = torch.ops.aten.add.Tensor(view_191, add_69);  view_191 = add_69 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_70, arg231_1);  add_70 = arg231_1 = None
        add_71 = torch.ops.aten.add.Tensor(mul_38, arg230_1);  mul_38 = arg230_1 = None
        view_192 = torch.ops.aten.view.default(add_71, [16384, 128])
        permute_96 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg233_1, view_192, permute_96);  arg233_1 = view_192 = permute_96 = None
        view_193 = torch.ops.aten.view.default(addmm_71, [128, 128, 512]);  addmm_71 = None
        relu_18 = torch.ops.aten.relu.default(view_193);  view_193 = None
        view_194 = torch.ops.aten.view.default(relu_18, [16384, 512]);  relu_18 = None
        permute_97 = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg235_1, view_194, permute_97);  arg235_1 = view_194 = permute_97 = None
        view_195 = torch.ops.aten.view.default(addmm_72, [128, 128, 128]);  addmm_72 = None
        add_72 = torch.ops.aten.add.Tensor(view_195, add_71);  view_195 = add_71 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_72, arg237_1);  add_72 = arg237_1 = None
        add_73 = torch.ops.aten.add.Tensor(mul_39, arg236_1);  mul_39 = arg236_1 = None
        view_196 = torch.ops.aten.view.default(add_73, [16384, 128])
        permute_98 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg203_1, view_196, permute_98);  arg203_1 = view_196 = permute_98 = None
        view_197 = torch.ops.aten.view.default(addmm_73, [128, 128, 512]);  addmm_73 = None
        relu_19 = torch.ops.aten.relu.default(view_197);  view_197 = None
        view_198 = torch.ops.aten.view.default(relu_19, [16384, 512]);  relu_19 = None
        permute_99 = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg205_1, view_198, permute_99);  arg205_1 = view_198 = permute_99 = None
        view_199 = torch.ops.aten.view.default(addmm_74, [128, 128, 128]);  addmm_74 = None
        add_74 = torch.ops.aten.add.Tensor(view_199, add_73);  view_199 = add_73 = None
        mul_40 = torch.ops.aten.mul.Tensor(add_74, arg207_1);  add_74 = arg207_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_40, arg206_1);  mul_40 = arg206_1 = None
        view_200 = torch.ops.aten.view.default(add_75, [16384, 128]);  add_75 = None
        permute_100 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg209_1, view_200, permute_100);  arg209_1 = view_200 = permute_100 = None
        view_201 = torch.ops.aten.view.default(addmm_75, [128, 128, 512]);  addmm_75 = None
        add_76 = torch.ops.aten.add.Tensor(view_201, add_62);  view_201 = add_62 = None
        mul_41 = torch.ops.aten.mul.Tensor(add_76, arg211_1);  add_76 = arg211_1 = None
        add_77 = torch.ops.aten.add.Tensor(mul_41, arg210_1);  mul_41 = arg210_1 = None
        view_202 = torch.ops.aten.view.default(add_77, [16384, 512])
        permute_101 = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg259_1, view_202, permute_101);  arg259_1 = view_202 = permute_101 = None
        view_203 = torch.ops.aten.view.default(addmm_76, [128, 128, 128]);  addmm_76 = None
        mul_42 = torch.ops.aten.mul.Tensor(view_203, arg261_1);  view_203 = arg261_1 = None
        add_78 = torch.ops.aten.add.Tensor(mul_42, arg260_1);  mul_42 = arg260_1 = None
        view_204 = torch.ops.aten.view.default(add_77, [16384, 512])
        permute_102 = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg263_1, view_204, permute_102);  arg263_1 = view_204 = permute_102 = None
        view_205 = torch.ops.aten.view.default(addmm_77, [128, 128, 128]);  addmm_77 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_205, arg265_1);  view_205 = arg265_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_43, arg264_1);  mul_43 = arg264_1 = None
        view_206 = torch.ops.aten.view.default(add_79, [16384, 128])
        permute_103 = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg239_1, view_206, permute_103);  arg239_1 = view_206 = permute_103 = None
        view_207 = torch.ops.aten.view.default(addmm_78, [128, 128, 128]);  addmm_78 = None
        view_208 = torch.ops.aten.view.default(add_79, [16384, 128]);  add_79 = None
        permute_104 = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg241_1, view_208, permute_104);  arg241_1 = view_208 = permute_104 = None
        view_209 = torch.ops.aten.view.default(addmm_79, [128, 128, 128]);  addmm_79 = None
        view_210 = torch.ops.aten.view.default(add_77, [16384, 512])
        permute_105 = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg243_1, view_210, permute_105);  arg243_1 = view_210 = permute_105 = None
        view_211 = torch.ops.aten.view.default(addmm_80, [128, 128, 128]);  addmm_80 = None
        view_212 = torch.ops.aten.view.default(view_207, [128, 128, 4, 32]);  view_207 = None
        view_213 = torch.ops.aten.view.default(view_209, [128, 128, 4, 32]);  view_209 = None
        view_214 = torch.ops.aten.view.default(view_211, [128, 128, 4, 32]);  view_211 = None
        permute_default_54 = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        permute_default_55 = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        permute_default_56 = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_54, permute_default_55, permute_default_56, None, False, scale = 0.17677669529663687);  permute_default_54 = permute_default_55 = permute_default_56 = None
        getitem_20 = _scaled_dot_product_efficient_attention_default_18[0];  _scaled_dot_product_efficient_attention_default_18 = None
        permute_110 = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
        clone_35 = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_221 = torch.ops.aten.view.default(clone_35, [128, 128, 128]);  clone_35 = None
        view_222 = torch.ops.aten.view.default(view_221, [16384, 128]);  view_221 = None
        permute_111 = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg245_1, view_222, permute_111);  arg245_1 = view_222 = permute_111 = None
        view_223 = torch.ops.aten.view.default(addmm_81, [128, 128, 128]);  addmm_81 = None
        add_81 = torch.ops.aten.add.Tensor(view_223, add_78);  view_223 = add_78 = None
        mul_44 = torch.ops.aten.mul.Tensor(add_81, arg247_1);  add_81 = arg247_1 = None
        add_82 = torch.ops.aten.add.Tensor(mul_44, arg246_1);  mul_44 = arg246_1 = None
        view_224 = torch.ops.aten.view.default(add_82, [16384, 128])
        permute_112 = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg267_1, view_224, permute_112);  arg267_1 = view_224 = permute_112 = None
        view_225 = torch.ops.aten.view.default(addmm_82, [128, 128, 512]);  addmm_82 = None
        relu_20 = torch.ops.aten.relu.default(view_225);  view_225 = None
        view_226 = torch.ops.aten.view.default(relu_20, [16384, 512]);  relu_20 = None
        permute_113 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg269_1, view_226, permute_113);  arg269_1 = view_226 = permute_113 = None
        view_227 = torch.ops.aten.view.default(addmm_83, [128, 128, 128]);  addmm_83 = None
        add_83 = torch.ops.aten.add.Tensor(view_227, add_82);  view_227 = add_82 = None
        mul_45 = torch.ops.aten.mul.Tensor(add_83, arg271_1);  add_83 = arg271_1 = None
        add_84 = torch.ops.aten.add.Tensor(mul_45, arg270_1);  mul_45 = arg270_1 = None
        view_228 = torch.ops.aten.view.default(add_84, [16384, 128])
        permute_114 = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg273_1, view_228, permute_114);  arg273_1 = view_228 = permute_114 = None
        view_229 = torch.ops.aten.view.default(addmm_84, [128, 128, 512]);  addmm_84 = None
        relu_21 = torch.ops.aten.relu.default(view_229);  view_229 = None
        view_230 = torch.ops.aten.view.default(relu_21, [16384, 512]);  relu_21 = None
        permute_115 = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg275_1, view_230, permute_115);  arg275_1 = view_230 = permute_115 = None
        view_231 = torch.ops.aten.view.default(addmm_85, [128, 128, 128]);  addmm_85 = None
        add_85 = torch.ops.aten.add.Tensor(view_231, add_84);  view_231 = add_84 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_85, arg277_1);  add_85 = arg277_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_46, arg276_1);  mul_46 = arg276_1 = None
        view_232 = torch.ops.aten.view.default(add_86, [16384, 128])
        permute_116 = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg279_1, view_232, permute_116);  arg279_1 = view_232 = permute_116 = None
        view_233 = torch.ops.aten.view.default(addmm_86, [128, 128, 512]);  addmm_86 = None
        relu_22 = torch.ops.aten.relu.default(view_233);  view_233 = None
        view_234 = torch.ops.aten.view.default(relu_22, [16384, 512]);  relu_22 = None
        permute_117 = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg281_1, view_234, permute_117);  arg281_1 = view_234 = permute_117 = None
        view_235 = torch.ops.aten.view.default(addmm_87, [128, 128, 128]);  addmm_87 = None
        add_87 = torch.ops.aten.add.Tensor(view_235, add_86);  view_235 = add_86 = None
        mul_47 = torch.ops.aten.mul.Tensor(add_87, arg283_1);  add_87 = arg283_1 = None
        add_88 = torch.ops.aten.add.Tensor(mul_47, arg282_1);  mul_47 = arg282_1 = None
        view_236 = torch.ops.aten.view.default(add_88, [16384, 128])
        permute_118 = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg249_1, view_236, permute_118);  arg249_1 = view_236 = permute_118 = None
        view_237 = torch.ops.aten.view.default(addmm_88, [128, 128, 512]);  addmm_88 = None
        relu_23 = torch.ops.aten.relu.default(view_237);  view_237 = None
        view_238 = torch.ops.aten.view.default(relu_23, [16384, 512]);  relu_23 = None
        permute_119 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg251_1, view_238, permute_119);  arg251_1 = view_238 = permute_119 = None
        view_239 = torch.ops.aten.view.default(addmm_89, [128, 128, 128]);  addmm_89 = None
        add_89 = torch.ops.aten.add.Tensor(view_239, add_88);  view_239 = add_88 = None
        mul_48 = torch.ops.aten.mul.Tensor(add_89, arg253_1);  add_89 = arg253_1 = None
        add_90 = torch.ops.aten.add.Tensor(mul_48, arg252_1);  mul_48 = arg252_1 = None
        view_240 = torch.ops.aten.view.default(add_90, [16384, 128]);  add_90 = None
        permute_120 = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg255_1, view_240, permute_120);  arg255_1 = view_240 = permute_120 = None
        view_241 = torch.ops.aten.view.default(addmm_90, [128, 128, 512]);  addmm_90 = None
        add_91 = torch.ops.aten.add.Tensor(view_241, add_77);  view_241 = add_77 = None
        mul_49 = torch.ops.aten.mul.Tensor(add_91, arg257_1);  add_91 = arg257_1 = None
        add_92 = torch.ops.aten.add.Tensor(mul_49, arg256_1);  mul_49 = arg256_1 = None
        view_242 = torch.ops.aten.view.default(add_92, [16384, 512])
        permute_121 = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg305_1, view_242, permute_121);  arg305_1 = view_242 = permute_121 = None
        view_243 = torch.ops.aten.view.default(addmm_91, [128, 128, 128]);  addmm_91 = None
        mul_50 = torch.ops.aten.mul.Tensor(view_243, arg307_1);  view_243 = arg307_1 = None
        add_93 = torch.ops.aten.add.Tensor(mul_50, arg306_1);  mul_50 = arg306_1 = None
        view_244 = torch.ops.aten.view.default(add_92, [16384, 512])
        permute_122 = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg309_1, view_244, permute_122);  arg309_1 = view_244 = permute_122 = None
        view_245 = torch.ops.aten.view.default(addmm_92, [128, 128, 128]);  addmm_92 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_245, arg311_1);  view_245 = arg311_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_51, arg310_1);  mul_51 = arg310_1 = None
        view_246 = torch.ops.aten.view.default(add_94, [16384, 128])
        permute_123 = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg285_1, view_246, permute_123);  arg285_1 = view_246 = permute_123 = None
        view_247 = torch.ops.aten.view.default(addmm_93, [128, 128, 128]);  addmm_93 = None
        view_248 = torch.ops.aten.view.default(add_94, [16384, 128]);  add_94 = None
        permute_124 = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg287_1, view_248, permute_124);  arg287_1 = view_248 = permute_124 = None
        view_249 = torch.ops.aten.view.default(addmm_94, [128, 128, 128]);  addmm_94 = None
        view_250 = torch.ops.aten.view.default(add_92, [16384, 512])
        permute_125 = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg289_1, view_250, permute_125);  arg289_1 = view_250 = permute_125 = None
        view_251 = torch.ops.aten.view.default(addmm_95, [128, 128, 128]);  addmm_95 = None
        view_252 = torch.ops.aten.view.default(view_247, [128, 128, 4, 32]);  view_247 = None
        view_253 = torch.ops.aten.view.default(view_249, [128, 128, 4, 32]);  view_249 = None
        view_254 = torch.ops.aten.view.default(view_251, [128, 128, 4, 32]);  view_251 = None
        permute_default_51 = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        permute_default_52 = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        permute_default_53 = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
        _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_51, permute_default_52, permute_default_53, None, False, scale = 0.17677669529663687);  permute_default_51 = permute_default_52 = permute_default_53 = None
        getitem_19 = _scaled_dot_product_efficient_attention_default_17[0];  _scaled_dot_product_efficient_attention_default_17 = None
        permute_130 = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3]);  getitem_19 = None
        clone_41 = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_261 = torch.ops.aten.view.default(clone_41, [128, 128, 128]);  clone_41 = None
        view_262 = torch.ops.aten.view.default(view_261, [16384, 128]);  view_261 = None
        permute_131 = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg291_1, view_262, permute_131);  arg291_1 = view_262 = permute_131 = None
        view_263 = torch.ops.aten.view.default(addmm_96, [128, 128, 128]);  addmm_96 = None
        add_96 = torch.ops.aten.add.Tensor(view_263, add_93);  view_263 = add_93 = None
        mul_52 = torch.ops.aten.mul.Tensor(add_96, arg293_1);  add_96 = arg293_1 = None
        add_97 = torch.ops.aten.add.Tensor(mul_52, arg292_1);  mul_52 = arg292_1 = None
        view_264 = torch.ops.aten.view.default(add_97, [16384, 128])
        permute_132 = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg313_1, view_264, permute_132);  arg313_1 = view_264 = permute_132 = None
        view_265 = torch.ops.aten.view.default(addmm_97, [128, 128, 512]);  addmm_97 = None
        relu_24 = torch.ops.aten.relu.default(view_265);  view_265 = None
        view_266 = torch.ops.aten.view.default(relu_24, [16384, 512]);  relu_24 = None
        permute_133 = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg315_1, view_266, permute_133);  arg315_1 = view_266 = permute_133 = None
        view_267 = torch.ops.aten.view.default(addmm_98, [128, 128, 128]);  addmm_98 = None
        add_98 = torch.ops.aten.add.Tensor(view_267, add_97);  view_267 = add_97 = None
        mul_53 = torch.ops.aten.mul.Tensor(add_98, arg317_1);  add_98 = arg317_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_53, arg316_1);  mul_53 = arg316_1 = None
        view_268 = torch.ops.aten.view.default(add_99, [16384, 128])
        permute_134 = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg319_1, view_268, permute_134);  arg319_1 = view_268 = permute_134 = None
        view_269 = torch.ops.aten.view.default(addmm_99, [128, 128, 512]);  addmm_99 = None
        relu_25 = torch.ops.aten.relu.default(view_269);  view_269 = None
        view_270 = torch.ops.aten.view.default(relu_25, [16384, 512]);  relu_25 = None
        permute_135 = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg321_1, view_270, permute_135);  arg321_1 = view_270 = permute_135 = None
        view_271 = torch.ops.aten.view.default(addmm_100, [128, 128, 128]);  addmm_100 = None
        add_100 = torch.ops.aten.add.Tensor(view_271, add_99);  view_271 = add_99 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_100, arg323_1);  add_100 = arg323_1 = None
        add_101 = torch.ops.aten.add.Tensor(mul_54, arg322_1);  mul_54 = arg322_1 = None
        view_272 = torch.ops.aten.view.default(add_101, [16384, 128])
        permute_136 = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg325_1, view_272, permute_136);  arg325_1 = view_272 = permute_136 = None
        view_273 = torch.ops.aten.view.default(addmm_101, [128, 128, 512]);  addmm_101 = None
        relu_26 = torch.ops.aten.relu.default(view_273);  view_273 = None
        view_274 = torch.ops.aten.view.default(relu_26, [16384, 512]);  relu_26 = None
        permute_137 = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg327_1, view_274, permute_137);  arg327_1 = view_274 = permute_137 = None
        view_275 = torch.ops.aten.view.default(addmm_102, [128, 128, 128]);  addmm_102 = None
        add_102 = torch.ops.aten.add.Tensor(view_275, add_101);  view_275 = add_101 = None
        mul_55 = torch.ops.aten.mul.Tensor(add_102, arg329_1);  add_102 = arg329_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_55, arg328_1);  mul_55 = arg328_1 = None
        view_276 = torch.ops.aten.view.default(add_103, [16384, 128])
        permute_138 = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg295_1, view_276, permute_138);  arg295_1 = view_276 = permute_138 = None
        view_277 = torch.ops.aten.view.default(addmm_103, [128, 128, 512]);  addmm_103 = None
        relu_27 = torch.ops.aten.relu.default(view_277);  view_277 = None
        view_278 = torch.ops.aten.view.default(relu_27, [16384, 512]);  relu_27 = None
        permute_139 = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg297_1, view_278, permute_139);  arg297_1 = view_278 = permute_139 = None
        view_279 = torch.ops.aten.view.default(addmm_104, [128, 128, 128]);  addmm_104 = None
        add_104 = torch.ops.aten.add.Tensor(view_279, add_103);  view_279 = add_103 = None
        mul_56 = torch.ops.aten.mul.Tensor(add_104, arg299_1);  add_104 = arg299_1 = None
        add_105 = torch.ops.aten.add.Tensor(mul_56, arg298_1);  mul_56 = arg298_1 = None
        view_280 = torch.ops.aten.view.default(add_105, [16384, 128]);  add_105 = None
        permute_140 = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg301_1, view_280, permute_140);  arg301_1 = view_280 = permute_140 = None
        view_281 = torch.ops.aten.view.default(addmm_105, [128, 128, 512]);  addmm_105 = None
        add_106 = torch.ops.aten.add.Tensor(view_281, add_92);  view_281 = add_92 = None
        mul_57 = torch.ops.aten.mul.Tensor(add_106, arg303_1);  add_106 = arg303_1 = None
        add_107 = torch.ops.aten.add.Tensor(mul_57, arg302_1);  mul_57 = arg302_1 = None
        view_282 = torch.ops.aten.view.default(add_107, [16384, 512])
        permute_141 = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg351_1, view_282, permute_141);  arg351_1 = view_282 = permute_141 = None
        view_283 = torch.ops.aten.view.default(addmm_106, [128, 128, 128]);  addmm_106 = None
        mul_58 = torch.ops.aten.mul.Tensor(view_283, arg353_1);  view_283 = arg353_1 = None
        add_108 = torch.ops.aten.add.Tensor(mul_58, arg352_1);  mul_58 = arg352_1 = None
        view_284 = torch.ops.aten.view.default(add_107, [16384, 512])
        permute_142 = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg355_1, view_284, permute_142);  arg355_1 = view_284 = permute_142 = None
        view_285 = torch.ops.aten.view.default(addmm_107, [128, 128, 128]);  addmm_107 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_285, arg357_1);  view_285 = arg357_1 = None
        add_109 = torch.ops.aten.add.Tensor(mul_59, arg356_1);  mul_59 = arg356_1 = None
        view_286 = torch.ops.aten.view.default(add_109, [16384, 128])
        permute_143 = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg331_1, view_286, permute_143);  arg331_1 = view_286 = permute_143 = None
        view_287 = torch.ops.aten.view.default(addmm_108, [128, 128, 128]);  addmm_108 = None
        view_288 = torch.ops.aten.view.default(add_109, [16384, 128]);  add_109 = None
        permute_144 = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg333_1, view_288, permute_144);  arg333_1 = view_288 = permute_144 = None
        view_289 = torch.ops.aten.view.default(addmm_109, [128, 128, 128]);  addmm_109 = None
        view_290 = torch.ops.aten.view.default(add_107, [16384, 512])
        permute_145 = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg335_1, view_290, permute_145);  arg335_1 = view_290 = permute_145 = None
        view_291 = torch.ops.aten.view.default(addmm_110, [128, 128, 128]);  addmm_110 = None
        view_292 = torch.ops.aten.view.default(view_287, [128, 128, 4, 32]);  view_287 = None
        view_293 = torch.ops.aten.view.default(view_289, [128, 128, 4, 32]);  view_289 = None
        view_294 = torch.ops.aten.view.default(view_291, [128, 128, 4, 32]);  view_291 = None
        permute_default_48 = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        permute_default_49 = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
        permute_default_50 = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_48, permute_default_49, permute_default_50, None, False, scale = 0.17677669529663687);  permute_default_48 = permute_default_49 = permute_default_50 = None
        getitem_18 = _scaled_dot_product_efficient_attention_default_16[0];  _scaled_dot_product_efficient_attention_default_16 = None
        permute_150 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        clone_47 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_301 = torch.ops.aten.view.default(clone_47, [128, 128, 128]);  clone_47 = None
        view_302 = torch.ops.aten.view.default(view_301, [16384, 128]);  view_301 = None
        permute_151 = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg337_1, view_302, permute_151);  arg337_1 = view_302 = permute_151 = None
        view_303 = torch.ops.aten.view.default(addmm_111, [128, 128, 128]);  addmm_111 = None
        add_111 = torch.ops.aten.add.Tensor(view_303, add_108);  view_303 = add_108 = None
        mul_60 = torch.ops.aten.mul.Tensor(add_111, arg339_1);  add_111 = arg339_1 = None
        add_112 = torch.ops.aten.add.Tensor(mul_60, arg338_1);  mul_60 = arg338_1 = None
        view_304 = torch.ops.aten.view.default(add_112, [16384, 128])
        permute_152 = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg359_1, view_304, permute_152);  arg359_1 = view_304 = permute_152 = None
        view_305 = torch.ops.aten.view.default(addmm_112, [128, 128, 512]);  addmm_112 = None
        relu_28 = torch.ops.aten.relu.default(view_305);  view_305 = None
        view_306 = torch.ops.aten.view.default(relu_28, [16384, 512]);  relu_28 = None
        permute_153 = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg361_1, view_306, permute_153);  arg361_1 = view_306 = permute_153 = None
        view_307 = torch.ops.aten.view.default(addmm_113, [128, 128, 128]);  addmm_113 = None
        add_113 = torch.ops.aten.add.Tensor(view_307, add_112);  view_307 = add_112 = None
        mul_61 = torch.ops.aten.mul.Tensor(add_113, arg363_1);  add_113 = arg363_1 = None
        add_114 = torch.ops.aten.add.Tensor(mul_61, arg362_1);  mul_61 = arg362_1 = None
        view_308 = torch.ops.aten.view.default(add_114, [16384, 128])
        permute_154 = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg365_1, view_308, permute_154);  arg365_1 = view_308 = permute_154 = None
        view_309 = torch.ops.aten.view.default(addmm_114, [128, 128, 512]);  addmm_114 = None
        relu_29 = torch.ops.aten.relu.default(view_309);  view_309 = None
        view_310 = torch.ops.aten.view.default(relu_29, [16384, 512]);  relu_29 = None
        permute_155 = torch.ops.aten.permute.default(arg366_1, [1, 0]);  arg366_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg367_1, view_310, permute_155);  arg367_1 = view_310 = permute_155 = None
        view_311 = torch.ops.aten.view.default(addmm_115, [128, 128, 128]);  addmm_115 = None
        add_115 = torch.ops.aten.add.Tensor(view_311, add_114);  view_311 = add_114 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_115, arg369_1);  add_115 = arg369_1 = None
        add_116 = torch.ops.aten.add.Tensor(mul_62, arg368_1);  mul_62 = arg368_1 = None
        view_312 = torch.ops.aten.view.default(add_116, [16384, 128])
        permute_156 = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg371_1, view_312, permute_156);  arg371_1 = view_312 = permute_156 = None
        view_313 = torch.ops.aten.view.default(addmm_116, [128, 128, 512]);  addmm_116 = None
        relu_30 = torch.ops.aten.relu.default(view_313);  view_313 = None
        view_314 = torch.ops.aten.view.default(relu_30, [16384, 512]);  relu_30 = None
        permute_157 = torch.ops.aten.permute.default(arg372_1, [1, 0]);  arg372_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg373_1, view_314, permute_157);  arg373_1 = view_314 = permute_157 = None
        view_315 = torch.ops.aten.view.default(addmm_117, [128, 128, 128]);  addmm_117 = None
        add_117 = torch.ops.aten.add.Tensor(view_315, add_116);  view_315 = add_116 = None
        mul_63 = torch.ops.aten.mul.Tensor(add_117, arg375_1);  add_117 = arg375_1 = None
        add_118 = torch.ops.aten.add.Tensor(mul_63, arg374_1);  mul_63 = arg374_1 = None
        view_316 = torch.ops.aten.view.default(add_118, [16384, 128])
        permute_158 = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg341_1, view_316, permute_158);  arg341_1 = view_316 = permute_158 = None
        view_317 = torch.ops.aten.view.default(addmm_118, [128, 128, 512]);  addmm_118 = None
        relu_31 = torch.ops.aten.relu.default(view_317);  view_317 = None
        view_318 = torch.ops.aten.view.default(relu_31, [16384, 512]);  relu_31 = None
        permute_159 = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg343_1, view_318, permute_159);  arg343_1 = view_318 = permute_159 = None
        view_319 = torch.ops.aten.view.default(addmm_119, [128, 128, 128]);  addmm_119 = None
        add_119 = torch.ops.aten.add.Tensor(view_319, add_118);  view_319 = add_118 = None
        mul_64 = torch.ops.aten.mul.Tensor(add_119, arg345_1);  add_119 = arg345_1 = None
        add_120 = torch.ops.aten.add.Tensor(mul_64, arg344_1);  mul_64 = arg344_1 = None
        view_320 = torch.ops.aten.view.default(add_120, [16384, 128]);  add_120 = None
        permute_160 = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg347_1, view_320, permute_160);  arg347_1 = view_320 = permute_160 = None
        view_321 = torch.ops.aten.view.default(addmm_120, [128, 128, 512]);  addmm_120 = None
        add_121 = torch.ops.aten.add.Tensor(view_321, add_107);  view_321 = add_107 = None
        mul_65 = torch.ops.aten.mul.Tensor(add_121, arg349_1);  add_121 = arg349_1 = None
        add_122 = torch.ops.aten.add.Tensor(mul_65, arg348_1);  mul_65 = arg348_1 = None
        view_322 = torch.ops.aten.view.default(add_122, [16384, 512])
        permute_161 = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg397_1, view_322, permute_161);  arg397_1 = view_322 = permute_161 = None
        view_323 = torch.ops.aten.view.default(addmm_121, [128, 128, 128]);  addmm_121 = None
        mul_66 = torch.ops.aten.mul.Tensor(view_323, arg399_1);  view_323 = arg399_1 = None
        add_123 = torch.ops.aten.add.Tensor(mul_66, arg398_1);  mul_66 = arg398_1 = None
        view_324 = torch.ops.aten.view.default(add_122, [16384, 512])
        permute_162 = torch.ops.aten.permute.default(arg400_1, [1, 0]);  arg400_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg401_1, view_324, permute_162);  arg401_1 = view_324 = permute_162 = None
        view_325 = torch.ops.aten.view.default(addmm_122, [128, 128, 128]);  addmm_122 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_325, arg403_1);  view_325 = arg403_1 = None
        add_124 = torch.ops.aten.add.Tensor(mul_67, arg402_1);  mul_67 = arg402_1 = None
        view_326 = torch.ops.aten.view.default(add_124, [16384, 128])
        permute_163 = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg377_1, view_326, permute_163);  arg377_1 = view_326 = permute_163 = None
        view_327 = torch.ops.aten.view.default(addmm_123, [128, 128, 128]);  addmm_123 = None
        view_328 = torch.ops.aten.view.default(add_124, [16384, 128]);  add_124 = None
        permute_164 = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg379_1, view_328, permute_164);  arg379_1 = view_328 = permute_164 = None
        view_329 = torch.ops.aten.view.default(addmm_124, [128, 128, 128]);  addmm_124 = None
        view_330 = torch.ops.aten.view.default(add_122, [16384, 512])
        permute_165 = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg381_1, view_330, permute_165);  arg381_1 = view_330 = permute_165 = None
        view_331 = torch.ops.aten.view.default(addmm_125, [128, 128, 128]);  addmm_125 = None
        view_332 = torch.ops.aten.view.default(view_327, [128, 128, 4, 32]);  view_327 = None
        view_333 = torch.ops.aten.view.default(view_329, [128, 128, 4, 32]);  view_329 = None
        view_334 = torch.ops.aten.view.default(view_331, [128, 128, 4, 32]);  view_331 = None
        permute_default_45 = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        permute_default_46 = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
        permute_default_47 = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_45, permute_default_46, permute_default_47, None, False, scale = 0.17677669529663687);  permute_default_45 = permute_default_46 = permute_default_47 = None
        getitem_17 = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
        permute_170 = torch.ops.aten.permute.default(getitem_17, [0, 2, 1, 3]);  getitem_17 = None
        clone_53 = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_341 = torch.ops.aten.view.default(clone_53, [128, 128, 128]);  clone_53 = None
        view_342 = torch.ops.aten.view.default(view_341, [16384, 128]);  view_341 = None
        permute_171 = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg383_1, view_342, permute_171);  arg383_1 = view_342 = permute_171 = None
        view_343 = torch.ops.aten.view.default(addmm_126, [128, 128, 128]);  addmm_126 = None
        add_126 = torch.ops.aten.add.Tensor(view_343, add_123);  view_343 = add_123 = None
        mul_68 = torch.ops.aten.mul.Tensor(add_126, arg385_1);  add_126 = arg385_1 = None
        add_127 = torch.ops.aten.add.Tensor(mul_68, arg384_1);  mul_68 = arg384_1 = None
        view_344 = torch.ops.aten.view.default(add_127, [16384, 128])
        permute_172 = torch.ops.aten.permute.default(arg404_1, [1, 0]);  arg404_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg405_1, view_344, permute_172);  arg405_1 = view_344 = permute_172 = None
        view_345 = torch.ops.aten.view.default(addmm_127, [128, 128, 512]);  addmm_127 = None
        relu_32 = torch.ops.aten.relu.default(view_345);  view_345 = None
        view_346 = torch.ops.aten.view.default(relu_32, [16384, 512]);  relu_32 = None
        permute_173 = torch.ops.aten.permute.default(arg406_1, [1, 0]);  arg406_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg407_1, view_346, permute_173);  arg407_1 = view_346 = permute_173 = None
        view_347 = torch.ops.aten.view.default(addmm_128, [128, 128, 128]);  addmm_128 = None
        add_128 = torch.ops.aten.add.Tensor(view_347, add_127);  view_347 = add_127 = None
        mul_69 = torch.ops.aten.mul.Tensor(add_128, arg409_1);  add_128 = arg409_1 = None
        add_129 = torch.ops.aten.add.Tensor(mul_69, arg408_1);  mul_69 = arg408_1 = None
        view_348 = torch.ops.aten.view.default(add_129, [16384, 128])
        permute_174 = torch.ops.aten.permute.default(arg410_1, [1, 0]);  arg410_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg411_1, view_348, permute_174);  arg411_1 = view_348 = permute_174 = None
        view_349 = torch.ops.aten.view.default(addmm_129, [128, 128, 512]);  addmm_129 = None
        relu_33 = torch.ops.aten.relu.default(view_349);  view_349 = None
        view_350 = torch.ops.aten.view.default(relu_33, [16384, 512]);  relu_33 = None
        permute_175 = torch.ops.aten.permute.default(arg412_1, [1, 0]);  arg412_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg413_1, view_350, permute_175);  arg413_1 = view_350 = permute_175 = None
        view_351 = torch.ops.aten.view.default(addmm_130, [128, 128, 128]);  addmm_130 = None
        add_130 = torch.ops.aten.add.Tensor(view_351, add_129);  view_351 = add_129 = None
        mul_70 = torch.ops.aten.mul.Tensor(add_130, arg415_1);  add_130 = arg415_1 = None
        add_131 = torch.ops.aten.add.Tensor(mul_70, arg414_1);  mul_70 = arg414_1 = None
        view_352 = torch.ops.aten.view.default(add_131, [16384, 128])
        permute_176 = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg417_1, view_352, permute_176);  arg417_1 = view_352 = permute_176 = None
        view_353 = torch.ops.aten.view.default(addmm_131, [128, 128, 512]);  addmm_131 = None
        relu_34 = torch.ops.aten.relu.default(view_353);  view_353 = None
        view_354 = torch.ops.aten.view.default(relu_34, [16384, 512]);  relu_34 = None
        permute_177 = torch.ops.aten.permute.default(arg418_1, [1, 0]);  arg418_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg419_1, view_354, permute_177);  arg419_1 = view_354 = permute_177 = None
        view_355 = torch.ops.aten.view.default(addmm_132, [128, 128, 128]);  addmm_132 = None
        add_132 = torch.ops.aten.add.Tensor(view_355, add_131);  view_355 = add_131 = None
        mul_71 = torch.ops.aten.mul.Tensor(add_132, arg421_1);  add_132 = arg421_1 = None
        add_133 = torch.ops.aten.add.Tensor(mul_71, arg420_1);  mul_71 = arg420_1 = None
        view_356 = torch.ops.aten.view.default(add_133, [16384, 128])
        permute_178 = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg387_1, view_356, permute_178);  arg387_1 = view_356 = permute_178 = None
        view_357 = torch.ops.aten.view.default(addmm_133, [128, 128, 512]);  addmm_133 = None
        relu_35 = torch.ops.aten.relu.default(view_357);  view_357 = None
        view_358 = torch.ops.aten.view.default(relu_35, [16384, 512]);  relu_35 = None
        permute_179 = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg389_1, view_358, permute_179);  arg389_1 = view_358 = permute_179 = None
        view_359 = torch.ops.aten.view.default(addmm_134, [128, 128, 128]);  addmm_134 = None
        add_134 = torch.ops.aten.add.Tensor(view_359, add_133);  view_359 = add_133 = None
        mul_72 = torch.ops.aten.mul.Tensor(add_134, arg391_1);  add_134 = arg391_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_72, arg390_1);  mul_72 = arg390_1 = None
        view_360 = torch.ops.aten.view.default(add_135, [16384, 128]);  add_135 = None
        permute_180 = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg393_1, view_360, permute_180);  arg393_1 = view_360 = permute_180 = None
        view_361 = torch.ops.aten.view.default(addmm_135, [128, 128, 512]);  addmm_135 = None
        add_136 = torch.ops.aten.add.Tensor(view_361, add_122);  view_361 = add_122 = None
        mul_73 = torch.ops.aten.mul.Tensor(add_136, arg395_1);  add_136 = arg395_1 = None
        add_137 = torch.ops.aten.add.Tensor(mul_73, arg394_1);  mul_73 = arg394_1 = None
        view_362 = torch.ops.aten.view.default(add_137, [16384, 512])
        permute_181 = torch.ops.aten.permute.default(arg442_1, [1, 0]);  arg442_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg443_1, view_362, permute_181);  arg443_1 = view_362 = permute_181 = None
        view_363 = torch.ops.aten.view.default(addmm_136, [128, 128, 128]);  addmm_136 = None
        mul_74 = torch.ops.aten.mul.Tensor(view_363, arg445_1);  view_363 = arg445_1 = None
        add_138 = torch.ops.aten.add.Tensor(mul_74, arg444_1);  mul_74 = arg444_1 = None
        view_364 = torch.ops.aten.view.default(add_137, [16384, 512])
        permute_182 = torch.ops.aten.permute.default(arg446_1, [1, 0]);  arg446_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg447_1, view_364, permute_182);  arg447_1 = view_364 = permute_182 = None
        view_365 = torch.ops.aten.view.default(addmm_137, [128, 128, 128]);  addmm_137 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_365, arg449_1);  view_365 = arg449_1 = None
        add_139 = torch.ops.aten.add.Tensor(mul_75, arg448_1);  mul_75 = arg448_1 = None
        view_366 = torch.ops.aten.view.default(add_139, [16384, 128])
        permute_183 = torch.ops.aten.permute.default(arg422_1, [1, 0]);  arg422_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg423_1, view_366, permute_183);  arg423_1 = view_366 = permute_183 = None
        view_367 = torch.ops.aten.view.default(addmm_138, [128, 128, 128]);  addmm_138 = None
        view_368 = torch.ops.aten.view.default(add_139, [16384, 128]);  add_139 = None
        permute_184 = torch.ops.aten.permute.default(arg424_1, [1, 0]);  arg424_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg425_1, view_368, permute_184);  arg425_1 = view_368 = permute_184 = None
        view_369 = torch.ops.aten.view.default(addmm_139, [128, 128, 128]);  addmm_139 = None
        view_370 = torch.ops.aten.view.default(add_137, [16384, 512])
        permute_185 = torch.ops.aten.permute.default(arg426_1, [1, 0]);  arg426_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg427_1, view_370, permute_185);  arg427_1 = view_370 = permute_185 = None
        view_371 = torch.ops.aten.view.default(addmm_140, [128, 128, 128]);  addmm_140 = None
        view_372 = torch.ops.aten.view.default(view_367, [128, 128, 4, 32]);  view_367 = None
        view_373 = torch.ops.aten.view.default(view_369, [128, 128, 4, 32]);  view_369 = None
        view_374 = torch.ops.aten.view.default(view_371, [128, 128, 4, 32]);  view_371 = None
        permute_default_42 = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        permute_default_43 = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        permute_default_44 = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_42, permute_default_43, permute_default_44, None, False, scale = 0.17677669529663687);  permute_default_42 = permute_default_43 = permute_default_44 = None
        getitem_16 = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
        permute_190 = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        clone_59 = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_381 = torch.ops.aten.view.default(clone_59, [128, 128, 128]);  clone_59 = None
        view_382 = torch.ops.aten.view.default(view_381, [16384, 128]);  view_381 = None
        permute_191 = torch.ops.aten.permute.default(arg428_1, [1, 0]);  arg428_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg429_1, view_382, permute_191);  arg429_1 = view_382 = permute_191 = None
        view_383 = torch.ops.aten.view.default(addmm_141, [128, 128, 128]);  addmm_141 = None
        add_141 = torch.ops.aten.add.Tensor(view_383, add_138);  view_383 = add_138 = None
        mul_76 = torch.ops.aten.mul.Tensor(add_141, arg431_1);  add_141 = arg431_1 = None
        add_142 = torch.ops.aten.add.Tensor(mul_76, arg430_1);  mul_76 = arg430_1 = None
        view_384 = torch.ops.aten.view.default(add_142, [16384, 128])
        permute_192 = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg451_1, view_384, permute_192);  arg451_1 = view_384 = permute_192 = None
        view_385 = torch.ops.aten.view.default(addmm_142, [128, 128, 512]);  addmm_142 = None
        relu_36 = torch.ops.aten.relu.default(view_385);  view_385 = None
        view_386 = torch.ops.aten.view.default(relu_36, [16384, 512]);  relu_36 = None
        permute_193 = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg453_1, view_386, permute_193);  arg453_1 = view_386 = permute_193 = None
        view_387 = torch.ops.aten.view.default(addmm_143, [128, 128, 128]);  addmm_143 = None
        add_143 = torch.ops.aten.add.Tensor(view_387, add_142);  view_387 = add_142 = None
        mul_77 = torch.ops.aten.mul.Tensor(add_143, arg455_1);  add_143 = arg455_1 = None
        add_144 = torch.ops.aten.add.Tensor(mul_77, arg454_1);  mul_77 = arg454_1 = None
        view_388 = torch.ops.aten.view.default(add_144, [16384, 128])
        permute_194 = torch.ops.aten.permute.default(arg456_1, [1, 0]);  arg456_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg457_1, view_388, permute_194);  arg457_1 = view_388 = permute_194 = None
        view_389 = torch.ops.aten.view.default(addmm_144, [128, 128, 512]);  addmm_144 = None
        relu_37 = torch.ops.aten.relu.default(view_389);  view_389 = None
        view_390 = torch.ops.aten.view.default(relu_37, [16384, 512]);  relu_37 = None
        permute_195 = torch.ops.aten.permute.default(arg458_1, [1, 0]);  arg458_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg459_1, view_390, permute_195);  arg459_1 = view_390 = permute_195 = None
        view_391 = torch.ops.aten.view.default(addmm_145, [128, 128, 128]);  addmm_145 = None
        add_145 = torch.ops.aten.add.Tensor(view_391, add_144);  view_391 = add_144 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_145, arg461_1);  add_145 = arg461_1 = None
        add_146 = torch.ops.aten.add.Tensor(mul_78, arg460_1);  mul_78 = arg460_1 = None
        view_392 = torch.ops.aten.view.default(add_146, [16384, 128])
        permute_196 = torch.ops.aten.permute.default(arg462_1, [1, 0]);  arg462_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg463_1, view_392, permute_196);  arg463_1 = view_392 = permute_196 = None
        view_393 = torch.ops.aten.view.default(addmm_146, [128, 128, 512]);  addmm_146 = None
        relu_38 = torch.ops.aten.relu.default(view_393);  view_393 = None
        view_394 = torch.ops.aten.view.default(relu_38, [16384, 512]);  relu_38 = None
        permute_197 = torch.ops.aten.permute.default(arg464_1, [1, 0]);  arg464_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg465_1, view_394, permute_197);  arg465_1 = view_394 = permute_197 = None
        view_395 = torch.ops.aten.view.default(addmm_147, [128, 128, 128]);  addmm_147 = None
        add_147 = torch.ops.aten.add.Tensor(view_395, add_146);  view_395 = add_146 = None
        mul_79 = torch.ops.aten.mul.Tensor(add_147, arg467_1);  add_147 = arg467_1 = None
        add_148 = torch.ops.aten.add.Tensor(mul_79, arg466_1);  mul_79 = arg466_1 = None
        view_396 = torch.ops.aten.view.default(add_148, [16384, 128])
        permute_198 = torch.ops.aten.permute.default(arg432_1, [1, 0]);  arg432_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg433_1, view_396, permute_198);  arg433_1 = view_396 = permute_198 = None
        view_397 = torch.ops.aten.view.default(addmm_148, [128, 128, 512]);  addmm_148 = None
        relu_39 = torch.ops.aten.relu.default(view_397);  view_397 = None
        view_398 = torch.ops.aten.view.default(relu_39, [16384, 512]);  relu_39 = None
        permute_199 = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg435_1, view_398, permute_199);  arg435_1 = view_398 = permute_199 = None
        view_399 = torch.ops.aten.view.default(addmm_149, [128, 128, 128]);  addmm_149 = None
        add_149 = torch.ops.aten.add.Tensor(view_399, add_148);  view_399 = add_148 = None
        mul_80 = torch.ops.aten.mul.Tensor(add_149, arg437_1);  add_149 = arg437_1 = None
        add_150 = torch.ops.aten.add.Tensor(mul_80, arg436_1);  mul_80 = arg436_1 = None
        view_400 = torch.ops.aten.view.default(add_150, [16384, 128]);  add_150 = None
        permute_200 = torch.ops.aten.permute.default(arg438_1, [1, 0]);  arg438_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg439_1, view_400, permute_200);  arg439_1 = view_400 = permute_200 = None
        view_401 = torch.ops.aten.view.default(addmm_150, [128, 128, 512]);  addmm_150 = None
        add_151 = torch.ops.aten.add.Tensor(view_401, add_137);  view_401 = add_137 = None
        mul_81 = torch.ops.aten.mul.Tensor(add_151, arg441_1);  add_151 = arg441_1 = None
        add_152 = torch.ops.aten.add.Tensor(mul_81, arg440_1);  mul_81 = arg440_1 = None
        view_402 = torch.ops.aten.view.default(add_152, [16384, 512])
        permute_201 = torch.ops.aten.permute.default(arg488_1, [1, 0]);  arg488_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg489_1, view_402, permute_201);  arg489_1 = view_402 = permute_201 = None
        view_403 = torch.ops.aten.view.default(addmm_151, [128, 128, 128]);  addmm_151 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_403, arg491_1);  view_403 = arg491_1 = None
        add_153 = torch.ops.aten.add.Tensor(mul_82, arg490_1);  mul_82 = arg490_1 = None
        view_404 = torch.ops.aten.view.default(add_152, [16384, 512])
        permute_202 = torch.ops.aten.permute.default(arg492_1, [1, 0]);  arg492_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg493_1, view_404, permute_202);  arg493_1 = view_404 = permute_202 = None
        view_405 = torch.ops.aten.view.default(addmm_152, [128, 128, 128]);  addmm_152 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_405, arg495_1);  view_405 = arg495_1 = None
        add_154 = torch.ops.aten.add.Tensor(mul_83, arg494_1);  mul_83 = arg494_1 = None
        view_406 = torch.ops.aten.view.default(add_154, [16384, 128])
        permute_203 = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg469_1, view_406, permute_203);  arg469_1 = view_406 = permute_203 = None
        view_407 = torch.ops.aten.view.default(addmm_153, [128, 128, 128]);  addmm_153 = None
        view_408 = torch.ops.aten.view.default(add_154, [16384, 128]);  add_154 = None
        permute_204 = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg471_1, view_408, permute_204);  arg471_1 = view_408 = permute_204 = None
        view_409 = torch.ops.aten.view.default(addmm_154, [128, 128, 128]);  addmm_154 = None
        view_410 = torch.ops.aten.view.default(add_152, [16384, 512])
        permute_205 = torch.ops.aten.permute.default(arg472_1, [1, 0]);  arg472_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg473_1, view_410, permute_205);  arg473_1 = view_410 = permute_205 = None
        view_411 = torch.ops.aten.view.default(addmm_155, [128, 128, 128]);  addmm_155 = None
        view_412 = torch.ops.aten.view.default(view_407, [128, 128, 4, 32]);  view_407 = None
        view_413 = torch.ops.aten.view.default(view_409, [128, 128, 4, 32]);  view_409 = None
        view_414 = torch.ops.aten.view.default(view_411, [128, 128, 4, 32]);  view_411 = None
        permute_default_39 = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        permute_default_40 = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
        permute_default_41 = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_39, permute_default_40, permute_default_41, None, False, scale = 0.17677669529663687);  permute_default_39 = permute_default_40 = permute_default_41 = None
        getitem_15 = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
        permute_210 = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
        clone_65 = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_421 = torch.ops.aten.view.default(clone_65, [128, 128, 128]);  clone_65 = None
        view_422 = torch.ops.aten.view.default(view_421, [16384, 128]);  view_421 = None
        permute_211 = torch.ops.aten.permute.default(arg474_1, [1, 0]);  arg474_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg475_1, view_422, permute_211);  arg475_1 = view_422 = permute_211 = None
        view_423 = torch.ops.aten.view.default(addmm_156, [128, 128, 128]);  addmm_156 = None
        add_156 = torch.ops.aten.add.Tensor(view_423, add_153);  view_423 = add_153 = None
        mul_84 = torch.ops.aten.mul.Tensor(add_156, arg477_1);  add_156 = arg477_1 = None
        add_157 = torch.ops.aten.add.Tensor(mul_84, arg476_1);  mul_84 = arg476_1 = None
        view_424 = torch.ops.aten.view.default(add_157, [16384, 128])
        permute_212 = torch.ops.aten.permute.default(arg496_1, [1, 0]);  arg496_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg497_1, view_424, permute_212);  arg497_1 = view_424 = permute_212 = None
        view_425 = torch.ops.aten.view.default(addmm_157, [128, 128, 512]);  addmm_157 = None
        relu_40 = torch.ops.aten.relu.default(view_425);  view_425 = None
        view_426 = torch.ops.aten.view.default(relu_40, [16384, 512]);  relu_40 = None
        permute_213 = torch.ops.aten.permute.default(arg498_1, [1, 0]);  arg498_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg499_1, view_426, permute_213);  arg499_1 = view_426 = permute_213 = None
        view_427 = torch.ops.aten.view.default(addmm_158, [128, 128, 128]);  addmm_158 = None
        add_158 = torch.ops.aten.add.Tensor(view_427, add_157);  view_427 = add_157 = None
        mul_85 = torch.ops.aten.mul.Tensor(add_158, arg501_1);  add_158 = arg501_1 = None
        add_159 = torch.ops.aten.add.Tensor(mul_85, arg500_1);  mul_85 = arg500_1 = None
        view_428 = torch.ops.aten.view.default(add_159, [16384, 128])
        permute_214 = torch.ops.aten.permute.default(arg502_1, [1, 0]);  arg502_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg503_1, view_428, permute_214);  arg503_1 = view_428 = permute_214 = None
        view_429 = torch.ops.aten.view.default(addmm_159, [128, 128, 512]);  addmm_159 = None
        relu_41 = torch.ops.aten.relu.default(view_429);  view_429 = None
        view_430 = torch.ops.aten.view.default(relu_41, [16384, 512]);  relu_41 = None
        permute_215 = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg505_1, view_430, permute_215);  arg505_1 = view_430 = permute_215 = None
        view_431 = torch.ops.aten.view.default(addmm_160, [128, 128, 128]);  addmm_160 = None
        add_160 = torch.ops.aten.add.Tensor(view_431, add_159);  view_431 = add_159 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_160, arg507_1);  add_160 = arg507_1 = None
        add_161 = torch.ops.aten.add.Tensor(mul_86, arg506_1);  mul_86 = arg506_1 = None
        view_432 = torch.ops.aten.view.default(add_161, [16384, 128])
        permute_216 = torch.ops.aten.permute.default(arg508_1, [1, 0]);  arg508_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg509_1, view_432, permute_216);  arg509_1 = view_432 = permute_216 = None
        view_433 = torch.ops.aten.view.default(addmm_161, [128, 128, 512]);  addmm_161 = None
        relu_42 = torch.ops.aten.relu.default(view_433);  view_433 = None
        view_434 = torch.ops.aten.view.default(relu_42, [16384, 512]);  relu_42 = None
        permute_217 = torch.ops.aten.permute.default(arg510_1, [1, 0]);  arg510_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg511_1, view_434, permute_217);  arg511_1 = view_434 = permute_217 = None
        view_435 = torch.ops.aten.view.default(addmm_162, [128, 128, 128]);  addmm_162 = None
        add_162 = torch.ops.aten.add.Tensor(view_435, add_161);  view_435 = add_161 = None
        mul_87 = torch.ops.aten.mul.Tensor(add_162, arg513_1);  add_162 = arg513_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_87, arg512_1);  mul_87 = arg512_1 = None
        view_436 = torch.ops.aten.view.default(add_163, [16384, 128])
        permute_218 = torch.ops.aten.permute.default(arg478_1, [1, 0]);  arg478_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg479_1, view_436, permute_218);  arg479_1 = view_436 = permute_218 = None
        view_437 = torch.ops.aten.view.default(addmm_163, [128, 128, 512]);  addmm_163 = None
        relu_43 = torch.ops.aten.relu.default(view_437);  view_437 = None
        view_438 = torch.ops.aten.view.default(relu_43, [16384, 512]);  relu_43 = None
        permute_219 = torch.ops.aten.permute.default(arg480_1, [1, 0]);  arg480_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg481_1, view_438, permute_219);  arg481_1 = view_438 = permute_219 = None
        view_439 = torch.ops.aten.view.default(addmm_164, [128, 128, 128]);  addmm_164 = None
        add_164 = torch.ops.aten.add.Tensor(view_439, add_163);  view_439 = add_163 = None
        mul_88 = torch.ops.aten.mul.Tensor(add_164, arg483_1);  add_164 = arg483_1 = None
        add_165 = torch.ops.aten.add.Tensor(mul_88, arg482_1);  mul_88 = arg482_1 = None
        view_440 = torch.ops.aten.view.default(add_165, [16384, 128]);  add_165 = None
        permute_220 = torch.ops.aten.permute.default(arg484_1, [1, 0]);  arg484_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg485_1, view_440, permute_220);  arg485_1 = view_440 = permute_220 = None
        view_441 = torch.ops.aten.view.default(addmm_165, [128, 128, 512]);  addmm_165 = None
        add_166 = torch.ops.aten.add.Tensor(view_441, add_152);  view_441 = add_152 = None
        mul_89 = torch.ops.aten.mul.Tensor(add_166, arg487_1);  add_166 = arg487_1 = None
        add_167 = torch.ops.aten.add.Tensor(mul_89, arg486_1);  mul_89 = arg486_1 = None
        view_442 = torch.ops.aten.view.default(add_167, [16384, 512])
        permute_221 = torch.ops.aten.permute.default(arg534_1, [1, 0]);  arg534_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg535_1, view_442, permute_221);  arg535_1 = view_442 = permute_221 = None
        view_443 = torch.ops.aten.view.default(addmm_166, [128, 128, 128]);  addmm_166 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_443, arg537_1);  view_443 = arg537_1 = None
        add_168 = torch.ops.aten.add.Tensor(mul_90, arg536_1);  mul_90 = arg536_1 = None
        view_444 = torch.ops.aten.view.default(add_167, [16384, 512])
        permute_222 = torch.ops.aten.permute.default(arg538_1, [1, 0]);  arg538_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg539_1, view_444, permute_222);  arg539_1 = view_444 = permute_222 = None
        view_445 = torch.ops.aten.view.default(addmm_167, [128, 128, 128]);  addmm_167 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_445, arg541_1);  view_445 = arg541_1 = None
        add_169 = torch.ops.aten.add.Tensor(mul_91, arg540_1);  mul_91 = arg540_1 = None
        view_446 = torch.ops.aten.view.default(add_169, [16384, 128])
        permute_223 = torch.ops.aten.permute.default(arg514_1, [1, 0]);  arg514_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg515_1, view_446, permute_223);  arg515_1 = view_446 = permute_223 = None
        view_447 = torch.ops.aten.view.default(addmm_168, [128, 128, 128]);  addmm_168 = None
        view_448 = torch.ops.aten.view.default(add_169, [16384, 128]);  add_169 = None
        permute_224 = torch.ops.aten.permute.default(arg516_1, [1, 0]);  arg516_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg517_1, view_448, permute_224);  arg517_1 = view_448 = permute_224 = None
        view_449 = torch.ops.aten.view.default(addmm_169, [128, 128, 128]);  addmm_169 = None
        view_450 = torch.ops.aten.view.default(add_167, [16384, 512])
        permute_225 = torch.ops.aten.permute.default(arg518_1, [1, 0]);  arg518_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg519_1, view_450, permute_225);  arg519_1 = view_450 = permute_225 = None
        view_451 = torch.ops.aten.view.default(addmm_170, [128, 128, 128]);  addmm_170 = None
        view_452 = torch.ops.aten.view.default(view_447, [128, 128, 4, 32]);  view_447 = None
        view_453 = torch.ops.aten.view.default(view_449, [128, 128, 4, 32]);  view_449 = None
        view_454 = torch.ops.aten.view.default(view_451, [128, 128, 4, 32]);  view_451 = None
        permute_default_36 = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        permute_default_37 = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        permute_default_38 = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_36, permute_default_37, permute_default_38, None, False, scale = 0.17677669529663687);  permute_default_36 = permute_default_37 = permute_default_38 = None
        getitem_14 = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
        permute_230 = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3]);  getitem_14 = None
        clone_71 = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_461 = torch.ops.aten.view.default(clone_71, [128, 128, 128]);  clone_71 = None
        view_462 = torch.ops.aten.view.default(view_461, [16384, 128]);  view_461 = None
        permute_231 = torch.ops.aten.permute.default(arg520_1, [1, 0]);  arg520_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg521_1, view_462, permute_231);  arg521_1 = view_462 = permute_231 = None
        view_463 = torch.ops.aten.view.default(addmm_171, [128, 128, 128]);  addmm_171 = None
        add_171 = torch.ops.aten.add.Tensor(view_463, add_168);  view_463 = add_168 = None
        mul_92 = torch.ops.aten.mul.Tensor(add_171, arg523_1);  add_171 = arg523_1 = None
        add_172 = torch.ops.aten.add.Tensor(mul_92, arg522_1);  mul_92 = arg522_1 = None
        view_464 = torch.ops.aten.view.default(add_172, [16384, 128])
        permute_232 = torch.ops.aten.permute.default(arg542_1, [1, 0]);  arg542_1 = None
        addmm_172 = torch.ops.aten.addmm.default(arg543_1, view_464, permute_232);  arg543_1 = view_464 = permute_232 = None
        view_465 = torch.ops.aten.view.default(addmm_172, [128, 128, 512]);  addmm_172 = None
        relu_44 = torch.ops.aten.relu.default(view_465);  view_465 = None
        view_466 = torch.ops.aten.view.default(relu_44, [16384, 512]);  relu_44 = None
        permute_233 = torch.ops.aten.permute.default(arg544_1, [1, 0]);  arg544_1 = None
        addmm_173 = torch.ops.aten.addmm.default(arg545_1, view_466, permute_233);  arg545_1 = view_466 = permute_233 = None
        view_467 = torch.ops.aten.view.default(addmm_173, [128, 128, 128]);  addmm_173 = None
        add_173 = torch.ops.aten.add.Tensor(view_467, add_172);  view_467 = add_172 = None
        mul_93 = torch.ops.aten.mul.Tensor(add_173, arg547_1);  add_173 = arg547_1 = None
        add_174 = torch.ops.aten.add.Tensor(mul_93, arg546_1);  mul_93 = arg546_1 = None
        view_468 = torch.ops.aten.view.default(add_174, [16384, 128])
        permute_234 = torch.ops.aten.permute.default(arg548_1, [1, 0]);  arg548_1 = None
        addmm_174 = torch.ops.aten.addmm.default(arg549_1, view_468, permute_234);  arg549_1 = view_468 = permute_234 = None
        view_469 = torch.ops.aten.view.default(addmm_174, [128, 128, 512]);  addmm_174 = None
        relu_45 = torch.ops.aten.relu.default(view_469);  view_469 = None
        view_470 = torch.ops.aten.view.default(relu_45, [16384, 512]);  relu_45 = None
        permute_235 = torch.ops.aten.permute.default(arg550_1, [1, 0]);  arg550_1 = None
        addmm_175 = torch.ops.aten.addmm.default(arg551_1, view_470, permute_235);  arg551_1 = view_470 = permute_235 = None
        view_471 = torch.ops.aten.view.default(addmm_175, [128, 128, 128]);  addmm_175 = None
        add_175 = torch.ops.aten.add.Tensor(view_471, add_174);  view_471 = add_174 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_175, arg553_1);  add_175 = arg553_1 = None
        add_176 = torch.ops.aten.add.Tensor(mul_94, arg552_1);  mul_94 = arg552_1 = None
        view_472 = torch.ops.aten.view.default(add_176, [16384, 128])
        permute_236 = torch.ops.aten.permute.default(arg554_1, [1, 0]);  arg554_1 = None
        addmm_176 = torch.ops.aten.addmm.default(arg555_1, view_472, permute_236);  arg555_1 = view_472 = permute_236 = None
        view_473 = torch.ops.aten.view.default(addmm_176, [128, 128, 512]);  addmm_176 = None
        relu_46 = torch.ops.aten.relu.default(view_473);  view_473 = None
        view_474 = torch.ops.aten.view.default(relu_46, [16384, 512]);  relu_46 = None
        permute_237 = torch.ops.aten.permute.default(arg556_1, [1, 0]);  arg556_1 = None
        addmm_177 = torch.ops.aten.addmm.default(arg557_1, view_474, permute_237);  arg557_1 = view_474 = permute_237 = None
        view_475 = torch.ops.aten.view.default(addmm_177, [128, 128, 128]);  addmm_177 = None
        add_177 = torch.ops.aten.add.Tensor(view_475, add_176);  view_475 = add_176 = None
        mul_95 = torch.ops.aten.mul.Tensor(add_177, arg559_1);  add_177 = arg559_1 = None
        add_178 = torch.ops.aten.add.Tensor(mul_95, arg558_1);  mul_95 = arg558_1 = None
        view_476 = torch.ops.aten.view.default(add_178, [16384, 128])
        permute_238 = torch.ops.aten.permute.default(arg524_1, [1, 0]);  arg524_1 = None
        addmm_178 = torch.ops.aten.addmm.default(arg525_1, view_476, permute_238);  arg525_1 = view_476 = permute_238 = None
        view_477 = torch.ops.aten.view.default(addmm_178, [128, 128, 512]);  addmm_178 = None
        relu_47 = torch.ops.aten.relu.default(view_477);  view_477 = None
        view_478 = torch.ops.aten.view.default(relu_47, [16384, 512]);  relu_47 = None
        permute_239 = torch.ops.aten.permute.default(arg526_1, [1, 0]);  arg526_1 = None
        addmm_179 = torch.ops.aten.addmm.default(arg527_1, view_478, permute_239);  arg527_1 = view_478 = permute_239 = None
        view_479 = torch.ops.aten.view.default(addmm_179, [128, 128, 128]);  addmm_179 = None
        add_179 = torch.ops.aten.add.Tensor(view_479, add_178);  view_479 = add_178 = None
        mul_96 = torch.ops.aten.mul.Tensor(add_179, arg529_1);  add_179 = arg529_1 = None
        add_180 = torch.ops.aten.add.Tensor(mul_96, arg528_1);  mul_96 = arg528_1 = None
        view_480 = torch.ops.aten.view.default(add_180, [16384, 128]);  add_180 = None
        permute_240 = torch.ops.aten.permute.default(arg530_1, [1, 0]);  arg530_1 = None
        addmm_180 = torch.ops.aten.addmm.default(arg531_1, view_480, permute_240);  arg531_1 = view_480 = permute_240 = None
        view_481 = torch.ops.aten.view.default(addmm_180, [128, 128, 512]);  addmm_180 = None
        add_181 = torch.ops.aten.add.Tensor(view_481, add_167);  view_481 = add_167 = None
        mul_97 = torch.ops.aten.mul.Tensor(add_181, arg533_1);  add_181 = arg533_1 = None
        add_182 = torch.ops.aten.add.Tensor(mul_97, arg532_1);  mul_97 = arg532_1 = None
        view_482 = torch.ops.aten.view.default(add_182, [16384, 512])
        permute_241 = torch.ops.aten.permute.default(arg580_1, [1, 0]);  arg580_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg581_1, view_482, permute_241);  arg581_1 = view_482 = permute_241 = None
        view_483 = torch.ops.aten.view.default(addmm_181, [128, 128, 128]);  addmm_181 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_483, arg583_1);  view_483 = arg583_1 = None
        add_183 = torch.ops.aten.add.Tensor(mul_98, arg582_1);  mul_98 = arg582_1 = None
        view_484 = torch.ops.aten.view.default(add_182, [16384, 512])
        permute_242 = torch.ops.aten.permute.default(arg584_1, [1, 0]);  arg584_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg585_1, view_484, permute_242);  arg585_1 = view_484 = permute_242 = None
        view_485 = torch.ops.aten.view.default(addmm_182, [128, 128, 128]);  addmm_182 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_485, arg587_1);  view_485 = arg587_1 = None
        add_184 = torch.ops.aten.add.Tensor(mul_99, arg586_1);  mul_99 = arg586_1 = None
        view_486 = torch.ops.aten.view.default(add_184, [16384, 128])
        permute_243 = torch.ops.aten.permute.default(arg560_1, [1, 0]);  arg560_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg561_1, view_486, permute_243);  arg561_1 = view_486 = permute_243 = None
        view_487 = torch.ops.aten.view.default(addmm_183, [128, 128, 128]);  addmm_183 = None
        view_488 = torch.ops.aten.view.default(add_184, [16384, 128]);  add_184 = None
        permute_244 = torch.ops.aten.permute.default(arg562_1, [1, 0]);  arg562_1 = None
        addmm_184 = torch.ops.aten.addmm.default(arg563_1, view_488, permute_244);  arg563_1 = view_488 = permute_244 = None
        view_489 = torch.ops.aten.view.default(addmm_184, [128, 128, 128]);  addmm_184 = None
        view_490 = torch.ops.aten.view.default(add_182, [16384, 512])
        permute_245 = torch.ops.aten.permute.default(arg564_1, [1, 0]);  arg564_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg565_1, view_490, permute_245);  arg565_1 = view_490 = permute_245 = None
        view_491 = torch.ops.aten.view.default(addmm_185, [128, 128, 128]);  addmm_185 = None
        view_492 = torch.ops.aten.view.default(view_487, [128, 128, 4, 32]);  view_487 = None
        view_493 = torch.ops.aten.view.default(view_489, [128, 128, 4, 32]);  view_489 = None
        view_494 = torch.ops.aten.view.default(view_491, [128, 128, 4, 32]);  view_491 = None
        permute_default_33 = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        permute_default_34 = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
        permute_default_35 = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.17677669529663687);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_13 = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        permute_250 = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3]);  getitem_13 = None
        clone_77 = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_501 = torch.ops.aten.view.default(clone_77, [128, 128, 128]);  clone_77 = None
        view_502 = torch.ops.aten.view.default(view_501, [16384, 128]);  view_501 = None
        permute_251 = torch.ops.aten.permute.default(arg566_1, [1, 0]);  arg566_1 = None
        addmm_186 = torch.ops.aten.addmm.default(arg567_1, view_502, permute_251);  arg567_1 = view_502 = permute_251 = None
        view_503 = torch.ops.aten.view.default(addmm_186, [128, 128, 128]);  addmm_186 = None
        add_186 = torch.ops.aten.add.Tensor(view_503, add_183);  view_503 = add_183 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_186, arg569_1);  add_186 = arg569_1 = None
        add_187 = torch.ops.aten.add.Tensor(mul_100, arg568_1);  mul_100 = arg568_1 = None
        view_504 = torch.ops.aten.view.default(add_187, [16384, 128])
        permute_252 = torch.ops.aten.permute.default(arg588_1, [1, 0]);  arg588_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg589_1, view_504, permute_252);  arg589_1 = view_504 = permute_252 = None
        view_505 = torch.ops.aten.view.default(addmm_187, [128, 128, 512]);  addmm_187 = None
        relu_48 = torch.ops.aten.relu.default(view_505);  view_505 = None
        view_506 = torch.ops.aten.view.default(relu_48, [16384, 512]);  relu_48 = None
        permute_253 = torch.ops.aten.permute.default(arg590_1, [1, 0]);  arg590_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg591_1, view_506, permute_253);  arg591_1 = view_506 = permute_253 = None
        view_507 = torch.ops.aten.view.default(addmm_188, [128, 128, 128]);  addmm_188 = None
        add_188 = torch.ops.aten.add.Tensor(view_507, add_187);  view_507 = add_187 = None
        mul_101 = torch.ops.aten.mul.Tensor(add_188, arg593_1);  add_188 = arg593_1 = None
        add_189 = torch.ops.aten.add.Tensor(mul_101, arg592_1);  mul_101 = arg592_1 = None
        view_508 = torch.ops.aten.view.default(add_189, [16384, 128])
        permute_254 = torch.ops.aten.permute.default(arg594_1, [1, 0]);  arg594_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg595_1, view_508, permute_254);  arg595_1 = view_508 = permute_254 = None
        view_509 = torch.ops.aten.view.default(addmm_189, [128, 128, 512]);  addmm_189 = None
        relu_49 = torch.ops.aten.relu.default(view_509);  view_509 = None
        view_510 = torch.ops.aten.view.default(relu_49, [16384, 512]);  relu_49 = None
        permute_255 = torch.ops.aten.permute.default(arg596_1, [1, 0]);  arg596_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg597_1, view_510, permute_255);  arg597_1 = view_510 = permute_255 = None
        view_511 = torch.ops.aten.view.default(addmm_190, [128, 128, 128]);  addmm_190 = None
        add_190 = torch.ops.aten.add.Tensor(view_511, add_189);  view_511 = add_189 = None
        mul_102 = torch.ops.aten.mul.Tensor(add_190, arg599_1);  add_190 = arg599_1 = None
        add_191 = torch.ops.aten.add.Tensor(mul_102, arg598_1);  mul_102 = arg598_1 = None
        view_512 = torch.ops.aten.view.default(add_191, [16384, 128])
        permute_256 = torch.ops.aten.permute.default(arg600_1, [1, 0]);  arg600_1 = None
        addmm_191 = torch.ops.aten.addmm.default(arg601_1, view_512, permute_256);  arg601_1 = view_512 = permute_256 = None
        view_513 = torch.ops.aten.view.default(addmm_191, [128, 128, 512]);  addmm_191 = None
        relu_50 = torch.ops.aten.relu.default(view_513);  view_513 = None
        view_514 = torch.ops.aten.view.default(relu_50, [16384, 512]);  relu_50 = None
        permute_257 = torch.ops.aten.permute.default(arg602_1, [1, 0]);  arg602_1 = None
        addmm_192 = torch.ops.aten.addmm.default(arg603_1, view_514, permute_257);  arg603_1 = view_514 = permute_257 = None
        view_515 = torch.ops.aten.view.default(addmm_192, [128, 128, 128]);  addmm_192 = None
        add_192 = torch.ops.aten.add.Tensor(view_515, add_191);  view_515 = add_191 = None
        mul_103 = torch.ops.aten.mul.Tensor(add_192, arg605_1);  add_192 = arg605_1 = None
        add_193 = torch.ops.aten.add.Tensor(mul_103, arg604_1);  mul_103 = arg604_1 = None
        view_516 = torch.ops.aten.view.default(add_193, [16384, 128])
        permute_258 = torch.ops.aten.permute.default(arg570_1, [1, 0]);  arg570_1 = None
        addmm_193 = torch.ops.aten.addmm.default(arg571_1, view_516, permute_258);  arg571_1 = view_516 = permute_258 = None
        view_517 = torch.ops.aten.view.default(addmm_193, [128, 128, 512]);  addmm_193 = None
        relu_51 = torch.ops.aten.relu.default(view_517);  view_517 = None
        view_518 = torch.ops.aten.view.default(relu_51, [16384, 512]);  relu_51 = None
        permute_259 = torch.ops.aten.permute.default(arg572_1, [1, 0]);  arg572_1 = None
        addmm_194 = torch.ops.aten.addmm.default(arg573_1, view_518, permute_259);  arg573_1 = view_518 = permute_259 = None
        view_519 = torch.ops.aten.view.default(addmm_194, [128, 128, 128]);  addmm_194 = None
        add_194 = torch.ops.aten.add.Tensor(view_519, add_193);  view_519 = add_193 = None
        mul_104 = torch.ops.aten.mul.Tensor(add_194, arg575_1);  add_194 = arg575_1 = None
        add_195 = torch.ops.aten.add.Tensor(mul_104, arg574_1);  mul_104 = arg574_1 = None
        view_520 = torch.ops.aten.view.default(add_195, [16384, 128]);  add_195 = None
        permute_260 = torch.ops.aten.permute.default(arg576_1, [1, 0]);  arg576_1 = None
        addmm_195 = torch.ops.aten.addmm.default(arg577_1, view_520, permute_260);  arg577_1 = view_520 = permute_260 = None
        view_521 = torch.ops.aten.view.default(addmm_195, [128, 128, 512]);  addmm_195 = None
        add_196 = torch.ops.aten.add.Tensor(view_521, add_182);  view_521 = add_182 = None
        mul_105 = torch.ops.aten.mul.Tensor(add_196, arg579_1);  add_196 = arg579_1 = None
        add_197 = torch.ops.aten.add.Tensor(mul_105, arg578_1);  mul_105 = arg578_1 = None
        view_522 = torch.ops.aten.view.default(add_197, [16384, 512])
        permute_261 = torch.ops.aten.permute.default(arg626_1, [1, 0]);  arg626_1 = None
        addmm_196 = torch.ops.aten.addmm.default(arg627_1, view_522, permute_261);  arg627_1 = view_522 = permute_261 = None
        view_523 = torch.ops.aten.view.default(addmm_196, [128, 128, 128]);  addmm_196 = None
        mul_106 = torch.ops.aten.mul.Tensor(view_523, arg629_1);  view_523 = arg629_1 = None
        add_198 = torch.ops.aten.add.Tensor(mul_106, arg628_1);  mul_106 = arg628_1 = None
        view_524 = torch.ops.aten.view.default(add_197, [16384, 512])
        permute_262 = torch.ops.aten.permute.default(arg630_1, [1, 0]);  arg630_1 = None
        addmm_197 = torch.ops.aten.addmm.default(arg631_1, view_524, permute_262);  arg631_1 = view_524 = permute_262 = None
        view_525 = torch.ops.aten.view.default(addmm_197, [128, 128, 128]);  addmm_197 = None
        mul_107 = torch.ops.aten.mul.Tensor(view_525, arg633_1);  view_525 = arg633_1 = None
        add_199 = torch.ops.aten.add.Tensor(mul_107, arg632_1);  mul_107 = arg632_1 = None
        view_526 = torch.ops.aten.view.default(add_199, [16384, 128])
        permute_263 = torch.ops.aten.permute.default(arg606_1, [1, 0]);  arg606_1 = None
        addmm_198 = torch.ops.aten.addmm.default(arg607_1, view_526, permute_263);  arg607_1 = view_526 = permute_263 = None
        view_527 = torch.ops.aten.view.default(addmm_198, [128, 128, 128]);  addmm_198 = None
        view_528 = torch.ops.aten.view.default(add_199, [16384, 128]);  add_199 = None
        permute_264 = torch.ops.aten.permute.default(arg608_1, [1, 0]);  arg608_1 = None
        addmm_199 = torch.ops.aten.addmm.default(arg609_1, view_528, permute_264);  arg609_1 = view_528 = permute_264 = None
        view_529 = torch.ops.aten.view.default(addmm_199, [128, 128, 128]);  addmm_199 = None
        view_530 = torch.ops.aten.view.default(add_197, [16384, 512])
        permute_265 = torch.ops.aten.permute.default(arg610_1, [1, 0]);  arg610_1 = None
        addmm_200 = torch.ops.aten.addmm.default(arg611_1, view_530, permute_265);  arg611_1 = view_530 = permute_265 = None
        view_531 = torch.ops.aten.view.default(addmm_200, [128, 128, 128]);  addmm_200 = None
        view_532 = torch.ops.aten.view.default(view_527, [128, 128, 4, 32]);  view_527 = None
        view_533 = torch.ops.aten.view.default(view_529, [128, 128, 4, 32]);  view_529 = None
        view_534 = torch.ops.aten.view.default(view_531, [128, 128, 4, 32]);  view_531 = None
        permute_default_30 = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        permute_default_31 = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
        permute_default_32 = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.17677669529663687);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_12 = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        permute_270 = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
        clone_83 = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
        view_541 = torch.ops.aten.view.default(clone_83, [128, 128, 128]);  clone_83 = None
        view_542 = torch.ops.aten.view.default(view_541, [16384, 128]);  view_541 = None
        permute_271 = torch.ops.aten.permute.default(arg612_1, [1, 0]);  arg612_1 = None
        addmm_201 = torch.ops.aten.addmm.default(arg613_1, view_542, permute_271);  arg613_1 = view_542 = permute_271 = None
        view_543 = torch.ops.aten.view.default(addmm_201, [128, 128, 128]);  addmm_201 = None
        add_201 = torch.ops.aten.add.Tensor(view_543, add_198);  view_543 = add_198 = None
        mul_108 = torch.ops.aten.mul.Tensor(add_201, arg615_1);  add_201 = arg615_1 = None
        add_202 = torch.ops.aten.add.Tensor(mul_108, arg614_1);  mul_108 = arg614_1 = None
        view_544 = torch.ops.aten.view.default(add_202, [16384, 128])
        permute_272 = torch.ops.aten.permute.default(arg634_1, [1, 0]);  arg634_1 = None
        addmm_202 = torch.ops.aten.addmm.default(arg635_1, view_544, permute_272);  arg635_1 = view_544 = permute_272 = None
        view_545 = torch.ops.aten.view.default(addmm_202, [128, 128, 512]);  addmm_202 = None
        relu_52 = torch.ops.aten.relu.default(view_545);  view_545 = None
        view_546 = torch.ops.aten.view.default(relu_52, [16384, 512]);  relu_52 = None
        permute_273 = torch.ops.aten.permute.default(arg636_1, [1, 0]);  arg636_1 = None
        addmm_203 = torch.ops.aten.addmm.default(arg637_1, view_546, permute_273);  arg637_1 = view_546 = permute_273 = None
        view_547 = torch.ops.aten.view.default(addmm_203, [128, 128, 128]);  addmm_203 = None
        add_203 = torch.ops.aten.add.Tensor(view_547, add_202);  view_547 = add_202 = None
        mul_109 = torch.ops.aten.mul.Tensor(add_203, arg639_1);  add_203 = arg639_1 = None
        add_204 = torch.ops.aten.add.Tensor(mul_109, arg638_1);  mul_109 = arg638_1 = None
        view_548 = torch.ops.aten.view.default(add_204, [16384, 128])
        permute_274 = torch.ops.aten.permute.default(arg640_1, [1, 0]);  arg640_1 = None
        addmm_204 = torch.ops.aten.addmm.default(arg641_1, view_548, permute_274);  arg641_1 = view_548 = permute_274 = None
        view_549 = torch.ops.aten.view.default(addmm_204, [128, 128, 512]);  addmm_204 = None
        relu_53 = torch.ops.aten.relu.default(view_549);  view_549 = None
        view_550 = torch.ops.aten.view.default(relu_53, [16384, 512]);  relu_53 = None
        permute_275 = torch.ops.aten.permute.default(arg642_1, [1, 0]);  arg642_1 = None
        addmm_205 = torch.ops.aten.addmm.default(arg643_1, view_550, permute_275);  arg643_1 = view_550 = permute_275 = None
        view_551 = torch.ops.aten.view.default(addmm_205, [128, 128, 128]);  addmm_205 = None
        add_205 = torch.ops.aten.add.Tensor(view_551, add_204);  view_551 = add_204 = None
        mul_110 = torch.ops.aten.mul.Tensor(add_205, arg645_1);  add_205 = arg645_1 = None
        add_206 = torch.ops.aten.add.Tensor(mul_110, arg644_1);  mul_110 = arg644_1 = None
        view_552 = torch.ops.aten.view.default(add_206, [16384, 128])
        permute_276 = torch.ops.aten.permute.default(arg646_1, [1, 0]);  arg646_1 = None
        addmm_206 = torch.ops.aten.addmm.default(arg647_1, view_552, permute_276);  arg647_1 = view_552 = permute_276 = None
        view_553 = torch.ops.aten.view.default(addmm_206, [128, 128, 512]);  addmm_206 = None
        relu_54 = torch.ops.aten.relu.default(view_553);  view_553 = None
        view_554 = torch.ops.aten.view.default(relu_54, [16384, 512]);  relu_54 = None
        permute_277 = torch.ops.aten.permute.default(arg648_1, [1, 0]);  arg648_1 = None
        addmm_207 = torch.ops.aten.addmm.default(arg649_1, view_554, permute_277);  arg649_1 = view_554 = permute_277 = None
        view_555 = torch.ops.aten.view.default(addmm_207, [128, 128, 128]);  addmm_207 = None
        add_207 = torch.ops.aten.add.Tensor(view_555, add_206);  view_555 = add_206 = None
        mul_111 = torch.ops.aten.mul.Tensor(add_207, arg651_1);  add_207 = arg651_1 = None
        add_208 = torch.ops.aten.add.Tensor(mul_111, arg650_1);  mul_111 = arg650_1 = None
        view_556 = torch.ops.aten.view.default(add_208, [16384, 128])
        permute_278 = torch.ops.aten.permute.default(arg616_1, [1, 0]);  arg616_1 = None
        addmm_208 = torch.ops.aten.addmm.default(arg617_1, view_556, permute_278);  arg617_1 = view_556 = permute_278 = None
        view_557 = torch.ops.aten.view.default(addmm_208, [128, 128, 512]);  addmm_208 = None
        relu_55 = torch.ops.aten.relu.default(view_557);  view_557 = None
        view_558 = torch.ops.aten.view.default(relu_55, [16384, 512]);  relu_55 = None
        permute_279 = torch.ops.aten.permute.default(arg618_1, [1, 0]);  arg618_1 = None
        addmm_209 = torch.ops.aten.addmm.default(arg619_1, view_558, permute_279);  arg619_1 = view_558 = permute_279 = None
        view_559 = torch.ops.aten.view.default(addmm_209, [128, 128, 128]);  addmm_209 = None
        add_209 = torch.ops.aten.add.Tensor(view_559, add_208);  view_559 = add_208 = None
        mul_112 = torch.ops.aten.mul.Tensor(add_209, arg621_1);  add_209 = arg621_1 = None
        add_210 = torch.ops.aten.add.Tensor(mul_112, arg620_1);  mul_112 = arg620_1 = None
        view_560 = torch.ops.aten.view.default(add_210, [16384, 128]);  add_210 = None
        permute_280 = torch.ops.aten.permute.default(arg622_1, [1, 0]);  arg622_1 = None
        addmm_210 = torch.ops.aten.addmm.default(arg623_1, view_560, permute_280);  arg623_1 = view_560 = permute_280 = None
        view_561 = torch.ops.aten.view.default(addmm_210, [128, 128, 512]);  addmm_210 = None
        add_211 = torch.ops.aten.add.Tensor(view_561, add_197);  view_561 = add_197 = None
        mul_113 = torch.ops.aten.mul.Tensor(add_211, arg625_1);  add_211 = arg625_1 = None
        add_212 = torch.ops.aten.add.Tensor(mul_113, arg624_1);  mul_113 = arg624_1 = None
        view_562 = torch.ops.aten.view.default(add_212, [16384, 512])
        permute_281 = torch.ops.aten.permute.default(arg672_1, [1, 0]);  arg672_1 = None
        addmm_211 = torch.ops.aten.addmm.default(arg673_1, view_562, permute_281);  arg673_1 = view_562 = permute_281 = None
        view_563 = torch.ops.aten.view.default(addmm_211, [128, 128, 128]);  addmm_211 = None
        mul_114 = torch.ops.aten.mul.Tensor(view_563, arg675_1);  view_563 = arg675_1 = None
        add_213 = torch.ops.aten.add.Tensor(mul_114, arg674_1);  mul_114 = arg674_1 = None
        view_564 = torch.ops.aten.view.default(add_212, [16384, 512])
        permute_282 = torch.ops.aten.permute.default(arg676_1, [1, 0]);  arg676_1 = None
        addmm_212 = torch.ops.aten.addmm.default(arg677_1, view_564, permute_282);  arg677_1 = view_564 = permute_282 = None
        view_565 = torch.ops.aten.view.default(addmm_212, [128, 128, 128]);  addmm_212 = None
        mul_115 = torch.ops.aten.mul.Tensor(view_565, arg679_1);  view_565 = arg679_1 = None
        add_214 = torch.ops.aten.add.Tensor(mul_115, arg678_1);  mul_115 = arg678_1 = None
        view_566 = torch.ops.aten.view.default(add_214, [16384, 128])
        permute_283 = torch.ops.aten.permute.default(arg652_1, [1, 0]);  arg652_1 = None
        addmm_213 = torch.ops.aten.addmm.default(arg653_1, view_566, permute_283);  arg653_1 = view_566 = permute_283 = None
        view_567 = torch.ops.aten.view.default(addmm_213, [128, 128, 128]);  addmm_213 = None
        view_568 = torch.ops.aten.view.default(add_214, [16384, 128]);  add_214 = None
        permute_284 = torch.ops.aten.permute.default(arg654_1, [1, 0]);  arg654_1 = None
        addmm_214 = torch.ops.aten.addmm.default(arg655_1, view_568, permute_284);  arg655_1 = view_568 = permute_284 = None
        view_569 = torch.ops.aten.view.default(addmm_214, [128, 128, 128]);  addmm_214 = None
        view_570 = torch.ops.aten.view.default(add_212, [16384, 512])
        permute_285 = torch.ops.aten.permute.default(arg656_1, [1, 0]);  arg656_1 = None
        addmm_215 = torch.ops.aten.addmm.default(arg657_1, view_570, permute_285);  arg657_1 = view_570 = permute_285 = None
        view_571 = torch.ops.aten.view.default(addmm_215, [128, 128, 128]);  addmm_215 = None
        view_572 = torch.ops.aten.view.default(view_567, [128, 128, 4, 32]);  view_567 = None
        view_573 = torch.ops.aten.view.default(view_569, [128, 128, 4, 32]);  view_569 = None
        view_574 = torch.ops.aten.view.default(view_571, [128, 128, 4, 32]);  view_571 = None
        permute_default_27 = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        permute_default_28 = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
        permute_default_29 = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.17677669529663687);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_11 = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        permute_290 = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
        clone_89 = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        view_581 = torch.ops.aten.view.default(clone_89, [128, 128, 128]);  clone_89 = None
        view_582 = torch.ops.aten.view.default(view_581, [16384, 128]);  view_581 = None
        permute_291 = torch.ops.aten.permute.default(arg658_1, [1, 0]);  arg658_1 = None
        addmm_216 = torch.ops.aten.addmm.default(arg659_1, view_582, permute_291);  arg659_1 = view_582 = permute_291 = None
        view_583 = torch.ops.aten.view.default(addmm_216, [128, 128, 128]);  addmm_216 = None
        add_216 = torch.ops.aten.add.Tensor(view_583, add_213);  view_583 = add_213 = None
        mul_116 = torch.ops.aten.mul.Tensor(add_216, arg661_1);  add_216 = arg661_1 = None
        add_217 = torch.ops.aten.add.Tensor(mul_116, arg660_1);  mul_116 = arg660_1 = None
        view_584 = torch.ops.aten.view.default(add_217, [16384, 128])
        permute_292 = torch.ops.aten.permute.default(arg680_1, [1, 0]);  arg680_1 = None
        addmm_217 = torch.ops.aten.addmm.default(arg681_1, view_584, permute_292);  arg681_1 = view_584 = permute_292 = None
        view_585 = torch.ops.aten.view.default(addmm_217, [128, 128, 512]);  addmm_217 = None
        relu_56 = torch.ops.aten.relu.default(view_585);  view_585 = None
        view_586 = torch.ops.aten.view.default(relu_56, [16384, 512]);  relu_56 = None
        permute_293 = torch.ops.aten.permute.default(arg682_1, [1, 0]);  arg682_1 = None
        addmm_218 = torch.ops.aten.addmm.default(arg683_1, view_586, permute_293);  arg683_1 = view_586 = permute_293 = None
        view_587 = torch.ops.aten.view.default(addmm_218, [128, 128, 128]);  addmm_218 = None
        add_218 = torch.ops.aten.add.Tensor(view_587, add_217);  view_587 = add_217 = None
        mul_117 = torch.ops.aten.mul.Tensor(add_218, arg685_1);  add_218 = arg685_1 = None
        add_219 = torch.ops.aten.add.Tensor(mul_117, arg684_1);  mul_117 = arg684_1 = None
        view_588 = torch.ops.aten.view.default(add_219, [16384, 128])
        permute_294 = torch.ops.aten.permute.default(arg686_1, [1, 0]);  arg686_1 = None
        addmm_219 = torch.ops.aten.addmm.default(arg687_1, view_588, permute_294);  arg687_1 = view_588 = permute_294 = None
        view_589 = torch.ops.aten.view.default(addmm_219, [128, 128, 512]);  addmm_219 = None
        relu_57 = torch.ops.aten.relu.default(view_589);  view_589 = None
        view_590 = torch.ops.aten.view.default(relu_57, [16384, 512]);  relu_57 = None
        permute_295 = torch.ops.aten.permute.default(arg688_1, [1, 0]);  arg688_1 = None
        addmm_220 = torch.ops.aten.addmm.default(arg689_1, view_590, permute_295);  arg689_1 = view_590 = permute_295 = None
        view_591 = torch.ops.aten.view.default(addmm_220, [128, 128, 128]);  addmm_220 = None
        add_220 = torch.ops.aten.add.Tensor(view_591, add_219);  view_591 = add_219 = None
        mul_118 = torch.ops.aten.mul.Tensor(add_220, arg691_1);  add_220 = arg691_1 = None
        add_221 = torch.ops.aten.add.Tensor(mul_118, arg690_1);  mul_118 = arg690_1 = None
        view_592 = torch.ops.aten.view.default(add_221, [16384, 128])
        permute_296 = torch.ops.aten.permute.default(arg692_1, [1, 0]);  arg692_1 = None
        addmm_221 = torch.ops.aten.addmm.default(arg693_1, view_592, permute_296);  arg693_1 = view_592 = permute_296 = None
        view_593 = torch.ops.aten.view.default(addmm_221, [128, 128, 512]);  addmm_221 = None
        relu_58 = torch.ops.aten.relu.default(view_593);  view_593 = None
        view_594 = torch.ops.aten.view.default(relu_58, [16384, 512]);  relu_58 = None
        permute_297 = torch.ops.aten.permute.default(arg694_1, [1, 0]);  arg694_1 = None
        addmm_222 = torch.ops.aten.addmm.default(arg695_1, view_594, permute_297);  arg695_1 = view_594 = permute_297 = None
        view_595 = torch.ops.aten.view.default(addmm_222, [128, 128, 128]);  addmm_222 = None
        add_222 = torch.ops.aten.add.Tensor(view_595, add_221);  view_595 = add_221 = None
        mul_119 = torch.ops.aten.mul.Tensor(add_222, arg697_1);  add_222 = arg697_1 = None
        add_223 = torch.ops.aten.add.Tensor(mul_119, arg696_1);  mul_119 = arg696_1 = None
        view_596 = torch.ops.aten.view.default(add_223, [16384, 128])
        permute_298 = torch.ops.aten.permute.default(arg662_1, [1, 0]);  arg662_1 = None
        addmm_223 = torch.ops.aten.addmm.default(arg663_1, view_596, permute_298);  arg663_1 = view_596 = permute_298 = None
        view_597 = torch.ops.aten.view.default(addmm_223, [128, 128, 512]);  addmm_223 = None
        relu_59 = torch.ops.aten.relu.default(view_597);  view_597 = None
        view_598 = torch.ops.aten.view.default(relu_59, [16384, 512]);  relu_59 = None
        permute_299 = torch.ops.aten.permute.default(arg664_1, [1, 0]);  arg664_1 = None
        addmm_224 = torch.ops.aten.addmm.default(arg665_1, view_598, permute_299);  arg665_1 = view_598 = permute_299 = None
        view_599 = torch.ops.aten.view.default(addmm_224, [128, 128, 128]);  addmm_224 = None
        add_224 = torch.ops.aten.add.Tensor(view_599, add_223);  view_599 = add_223 = None
        mul_120 = torch.ops.aten.mul.Tensor(add_224, arg667_1);  add_224 = arg667_1 = None
        add_225 = torch.ops.aten.add.Tensor(mul_120, arg666_1);  mul_120 = arg666_1 = None
        view_600 = torch.ops.aten.view.default(add_225, [16384, 128]);  add_225 = None
        permute_300 = torch.ops.aten.permute.default(arg668_1, [1, 0]);  arg668_1 = None
        addmm_225 = torch.ops.aten.addmm.default(arg669_1, view_600, permute_300);  arg669_1 = view_600 = permute_300 = None
        view_601 = torch.ops.aten.view.default(addmm_225, [128, 128, 512]);  addmm_225 = None
        add_226 = torch.ops.aten.add.Tensor(view_601, add_212);  view_601 = add_212 = None
        mul_121 = torch.ops.aten.mul.Tensor(add_226, arg671_1);  add_226 = arg671_1 = None
        add_227 = torch.ops.aten.add.Tensor(mul_121, arg670_1);  mul_121 = arg670_1 = None
        view_602 = torch.ops.aten.view.default(add_227, [16384, 512])
        permute_301 = torch.ops.aten.permute.default(arg718_1, [1, 0]);  arg718_1 = None
        addmm_226 = torch.ops.aten.addmm.default(arg719_1, view_602, permute_301);  arg719_1 = view_602 = permute_301 = None
        view_603 = torch.ops.aten.view.default(addmm_226, [128, 128, 128]);  addmm_226 = None
        mul_122 = torch.ops.aten.mul.Tensor(view_603, arg721_1);  view_603 = arg721_1 = None
        add_228 = torch.ops.aten.add.Tensor(mul_122, arg720_1);  mul_122 = arg720_1 = None
        view_604 = torch.ops.aten.view.default(add_227, [16384, 512])
        permute_302 = torch.ops.aten.permute.default(arg722_1, [1, 0]);  arg722_1 = None
        addmm_227 = torch.ops.aten.addmm.default(arg723_1, view_604, permute_302);  arg723_1 = view_604 = permute_302 = None
        view_605 = torch.ops.aten.view.default(addmm_227, [128, 128, 128]);  addmm_227 = None
        mul_123 = torch.ops.aten.mul.Tensor(view_605, arg725_1);  view_605 = arg725_1 = None
        add_229 = torch.ops.aten.add.Tensor(mul_123, arg724_1);  mul_123 = arg724_1 = None
        view_606 = torch.ops.aten.view.default(add_229, [16384, 128])
        permute_303 = torch.ops.aten.permute.default(arg698_1, [1, 0]);  arg698_1 = None
        addmm_228 = torch.ops.aten.addmm.default(arg699_1, view_606, permute_303);  arg699_1 = view_606 = permute_303 = None
        view_607 = torch.ops.aten.view.default(addmm_228, [128, 128, 128]);  addmm_228 = None
        view_608 = torch.ops.aten.view.default(add_229, [16384, 128]);  add_229 = None
        permute_304 = torch.ops.aten.permute.default(arg700_1, [1, 0]);  arg700_1 = None
        addmm_229 = torch.ops.aten.addmm.default(arg701_1, view_608, permute_304);  arg701_1 = view_608 = permute_304 = None
        view_609 = torch.ops.aten.view.default(addmm_229, [128, 128, 128]);  addmm_229 = None
        view_610 = torch.ops.aten.view.default(add_227, [16384, 512])
        permute_305 = torch.ops.aten.permute.default(arg702_1, [1, 0]);  arg702_1 = None
        addmm_230 = torch.ops.aten.addmm.default(arg703_1, view_610, permute_305);  arg703_1 = view_610 = permute_305 = None
        view_611 = torch.ops.aten.view.default(addmm_230, [128, 128, 128]);  addmm_230 = None
        view_612 = torch.ops.aten.view.default(view_607, [128, 128, 4, 32]);  view_607 = None
        view_613 = torch.ops.aten.view.default(view_609, [128, 128, 4, 32]);  view_609 = None
        view_614 = torch.ops.aten.view.default(view_611, [128, 128, 4, 32]);  view_611 = None
        permute_default_24 = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
        permute_default_25 = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
        permute_default_26 = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.17677669529663687);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_10 = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        permute_310 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        clone_95 = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_621 = torch.ops.aten.view.default(clone_95, [128, 128, 128]);  clone_95 = None
        view_622 = torch.ops.aten.view.default(view_621, [16384, 128]);  view_621 = None
        permute_311 = torch.ops.aten.permute.default(arg704_1, [1, 0]);  arg704_1 = None
        addmm_231 = torch.ops.aten.addmm.default(arg705_1, view_622, permute_311);  arg705_1 = view_622 = permute_311 = None
        view_623 = torch.ops.aten.view.default(addmm_231, [128, 128, 128]);  addmm_231 = None
        add_231 = torch.ops.aten.add.Tensor(view_623, add_228);  view_623 = add_228 = None
        mul_124 = torch.ops.aten.mul.Tensor(add_231, arg707_1);  add_231 = arg707_1 = None
        add_232 = torch.ops.aten.add.Tensor(mul_124, arg706_1);  mul_124 = arg706_1 = None
        view_624 = torch.ops.aten.view.default(add_232, [16384, 128])
        permute_312 = torch.ops.aten.permute.default(arg726_1, [1, 0]);  arg726_1 = None
        addmm_232 = torch.ops.aten.addmm.default(arg727_1, view_624, permute_312);  arg727_1 = view_624 = permute_312 = None
        view_625 = torch.ops.aten.view.default(addmm_232, [128, 128, 512]);  addmm_232 = None
        relu_60 = torch.ops.aten.relu.default(view_625);  view_625 = None
        view_626 = torch.ops.aten.view.default(relu_60, [16384, 512]);  relu_60 = None
        permute_313 = torch.ops.aten.permute.default(arg728_1, [1, 0]);  arg728_1 = None
        addmm_233 = torch.ops.aten.addmm.default(arg729_1, view_626, permute_313);  arg729_1 = view_626 = permute_313 = None
        view_627 = torch.ops.aten.view.default(addmm_233, [128, 128, 128]);  addmm_233 = None
        add_233 = torch.ops.aten.add.Tensor(view_627, add_232);  view_627 = add_232 = None
        mul_125 = torch.ops.aten.mul.Tensor(add_233, arg731_1);  add_233 = arg731_1 = None
        add_234 = torch.ops.aten.add.Tensor(mul_125, arg730_1);  mul_125 = arg730_1 = None
        view_628 = torch.ops.aten.view.default(add_234, [16384, 128])
        permute_314 = torch.ops.aten.permute.default(arg732_1, [1, 0]);  arg732_1 = None
        addmm_234 = torch.ops.aten.addmm.default(arg733_1, view_628, permute_314);  arg733_1 = view_628 = permute_314 = None
        view_629 = torch.ops.aten.view.default(addmm_234, [128, 128, 512]);  addmm_234 = None
        relu_61 = torch.ops.aten.relu.default(view_629);  view_629 = None
        view_630 = torch.ops.aten.view.default(relu_61, [16384, 512]);  relu_61 = None
        permute_315 = torch.ops.aten.permute.default(arg734_1, [1, 0]);  arg734_1 = None
        addmm_235 = torch.ops.aten.addmm.default(arg735_1, view_630, permute_315);  arg735_1 = view_630 = permute_315 = None
        view_631 = torch.ops.aten.view.default(addmm_235, [128, 128, 128]);  addmm_235 = None
        add_235 = torch.ops.aten.add.Tensor(view_631, add_234);  view_631 = add_234 = None
        mul_126 = torch.ops.aten.mul.Tensor(add_235, arg737_1);  add_235 = arg737_1 = None
        add_236 = torch.ops.aten.add.Tensor(mul_126, arg736_1);  mul_126 = arg736_1 = None
        view_632 = torch.ops.aten.view.default(add_236, [16384, 128])
        permute_316 = torch.ops.aten.permute.default(arg738_1, [1, 0]);  arg738_1 = None
        addmm_236 = torch.ops.aten.addmm.default(arg739_1, view_632, permute_316);  arg739_1 = view_632 = permute_316 = None
        view_633 = torch.ops.aten.view.default(addmm_236, [128, 128, 512]);  addmm_236 = None
        relu_62 = torch.ops.aten.relu.default(view_633);  view_633 = None
        view_634 = torch.ops.aten.view.default(relu_62, [16384, 512]);  relu_62 = None
        permute_317 = torch.ops.aten.permute.default(arg740_1, [1, 0]);  arg740_1 = None
        addmm_237 = torch.ops.aten.addmm.default(arg741_1, view_634, permute_317);  arg741_1 = view_634 = permute_317 = None
        view_635 = torch.ops.aten.view.default(addmm_237, [128, 128, 128]);  addmm_237 = None
        add_237 = torch.ops.aten.add.Tensor(view_635, add_236);  view_635 = add_236 = None
        mul_127 = torch.ops.aten.mul.Tensor(add_237, arg743_1);  add_237 = arg743_1 = None
        add_238 = torch.ops.aten.add.Tensor(mul_127, arg742_1);  mul_127 = arg742_1 = None
        view_636 = torch.ops.aten.view.default(add_238, [16384, 128])
        permute_318 = torch.ops.aten.permute.default(arg708_1, [1, 0]);  arg708_1 = None
        addmm_238 = torch.ops.aten.addmm.default(arg709_1, view_636, permute_318);  arg709_1 = view_636 = permute_318 = None
        view_637 = torch.ops.aten.view.default(addmm_238, [128, 128, 512]);  addmm_238 = None
        relu_63 = torch.ops.aten.relu.default(view_637);  view_637 = None
        view_638 = torch.ops.aten.view.default(relu_63, [16384, 512]);  relu_63 = None
        permute_319 = torch.ops.aten.permute.default(arg710_1, [1, 0]);  arg710_1 = None
        addmm_239 = torch.ops.aten.addmm.default(arg711_1, view_638, permute_319);  arg711_1 = view_638 = permute_319 = None
        view_639 = torch.ops.aten.view.default(addmm_239, [128, 128, 128]);  addmm_239 = None
        add_239 = torch.ops.aten.add.Tensor(view_639, add_238);  view_639 = add_238 = None
        mul_128 = torch.ops.aten.mul.Tensor(add_239, arg713_1);  add_239 = arg713_1 = None
        add_240 = torch.ops.aten.add.Tensor(mul_128, arg712_1);  mul_128 = arg712_1 = None
        view_640 = torch.ops.aten.view.default(add_240, [16384, 128]);  add_240 = None
        permute_320 = torch.ops.aten.permute.default(arg714_1, [1, 0]);  arg714_1 = None
        addmm_240 = torch.ops.aten.addmm.default(arg715_1, view_640, permute_320);  arg715_1 = view_640 = permute_320 = None
        view_641 = torch.ops.aten.view.default(addmm_240, [128, 128, 512]);  addmm_240 = None
        add_241 = torch.ops.aten.add.Tensor(view_641, add_227);  view_641 = add_227 = None
        mul_129 = torch.ops.aten.mul.Tensor(add_241, arg717_1);  add_241 = arg717_1 = None
        add_242 = torch.ops.aten.add.Tensor(mul_129, arg716_1);  mul_129 = arg716_1 = None
        view_642 = torch.ops.aten.view.default(add_242, [16384, 512])
        permute_321 = torch.ops.aten.permute.default(arg764_1, [1, 0]);  arg764_1 = None
        addmm_241 = torch.ops.aten.addmm.default(arg765_1, view_642, permute_321);  arg765_1 = view_642 = permute_321 = None
        view_643 = torch.ops.aten.view.default(addmm_241, [128, 128, 128]);  addmm_241 = None
        mul_130 = torch.ops.aten.mul.Tensor(view_643, arg767_1);  view_643 = arg767_1 = None
        add_243 = torch.ops.aten.add.Tensor(mul_130, arg766_1);  mul_130 = arg766_1 = None
        view_644 = torch.ops.aten.view.default(add_242, [16384, 512])
        permute_322 = torch.ops.aten.permute.default(arg768_1, [1, 0]);  arg768_1 = None
        addmm_242 = torch.ops.aten.addmm.default(arg769_1, view_644, permute_322);  arg769_1 = view_644 = permute_322 = None
        view_645 = torch.ops.aten.view.default(addmm_242, [128, 128, 128]);  addmm_242 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_645, arg771_1);  view_645 = arg771_1 = None
        add_244 = torch.ops.aten.add.Tensor(mul_131, arg770_1);  mul_131 = arg770_1 = None
        view_646 = torch.ops.aten.view.default(add_244, [16384, 128])
        permute_323 = torch.ops.aten.permute.default(arg744_1, [1, 0]);  arg744_1 = None
        addmm_243 = torch.ops.aten.addmm.default(arg745_1, view_646, permute_323);  arg745_1 = view_646 = permute_323 = None
        view_647 = torch.ops.aten.view.default(addmm_243, [128, 128, 128]);  addmm_243 = None
        view_648 = torch.ops.aten.view.default(add_244, [16384, 128]);  add_244 = None
        permute_324 = torch.ops.aten.permute.default(arg746_1, [1, 0]);  arg746_1 = None
        addmm_244 = torch.ops.aten.addmm.default(arg747_1, view_648, permute_324);  arg747_1 = view_648 = permute_324 = None
        view_649 = torch.ops.aten.view.default(addmm_244, [128, 128, 128]);  addmm_244 = None
        view_650 = torch.ops.aten.view.default(add_242, [16384, 512])
        permute_325 = torch.ops.aten.permute.default(arg748_1, [1, 0]);  arg748_1 = None
        addmm_245 = torch.ops.aten.addmm.default(arg749_1, view_650, permute_325);  arg749_1 = view_650 = permute_325 = None
        view_651 = torch.ops.aten.view.default(addmm_245, [128, 128, 128]);  addmm_245 = None
        view_652 = torch.ops.aten.view.default(view_647, [128, 128, 4, 32]);  view_647 = None
        view_653 = torch.ops.aten.view.default(view_649, [128, 128, 4, 32]);  view_649 = None
        view_654 = torch.ops.aten.view.default(view_651, [128, 128, 4, 32]);  view_651 = None
        permute_default_21 = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
        permute_default_22 = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
        permute_default_23 = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.17677669529663687);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_9 = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        permute_330 = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
        clone_101 = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_661 = torch.ops.aten.view.default(clone_101, [128, 128, 128]);  clone_101 = None
        view_662 = torch.ops.aten.view.default(view_661, [16384, 128]);  view_661 = None
        permute_331 = torch.ops.aten.permute.default(arg750_1, [1, 0]);  arg750_1 = None
        addmm_246 = torch.ops.aten.addmm.default(arg751_1, view_662, permute_331);  arg751_1 = view_662 = permute_331 = None
        view_663 = torch.ops.aten.view.default(addmm_246, [128, 128, 128]);  addmm_246 = None
        add_246 = torch.ops.aten.add.Tensor(view_663, add_243);  view_663 = add_243 = None
        mul_132 = torch.ops.aten.mul.Tensor(add_246, arg753_1);  add_246 = arg753_1 = None
        add_247 = torch.ops.aten.add.Tensor(mul_132, arg752_1);  mul_132 = arg752_1 = None
        view_664 = torch.ops.aten.view.default(add_247, [16384, 128])
        permute_332 = torch.ops.aten.permute.default(arg772_1, [1, 0]);  arg772_1 = None
        addmm_247 = torch.ops.aten.addmm.default(arg773_1, view_664, permute_332);  arg773_1 = view_664 = permute_332 = None
        view_665 = torch.ops.aten.view.default(addmm_247, [128, 128, 512]);  addmm_247 = None
        relu_64 = torch.ops.aten.relu.default(view_665);  view_665 = None
        view_666 = torch.ops.aten.view.default(relu_64, [16384, 512]);  relu_64 = None
        permute_333 = torch.ops.aten.permute.default(arg774_1, [1, 0]);  arg774_1 = None
        addmm_248 = torch.ops.aten.addmm.default(arg775_1, view_666, permute_333);  arg775_1 = view_666 = permute_333 = None
        view_667 = torch.ops.aten.view.default(addmm_248, [128, 128, 128]);  addmm_248 = None
        add_248 = torch.ops.aten.add.Tensor(view_667, add_247);  view_667 = add_247 = None
        mul_133 = torch.ops.aten.mul.Tensor(add_248, arg777_1);  add_248 = arg777_1 = None
        add_249 = torch.ops.aten.add.Tensor(mul_133, arg776_1);  mul_133 = arg776_1 = None
        view_668 = torch.ops.aten.view.default(add_249, [16384, 128])
        permute_334 = torch.ops.aten.permute.default(arg778_1, [1, 0]);  arg778_1 = None
        addmm_249 = torch.ops.aten.addmm.default(arg779_1, view_668, permute_334);  arg779_1 = view_668 = permute_334 = None
        view_669 = torch.ops.aten.view.default(addmm_249, [128, 128, 512]);  addmm_249 = None
        relu_65 = torch.ops.aten.relu.default(view_669);  view_669 = None
        view_670 = torch.ops.aten.view.default(relu_65, [16384, 512]);  relu_65 = None
        permute_335 = torch.ops.aten.permute.default(arg780_1, [1, 0]);  arg780_1 = None
        addmm_250 = torch.ops.aten.addmm.default(arg781_1, view_670, permute_335);  arg781_1 = view_670 = permute_335 = None
        view_671 = torch.ops.aten.view.default(addmm_250, [128, 128, 128]);  addmm_250 = None
        add_250 = torch.ops.aten.add.Tensor(view_671, add_249);  view_671 = add_249 = None
        mul_134 = torch.ops.aten.mul.Tensor(add_250, arg783_1);  add_250 = arg783_1 = None
        add_251 = torch.ops.aten.add.Tensor(mul_134, arg782_1);  mul_134 = arg782_1 = None
        view_672 = torch.ops.aten.view.default(add_251, [16384, 128])
        permute_336 = torch.ops.aten.permute.default(arg784_1, [1, 0]);  arg784_1 = None
        addmm_251 = torch.ops.aten.addmm.default(arg785_1, view_672, permute_336);  arg785_1 = view_672 = permute_336 = None
        view_673 = torch.ops.aten.view.default(addmm_251, [128, 128, 512]);  addmm_251 = None
        relu_66 = torch.ops.aten.relu.default(view_673);  view_673 = None
        view_674 = torch.ops.aten.view.default(relu_66, [16384, 512]);  relu_66 = None
        permute_337 = torch.ops.aten.permute.default(arg786_1, [1, 0]);  arg786_1 = None
        addmm_252 = torch.ops.aten.addmm.default(arg787_1, view_674, permute_337);  arg787_1 = view_674 = permute_337 = None
        view_675 = torch.ops.aten.view.default(addmm_252, [128, 128, 128]);  addmm_252 = None
        add_252 = torch.ops.aten.add.Tensor(view_675, add_251);  view_675 = add_251 = None
        mul_135 = torch.ops.aten.mul.Tensor(add_252, arg789_1);  add_252 = arg789_1 = None
        add_253 = torch.ops.aten.add.Tensor(mul_135, arg788_1);  mul_135 = arg788_1 = None
        view_676 = torch.ops.aten.view.default(add_253, [16384, 128])
        permute_338 = torch.ops.aten.permute.default(arg754_1, [1, 0]);  arg754_1 = None
        addmm_253 = torch.ops.aten.addmm.default(arg755_1, view_676, permute_338);  arg755_1 = view_676 = permute_338 = None
        view_677 = torch.ops.aten.view.default(addmm_253, [128, 128, 512]);  addmm_253 = None
        relu_67 = torch.ops.aten.relu.default(view_677);  view_677 = None
        view_678 = torch.ops.aten.view.default(relu_67, [16384, 512]);  relu_67 = None
        permute_339 = torch.ops.aten.permute.default(arg756_1, [1, 0]);  arg756_1 = None
        addmm_254 = torch.ops.aten.addmm.default(arg757_1, view_678, permute_339);  arg757_1 = view_678 = permute_339 = None
        view_679 = torch.ops.aten.view.default(addmm_254, [128, 128, 128]);  addmm_254 = None
        add_254 = torch.ops.aten.add.Tensor(view_679, add_253);  view_679 = add_253 = None
        mul_136 = torch.ops.aten.mul.Tensor(add_254, arg759_1);  add_254 = arg759_1 = None
        add_255 = torch.ops.aten.add.Tensor(mul_136, arg758_1);  mul_136 = arg758_1 = None
        view_680 = torch.ops.aten.view.default(add_255, [16384, 128]);  add_255 = None
        permute_340 = torch.ops.aten.permute.default(arg760_1, [1, 0]);  arg760_1 = None
        addmm_255 = torch.ops.aten.addmm.default(arg761_1, view_680, permute_340);  arg761_1 = view_680 = permute_340 = None
        view_681 = torch.ops.aten.view.default(addmm_255, [128, 128, 512]);  addmm_255 = None
        add_256 = torch.ops.aten.add.Tensor(view_681, add_242);  view_681 = add_242 = None
        mul_137 = torch.ops.aten.mul.Tensor(add_256, arg763_1);  add_256 = arg763_1 = None
        add_257 = torch.ops.aten.add.Tensor(mul_137, arg762_1);  mul_137 = arg762_1 = None
        view_682 = torch.ops.aten.view.default(add_257, [16384, 512])
        permute_341 = torch.ops.aten.permute.default(arg810_1, [1, 0]);  arg810_1 = None
        addmm_256 = torch.ops.aten.addmm.default(arg811_1, view_682, permute_341);  arg811_1 = view_682 = permute_341 = None
        view_683 = torch.ops.aten.view.default(addmm_256, [128, 128, 128]);  addmm_256 = None
        mul_138 = torch.ops.aten.mul.Tensor(view_683, arg813_1);  view_683 = arg813_1 = None
        add_258 = torch.ops.aten.add.Tensor(mul_138, arg812_1);  mul_138 = arg812_1 = None
        view_684 = torch.ops.aten.view.default(add_257, [16384, 512])
        permute_342 = torch.ops.aten.permute.default(arg814_1, [1, 0]);  arg814_1 = None
        addmm_257 = torch.ops.aten.addmm.default(arg815_1, view_684, permute_342);  arg815_1 = view_684 = permute_342 = None
        view_685 = torch.ops.aten.view.default(addmm_257, [128, 128, 128]);  addmm_257 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_685, arg817_1);  view_685 = arg817_1 = None
        add_259 = torch.ops.aten.add.Tensor(mul_139, arg816_1);  mul_139 = arg816_1 = None
        view_686 = torch.ops.aten.view.default(add_259, [16384, 128])
        permute_343 = torch.ops.aten.permute.default(arg790_1, [1, 0]);  arg790_1 = None
        addmm_258 = torch.ops.aten.addmm.default(arg791_1, view_686, permute_343);  arg791_1 = view_686 = permute_343 = None
        view_687 = torch.ops.aten.view.default(addmm_258, [128, 128, 128]);  addmm_258 = None
        view_688 = torch.ops.aten.view.default(add_259, [16384, 128]);  add_259 = None
        permute_344 = torch.ops.aten.permute.default(arg792_1, [1, 0]);  arg792_1 = None
        addmm_259 = torch.ops.aten.addmm.default(arg793_1, view_688, permute_344);  arg793_1 = view_688 = permute_344 = None
        view_689 = torch.ops.aten.view.default(addmm_259, [128, 128, 128]);  addmm_259 = None
        view_690 = torch.ops.aten.view.default(add_257, [16384, 512])
        permute_345 = torch.ops.aten.permute.default(arg794_1, [1, 0]);  arg794_1 = None
        addmm_260 = torch.ops.aten.addmm.default(arg795_1, view_690, permute_345);  arg795_1 = view_690 = permute_345 = None
        view_691 = torch.ops.aten.view.default(addmm_260, [128, 128, 128]);  addmm_260 = None
        view_692 = torch.ops.aten.view.default(view_687, [128, 128, 4, 32]);  view_687 = None
        view_693 = torch.ops.aten.view.default(view_689, [128, 128, 4, 32]);  view_689 = None
        view_694 = torch.ops.aten.view.default(view_691, [128, 128, 4, 32]);  view_691 = None
        permute_default_18 = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
        permute_default_19 = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
        permute_default_20 = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.17677669529663687);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_8 = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        permute_350 = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        clone_107 = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
        view_701 = torch.ops.aten.view.default(clone_107, [128, 128, 128]);  clone_107 = None
        view_702 = torch.ops.aten.view.default(view_701, [16384, 128]);  view_701 = None
        permute_351 = torch.ops.aten.permute.default(arg796_1, [1, 0]);  arg796_1 = None
        addmm_261 = torch.ops.aten.addmm.default(arg797_1, view_702, permute_351);  arg797_1 = view_702 = permute_351 = None
        view_703 = torch.ops.aten.view.default(addmm_261, [128, 128, 128]);  addmm_261 = None
        add_261 = torch.ops.aten.add.Tensor(view_703, add_258);  view_703 = add_258 = None
        mul_140 = torch.ops.aten.mul.Tensor(add_261, arg799_1);  add_261 = arg799_1 = None
        add_262 = torch.ops.aten.add.Tensor(mul_140, arg798_1);  mul_140 = arg798_1 = None
        view_704 = torch.ops.aten.view.default(add_262, [16384, 128])
        permute_352 = torch.ops.aten.permute.default(arg818_1, [1, 0]);  arg818_1 = None
        addmm_262 = torch.ops.aten.addmm.default(arg819_1, view_704, permute_352);  arg819_1 = view_704 = permute_352 = None
        view_705 = torch.ops.aten.view.default(addmm_262, [128, 128, 512]);  addmm_262 = None
        relu_68 = torch.ops.aten.relu.default(view_705);  view_705 = None
        view_706 = torch.ops.aten.view.default(relu_68, [16384, 512]);  relu_68 = None
        permute_353 = torch.ops.aten.permute.default(arg820_1, [1, 0]);  arg820_1 = None
        addmm_263 = torch.ops.aten.addmm.default(arg821_1, view_706, permute_353);  arg821_1 = view_706 = permute_353 = None
        view_707 = torch.ops.aten.view.default(addmm_263, [128, 128, 128]);  addmm_263 = None
        add_263 = torch.ops.aten.add.Tensor(view_707, add_262);  view_707 = add_262 = None
        mul_141 = torch.ops.aten.mul.Tensor(add_263, arg823_1);  add_263 = arg823_1 = None
        add_264 = torch.ops.aten.add.Tensor(mul_141, arg822_1);  mul_141 = arg822_1 = None
        view_708 = torch.ops.aten.view.default(add_264, [16384, 128])
        permute_354 = torch.ops.aten.permute.default(arg824_1, [1, 0]);  arg824_1 = None
        addmm_264 = torch.ops.aten.addmm.default(arg825_1, view_708, permute_354);  arg825_1 = view_708 = permute_354 = None
        view_709 = torch.ops.aten.view.default(addmm_264, [128, 128, 512]);  addmm_264 = None
        relu_69 = torch.ops.aten.relu.default(view_709);  view_709 = None
        view_710 = torch.ops.aten.view.default(relu_69, [16384, 512]);  relu_69 = None
        permute_355 = torch.ops.aten.permute.default(arg826_1, [1, 0]);  arg826_1 = None
        addmm_265 = torch.ops.aten.addmm.default(arg827_1, view_710, permute_355);  arg827_1 = view_710 = permute_355 = None
        view_711 = torch.ops.aten.view.default(addmm_265, [128, 128, 128]);  addmm_265 = None
        add_265 = torch.ops.aten.add.Tensor(view_711, add_264);  view_711 = add_264 = None
        mul_142 = torch.ops.aten.mul.Tensor(add_265, arg829_1);  add_265 = arg829_1 = None
        add_266 = torch.ops.aten.add.Tensor(mul_142, arg828_1);  mul_142 = arg828_1 = None
        view_712 = torch.ops.aten.view.default(add_266, [16384, 128])
        permute_356 = torch.ops.aten.permute.default(arg830_1, [1, 0]);  arg830_1 = None
        addmm_266 = torch.ops.aten.addmm.default(arg831_1, view_712, permute_356);  arg831_1 = view_712 = permute_356 = None
        view_713 = torch.ops.aten.view.default(addmm_266, [128, 128, 512]);  addmm_266 = None
        relu_70 = torch.ops.aten.relu.default(view_713);  view_713 = None
        view_714 = torch.ops.aten.view.default(relu_70, [16384, 512]);  relu_70 = None
        permute_357 = torch.ops.aten.permute.default(arg832_1, [1, 0]);  arg832_1 = None
        addmm_267 = torch.ops.aten.addmm.default(arg833_1, view_714, permute_357);  arg833_1 = view_714 = permute_357 = None
        view_715 = torch.ops.aten.view.default(addmm_267, [128, 128, 128]);  addmm_267 = None
        add_267 = torch.ops.aten.add.Tensor(view_715, add_266);  view_715 = add_266 = None
        mul_143 = torch.ops.aten.mul.Tensor(add_267, arg835_1);  add_267 = arg835_1 = None
        add_268 = torch.ops.aten.add.Tensor(mul_143, arg834_1);  mul_143 = arg834_1 = None
        view_716 = torch.ops.aten.view.default(add_268, [16384, 128])
        permute_358 = torch.ops.aten.permute.default(arg800_1, [1, 0]);  arg800_1 = None
        addmm_268 = torch.ops.aten.addmm.default(arg801_1, view_716, permute_358);  arg801_1 = view_716 = permute_358 = None
        view_717 = torch.ops.aten.view.default(addmm_268, [128, 128, 512]);  addmm_268 = None
        relu_71 = torch.ops.aten.relu.default(view_717);  view_717 = None
        view_718 = torch.ops.aten.view.default(relu_71, [16384, 512]);  relu_71 = None
        permute_359 = torch.ops.aten.permute.default(arg802_1, [1, 0]);  arg802_1 = None
        addmm_269 = torch.ops.aten.addmm.default(arg803_1, view_718, permute_359);  arg803_1 = view_718 = permute_359 = None
        view_719 = torch.ops.aten.view.default(addmm_269, [128, 128, 128]);  addmm_269 = None
        add_269 = torch.ops.aten.add.Tensor(view_719, add_268);  view_719 = add_268 = None
        mul_144 = torch.ops.aten.mul.Tensor(add_269, arg805_1);  add_269 = arg805_1 = None
        add_270 = torch.ops.aten.add.Tensor(mul_144, arg804_1);  mul_144 = arg804_1 = None
        view_720 = torch.ops.aten.view.default(add_270, [16384, 128]);  add_270 = None
        permute_360 = torch.ops.aten.permute.default(arg806_1, [1, 0]);  arg806_1 = None
        addmm_270 = torch.ops.aten.addmm.default(arg807_1, view_720, permute_360);  arg807_1 = view_720 = permute_360 = None
        view_721 = torch.ops.aten.view.default(addmm_270, [128, 128, 512]);  addmm_270 = None
        add_271 = torch.ops.aten.add.Tensor(view_721, add_257);  view_721 = add_257 = None
        mul_145 = torch.ops.aten.mul.Tensor(add_271, arg809_1);  add_271 = arg809_1 = None
        add_272 = torch.ops.aten.add.Tensor(mul_145, arg808_1);  mul_145 = arg808_1 = None
        view_722 = torch.ops.aten.view.default(add_272, [16384, 512])
        permute_361 = torch.ops.aten.permute.default(arg856_1, [1, 0]);  arg856_1 = None
        addmm_271 = torch.ops.aten.addmm.default(arg857_1, view_722, permute_361);  arg857_1 = view_722 = permute_361 = None
        view_723 = torch.ops.aten.view.default(addmm_271, [128, 128, 128]);  addmm_271 = None
        mul_146 = torch.ops.aten.mul.Tensor(view_723, arg859_1);  view_723 = arg859_1 = None
        add_273 = torch.ops.aten.add.Tensor(mul_146, arg858_1);  mul_146 = arg858_1 = None
        view_724 = torch.ops.aten.view.default(add_272, [16384, 512])
        permute_362 = torch.ops.aten.permute.default(arg860_1, [1, 0]);  arg860_1 = None
        addmm_272 = torch.ops.aten.addmm.default(arg861_1, view_724, permute_362);  arg861_1 = view_724 = permute_362 = None
        view_725 = torch.ops.aten.view.default(addmm_272, [128, 128, 128]);  addmm_272 = None
        mul_147 = torch.ops.aten.mul.Tensor(view_725, arg863_1);  view_725 = arg863_1 = None
        add_274 = torch.ops.aten.add.Tensor(mul_147, arg862_1);  mul_147 = arg862_1 = None
        view_726 = torch.ops.aten.view.default(add_274, [16384, 128])
        permute_363 = torch.ops.aten.permute.default(arg836_1, [1, 0]);  arg836_1 = None
        addmm_273 = torch.ops.aten.addmm.default(arg837_1, view_726, permute_363);  arg837_1 = view_726 = permute_363 = None
        view_727 = torch.ops.aten.view.default(addmm_273, [128, 128, 128]);  addmm_273 = None
        view_728 = torch.ops.aten.view.default(add_274, [16384, 128]);  add_274 = None
        permute_364 = torch.ops.aten.permute.default(arg838_1, [1, 0]);  arg838_1 = None
        addmm_274 = torch.ops.aten.addmm.default(arg839_1, view_728, permute_364);  arg839_1 = view_728 = permute_364 = None
        view_729 = torch.ops.aten.view.default(addmm_274, [128, 128, 128]);  addmm_274 = None
        view_730 = torch.ops.aten.view.default(add_272, [16384, 512])
        permute_365 = torch.ops.aten.permute.default(arg840_1, [1, 0]);  arg840_1 = None
        addmm_275 = torch.ops.aten.addmm.default(arg841_1, view_730, permute_365);  arg841_1 = view_730 = permute_365 = None
        view_731 = torch.ops.aten.view.default(addmm_275, [128, 128, 128]);  addmm_275 = None
        view_732 = torch.ops.aten.view.default(view_727, [128, 128, 4, 32]);  view_727 = None
        view_733 = torch.ops.aten.view.default(view_729, [128, 128, 4, 32]);  view_729 = None
        view_734 = torch.ops.aten.view.default(view_731, [128, 128, 4, 32]);  view_731 = None
        permute_default_15 = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
        permute_default_16 = torch.ops.aten.permute.default(view_733, [0, 2, 1, 3]);  view_733 = None
        permute_default_17 = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.17677669529663687);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_7 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        permute_370 = torch.ops.aten.permute.default(getitem_7, [0, 2, 1, 3]);  getitem_7 = None
        clone_113 = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
        view_741 = torch.ops.aten.view.default(clone_113, [128, 128, 128]);  clone_113 = None
        view_742 = torch.ops.aten.view.default(view_741, [16384, 128]);  view_741 = None
        permute_371 = torch.ops.aten.permute.default(arg842_1, [1, 0]);  arg842_1 = None
        addmm_276 = torch.ops.aten.addmm.default(arg843_1, view_742, permute_371);  arg843_1 = view_742 = permute_371 = None
        view_743 = torch.ops.aten.view.default(addmm_276, [128, 128, 128]);  addmm_276 = None
        add_276 = torch.ops.aten.add.Tensor(view_743, add_273);  view_743 = add_273 = None
        mul_148 = torch.ops.aten.mul.Tensor(add_276, arg845_1);  add_276 = arg845_1 = None
        add_277 = torch.ops.aten.add.Tensor(mul_148, arg844_1);  mul_148 = arg844_1 = None
        view_744 = torch.ops.aten.view.default(add_277, [16384, 128])
        permute_372 = torch.ops.aten.permute.default(arg864_1, [1, 0]);  arg864_1 = None
        addmm_277 = torch.ops.aten.addmm.default(arg865_1, view_744, permute_372);  arg865_1 = view_744 = permute_372 = None
        view_745 = torch.ops.aten.view.default(addmm_277, [128, 128, 512]);  addmm_277 = None
        relu_72 = torch.ops.aten.relu.default(view_745);  view_745 = None
        view_746 = torch.ops.aten.view.default(relu_72, [16384, 512]);  relu_72 = None
        permute_373 = torch.ops.aten.permute.default(arg866_1, [1, 0]);  arg866_1 = None
        addmm_278 = torch.ops.aten.addmm.default(arg867_1, view_746, permute_373);  arg867_1 = view_746 = permute_373 = None
        view_747 = torch.ops.aten.view.default(addmm_278, [128, 128, 128]);  addmm_278 = None
        add_278 = torch.ops.aten.add.Tensor(view_747, add_277);  view_747 = add_277 = None
        mul_149 = torch.ops.aten.mul.Tensor(add_278, arg869_1);  add_278 = arg869_1 = None
        add_279 = torch.ops.aten.add.Tensor(mul_149, arg868_1);  mul_149 = arg868_1 = None
        view_748 = torch.ops.aten.view.default(add_279, [16384, 128])
        permute_374 = torch.ops.aten.permute.default(arg870_1, [1, 0]);  arg870_1 = None
        addmm_279 = torch.ops.aten.addmm.default(arg871_1, view_748, permute_374);  arg871_1 = view_748 = permute_374 = None
        view_749 = torch.ops.aten.view.default(addmm_279, [128, 128, 512]);  addmm_279 = None
        relu_73 = torch.ops.aten.relu.default(view_749);  view_749 = None
        view_750 = torch.ops.aten.view.default(relu_73, [16384, 512]);  relu_73 = None
        permute_375 = torch.ops.aten.permute.default(arg872_1, [1, 0]);  arg872_1 = None
        addmm_280 = torch.ops.aten.addmm.default(arg873_1, view_750, permute_375);  arg873_1 = view_750 = permute_375 = None
        view_751 = torch.ops.aten.view.default(addmm_280, [128, 128, 128]);  addmm_280 = None
        add_280 = torch.ops.aten.add.Tensor(view_751, add_279);  view_751 = add_279 = None
        mul_150 = torch.ops.aten.mul.Tensor(add_280, arg875_1);  add_280 = arg875_1 = None
        add_281 = torch.ops.aten.add.Tensor(mul_150, arg874_1);  mul_150 = arg874_1 = None
        view_752 = torch.ops.aten.view.default(add_281, [16384, 128])
        permute_376 = torch.ops.aten.permute.default(arg876_1, [1, 0]);  arg876_1 = None
        addmm_281 = torch.ops.aten.addmm.default(arg877_1, view_752, permute_376);  arg877_1 = view_752 = permute_376 = None
        view_753 = torch.ops.aten.view.default(addmm_281, [128, 128, 512]);  addmm_281 = None
        relu_74 = torch.ops.aten.relu.default(view_753);  view_753 = None
        view_754 = torch.ops.aten.view.default(relu_74, [16384, 512]);  relu_74 = None
        permute_377 = torch.ops.aten.permute.default(arg878_1, [1, 0]);  arg878_1 = None
        addmm_282 = torch.ops.aten.addmm.default(arg879_1, view_754, permute_377);  arg879_1 = view_754 = permute_377 = None
        view_755 = torch.ops.aten.view.default(addmm_282, [128, 128, 128]);  addmm_282 = None
        add_282 = torch.ops.aten.add.Tensor(view_755, add_281);  view_755 = add_281 = None
        mul_151 = torch.ops.aten.mul.Tensor(add_282, arg881_1);  add_282 = arg881_1 = None
        add_283 = torch.ops.aten.add.Tensor(mul_151, arg880_1);  mul_151 = arg880_1 = None
        view_756 = torch.ops.aten.view.default(add_283, [16384, 128])
        permute_378 = torch.ops.aten.permute.default(arg846_1, [1, 0]);  arg846_1 = None
        addmm_283 = torch.ops.aten.addmm.default(arg847_1, view_756, permute_378);  arg847_1 = view_756 = permute_378 = None
        view_757 = torch.ops.aten.view.default(addmm_283, [128, 128, 512]);  addmm_283 = None
        relu_75 = torch.ops.aten.relu.default(view_757);  view_757 = None
        view_758 = torch.ops.aten.view.default(relu_75, [16384, 512]);  relu_75 = None
        permute_379 = torch.ops.aten.permute.default(arg848_1, [1, 0]);  arg848_1 = None
        addmm_284 = torch.ops.aten.addmm.default(arg849_1, view_758, permute_379);  arg849_1 = view_758 = permute_379 = None
        view_759 = torch.ops.aten.view.default(addmm_284, [128, 128, 128]);  addmm_284 = None
        add_284 = torch.ops.aten.add.Tensor(view_759, add_283);  view_759 = add_283 = None
        mul_152 = torch.ops.aten.mul.Tensor(add_284, arg851_1);  add_284 = arg851_1 = None
        add_285 = torch.ops.aten.add.Tensor(mul_152, arg850_1);  mul_152 = arg850_1 = None
        view_760 = torch.ops.aten.view.default(add_285, [16384, 128]);  add_285 = None
        permute_380 = torch.ops.aten.permute.default(arg852_1, [1, 0]);  arg852_1 = None
        addmm_285 = torch.ops.aten.addmm.default(arg853_1, view_760, permute_380);  arg853_1 = view_760 = permute_380 = None
        view_761 = torch.ops.aten.view.default(addmm_285, [128, 128, 512]);  addmm_285 = None
        add_286 = torch.ops.aten.add.Tensor(view_761, add_272);  view_761 = add_272 = None
        mul_153 = torch.ops.aten.mul.Tensor(add_286, arg855_1);  add_286 = arg855_1 = None
        add_287 = torch.ops.aten.add.Tensor(mul_153, arg854_1);  mul_153 = arg854_1 = None
        view_762 = torch.ops.aten.view.default(add_287, [16384, 512])
        permute_381 = torch.ops.aten.permute.default(arg902_1, [1, 0]);  arg902_1 = None
        addmm_286 = torch.ops.aten.addmm.default(arg903_1, view_762, permute_381);  arg903_1 = view_762 = permute_381 = None
        view_763 = torch.ops.aten.view.default(addmm_286, [128, 128, 128]);  addmm_286 = None
        mul_154 = torch.ops.aten.mul.Tensor(view_763, arg905_1);  view_763 = arg905_1 = None
        add_288 = torch.ops.aten.add.Tensor(mul_154, arg904_1);  mul_154 = arg904_1 = None
        view_764 = torch.ops.aten.view.default(add_287, [16384, 512])
        permute_382 = torch.ops.aten.permute.default(arg906_1, [1, 0]);  arg906_1 = None
        addmm_287 = torch.ops.aten.addmm.default(arg907_1, view_764, permute_382);  arg907_1 = view_764 = permute_382 = None
        view_765 = torch.ops.aten.view.default(addmm_287, [128, 128, 128]);  addmm_287 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_765, arg909_1);  view_765 = arg909_1 = None
        add_289 = torch.ops.aten.add.Tensor(mul_155, arg908_1);  mul_155 = arg908_1 = None
        view_766 = torch.ops.aten.view.default(add_289, [16384, 128])
        permute_383 = torch.ops.aten.permute.default(arg882_1, [1, 0]);  arg882_1 = None
        addmm_288 = torch.ops.aten.addmm.default(arg883_1, view_766, permute_383);  arg883_1 = view_766 = permute_383 = None
        view_767 = torch.ops.aten.view.default(addmm_288, [128, 128, 128]);  addmm_288 = None
        view_768 = torch.ops.aten.view.default(add_289, [16384, 128]);  add_289 = None
        permute_384 = torch.ops.aten.permute.default(arg884_1, [1, 0]);  arg884_1 = None
        addmm_289 = torch.ops.aten.addmm.default(arg885_1, view_768, permute_384);  arg885_1 = view_768 = permute_384 = None
        view_769 = torch.ops.aten.view.default(addmm_289, [128, 128, 128]);  addmm_289 = None
        view_770 = torch.ops.aten.view.default(add_287, [16384, 512])
        permute_385 = torch.ops.aten.permute.default(arg886_1, [1, 0]);  arg886_1 = None
        addmm_290 = torch.ops.aten.addmm.default(arg887_1, view_770, permute_385);  arg887_1 = view_770 = permute_385 = None
        view_771 = torch.ops.aten.view.default(addmm_290, [128, 128, 128]);  addmm_290 = None
        view_772 = torch.ops.aten.view.default(view_767, [128, 128, 4, 32]);  view_767 = None
        view_773 = torch.ops.aten.view.default(view_769, [128, 128, 4, 32]);  view_769 = None
        view_774 = torch.ops.aten.view.default(view_771, [128, 128, 4, 32]);  view_771 = None
        permute_default_12 = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
        permute_default_13 = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None
        permute_default_14 = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.17677669529663687);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_6 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        permute_390 = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
        clone_119 = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
        view_781 = torch.ops.aten.view.default(clone_119, [128, 128, 128]);  clone_119 = None
        view_782 = torch.ops.aten.view.default(view_781, [16384, 128]);  view_781 = None
        permute_391 = torch.ops.aten.permute.default(arg888_1, [1, 0]);  arg888_1 = None
        addmm_291 = torch.ops.aten.addmm.default(arg889_1, view_782, permute_391);  arg889_1 = view_782 = permute_391 = None
        view_783 = torch.ops.aten.view.default(addmm_291, [128, 128, 128]);  addmm_291 = None
        add_291 = torch.ops.aten.add.Tensor(view_783, add_288);  view_783 = add_288 = None
        mul_156 = torch.ops.aten.mul.Tensor(add_291, arg891_1);  add_291 = arg891_1 = None
        add_292 = torch.ops.aten.add.Tensor(mul_156, arg890_1);  mul_156 = arg890_1 = None
        view_784 = torch.ops.aten.view.default(add_292, [16384, 128])
        permute_392 = torch.ops.aten.permute.default(arg910_1, [1, 0]);  arg910_1 = None
        addmm_292 = torch.ops.aten.addmm.default(arg911_1, view_784, permute_392);  arg911_1 = view_784 = permute_392 = None
        view_785 = torch.ops.aten.view.default(addmm_292, [128, 128, 512]);  addmm_292 = None
        relu_76 = torch.ops.aten.relu.default(view_785);  view_785 = None
        view_786 = torch.ops.aten.view.default(relu_76, [16384, 512]);  relu_76 = None
        permute_393 = torch.ops.aten.permute.default(arg912_1, [1, 0]);  arg912_1 = None
        addmm_293 = torch.ops.aten.addmm.default(arg913_1, view_786, permute_393);  arg913_1 = view_786 = permute_393 = None
        view_787 = torch.ops.aten.view.default(addmm_293, [128, 128, 128]);  addmm_293 = None
        add_293 = torch.ops.aten.add.Tensor(view_787, add_292);  view_787 = add_292 = None
        mul_157 = torch.ops.aten.mul.Tensor(add_293, arg915_1);  add_293 = arg915_1 = None
        add_294 = torch.ops.aten.add.Tensor(mul_157, arg914_1);  mul_157 = arg914_1 = None
        view_788 = torch.ops.aten.view.default(add_294, [16384, 128])
        permute_394 = torch.ops.aten.permute.default(arg916_1, [1, 0]);  arg916_1 = None
        addmm_294 = torch.ops.aten.addmm.default(arg917_1, view_788, permute_394);  arg917_1 = view_788 = permute_394 = None
        view_789 = torch.ops.aten.view.default(addmm_294, [128, 128, 512]);  addmm_294 = None
        relu_77 = torch.ops.aten.relu.default(view_789);  view_789 = None
        view_790 = torch.ops.aten.view.default(relu_77, [16384, 512]);  relu_77 = None
        permute_395 = torch.ops.aten.permute.default(arg918_1, [1, 0]);  arg918_1 = None
        addmm_295 = torch.ops.aten.addmm.default(arg919_1, view_790, permute_395);  arg919_1 = view_790 = permute_395 = None
        view_791 = torch.ops.aten.view.default(addmm_295, [128, 128, 128]);  addmm_295 = None
        add_295 = torch.ops.aten.add.Tensor(view_791, add_294);  view_791 = add_294 = None
        mul_158 = torch.ops.aten.mul.Tensor(add_295, arg921_1);  add_295 = arg921_1 = None
        add_296 = torch.ops.aten.add.Tensor(mul_158, arg920_1);  mul_158 = arg920_1 = None
        view_792 = torch.ops.aten.view.default(add_296, [16384, 128])
        permute_396 = torch.ops.aten.permute.default(arg922_1, [1, 0]);  arg922_1 = None
        addmm_296 = torch.ops.aten.addmm.default(arg923_1, view_792, permute_396);  arg923_1 = view_792 = permute_396 = None
        view_793 = torch.ops.aten.view.default(addmm_296, [128, 128, 512]);  addmm_296 = None
        relu_78 = torch.ops.aten.relu.default(view_793);  view_793 = None
        view_794 = torch.ops.aten.view.default(relu_78, [16384, 512]);  relu_78 = None
        permute_397 = torch.ops.aten.permute.default(arg924_1, [1, 0]);  arg924_1 = None
        addmm_297 = torch.ops.aten.addmm.default(arg925_1, view_794, permute_397);  arg925_1 = view_794 = permute_397 = None
        view_795 = torch.ops.aten.view.default(addmm_297, [128, 128, 128]);  addmm_297 = None
        add_297 = torch.ops.aten.add.Tensor(view_795, add_296);  view_795 = add_296 = None
        mul_159 = torch.ops.aten.mul.Tensor(add_297, arg927_1);  add_297 = arg927_1 = None
        add_298 = torch.ops.aten.add.Tensor(mul_159, arg926_1);  mul_159 = arg926_1 = None
        view_796 = torch.ops.aten.view.default(add_298, [16384, 128])
        permute_398 = torch.ops.aten.permute.default(arg892_1, [1, 0]);  arg892_1 = None
        addmm_298 = torch.ops.aten.addmm.default(arg893_1, view_796, permute_398);  arg893_1 = view_796 = permute_398 = None
        view_797 = torch.ops.aten.view.default(addmm_298, [128, 128, 512]);  addmm_298 = None
        relu_79 = torch.ops.aten.relu.default(view_797);  view_797 = None
        view_798 = torch.ops.aten.view.default(relu_79, [16384, 512]);  relu_79 = None
        permute_399 = torch.ops.aten.permute.default(arg894_1, [1, 0]);  arg894_1 = None
        addmm_299 = torch.ops.aten.addmm.default(arg895_1, view_798, permute_399);  arg895_1 = view_798 = permute_399 = None
        view_799 = torch.ops.aten.view.default(addmm_299, [128, 128, 128]);  addmm_299 = None
        add_299 = torch.ops.aten.add.Tensor(view_799, add_298);  view_799 = add_298 = None
        mul_160 = torch.ops.aten.mul.Tensor(add_299, arg897_1);  add_299 = arg897_1 = None
        add_300 = torch.ops.aten.add.Tensor(mul_160, arg896_1);  mul_160 = arg896_1 = None
        view_800 = torch.ops.aten.view.default(add_300, [16384, 128]);  add_300 = None
        permute_400 = torch.ops.aten.permute.default(arg898_1, [1, 0]);  arg898_1 = None
        addmm_300 = torch.ops.aten.addmm.default(arg899_1, view_800, permute_400);  arg899_1 = view_800 = permute_400 = None
        view_801 = torch.ops.aten.view.default(addmm_300, [128, 128, 512]);  addmm_300 = None
        add_301 = torch.ops.aten.add.Tensor(view_801, add_287);  view_801 = add_287 = None
        mul_161 = torch.ops.aten.mul.Tensor(add_301, arg901_1);  add_301 = arg901_1 = None
        add_302 = torch.ops.aten.add.Tensor(mul_161, arg900_1);  mul_161 = arg900_1 = None
        view_802 = torch.ops.aten.view.default(add_302, [16384, 512])
        permute_401 = torch.ops.aten.permute.default(arg948_1, [1, 0]);  arg948_1 = None
        addmm_301 = torch.ops.aten.addmm.default(arg949_1, view_802, permute_401);  arg949_1 = view_802 = permute_401 = None
        view_803 = torch.ops.aten.view.default(addmm_301, [128, 128, 128]);  addmm_301 = None
        mul_162 = torch.ops.aten.mul.Tensor(view_803, arg951_1);  view_803 = arg951_1 = None
        add_303 = torch.ops.aten.add.Tensor(mul_162, arg950_1);  mul_162 = arg950_1 = None
        view_804 = torch.ops.aten.view.default(add_302, [16384, 512])
        permute_402 = torch.ops.aten.permute.default(arg952_1, [1, 0]);  arg952_1 = None
        addmm_302 = torch.ops.aten.addmm.default(arg953_1, view_804, permute_402);  arg953_1 = view_804 = permute_402 = None
        view_805 = torch.ops.aten.view.default(addmm_302, [128, 128, 128]);  addmm_302 = None
        mul_163 = torch.ops.aten.mul.Tensor(view_805, arg955_1);  view_805 = arg955_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_163, arg954_1);  mul_163 = arg954_1 = None
        view_806 = torch.ops.aten.view.default(add_304, [16384, 128])
        permute_403 = torch.ops.aten.permute.default(arg928_1, [1, 0]);  arg928_1 = None
        addmm_303 = torch.ops.aten.addmm.default(arg929_1, view_806, permute_403);  arg929_1 = view_806 = permute_403 = None
        view_807 = torch.ops.aten.view.default(addmm_303, [128, 128, 128]);  addmm_303 = None
        view_808 = torch.ops.aten.view.default(add_304, [16384, 128]);  add_304 = None
        permute_404 = torch.ops.aten.permute.default(arg930_1, [1, 0]);  arg930_1 = None
        addmm_304 = torch.ops.aten.addmm.default(arg931_1, view_808, permute_404);  arg931_1 = view_808 = permute_404 = None
        view_809 = torch.ops.aten.view.default(addmm_304, [128, 128, 128]);  addmm_304 = None
        view_810 = torch.ops.aten.view.default(add_302, [16384, 512])
        permute_405 = torch.ops.aten.permute.default(arg932_1, [1, 0]);  arg932_1 = None
        addmm_305 = torch.ops.aten.addmm.default(arg933_1, view_810, permute_405);  arg933_1 = view_810 = permute_405 = None
        view_811 = torch.ops.aten.view.default(addmm_305, [128, 128, 128]);  addmm_305 = None
        view_812 = torch.ops.aten.view.default(view_807, [128, 128, 4, 32]);  view_807 = None
        view_813 = torch.ops.aten.view.default(view_809, [128, 128, 4, 32]);  view_809 = None
        view_814 = torch.ops.aten.view.default(view_811, [128, 128, 4, 32]);  view_811 = None
        permute_default_9 = torch.ops.aten.permute.default(view_812, [0, 2, 1, 3]);  view_812 = None
        permute_default_10 = torch.ops.aten.permute.default(view_813, [0, 2, 1, 3]);  view_813 = None
        permute_default_11 = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.17677669529663687);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_5 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        permute_410 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        clone_125 = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
        view_821 = torch.ops.aten.view.default(clone_125, [128, 128, 128]);  clone_125 = None
        view_822 = torch.ops.aten.view.default(view_821, [16384, 128]);  view_821 = None
        permute_411 = torch.ops.aten.permute.default(arg934_1, [1, 0]);  arg934_1 = None
        addmm_306 = torch.ops.aten.addmm.default(arg935_1, view_822, permute_411);  arg935_1 = view_822 = permute_411 = None
        view_823 = torch.ops.aten.view.default(addmm_306, [128, 128, 128]);  addmm_306 = None
        add_306 = torch.ops.aten.add.Tensor(view_823, add_303);  view_823 = add_303 = None
        mul_164 = torch.ops.aten.mul.Tensor(add_306, arg937_1);  add_306 = arg937_1 = None
        add_307 = torch.ops.aten.add.Tensor(mul_164, arg936_1);  mul_164 = arg936_1 = None
        view_824 = torch.ops.aten.view.default(add_307, [16384, 128])
        permute_412 = torch.ops.aten.permute.default(arg956_1, [1, 0]);  arg956_1 = None
        addmm_307 = torch.ops.aten.addmm.default(arg957_1, view_824, permute_412);  arg957_1 = view_824 = permute_412 = None
        view_825 = torch.ops.aten.view.default(addmm_307, [128, 128, 512]);  addmm_307 = None
        relu_80 = torch.ops.aten.relu.default(view_825);  view_825 = None
        view_826 = torch.ops.aten.view.default(relu_80, [16384, 512]);  relu_80 = None
        permute_413 = torch.ops.aten.permute.default(arg958_1, [1, 0]);  arg958_1 = None
        addmm_308 = torch.ops.aten.addmm.default(arg959_1, view_826, permute_413);  arg959_1 = view_826 = permute_413 = None
        view_827 = torch.ops.aten.view.default(addmm_308, [128, 128, 128]);  addmm_308 = None
        add_308 = torch.ops.aten.add.Tensor(view_827, add_307);  view_827 = add_307 = None
        mul_165 = torch.ops.aten.mul.Tensor(add_308, arg961_1);  add_308 = arg961_1 = None
        add_309 = torch.ops.aten.add.Tensor(mul_165, arg960_1);  mul_165 = arg960_1 = None
        view_828 = torch.ops.aten.view.default(add_309, [16384, 128])
        permute_414 = torch.ops.aten.permute.default(arg962_1, [1, 0]);  arg962_1 = None
        addmm_309 = torch.ops.aten.addmm.default(arg963_1, view_828, permute_414);  arg963_1 = view_828 = permute_414 = None
        view_829 = torch.ops.aten.view.default(addmm_309, [128, 128, 512]);  addmm_309 = None
        relu_81 = torch.ops.aten.relu.default(view_829);  view_829 = None
        view_830 = torch.ops.aten.view.default(relu_81, [16384, 512]);  relu_81 = None
        permute_415 = torch.ops.aten.permute.default(arg964_1, [1, 0]);  arg964_1 = None
        addmm_310 = torch.ops.aten.addmm.default(arg965_1, view_830, permute_415);  arg965_1 = view_830 = permute_415 = None
        view_831 = torch.ops.aten.view.default(addmm_310, [128, 128, 128]);  addmm_310 = None
        add_310 = torch.ops.aten.add.Tensor(view_831, add_309);  view_831 = add_309 = None
        mul_166 = torch.ops.aten.mul.Tensor(add_310, arg967_1);  add_310 = arg967_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_166, arg966_1);  mul_166 = arg966_1 = None
        view_832 = torch.ops.aten.view.default(add_311, [16384, 128])
        permute_416 = torch.ops.aten.permute.default(arg968_1, [1, 0]);  arg968_1 = None
        addmm_311 = torch.ops.aten.addmm.default(arg969_1, view_832, permute_416);  arg969_1 = view_832 = permute_416 = None
        view_833 = torch.ops.aten.view.default(addmm_311, [128, 128, 512]);  addmm_311 = None
        relu_82 = torch.ops.aten.relu.default(view_833);  view_833 = None
        view_834 = torch.ops.aten.view.default(relu_82, [16384, 512]);  relu_82 = None
        permute_417 = torch.ops.aten.permute.default(arg970_1, [1, 0]);  arg970_1 = None
        addmm_312 = torch.ops.aten.addmm.default(arg971_1, view_834, permute_417);  arg971_1 = view_834 = permute_417 = None
        view_835 = torch.ops.aten.view.default(addmm_312, [128, 128, 128]);  addmm_312 = None
        add_312 = torch.ops.aten.add.Tensor(view_835, add_311);  view_835 = add_311 = None
        mul_167 = torch.ops.aten.mul.Tensor(add_312, arg973_1);  add_312 = arg973_1 = None
        add_313 = torch.ops.aten.add.Tensor(mul_167, arg972_1);  mul_167 = arg972_1 = None
        view_836 = torch.ops.aten.view.default(add_313, [16384, 128])
        permute_418 = torch.ops.aten.permute.default(arg938_1, [1, 0]);  arg938_1 = None
        addmm_313 = torch.ops.aten.addmm.default(arg939_1, view_836, permute_418);  arg939_1 = view_836 = permute_418 = None
        view_837 = torch.ops.aten.view.default(addmm_313, [128, 128, 512]);  addmm_313 = None
        relu_83 = torch.ops.aten.relu.default(view_837);  view_837 = None
        view_838 = torch.ops.aten.view.default(relu_83, [16384, 512]);  relu_83 = None
        permute_419 = torch.ops.aten.permute.default(arg940_1, [1, 0]);  arg940_1 = None
        addmm_314 = torch.ops.aten.addmm.default(arg941_1, view_838, permute_419);  arg941_1 = view_838 = permute_419 = None
        view_839 = torch.ops.aten.view.default(addmm_314, [128, 128, 128]);  addmm_314 = None
        add_314 = torch.ops.aten.add.Tensor(view_839, add_313);  view_839 = add_313 = None
        mul_168 = torch.ops.aten.mul.Tensor(add_314, arg943_1);  add_314 = arg943_1 = None
        add_315 = torch.ops.aten.add.Tensor(mul_168, arg942_1);  mul_168 = arg942_1 = None
        view_840 = torch.ops.aten.view.default(add_315, [16384, 128]);  add_315 = None
        permute_420 = torch.ops.aten.permute.default(arg944_1, [1, 0]);  arg944_1 = None
        addmm_315 = torch.ops.aten.addmm.default(arg945_1, view_840, permute_420);  arg945_1 = view_840 = permute_420 = None
        view_841 = torch.ops.aten.view.default(addmm_315, [128, 128, 512]);  addmm_315 = None
        add_316 = torch.ops.aten.add.Tensor(view_841, add_302);  view_841 = add_302 = None
        mul_169 = torch.ops.aten.mul.Tensor(add_316, arg947_1);  add_316 = arg947_1 = None
        add_317 = torch.ops.aten.add.Tensor(mul_169, arg946_1);  mul_169 = arg946_1 = None
        view_842 = torch.ops.aten.view.default(add_317, [16384, 512])
        permute_421 = torch.ops.aten.permute.default(arg994_1, [1, 0]);  arg994_1 = None
        addmm_316 = torch.ops.aten.addmm.default(arg995_1, view_842, permute_421);  arg995_1 = view_842 = permute_421 = None
        view_843 = torch.ops.aten.view.default(addmm_316, [128, 128, 128]);  addmm_316 = None
        mul_170 = torch.ops.aten.mul.Tensor(view_843, arg997_1);  view_843 = arg997_1 = None
        add_318 = torch.ops.aten.add.Tensor(mul_170, arg996_1);  mul_170 = arg996_1 = None
        view_844 = torch.ops.aten.view.default(add_317, [16384, 512])
        permute_422 = torch.ops.aten.permute.default(arg998_1, [1, 0]);  arg998_1 = None
        addmm_317 = torch.ops.aten.addmm.default(arg999_1, view_844, permute_422);  arg999_1 = view_844 = permute_422 = None
        view_845 = torch.ops.aten.view.default(addmm_317, [128, 128, 128]);  addmm_317 = None
        mul_171 = torch.ops.aten.mul.Tensor(view_845, arg1001_1);  view_845 = arg1001_1 = None
        add_319 = torch.ops.aten.add.Tensor(mul_171, arg1000_1);  mul_171 = arg1000_1 = None
        view_846 = torch.ops.aten.view.default(add_319, [16384, 128])
        permute_423 = torch.ops.aten.permute.default(arg974_1, [1, 0]);  arg974_1 = None
        addmm_318 = torch.ops.aten.addmm.default(arg975_1, view_846, permute_423);  arg975_1 = view_846 = permute_423 = None
        view_847 = torch.ops.aten.view.default(addmm_318, [128, 128, 128]);  addmm_318 = None
        view_848 = torch.ops.aten.view.default(add_319, [16384, 128]);  add_319 = None
        permute_424 = torch.ops.aten.permute.default(arg976_1, [1, 0]);  arg976_1 = None
        addmm_319 = torch.ops.aten.addmm.default(arg977_1, view_848, permute_424);  arg977_1 = view_848 = permute_424 = None
        view_849 = torch.ops.aten.view.default(addmm_319, [128, 128, 128]);  addmm_319 = None
        view_850 = torch.ops.aten.view.default(add_317, [16384, 512])
        permute_425 = torch.ops.aten.permute.default(arg978_1, [1, 0]);  arg978_1 = None
        addmm_320 = torch.ops.aten.addmm.default(arg979_1, view_850, permute_425);  arg979_1 = view_850 = permute_425 = None
        view_851 = torch.ops.aten.view.default(addmm_320, [128, 128, 128]);  addmm_320 = None
        view_852 = torch.ops.aten.view.default(view_847, [128, 128, 4, 32]);  view_847 = None
        view_853 = torch.ops.aten.view.default(view_849, [128, 128, 4, 32]);  view_849 = None
        view_854 = torch.ops.aten.view.default(view_851, [128, 128, 4, 32]);  view_851 = None
        permute_default_6 = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
        permute_default_7 = torch.ops.aten.permute.default(view_853, [0, 2, 1, 3]);  view_853 = None
        permute_default_8 = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.17677669529663687);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_4 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        permute_430 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        clone_131 = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None
        view_861 = torch.ops.aten.view.default(clone_131, [128, 128, 128]);  clone_131 = None
        view_862 = torch.ops.aten.view.default(view_861, [16384, 128]);  view_861 = None
        permute_431 = torch.ops.aten.permute.default(arg980_1, [1, 0]);  arg980_1 = None
        addmm_321 = torch.ops.aten.addmm.default(arg981_1, view_862, permute_431);  arg981_1 = view_862 = permute_431 = None
        view_863 = torch.ops.aten.view.default(addmm_321, [128, 128, 128]);  addmm_321 = None
        add_321 = torch.ops.aten.add.Tensor(view_863, add_318);  view_863 = add_318 = None
        mul_172 = torch.ops.aten.mul.Tensor(add_321, arg983_1);  add_321 = arg983_1 = None
        add_322 = torch.ops.aten.add.Tensor(mul_172, arg982_1);  mul_172 = arg982_1 = None
        view_864 = torch.ops.aten.view.default(add_322, [16384, 128])
        permute_432 = torch.ops.aten.permute.default(arg1002_1, [1, 0]);  arg1002_1 = None
        addmm_322 = torch.ops.aten.addmm.default(arg1003_1, view_864, permute_432);  arg1003_1 = view_864 = permute_432 = None
        view_865 = torch.ops.aten.view.default(addmm_322, [128, 128, 512]);  addmm_322 = None
        relu_84 = torch.ops.aten.relu.default(view_865);  view_865 = None
        view_866 = torch.ops.aten.view.default(relu_84, [16384, 512]);  relu_84 = None
        permute_433 = torch.ops.aten.permute.default(arg1004_1, [1, 0]);  arg1004_1 = None
        addmm_323 = torch.ops.aten.addmm.default(arg1005_1, view_866, permute_433);  arg1005_1 = view_866 = permute_433 = None
        view_867 = torch.ops.aten.view.default(addmm_323, [128, 128, 128]);  addmm_323 = None
        add_323 = torch.ops.aten.add.Tensor(view_867, add_322);  view_867 = add_322 = None
        mul_173 = torch.ops.aten.mul.Tensor(add_323, arg1007_1);  add_323 = arg1007_1 = None
        add_324 = torch.ops.aten.add.Tensor(mul_173, arg1006_1);  mul_173 = arg1006_1 = None
        view_868 = torch.ops.aten.view.default(add_324, [16384, 128])
        permute_434 = torch.ops.aten.permute.default(arg1008_1, [1, 0]);  arg1008_1 = None
        addmm_324 = torch.ops.aten.addmm.default(arg1009_1, view_868, permute_434);  arg1009_1 = view_868 = permute_434 = None
        view_869 = torch.ops.aten.view.default(addmm_324, [128, 128, 512]);  addmm_324 = None
        relu_85 = torch.ops.aten.relu.default(view_869);  view_869 = None
        view_870 = torch.ops.aten.view.default(relu_85, [16384, 512]);  relu_85 = None
        permute_435 = torch.ops.aten.permute.default(arg1010_1, [1, 0]);  arg1010_1 = None
        addmm_325 = torch.ops.aten.addmm.default(arg1011_1, view_870, permute_435);  arg1011_1 = view_870 = permute_435 = None
        view_871 = torch.ops.aten.view.default(addmm_325, [128, 128, 128]);  addmm_325 = None
        add_325 = torch.ops.aten.add.Tensor(view_871, add_324);  view_871 = add_324 = None
        mul_174 = torch.ops.aten.mul.Tensor(add_325, arg1013_1);  add_325 = arg1013_1 = None
        add_326 = torch.ops.aten.add.Tensor(mul_174, arg1012_1);  mul_174 = arg1012_1 = None
        view_872 = torch.ops.aten.view.default(add_326, [16384, 128])
        permute_436 = torch.ops.aten.permute.default(arg1014_1, [1, 0]);  arg1014_1 = None
        addmm_326 = torch.ops.aten.addmm.default(arg1015_1, view_872, permute_436);  arg1015_1 = view_872 = permute_436 = None
        view_873 = torch.ops.aten.view.default(addmm_326, [128, 128, 512]);  addmm_326 = None
        relu_86 = torch.ops.aten.relu.default(view_873);  view_873 = None
        view_874 = torch.ops.aten.view.default(relu_86, [16384, 512]);  relu_86 = None
        permute_437 = torch.ops.aten.permute.default(arg1016_1, [1, 0]);  arg1016_1 = None
        addmm_327 = torch.ops.aten.addmm.default(arg1017_1, view_874, permute_437);  arg1017_1 = view_874 = permute_437 = None
        view_875 = torch.ops.aten.view.default(addmm_327, [128, 128, 128]);  addmm_327 = None
        add_327 = torch.ops.aten.add.Tensor(view_875, add_326);  view_875 = add_326 = None
        mul_175 = torch.ops.aten.mul.Tensor(add_327, arg1019_1);  add_327 = arg1019_1 = None
        add_328 = torch.ops.aten.add.Tensor(mul_175, arg1018_1);  mul_175 = arg1018_1 = None
        view_876 = torch.ops.aten.view.default(add_328, [16384, 128])
        permute_438 = torch.ops.aten.permute.default(arg984_1, [1, 0]);  arg984_1 = None
        addmm_328 = torch.ops.aten.addmm.default(arg985_1, view_876, permute_438);  arg985_1 = view_876 = permute_438 = None
        view_877 = torch.ops.aten.view.default(addmm_328, [128, 128, 512]);  addmm_328 = None
        relu_87 = torch.ops.aten.relu.default(view_877);  view_877 = None
        view_878 = torch.ops.aten.view.default(relu_87, [16384, 512]);  relu_87 = None
        permute_439 = torch.ops.aten.permute.default(arg986_1, [1, 0]);  arg986_1 = None
        addmm_329 = torch.ops.aten.addmm.default(arg987_1, view_878, permute_439);  arg987_1 = view_878 = permute_439 = None
        view_879 = torch.ops.aten.view.default(addmm_329, [128, 128, 128]);  addmm_329 = None
        add_329 = torch.ops.aten.add.Tensor(view_879, add_328);  view_879 = add_328 = None
        mul_176 = torch.ops.aten.mul.Tensor(add_329, arg989_1);  add_329 = arg989_1 = None
        add_330 = torch.ops.aten.add.Tensor(mul_176, arg988_1);  mul_176 = arg988_1 = None
        view_880 = torch.ops.aten.view.default(add_330, [16384, 128]);  add_330 = None
        permute_440 = torch.ops.aten.permute.default(arg990_1, [1, 0]);  arg990_1 = None
        addmm_330 = torch.ops.aten.addmm.default(arg991_1, view_880, permute_440);  arg991_1 = view_880 = permute_440 = None
        view_881 = torch.ops.aten.view.default(addmm_330, [128, 128, 512]);  addmm_330 = None
        add_331 = torch.ops.aten.add.Tensor(view_881, add_317);  view_881 = add_317 = None
        mul_177 = torch.ops.aten.mul.Tensor(add_331, arg993_1);  add_331 = arg993_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_177, arg992_1);  mul_177 = arg992_1 = None
        view_882 = torch.ops.aten.view.default(add_332, [16384, 512])
        permute_441 = torch.ops.aten.permute.default(arg1040_1, [1, 0]);  arg1040_1 = None
        addmm_331 = torch.ops.aten.addmm.default(arg1041_1, view_882, permute_441);  arg1041_1 = view_882 = permute_441 = None
        view_883 = torch.ops.aten.view.default(addmm_331, [128, 128, 128]);  addmm_331 = None
        mul_178 = torch.ops.aten.mul.Tensor(view_883, arg1043_1);  view_883 = arg1043_1 = None
        add_333 = torch.ops.aten.add.Tensor(mul_178, arg1042_1);  mul_178 = arg1042_1 = None
        view_884 = torch.ops.aten.view.default(add_332, [16384, 512])
        permute_442 = torch.ops.aten.permute.default(arg1044_1, [1, 0]);  arg1044_1 = None
        addmm_332 = torch.ops.aten.addmm.default(arg1045_1, view_884, permute_442);  arg1045_1 = view_884 = permute_442 = None
        view_885 = torch.ops.aten.view.default(addmm_332, [128, 128, 128]);  addmm_332 = None
        mul_179 = torch.ops.aten.mul.Tensor(view_885, arg1047_1);  view_885 = arg1047_1 = None
        add_334 = torch.ops.aten.add.Tensor(mul_179, arg1046_1);  mul_179 = arg1046_1 = None
        view_886 = torch.ops.aten.view.default(add_334, [16384, 128])
        permute_443 = torch.ops.aten.permute.default(arg1020_1, [1, 0]);  arg1020_1 = None
        addmm_333 = torch.ops.aten.addmm.default(arg1021_1, view_886, permute_443);  arg1021_1 = view_886 = permute_443 = None
        view_887 = torch.ops.aten.view.default(addmm_333, [128, 128, 128]);  addmm_333 = None
        view_888 = torch.ops.aten.view.default(add_334, [16384, 128]);  add_334 = None
        permute_444 = torch.ops.aten.permute.default(arg1022_1, [1, 0]);  arg1022_1 = None
        addmm_334 = torch.ops.aten.addmm.default(arg1023_1, view_888, permute_444);  arg1023_1 = view_888 = permute_444 = None
        view_889 = torch.ops.aten.view.default(addmm_334, [128, 128, 128]);  addmm_334 = None
        view_890 = torch.ops.aten.view.default(add_332, [16384, 512])
        permute_445 = torch.ops.aten.permute.default(arg1024_1, [1, 0]);  arg1024_1 = None
        addmm_335 = torch.ops.aten.addmm.default(arg1025_1, view_890, permute_445);  arg1025_1 = view_890 = permute_445 = None
        view_891 = torch.ops.aten.view.default(addmm_335, [128, 128, 128]);  addmm_335 = None
        view_892 = torch.ops.aten.view.default(view_887, [128, 128, 4, 32]);  view_887 = None
        view_893 = torch.ops.aten.view.default(view_889, [128, 128, 4, 32]);  view_889 = None
        view_894 = torch.ops.aten.view.default(view_891, [128, 128, 4, 32]);  view_891 = None
        permute_default_3 = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
        permute_default_4 = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None
        permute_default_5 = torch.ops.aten.permute.default(view_894, [0, 2, 1, 3]);  view_894 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.17677669529663687);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_3 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        permute_450 = torch.ops.aten.permute.default(getitem_3, [0, 2, 1, 3]);  getitem_3 = None
        clone_137 = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
        view_901 = torch.ops.aten.view.default(clone_137, [128, 128, 128]);  clone_137 = None
        view_902 = torch.ops.aten.view.default(view_901, [16384, 128]);  view_901 = None
        permute_451 = torch.ops.aten.permute.default(arg1026_1, [1, 0]);  arg1026_1 = None
        addmm_336 = torch.ops.aten.addmm.default(arg1027_1, view_902, permute_451);  arg1027_1 = view_902 = permute_451 = None
        view_903 = torch.ops.aten.view.default(addmm_336, [128, 128, 128]);  addmm_336 = None
        add_336 = torch.ops.aten.add.Tensor(view_903, add_333);  view_903 = add_333 = None
        mul_180 = torch.ops.aten.mul.Tensor(add_336, arg1029_1);  add_336 = arg1029_1 = None
        add_337 = torch.ops.aten.add.Tensor(mul_180, arg1028_1);  mul_180 = arg1028_1 = None
        view_904 = torch.ops.aten.view.default(add_337, [16384, 128])
        permute_452 = torch.ops.aten.permute.default(arg1048_1, [1, 0]);  arg1048_1 = None
        addmm_337 = torch.ops.aten.addmm.default(arg1049_1, view_904, permute_452);  arg1049_1 = view_904 = permute_452 = None
        view_905 = torch.ops.aten.view.default(addmm_337, [128, 128, 512]);  addmm_337 = None
        relu_88 = torch.ops.aten.relu.default(view_905);  view_905 = None
        view_906 = torch.ops.aten.view.default(relu_88, [16384, 512]);  relu_88 = None
        permute_453 = torch.ops.aten.permute.default(arg1050_1, [1, 0]);  arg1050_1 = None
        addmm_338 = torch.ops.aten.addmm.default(arg1051_1, view_906, permute_453);  arg1051_1 = view_906 = permute_453 = None
        view_907 = torch.ops.aten.view.default(addmm_338, [128, 128, 128]);  addmm_338 = None
        add_338 = torch.ops.aten.add.Tensor(view_907, add_337);  view_907 = add_337 = None
        mul_181 = torch.ops.aten.mul.Tensor(add_338, arg1053_1);  add_338 = arg1053_1 = None
        add_339 = torch.ops.aten.add.Tensor(mul_181, arg1052_1);  mul_181 = arg1052_1 = None
        view_908 = torch.ops.aten.view.default(add_339, [16384, 128])
        permute_454 = torch.ops.aten.permute.default(arg1054_1, [1, 0]);  arg1054_1 = None
        addmm_339 = torch.ops.aten.addmm.default(arg1055_1, view_908, permute_454);  arg1055_1 = view_908 = permute_454 = None
        view_909 = torch.ops.aten.view.default(addmm_339, [128, 128, 512]);  addmm_339 = None
        relu_89 = torch.ops.aten.relu.default(view_909);  view_909 = None
        view_910 = torch.ops.aten.view.default(relu_89, [16384, 512]);  relu_89 = None
        permute_455 = torch.ops.aten.permute.default(arg1056_1, [1, 0]);  arg1056_1 = None
        addmm_340 = torch.ops.aten.addmm.default(arg1057_1, view_910, permute_455);  arg1057_1 = view_910 = permute_455 = None
        view_911 = torch.ops.aten.view.default(addmm_340, [128, 128, 128]);  addmm_340 = None
        add_340 = torch.ops.aten.add.Tensor(view_911, add_339);  view_911 = add_339 = None
        mul_182 = torch.ops.aten.mul.Tensor(add_340, arg1059_1);  add_340 = arg1059_1 = None
        add_341 = torch.ops.aten.add.Tensor(mul_182, arg1058_1);  mul_182 = arg1058_1 = None
        view_912 = torch.ops.aten.view.default(add_341, [16384, 128])
        permute_456 = torch.ops.aten.permute.default(arg1060_1, [1, 0]);  arg1060_1 = None
        addmm_341 = torch.ops.aten.addmm.default(arg1061_1, view_912, permute_456);  arg1061_1 = view_912 = permute_456 = None
        view_913 = torch.ops.aten.view.default(addmm_341, [128, 128, 512]);  addmm_341 = None
        relu_90 = torch.ops.aten.relu.default(view_913);  view_913 = None
        view_914 = torch.ops.aten.view.default(relu_90, [16384, 512]);  relu_90 = None
        permute_457 = torch.ops.aten.permute.default(arg1062_1, [1, 0]);  arg1062_1 = None
        addmm_342 = torch.ops.aten.addmm.default(arg1063_1, view_914, permute_457);  arg1063_1 = view_914 = permute_457 = None
        view_915 = torch.ops.aten.view.default(addmm_342, [128, 128, 128]);  addmm_342 = None
        add_342 = torch.ops.aten.add.Tensor(view_915, add_341);  view_915 = add_341 = None
        mul_183 = torch.ops.aten.mul.Tensor(add_342, arg1065_1);  add_342 = arg1065_1 = None
        add_343 = torch.ops.aten.add.Tensor(mul_183, arg1064_1);  mul_183 = arg1064_1 = None
        view_916 = torch.ops.aten.view.default(add_343, [16384, 128])
        permute_458 = torch.ops.aten.permute.default(arg1030_1, [1, 0]);  arg1030_1 = None
        addmm_343 = torch.ops.aten.addmm.default(arg1031_1, view_916, permute_458);  arg1031_1 = view_916 = permute_458 = None
        view_917 = torch.ops.aten.view.default(addmm_343, [128, 128, 512]);  addmm_343 = None
        relu_91 = torch.ops.aten.relu.default(view_917);  view_917 = None
        view_918 = torch.ops.aten.view.default(relu_91, [16384, 512]);  relu_91 = None
        permute_459 = torch.ops.aten.permute.default(arg1032_1, [1, 0]);  arg1032_1 = None
        addmm_344 = torch.ops.aten.addmm.default(arg1033_1, view_918, permute_459);  arg1033_1 = view_918 = permute_459 = None
        view_919 = torch.ops.aten.view.default(addmm_344, [128, 128, 128]);  addmm_344 = None
        add_344 = torch.ops.aten.add.Tensor(view_919, add_343);  view_919 = add_343 = None
        mul_184 = torch.ops.aten.mul.Tensor(add_344, arg1035_1);  add_344 = arg1035_1 = None
        add_345 = torch.ops.aten.add.Tensor(mul_184, arg1034_1);  mul_184 = arg1034_1 = None
        view_920 = torch.ops.aten.view.default(add_345, [16384, 128]);  add_345 = None
        permute_460 = torch.ops.aten.permute.default(arg1036_1, [1, 0]);  arg1036_1 = None
        addmm_345 = torch.ops.aten.addmm.default(arg1037_1, view_920, permute_460);  arg1037_1 = view_920 = permute_460 = None
        view_921 = torch.ops.aten.view.default(addmm_345, [128, 128, 512]);  addmm_345 = None
        add_346 = torch.ops.aten.add.Tensor(view_921, add_332);  view_921 = add_332 = None
        mul_185 = torch.ops.aten.mul.Tensor(add_346, arg1039_1);  add_346 = arg1039_1 = None
        add_347 = torch.ops.aten.add.Tensor(mul_185, arg1038_1);  mul_185 = arg1038_1 = None
        view_922 = torch.ops.aten.view.default(add_347, [16384, 512])
        permute_461 = torch.ops.aten.permute.default(arg1086_1, [1, 0]);  arg1086_1 = None
        addmm_346 = torch.ops.aten.addmm.default(arg1087_1, view_922, permute_461);  arg1087_1 = view_922 = permute_461 = None
        view_923 = torch.ops.aten.view.default(addmm_346, [128, 128, 128]);  addmm_346 = None
        mul_186 = torch.ops.aten.mul.Tensor(view_923, arg1089_1);  view_923 = arg1089_1 = None
        add_348 = torch.ops.aten.add.Tensor(mul_186, arg1088_1);  mul_186 = arg1088_1 = None
        view_924 = torch.ops.aten.view.default(add_347, [16384, 512])
        permute_462 = torch.ops.aten.permute.default(arg1090_1, [1, 0]);  arg1090_1 = None
        addmm_347 = torch.ops.aten.addmm.default(arg1091_1, view_924, permute_462);  arg1091_1 = view_924 = permute_462 = None
        view_925 = torch.ops.aten.view.default(addmm_347, [128, 128, 128]);  addmm_347 = None
        mul_187 = torch.ops.aten.mul.Tensor(view_925, arg1093_1);  view_925 = arg1093_1 = None
        add_349 = torch.ops.aten.add.Tensor(mul_187, arg1092_1);  mul_187 = arg1092_1 = None
        view_926 = torch.ops.aten.view.default(add_349, [16384, 128])
        permute_463 = torch.ops.aten.permute.default(arg1066_1, [1, 0]);  arg1066_1 = None
        addmm_348 = torch.ops.aten.addmm.default(arg1067_1, view_926, permute_463);  arg1067_1 = view_926 = permute_463 = None
        view_927 = torch.ops.aten.view.default(addmm_348, [128, 128, 128]);  addmm_348 = None
        view_928 = torch.ops.aten.view.default(add_349, [16384, 128]);  add_349 = None
        permute_464 = torch.ops.aten.permute.default(arg1068_1, [1, 0]);  arg1068_1 = None
        addmm_349 = torch.ops.aten.addmm.default(arg1069_1, view_928, permute_464);  arg1069_1 = view_928 = permute_464 = None
        view_929 = torch.ops.aten.view.default(addmm_349, [128, 128, 128]);  addmm_349 = None
        view_930 = torch.ops.aten.view.default(add_347, [16384, 512])
        permute_465 = torch.ops.aten.permute.default(arg1070_1, [1, 0]);  arg1070_1 = None
        addmm_350 = torch.ops.aten.addmm.default(arg1071_1, view_930, permute_465);  arg1071_1 = view_930 = permute_465 = None
        view_931 = torch.ops.aten.view.default(addmm_350, [128, 128, 128]);  addmm_350 = None
        view_932 = torch.ops.aten.view.default(view_927, [128, 128, 4, 32]);  view_927 = None
        view_933 = torch.ops.aten.view.default(view_929, [128, 128, 4, 32]);  view_929 = None
        view_934 = torch.ops.aten.view.default(view_931, [128, 128, 4, 32]);  view_931 = None
        permute_default = torch.ops.aten.permute.default(view_932, [0, 2, 1, 3]);  view_932 = None
        permute_default_1 = torch.ops.aten.permute.default(view_933, [0, 2, 1, 3]);  view_933 = None
        permute_default_2 = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.17677669529663687);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_2 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        permute_470 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_143 = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
        view_941 = torch.ops.aten.view.default(clone_143, [128, 128, 128]);  clone_143 = None
        view_942 = torch.ops.aten.view.default(view_941, [16384, 128]);  view_941 = None
        permute_471 = torch.ops.aten.permute.default(arg1072_1, [1, 0]);  arg1072_1 = None
        addmm_351 = torch.ops.aten.addmm.default(arg1073_1, view_942, permute_471);  arg1073_1 = view_942 = permute_471 = None
        view_943 = torch.ops.aten.view.default(addmm_351, [128, 128, 128]);  addmm_351 = None
        add_351 = torch.ops.aten.add.Tensor(view_943, add_348);  view_943 = add_348 = None
        mul_188 = torch.ops.aten.mul.Tensor(add_351, arg1075_1);  add_351 = arg1075_1 = None
        add_352 = torch.ops.aten.add.Tensor(mul_188, arg1074_1);  mul_188 = arg1074_1 = None
        view_944 = torch.ops.aten.view.default(add_352, [16384, 128])
        permute_472 = torch.ops.aten.permute.default(arg1094_1, [1, 0]);  arg1094_1 = None
        addmm_352 = torch.ops.aten.addmm.default(arg1095_1, view_944, permute_472);  arg1095_1 = view_944 = permute_472 = None
        view_945 = torch.ops.aten.view.default(addmm_352, [128, 128, 512]);  addmm_352 = None
        relu_92 = torch.ops.aten.relu.default(view_945);  view_945 = None
        view_946 = torch.ops.aten.view.default(relu_92, [16384, 512]);  relu_92 = None
        permute_473 = torch.ops.aten.permute.default(arg1096_1, [1, 0]);  arg1096_1 = None
        addmm_353 = torch.ops.aten.addmm.default(arg1097_1, view_946, permute_473);  arg1097_1 = view_946 = permute_473 = None
        view_947 = torch.ops.aten.view.default(addmm_353, [128, 128, 128]);  addmm_353 = None
        add_353 = torch.ops.aten.add.Tensor(view_947, add_352);  view_947 = add_352 = None
        mul_189 = torch.ops.aten.mul.Tensor(add_353, arg1099_1);  add_353 = arg1099_1 = None
        add_354 = torch.ops.aten.add.Tensor(mul_189, arg1098_1);  mul_189 = arg1098_1 = None
        view_948 = torch.ops.aten.view.default(add_354, [16384, 128])
        permute_474 = torch.ops.aten.permute.default(arg1100_1, [1, 0]);  arg1100_1 = None
        addmm_354 = torch.ops.aten.addmm.default(arg1101_1, view_948, permute_474);  arg1101_1 = view_948 = permute_474 = None
        view_949 = torch.ops.aten.view.default(addmm_354, [128, 128, 512]);  addmm_354 = None
        relu_93 = torch.ops.aten.relu.default(view_949);  view_949 = None
        view_950 = torch.ops.aten.view.default(relu_93, [16384, 512]);  relu_93 = None
        permute_475 = torch.ops.aten.permute.default(arg1102_1, [1, 0]);  arg1102_1 = None
        addmm_355 = torch.ops.aten.addmm.default(arg1103_1, view_950, permute_475);  arg1103_1 = view_950 = permute_475 = None
        view_951 = torch.ops.aten.view.default(addmm_355, [128, 128, 128]);  addmm_355 = None
        add_355 = torch.ops.aten.add.Tensor(view_951, add_354);  view_951 = add_354 = None
        mul_190 = torch.ops.aten.mul.Tensor(add_355, arg1105_1);  add_355 = arg1105_1 = None
        add_356 = torch.ops.aten.add.Tensor(mul_190, arg1104_1);  mul_190 = arg1104_1 = None
        view_952 = torch.ops.aten.view.default(add_356, [16384, 128])
        permute_476 = torch.ops.aten.permute.default(arg1106_1, [1, 0]);  arg1106_1 = None
        addmm_356 = torch.ops.aten.addmm.default(arg1107_1, view_952, permute_476);  arg1107_1 = view_952 = permute_476 = None
        view_953 = torch.ops.aten.view.default(addmm_356, [128, 128, 512]);  addmm_356 = None
        relu_94 = torch.ops.aten.relu.default(view_953);  view_953 = None
        view_954 = torch.ops.aten.view.default(relu_94, [16384, 512]);  relu_94 = None
        permute_477 = torch.ops.aten.permute.default(arg1108_1, [1, 0]);  arg1108_1 = None
        addmm_357 = torch.ops.aten.addmm.default(arg1109_1, view_954, permute_477);  arg1109_1 = view_954 = permute_477 = None
        view_955 = torch.ops.aten.view.default(addmm_357, [128, 128, 128]);  addmm_357 = None
        add_357 = torch.ops.aten.add.Tensor(view_955, add_356);  view_955 = add_356 = None
        mul_191 = torch.ops.aten.mul.Tensor(add_357, arg1111_1);  add_357 = arg1111_1 = None
        add_358 = torch.ops.aten.add.Tensor(mul_191, arg1110_1);  mul_191 = arg1110_1 = None
        view_956 = torch.ops.aten.view.default(add_358, [16384, 128])
        permute_478 = torch.ops.aten.permute.default(arg1076_1, [1, 0]);  arg1076_1 = None
        addmm_358 = torch.ops.aten.addmm.default(arg1077_1, view_956, permute_478);  arg1077_1 = view_956 = permute_478 = None
        view_957 = torch.ops.aten.view.default(addmm_358, [128, 128, 512]);  addmm_358 = None
        relu_95 = torch.ops.aten.relu.default(view_957);  view_957 = None
        view_958 = torch.ops.aten.view.default(relu_95, [16384, 512]);  relu_95 = None
        permute_479 = torch.ops.aten.permute.default(arg1078_1, [1, 0]);  arg1078_1 = None
        addmm_359 = torch.ops.aten.addmm.default(arg1079_1, view_958, permute_479);  arg1079_1 = view_958 = permute_479 = None
        view_959 = torch.ops.aten.view.default(addmm_359, [128, 128, 128]);  addmm_359 = None
        add_359 = torch.ops.aten.add.Tensor(view_959, add_358);  view_959 = add_358 = None
        mul_192 = torch.ops.aten.mul.Tensor(add_359, arg1081_1);  add_359 = arg1081_1 = None
        add_360 = torch.ops.aten.add.Tensor(mul_192, arg1080_1);  mul_192 = arg1080_1 = None
        view_960 = torch.ops.aten.view.default(add_360, [16384, 128]);  add_360 = None
        permute_480 = torch.ops.aten.permute.default(arg1082_1, [1, 0]);  arg1082_1 = None
        addmm_360 = torch.ops.aten.addmm.default(arg1083_1, view_960, permute_480);  arg1083_1 = view_960 = permute_480 = None
        view_961 = torch.ops.aten.view.default(addmm_360, [128, 128, 512]);  addmm_360 = None
        add_361 = torch.ops.aten.add.Tensor(view_961, add_347);  view_961 = add_347 = None
        mul_193 = torch.ops.aten.mul.Tensor(add_361, arg1085_1);  add_361 = arg1085_1 = None
        add_362 = torch.ops.aten.add.Tensor(mul_193, arg1084_1);  mul_193 = arg1084_1 = None
        view_962 = torch.ops.aten.view.default(add_362, [16384, 512]);  add_362 = None
        permute_481 = torch.ops.aten.permute.default(arg1113_1, [1, 0]);  arg1113_1 = None
        addmm_361 = torch.ops.aten.addmm.default(arg1114_1, view_962, permute_481);  arg1114_1 = view_962 = permute_481 = None
        view_963 = torch.ops.aten.view.default(addmm_361, [128, 128, 2]);  addmm_361 = None
        split = torch.ops.aten.split.Tensor(view_963, 1, -1);  view_963 = None
        getitem = split[0]
        getitem_1 = split[1];  split = None
        squeeze = torch.ops.aten.squeeze.dim(getitem, -1);  getitem = None
        clone_145 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        squeeze_1 = torch.ops.aten.squeeze.dim(getitem_1, -1);  getitem_1 = None
        clone_146 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(arg1115_1, 0);  arg1115_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(arg1116_1, 0);  arg1116_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
        amax_24 = torch.ops.aten.amax.default(clone_145, [1], True)
        sub_25 = torch.ops.aten.sub.Tensor(clone_145, amax_24);  amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_25)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_26 = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        ne = torch.ops.aten.ne.Scalar(clamp_max, 128)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, clamp_max, full_default_2);  ne = full_default_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_26, 1, unsqueeze_2);  sub_26 = unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        ne_1 = torch.ops.aten.ne.Scalar(clamp_max, 128)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(clamp_max, 128);  clamp_max = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_48 = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
        amax_25 = torch.ops.aten.amax.default(clone_146, [1], True)
        sub_27 = torch.ops.aten.sub.Tensor(clone_146, amax_25);  amax_25 = None
        exp_25 = torch.ops.aten.exp.default(sub_27)
        sum_28 = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
        log_1 = torch.ops.aten.log.default(sum_28);  sum_28 = None
        sub_28 = torch.ops.aten.sub.Tensor(sub_27, log_1);  sub_27 = log_1 = None
        ne_3 = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
        full_default_4 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_4);  ne_3 = full_default_4 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1 = torch.ops.aten.gather.default(sub_28, 1, unsqueeze_3);  sub_28 = unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        ne_4 = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
        full_default_5 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne_4, neg_1, full_default_5);  ne_4 = neg_1 = full_default_5 = None
        ne_5 = torch.ops.aten.ne.Scalar(clamp_max_1, 128);  clamp_max_1 = None
        sum_29 = torch.ops.aten.sum.default(ne_5);  ne_5 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
        sum_30 = torch.ops.aten.sum.default(where_3);  where_3 = None
        div_49 = torch.ops.aten.div.Tensor(sum_30, convert_element_type_1);  sum_30 = convert_element_type_1 = None
        add_363 = torch.ops.aten.add.Tensor(div_48, div_49);  div_48 = div_49 = None
        div_50 = torch.ops.aten.div.Tensor(add_363, 2);  add_363 = None
        return (div_50, clone_145, clone_146)
        
def load_args(reader):
    buf0 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (128, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 15627264, device=device(type='cuda', index=0))
    reader.tensor(buf1, (30522, 128), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 512), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf3, (2, 512), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf4, (512, 384), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128, 128), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128, 128), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128, 512), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128, 128), is_leaf=True)  # arg14_1
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
    buf23 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512, 128), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128, 512), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf31, (128,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128, 512), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf34, (128,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf35, (128,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf36, (512, 128), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128, 512), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf39, (128,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf40, (128,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf41, (128,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512, 128), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf44, (128, 512), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf45, (128,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf46, (128,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf47, (128,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf48, (512, 128), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf50, (128, 512), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf51, (128,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf52, (128,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf53, (128,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf54, (128, 128), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf55, (128,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf56, (128, 128), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf57, (128,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf58, (128, 512), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf59, (128,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf60, (128, 128), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512, 128), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf66, (128, 512), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf69, (128,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512, 128), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128, 512), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128, 512), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf79, (128,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (128,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf81, (128,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512, 128), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128, 512), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf87, (128,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512, 128), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128, 512), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf94, (512, 128), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf95, (512,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf96, (128, 512), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf99, (128,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf100, (128, 128), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf101, (128,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf102, (128, 128), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf103, (128,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf104, (128, 512), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf105, (128,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128, 128), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (128,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf109, (128,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512, 128), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf112, (128, 512), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf113, (128,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf114, (128,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512, 128), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf120, (128, 512), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf121, (128,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf122, (128,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf123, (128,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf124, (128, 512), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf125, (128,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf126, (128,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf127, (128,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512, 128), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf130, (128, 512), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf131, (128,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf132, (128,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf133, (128,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf134, (512, 128), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf135, (512,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf136, (128, 512), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf137, (128,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf138, (128,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf139, (128,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512, 128), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf141, (512,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf142, (128, 512), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf143, (128,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf144, (128,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf145, (128,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf146, (128, 128), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf147, (128,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf148, (128, 128), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf149, (128,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf150, (128, 512), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf151, (128,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf152, (128, 128), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf153, (128,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf154, (128,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf155, (128,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf156, (512, 128), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf157, (512,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf158, (128, 512), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf159, (128,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf160, (128,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf161, (128,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512, 128), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf164, (512,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf165, (512,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf166, (128, 512), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf167, (128,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf168, (128,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf169, (128,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf170, (128, 512), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf171, (128,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf172, (128,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf173, (128,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512, 128), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf176, (128, 512), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf177, (128,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf178, (128,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf179, (128,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512, 128), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf182, (128, 512), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf183, (128,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf184, (128,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf185, (128,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512, 128), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf187, (512,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf188, (128, 512), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf189, (128,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf190, (128,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf191, (128,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf192, (128, 128), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf193, (128,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf194, (128, 128), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf195, (128,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf196, (128, 512), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf197, (128,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf198, (128, 128), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf199, (128,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf200, (128,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf201, (128,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf202, (512, 128), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf203, (512,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf204, (128, 512), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf205, (128,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf206, (128,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf207, (128,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf208, (512, 128), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf209, (512,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf210, (512,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf211, (512,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf212, (128, 512), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf213, (128,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf214, (128,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf215, (128,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf216, (128, 512), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf217, (128,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf218, (128,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf219, (128,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf220, (512, 128), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf222, (128, 512), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf223, (128,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf224, (128,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf225, (128,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf226, (512, 128), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf227, (512,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf228, (128, 512), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf229, (128,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf230, (128,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf231, (128,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf232, (512, 128), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf233, (512,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf234, (128, 512), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf235, (128,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf236, (128,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf237, (128,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf238, (128, 128), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf239, (128,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf240, (128, 128), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf241, (128,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf242, (128, 512), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf243, (128,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf244, (128, 128), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf245, (128,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf246, (128,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf247, (128,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf248, (512, 128), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf249, (512,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf250, (128, 512), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf251, (128,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf252, (128,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf253, (128,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf254, (512, 128), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf255, (512,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf256, (512,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf257, (512,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf258, (128, 512), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf259, (128,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf260, (128,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf261, (128,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf262, (128, 512), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf263, (128,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf264, (128,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf265, (128,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf266, (512, 128), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf267, (512,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf268, (128, 512), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf269, (128,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf270, (128,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf271, (128,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf272, (512, 128), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf273, (512,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf274, (128, 512), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf275, (128,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf276, (128,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf277, (128,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf278, (512, 128), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf279, (512,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf280, (128, 512), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf281, (128,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf282, (128,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf283, (128,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf284, (128, 128), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf285, (128,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf286, (128, 128), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf287, (128,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf288, (128, 512), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf289, (128,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf290, (128, 128), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf291, (128,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf292, (128,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf293, (128,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf294, (512, 128), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf295, (512,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf296, (128, 512), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf297, (128,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf298, (128,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf299, (128,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf300, (512, 128), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf301, (512,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf302, (512,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf303, (512,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf304, (128, 512), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf305, (128,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf306, (128,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf307, (128,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf308, (128, 512), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf309, (128,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf310, (128,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf311, (128,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf312, (512, 128), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf313, (512,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf314, (128, 512), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf315, (128,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf316, (128,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf317, (128,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf318, (512, 128), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf319, (512,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf320, (128, 512), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf321, (128,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf322, (128,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf323, (128,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf324, (512, 128), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf325, (512,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf326, (128, 512), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf327, (128,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf328, (128,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf329, (128,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf330, (128, 128), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf331, (128,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf332, (128, 128), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf333, (128,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf334, (128, 512), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf335, (128,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf336, (128, 128), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf337, (128,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf338, (128,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf339, (128,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf340, (512, 128), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf341, (512,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf342, (128, 512), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf343, (128,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf344, (128,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf345, (128,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf346, (512, 128), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf347, (512,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf348, (512,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf349, (512,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf350, (128, 512), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf351, (128,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf352, (128,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf353, (128,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf354, (128, 512), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf355, (128,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf356, (128,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf357, (128,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf358, (512, 128), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf359, (512,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf360, (128, 512), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf361, (128,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf362, (128,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf363, (128,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf364, (512, 128), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf365, (512,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf366, (128, 512), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf367, (128,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf368, (128,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf369, (128,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf370, (512, 128), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf371, (512,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf372, (128, 512), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf373, (128,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf374, (128,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf375, (128,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf376, (128, 128), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf377, (128,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf378, (128, 128), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf379, (128,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf380, (128, 512), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf381, (128,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf382, (128, 128), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf383, (128,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf384, (128,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf385, (128,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf386, (512, 128), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf387, (512,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf388, (128, 512), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf389, (128,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf390, (128,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf391, (128,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf392, (512, 128), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf393, (512,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf394, (512,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf395, (512,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf396, (128, 512), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf397, (128,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf398, (128,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf399, (128,), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf400, (128, 512), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf401, (128,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf402, (128,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf403, (128,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf404, (512, 128), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf405, (512,), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf406, (128, 512), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf407, (128,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf408, (128,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf409, (128,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf410, (512, 128), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf411, (512,), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf412, (128, 512), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf413, (128,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf414, (128,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf415, (128,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf416, (512, 128), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf417, (512,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf418, (128, 512), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf419, (128,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf420, (128,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf421, (128,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf422, (128, 128), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf423, (128,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf424, (128, 128), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf425, (128,), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf426, (128, 512), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf427, (128,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf428, (128, 128), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf429, (128,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf430, (128,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf431, (128,), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf432, (512, 128), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf433, (512,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf434, (128, 512), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf435, (128,), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf436, (128,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf437, (128,), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf438, (512, 128), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf439, (512,), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf440, (512,), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf441, (512,), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf442, (128, 512), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf443, (128,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf444, (128,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf445, (128,), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf446, (128, 512), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf447, (128,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf448, (128,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf449, (128,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf450, (512, 128), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf451, (512,), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf452, (128, 512), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf453, (128,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf454, (128,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf455, (128,), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf456, (512, 128), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf457, (512,), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf458, (128, 512), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf459, (128,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf460, (128,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf461, (128,), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf462, (512, 128), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf463, (512,), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf464, (128, 512), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf465, (128,), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf466, (128,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf467, (128,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf468, (128, 128), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf469, (128,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf470, (128, 128), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf471, (128,), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf472, (128, 512), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf473, (128,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf474, (128, 128), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf475, (128,), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf476, (128,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf477, (128,), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf478, (512, 128), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf479, (512,), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf480, (128, 512), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf481, (128,), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf482, (128,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf483, (128,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf484, (512, 128), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf485, (512,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf486, (512,), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf487, (512,), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf488, (128, 512), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf489, (128,), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf490, (128,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf491, (128,), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf492, (128, 512), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf493, (128,), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf494, (128,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf495, (128,), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf496, (512, 128), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf497, (512,), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf498, (128, 512), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf499, (128,), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf500, (128,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf501, (128,), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf502, (512, 128), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf503, (512,), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf504, (128, 512), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf505, (128,), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf506, (128,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf507, (128,), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf508, (512, 128), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf509, (512,), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf510, (128, 512), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf511, (128,), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf512, (128,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf513, (128,), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf514, (128, 128), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf515, (128,), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf516, (128, 128), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf517, (128,), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf518, (128, 512), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf519, (128,), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf520, (128, 128), is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf521, (128,), is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf522, (128,), is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf523, (128,), is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf524, (512, 128), is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf525, (512,), is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf526, (128, 512), is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf527, (128,), is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf528, (128,), is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf529, (128,), is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf530, (512, 128), is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf531, (512,), is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf532, (512,), is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf533, (512,), is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf534, (128, 512), is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf535, (128,), is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf536, (128,), is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf537, (128,), is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf538, (128, 512), is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf539, (128,), is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf540, (128,), is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf541, (128,), is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf542, (512, 128), is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf543, (512,), is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf544, (128, 512), is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf545, (128,), is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf546, (128,), is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf547, (128,), is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf548, (512, 128), is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf549, (512,), is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf550, (128, 512), is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf551, (128,), is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf552, (128,), is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf553, (128,), is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf554, (512, 128), is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf555, (512,), is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf556, (128, 512), is_leaf=True)  # arg556_1
    buf557 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf557, (128,), is_leaf=True)  # arg557_1
    buf558 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf558, (128,), is_leaf=True)  # arg558_1
    buf559 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf559, (128,), is_leaf=True)  # arg559_1
    buf560 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf560, (128, 128), is_leaf=True)  # arg560_1
    buf561 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf561, (128,), is_leaf=True)  # arg561_1
    buf562 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf562, (128, 128), is_leaf=True)  # arg562_1
    buf563 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf563, (128,), is_leaf=True)  # arg563_1
    buf564 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf564, (128, 512), is_leaf=True)  # arg564_1
    buf565 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf565, (128,), is_leaf=True)  # arg565_1
    buf566 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf566, (128, 128), is_leaf=True)  # arg566_1
    buf567 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf567, (128,), is_leaf=True)  # arg567_1
    buf568 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf568, (128,), is_leaf=True)  # arg568_1
    buf569 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf569, (128,), is_leaf=True)  # arg569_1
    buf570 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf570, (512, 128), is_leaf=True)  # arg570_1
    buf571 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf571, (512,), is_leaf=True)  # arg571_1
    buf572 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf572, (128, 512), is_leaf=True)  # arg572_1
    buf573 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf573, (128,), is_leaf=True)  # arg573_1
    buf574 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf574, (128,), is_leaf=True)  # arg574_1
    buf575 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf575, (128,), is_leaf=True)  # arg575_1
    buf576 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf576, (512, 128), is_leaf=True)  # arg576_1
    buf577 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf577, (512,), is_leaf=True)  # arg577_1
    buf578 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf578, (512,), is_leaf=True)  # arg578_1
    buf579 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf579, (512,), is_leaf=True)  # arg579_1
    buf580 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf580, (128, 512), is_leaf=True)  # arg580_1
    buf581 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf581, (128,), is_leaf=True)  # arg581_1
    buf582 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf582, (128,), is_leaf=True)  # arg582_1
    buf583 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf583, (128,), is_leaf=True)  # arg583_1
    buf584 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf584, (128, 512), is_leaf=True)  # arg584_1
    buf585 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf585, (128,), is_leaf=True)  # arg585_1
    buf586 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf586, (128,), is_leaf=True)  # arg586_1
    buf587 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf587, (128,), is_leaf=True)  # arg587_1
    buf588 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf588, (512, 128), is_leaf=True)  # arg588_1
    buf589 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf589, (512,), is_leaf=True)  # arg589_1
    buf590 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf590, (128, 512), is_leaf=True)  # arg590_1
    buf591 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf591, (128,), is_leaf=True)  # arg591_1
    buf592 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf592, (128,), is_leaf=True)  # arg592_1
    buf593 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf593, (128,), is_leaf=True)  # arg593_1
    buf594 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf594, (512, 128), is_leaf=True)  # arg594_1
    buf595 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf595, (512,), is_leaf=True)  # arg595_1
    buf596 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf596, (128, 512), is_leaf=True)  # arg596_1
    buf597 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf597, (128,), is_leaf=True)  # arg597_1
    buf598 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf598, (128,), is_leaf=True)  # arg598_1
    buf599 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf599, (128,), is_leaf=True)  # arg599_1
    buf600 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf600, (512, 128), is_leaf=True)  # arg600_1
    buf601 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf601, (512,), is_leaf=True)  # arg601_1
    buf602 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf602, (128, 512), is_leaf=True)  # arg602_1
    buf603 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf603, (128,), is_leaf=True)  # arg603_1
    buf604 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf604, (128,), is_leaf=True)  # arg604_1
    buf605 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf605, (128,), is_leaf=True)  # arg605_1
    buf606 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf606, (128, 128), is_leaf=True)  # arg606_1
    buf607 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf607, (128,), is_leaf=True)  # arg607_1
    buf608 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf608, (128, 128), is_leaf=True)  # arg608_1
    buf609 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf609, (128,), is_leaf=True)  # arg609_1
    buf610 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf610, (128, 512), is_leaf=True)  # arg610_1
    buf611 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf611, (128,), is_leaf=True)  # arg611_1
    buf612 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf612, (128, 128), is_leaf=True)  # arg612_1
    buf613 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf613, (128,), is_leaf=True)  # arg613_1
    buf614 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf614, (128,), is_leaf=True)  # arg614_1
    buf615 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf615, (128,), is_leaf=True)  # arg615_1
    buf616 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf616, (512, 128), is_leaf=True)  # arg616_1
    buf617 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf617, (512,), is_leaf=True)  # arg617_1
    buf618 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf618, (128, 512), is_leaf=True)  # arg618_1
    buf619 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf619, (128,), is_leaf=True)  # arg619_1
    buf620 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf620, (128,), is_leaf=True)  # arg620_1
    buf621 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf621, (128,), is_leaf=True)  # arg621_1
    buf622 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf622, (512, 128), is_leaf=True)  # arg622_1
    buf623 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf623, (512,), is_leaf=True)  # arg623_1
    buf624 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf624, (512,), is_leaf=True)  # arg624_1
    buf625 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf625, (512,), is_leaf=True)  # arg625_1
    buf626 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf626, (128, 512), is_leaf=True)  # arg626_1
    buf627 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf627, (128,), is_leaf=True)  # arg627_1
    buf628 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf628, (128,), is_leaf=True)  # arg628_1
    buf629 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf629, (128,), is_leaf=True)  # arg629_1
    buf630 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf630, (128, 512), is_leaf=True)  # arg630_1
    buf631 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf631, (128,), is_leaf=True)  # arg631_1
    buf632 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf632, (128,), is_leaf=True)  # arg632_1
    buf633 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf633, (128,), is_leaf=True)  # arg633_1
    buf634 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf634, (512, 128), is_leaf=True)  # arg634_1
    buf635 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf635, (512,), is_leaf=True)  # arg635_1
    buf636 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf636, (128, 512), is_leaf=True)  # arg636_1
    buf637 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf637, (128,), is_leaf=True)  # arg637_1
    buf638 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf638, (128,), is_leaf=True)  # arg638_1
    buf639 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf639, (128,), is_leaf=True)  # arg639_1
    buf640 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf640, (512, 128), is_leaf=True)  # arg640_1
    buf641 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf641, (512,), is_leaf=True)  # arg641_1
    buf642 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf642, (128, 512), is_leaf=True)  # arg642_1
    buf643 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf643, (128,), is_leaf=True)  # arg643_1
    buf644 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf644, (128,), is_leaf=True)  # arg644_1
    buf645 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf645, (128,), is_leaf=True)  # arg645_1
    buf646 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf646, (512, 128), is_leaf=True)  # arg646_1
    buf647 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf647, (512,), is_leaf=True)  # arg647_1
    buf648 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf648, (128, 512), is_leaf=True)  # arg648_1
    buf649 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf649, (128,), is_leaf=True)  # arg649_1
    buf650 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf650, (128,), is_leaf=True)  # arg650_1
    buf651 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf651, (128,), is_leaf=True)  # arg651_1
    buf652 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf652, (128, 128), is_leaf=True)  # arg652_1
    buf653 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf653, (128,), is_leaf=True)  # arg653_1
    buf654 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf654, (128, 128), is_leaf=True)  # arg654_1
    buf655 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf655, (128,), is_leaf=True)  # arg655_1
    buf656 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf656, (128, 512), is_leaf=True)  # arg656_1
    buf657 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf657, (128,), is_leaf=True)  # arg657_1
    buf658 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf658, (128, 128), is_leaf=True)  # arg658_1
    buf659 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf659, (128,), is_leaf=True)  # arg659_1
    buf660 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf660, (128,), is_leaf=True)  # arg660_1
    buf661 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf661, (128,), is_leaf=True)  # arg661_1
    buf662 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf662, (512, 128), is_leaf=True)  # arg662_1
    buf663 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf663, (512,), is_leaf=True)  # arg663_1
    buf664 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf664, (128, 512), is_leaf=True)  # arg664_1
    buf665 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf665, (128,), is_leaf=True)  # arg665_1
    buf666 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf666, (128,), is_leaf=True)  # arg666_1
    buf667 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf667, (128,), is_leaf=True)  # arg667_1
    buf668 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf668, (512, 128), is_leaf=True)  # arg668_1
    buf669 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf669, (512,), is_leaf=True)  # arg669_1
    buf670 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf670, (512,), is_leaf=True)  # arg670_1
    buf671 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf671, (512,), is_leaf=True)  # arg671_1
    buf672 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf672, (128, 512), is_leaf=True)  # arg672_1
    buf673 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf673, (128,), is_leaf=True)  # arg673_1
    buf674 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf674, (128,), is_leaf=True)  # arg674_1
    buf675 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf675, (128,), is_leaf=True)  # arg675_1
    buf676 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf676, (128, 512), is_leaf=True)  # arg676_1
    buf677 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf677, (128,), is_leaf=True)  # arg677_1
    buf678 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf678, (128,), is_leaf=True)  # arg678_1
    buf679 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf679, (128,), is_leaf=True)  # arg679_1
    buf680 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf680, (512, 128), is_leaf=True)  # arg680_1
    buf681 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf681, (512,), is_leaf=True)  # arg681_1
    buf682 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf682, (128, 512), is_leaf=True)  # arg682_1
    buf683 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf683, (128,), is_leaf=True)  # arg683_1
    buf684 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf684, (128,), is_leaf=True)  # arg684_1
    buf685 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf685, (128,), is_leaf=True)  # arg685_1
    buf686 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf686, (512, 128), is_leaf=True)  # arg686_1
    buf687 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf687, (512,), is_leaf=True)  # arg687_1
    buf688 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf688, (128, 512), is_leaf=True)  # arg688_1
    buf689 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf689, (128,), is_leaf=True)  # arg689_1
    buf690 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf690, (128,), is_leaf=True)  # arg690_1
    buf691 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf691, (128,), is_leaf=True)  # arg691_1
    buf692 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf692, (512, 128), is_leaf=True)  # arg692_1
    buf693 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf693, (512,), is_leaf=True)  # arg693_1
    buf694 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf694, (128, 512), is_leaf=True)  # arg694_1
    buf695 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf695, (128,), is_leaf=True)  # arg695_1
    buf696 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf696, (128,), is_leaf=True)  # arg696_1
    buf697 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf697, (128,), is_leaf=True)  # arg697_1
    buf698 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf698, (128, 128), is_leaf=True)  # arg698_1
    buf699 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf699, (128,), is_leaf=True)  # arg699_1
    buf700 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf700, (128, 128), is_leaf=True)  # arg700_1
    buf701 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf701, (128,), is_leaf=True)  # arg701_1
    buf702 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf702, (128, 512), is_leaf=True)  # arg702_1
    buf703 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf703, (128,), is_leaf=True)  # arg703_1
    buf704 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf704, (128, 128), is_leaf=True)  # arg704_1
    buf705 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf705, (128,), is_leaf=True)  # arg705_1
    buf706 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf706, (128,), is_leaf=True)  # arg706_1
    buf707 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf707, (128,), is_leaf=True)  # arg707_1
    buf708 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf708, (512, 128), is_leaf=True)  # arg708_1
    buf709 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf709, (512,), is_leaf=True)  # arg709_1
    buf710 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf710, (128, 512), is_leaf=True)  # arg710_1
    buf711 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf711, (128,), is_leaf=True)  # arg711_1
    buf712 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf712, (128,), is_leaf=True)  # arg712_1
    buf713 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf713, (128,), is_leaf=True)  # arg713_1
    buf714 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf714, (512, 128), is_leaf=True)  # arg714_1
    buf715 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf715, (512,), is_leaf=True)  # arg715_1
    buf716 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf716, (512,), is_leaf=True)  # arg716_1
    buf717 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf717, (512,), is_leaf=True)  # arg717_1
    buf718 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf718, (128, 512), is_leaf=True)  # arg718_1
    buf719 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf719, (128,), is_leaf=True)  # arg719_1
    buf720 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf720, (128,), is_leaf=True)  # arg720_1
    buf721 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf721, (128,), is_leaf=True)  # arg721_1
    buf722 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf722, (128, 512), is_leaf=True)  # arg722_1
    buf723 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf723, (128,), is_leaf=True)  # arg723_1
    buf724 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf724, (128,), is_leaf=True)  # arg724_1
    buf725 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf725, (128,), is_leaf=True)  # arg725_1
    buf726 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf726, (512, 128), is_leaf=True)  # arg726_1
    buf727 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf727, (512,), is_leaf=True)  # arg727_1
    buf728 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf728, (128, 512), is_leaf=True)  # arg728_1
    buf729 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf729, (128,), is_leaf=True)  # arg729_1
    buf730 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf730, (128,), is_leaf=True)  # arg730_1
    buf731 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf731, (128,), is_leaf=True)  # arg731_1
    buf732 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf732, (512, 128), is_leaf=True)  # arg732_1
    buf733 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf733, (512,), is_leaf=True)  # arg733_1
    buf734 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf734, (128, 512), is_leaf=True)  # arg734_1
    buf735 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf735, (128,), is_leaf=True)  # arg735_1
    buf736 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf736, (128,), is_leaf=True)  # arg736_1
    buf737 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf737, (128,), is_leaf=True)  # arg737_1
    buf738 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf738, (512, 128), is_leaf=True)  # arg738_1
    buf739 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf739, (512,), is_leaf=True)  # arg739_1
    buf740 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf740, (128, 512), is_leaf=True)  # arg740_1
    buf741 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf741, (128,), is_leaf=True)  # arg741_1
    buf742 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf742, (128,), is_leaf=True)  # arg742_1
    buf743 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf743, (128,), is_leaf=True)  # arg743_1
    buf744 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf744, (128, 128), is_leaf=True)  # arg744_1
    buf745 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf745, (128,), is_leaf=True)  # arg745_1
    buf746 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf746, (128, 128), is_leaf=True)  # arg746_1
    buf747 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf747, (128,), is_leaf=True)  # arg747_1
    buf748 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf748, (128, 512), is_leaf=True)  # arg748_1
    buf749 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf749, (128,), is_leaf=True)  # arg749_1
    buf750 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf750, (128, 128), is_leaf=True)  # arg750_1
    buf751 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf751, (128,), is_leaf=True)  # arg751_1
    buf752 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf752, (128,), is_leaf=True)  # arg752_1
    buf753 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf753, (128,), is_leaf=True)  # arg753_1
    buf754 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf754, (512, 128), is_leaf=True)  # arg754_1
    buf755 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf755, (512,), is_leaf=True)  # arg755_1
    buf756 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf756, (128, 512), is_leaf=True)  # arg756_1
    buf757 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf757, (128,), is_leaf=True)  # arg757_1
    buf758 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf758, (128,), is_leaf=True)  # arg758_1
    buf759 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf759, (128,), is_leaf=True)  # arg759_1
    buf760 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf760, (512, 128), is_leaf=True)  # arg760_1
    buf761 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf761, (512,), is_leaf=True)  # arg761_1
    buf762 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf762, (512,), is_leaf=True)  # arg762_1
    buf763 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf763, (512,), is_leaf=True)  # arg763_1
    buf764 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf764, (128, 512), is_leaf=True)  # arg764_1
    buf765 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf765, (128,), is_leaf=True)  # arg765_1
    buf766 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf766, (128,), is_leaf=True)  # arg766_1
    buf767 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf767, (128,), is_leaf=True)  # arg767_1
    buf768 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf768, (128, 512), is_leaf=True)  # arg768_1
    buf769 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf769, (128,), is_leaf=True)  # arg769_1
    buf770 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf770, (128,), is_leaf=True)  # arg770_1
    buf771 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf771, (128,), is_leaf=True)  # arg771_1
    buf772 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf772, (512, 128), is_leaf=True)  # arg772_1
    buf773 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf773, (512,), is_leaf=True)  # arg773_1
    buf774 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf774, (128, 512), is_leaf=True)  # arg774_1
    buf775 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf775, (128,), is_leaf=True)  # arg775_1
    buf776 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf776, (128,), is_leaf=True)  # arg776_1
    buf777 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf777, (128,), is_leaf=True)  # arg777_1
    buf778 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf778, (512, 128), is_leaf=True)  # arg778_1
    buf779 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf779, (512,), is_leaf=True)  # arg779_1
    buf780 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf780, (128, 512), is_leaf=True)  # arg780_1
    buf781 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf781, (128,), is_leaf=True)  # arg781_1
    buf782 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf782, (128,), is_leaf=True)  # arg782_1
    buf783 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf783, (128,), is_leaf=True)  # arg783_1
    buf784 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf784, (512, 128), is_leaf=True)  # arg784_1
    buf785 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf785, (512,), is_leaf=True)  # arg785_1
    buf786 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf786, (128, 512), is_leaf=True)  # arg786_1
    buf787 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf787, (128,), is_leaf=True)  # arg787_1
    buf788 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf788, (128,), is_leaf=True)  # arg788_1
    buf789 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf789, (128,), is_leaf=True)  # arg789_1
    buf790 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf790, (128, 128), is_leaf=True)  # arg790_1
    buf791 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf791, (128,), is_leaf=True)  # arg791_1
    buf792 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf792, (128, 128), is_leaf=True)  # arg792_1
    buf793 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf793, (128,), is_leaf=True)  # arg793_1
    buf794 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf794, (128, 512), is_leaf=True)  # arg794_1
    buf795 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf795, (128,), is_leaf=True)  # arg795_1
    buf796 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf796, (128, 128), is_leaf=True)  # arg796_1
    buf797 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf797, (128,), is_leaf=True)  # arg797_1
    buf798 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf798, (128,), is_leaf=True)  # arg798_1
    buf799 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf799, (128,), is_leaf=True)  # arg799_1
    buf800 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf800, (512, 128), is_leaf=True)  # arg800_1
    buf801 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf801, (512,), is_leaf=True)  # arg801_1
    buf802 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf802, (128, 512), is_leaf=True)  # arg802_1
    buf803 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf803, (128,), is_leaf=True)  # arg803_1
    buf804 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf804, (128,), is_leaf=True)  # arg804_1
    buf805 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf805, (128,), is_leaf=True)  # arg805_1
    buf806 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf806, (512, 128), is_leaf=True)  # arg806_1
    buf807 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf807, (512,), is_leaf=True)  # arg807_1
    buf808 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf808, (512,), is_leaf=True)  # arg808_1
    buf809 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf809, (512,), is_leaf=True)  # arg809_1
    buf810 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf810, (128, 512), is_leaf=True)  # arg810_1
    buf811 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf811, (128,), is_leaf=True)  # arg811_1
    buf812 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf812, (128,), is_leaf=True)  # arg812_1
    buf813 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf813, (128,), is_leaf=True)  # arg813_1
    buf814 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf814, (128, 512), is_leaf=True)  # arg814_1
    buf815 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf815, (128,), is_leaf=True)  # arg815_1
    buf816 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf816, (128,), is_leaf=True)  # arg816_1
    buf817 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf817, (128,), is_leaf=True)  # arg817_1
    buf818 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf818, (512, 128), is_leaf=True)  # arg818_1
    buf819 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf819, (512,), is_leaf=True)  # arg819_1
    buf820 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf820, (128, 512), is_leaf=True)  # arg820_1
    buf821 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf821, (128,), is_leaf=True)  # arg821_1
    buf822 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf822, (128,), is_leaf=True)  # arg822_1
    buf823 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf823, (128,), is_leaf=True)  # arg823_1
    buf824 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf824, (512, 128), is_leaf=True)  # arg824_1
    buf825 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf825, (512,), is_leaf=True)  # arg825_1
    buf826 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf826, (128, 512), is_leaf=True)  # arg826_1
    buf827 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf827, (128,), is_leaf=True)  # arg827_1
    buf828 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf828, (128,), is_leaf=True)  # arg828_1
    buf829 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf829, (128,), is_leaf=True)  # arg829_1
    buf830 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf830, (512, 128), is_leaf=True)  # arg830_1
    buf831 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf831, (512,), is_leaf=True)  # arg831_1
    buf832 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf832, (128, 512), is_leaf=True)  # arg832_1
    buf833 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf833, (128,), is_leaf=True)  # arg833_1
    buf834 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf834, (128,), is_leaf=True)  # arg834_1
    buf835 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf835, (128,), is_leaf=True)  # arg835_1
    buf836 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf836, (128, 128), is_leaf=True)  # arg836_1
    buf837 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf837, (128,), is_leaf=True)  # arg837_1
    buf838 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf838, (128, 128), is_leaf=True)  # arg838_1
    buf839 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf839, (128,), is_leaf=True)  # arg839_1
    buf840 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf840, (128, 512), is_leaf=True)  # arg840_1
    buf841 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf841, (128,), is_leaf=True)  # arg841_1
    buf842 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf842, (128, 128), is_leaf=True)  # arg842_1
    buf843 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf843, (128,), is_leaf=True)  # arg843_1
    buf844 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf844, (128,), is_leaf=True)  # arg844_1
    buf845 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf845, (128,), is_leaf=True)  # arg845_1
    buf846 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf846, (512, 128), is_leaf=True)  # arg846_1
    buf847 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf847, (512,), is_leaf=True)  # arg847_1
    buf848 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf848, (128, 512), is_leaf=True)  # arg848_1
    buf849 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf849, (128,), is_leaf=True)  # arg849_1
    buf850 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf850, (128,), is_leaf=True)  # arg850_1
    buf851 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf851, (128,), is_leaf=True)  # arg851_1
    buf852 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf852, (512, 128), is_leaf=True)  # arg852_1
    buf853 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf853, (512,), is_leaf=True)  # arg853_1
    buf854 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf854, (512,), is_leaf=True)  # arg854_1
    buf855 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf855, (512,), is_leaf=True)  # arg855_1
    buf856 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf856, (128, 512), is_leaf=True)  # arg856_1
    buf857 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf857, (128,), is_leaf=True)  # arg857_1
    buf858 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf858, (128,), is_leaf=True)  # arg858_1
    buf859 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf859, (128,), is_leaf=True)  # arg859_1
    buf860 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf860, (128, 512), is_leaf=True)  # arg860_1
    buf861 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf861, (128,), is_leaf=True)  # arg861_1
    buf862 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf862, (128,), is_leaf=True)  # arg862_1
    buf863 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf863, (128,), is_leaf=True)  # arg863_1
    buf864 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf864, (512, 128), is_leaf=True)  # arg864_1
    buf865 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf865, (512,), is_leaf=True)  # arg865_1
    buf866 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf866, (128, 512), is_leaf=True)  # arg866_1
    buf867 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf867, (128,), is_leaf=True)  # arg867_1
    buf868 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf868, (128,), is_leaf=True)  # arg868_1
    buf869 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf869, (128,), is_leaf=True)  # arg869_1
    buf870 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf870, (512, 128), is_leaf=True)  # arg870_1
    buf871 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf871, (512,), is_leaf=True)  # arg871_1
    buf872 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf872, (128, 512), is_leaf=True)  # arg872_1
    buf873 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf873, (128,), is_leaf=True)  # arg873_1
    buf874 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf874, (128,), is_leaf=True)  # arg874_1
    buf875 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf875, (128,), is_leaf=True)  # arg875_1
    buf876 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf876, (512, 128), is_leaf=True)  # arg876_1
    buf877 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf877, (512,), is_leaf=True)  # arg877_1
    buf878 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf878, (128, 512), is_leaf=True)  # arg878_1
    buf879 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf879, (128,), is_leaf=True)  # arg879_1
    buf880 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf880, (128,), is_leaf=True)  # arg880_1
    buf881 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf881, (128,), is_leaf=True)  # arg881_1
    buf882 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf882, (128, 128), is_leaf=True)  # arg882_1
    buf883 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf883, (128,), is_leaf=True)  # arg883_1
    buf884 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf884, (128, 128), is_leaf=True)  # arg884_1
    buf885 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf885, (128,), is_leaf=True)  # arg885_1
    buf886 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf886, (128, 512), is_leaf=True)  # arg886_1
    buf887 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf887, (128,), is_leaf=True)  # arg887_1
    buf888 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf888, (128, 128), is_leaf=True)  # arg888_1
    buf889 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf889, (128,), is_leaf=True)  # arg889_1
    buf890 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf890, (128,), is_leaf=True)  # arg890_1
    buf891 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf891, (128,), is_leaf=True)  # arg891_1
    buf892 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf892, (512, 128), is_leaf=True)  # arg892_1
    buf893 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf893, (512,), is_leaf=True)  # arg893_1
    buf894 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf894, (128, 512), is_leaf=True)  # arg894_1
    buf895 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf895, (128,), is_leaf=True)  # arg895_1
    buf896 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf896, (128,), is_leaf=True)  # arg896_1
    buf897 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf897, (128,), is_leaf=True)  # arg897_1
    buf898 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf898, (512, 128), is_leaf=True)  # arg898_1
    buf899 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf899, (512,), is_leaf=True)  # arg899_1
    buf900 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf900, (512,), is_leaf=True)  # arg900_1
    buf901 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf901, (512,), is_leaf=True)  # arg901_1
    buf902 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf902, (128, 512), is_leaf=True)  # arg902_1
    buf903 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf903, (128,), is_leaf=True)  # arg903_1
    buf904 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf904, (128,), is_leaf=True)  # arg904_1
    buf905 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf905, (128,), is_leaf=True)  # arg905_1
    buf906 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf906, (128, 512), is_leaf=True)  # arg906_1
    buf907 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf907, (128,), is_leaf=True)  # arg907_1
    buf908 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf908, (128,), is_leaf=True)  # arg908_1
    buf909 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf909, (128,), is_leaf=True)  # arg909_1
    buf910 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf910, (512, 128), is_leaf=True)  # arg910_1
    buf911 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf911, (512,), is_leaf=True)  # arg911_1
    buf912 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf912, (128, 512), is_leaf=True)  # arg912_1
    buf913 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf913, (128,), is_leaf=True)  # arg913_1
    buf914 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf914, (128,), is_leaf=True)  # arg914_1
    buf915 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf915, (128,), is_leaf=True)  # arg915_1
    buf916 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf916, (512, 128), is_leaf=True)  # arg916_1
    buf917 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf917, (512,), is_leaf=True)  # arg917_1
    buf918 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf918, (128, 512), is_leaf=True)  # arg918_1
    buf919 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf919, (128,), is_leaf=True)  # arg919_1
    buf920 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf920, (128,), is_leaf=True)  # arg920_1
    buf921 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf921, (128,), is_leaf=True)  # arg921_1
    buf922 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf922, (512, 128), is_leaf=True)  # arg922_1
    buf923 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf923, (512,), is_leaf=True)  # arg923_1
    buf924 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf924, (128, 512), is_leaf=True)  # arg924_1
    buf925 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf925, (128,), is_leaf=True)  # arg925_1
    buf926 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf926, (128,), is_leaf=True)  # arg926_1
    buf927 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf927, (128,), is_leaf=True)  # arg927_1
    buf928 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf928, (128, 128), is_leaf=True)  # arg928_1
    buf929 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf929, (128,), is_leaf=True)  # arg929_1
    buf930 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf930, (128, 128), is_leaf=True)  # arg930_1
    buf931 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf931, (128,), is_leaf=True)  # arg931_1
    buf932 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf932, (128, 512), is_leaf=True)  # arg932_1
    buf933 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf933, (128,), is_leaf=True)  # arg933_1
    buf934 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf934, (128, 128), is_leaf=True)  # arg934_1
    buf935 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf935, (128,), is_leaf=True)  # arg935_1
    buf936 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf936, (128,), is_leaf=True)  # arg936_1
    buf937 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf937, (128,), is_leaf=True)  # arg937_1
    buf938 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf938, (512, 128), is_leaf=True)  # arg938_1
    buf939 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf939, (512,), is_leaf=True)  # arg939_1
    buf940 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf940, (128, 512), is_leaf=True)  # arg940_1
    buf941 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf941, (128,), is_leaf=True)  # arg941_1
    buf942 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf942, (128,), is_leaf=True)  # arg942_1
    buf943 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf943, (128,), is_leaf=True)  # arg943_1
    buf944 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf944, (512, 128), is_leaf=True)  # arg944_1
    buf945 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf945, (512,), is_leaf=True)  # arg945_1
    buf946 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf946, (512,), is_leaf=True)  # arg946_1
    buf947 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf947, (512,), is_leaf=True)  # arg947_1
    buf948 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf948, (128, 512), is_leaf=True)  # arg948_1
    buf949 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf949, (128,), is_leaf=True)  # arg949_1
    buf950 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf950, (128,), is_leaf=True)  # arg950_1
    buf951 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf951, (128,), is_leaf=True)  # arg951_1
    buf952 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf952, (128, 512), is_leaf=True)  # arg952_1
    buf953 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf953, (128,), is_leaf=True)  # arg953_1
    buf954 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf954, (128,), is_leaf=True)  # arg954_1
    buf955 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf955, (128,), is_leaf=True)  # arg955_1
    buf956 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf956, (512, 128), is_leaf=True)  # arg956_1
    buf957 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf957, (512,), is_leaf=True)  # arg957_1
    buf958 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf958, (128, 512), is_leaf=True)  # arg958_1
    buf959 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf959, (128,), is_leaf=True)  # arg959_1
    buf960 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf960, (128,), is_leaf=True)  # arg960_1
    buf961 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf961, (128,), is_leaf=True)  # arg961_1
    buf962 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf962, (512, 128), is_leaf=True)  # arg962_1
    buf963 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf963, (512,), is_leaf=True)  # arg963_1
    buf964 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf964, (128, 512), is_leaf=True)  # arg964_1
    buf965 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf965, (128,), is_leaf=True)  # arg965_1
    buf966 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf966, (128,), is_leaf=True)  # arg966_1
    buf967 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf967, (128,), is_leaf=True)  # arg967_1
    buf968 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf968, (512, 128), is_leaf=True)  # arg968_1
    buf969 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf969, (512,), is_leaf=True)  # arg969_1
    buf970 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf970, (128, 512), is_leaf=True)  # arg970_1
    buf971 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf971, (128,), is_leaf=True)  # arg971_1
    buf972 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf972, (128,), is_leaf=True)  # arg972_1
    buf973 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf973, (128,), is_leaf=True)  # arg973_1
    buf974 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf974, (128, 128), is_leaf=True)  # arg974_1
    buf975 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf975, (128,), is_leaf=True)  # arg975_1
    buf976 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf976, (128, 128), is_leaf=True)  # arg976_1
    buf977 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf977, (128,), is_leaf=True)  # arg977_1
    buf978 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf978, (128, 512), is_leaf=True)  # arg978_1
    buf979 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf979, (128,), is_leaf=True)  # arg979_1
    buf980 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf980, (128, 128), is_leaf=True)  # arg980_1
    buf981 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf981, (128,), is_leaf=True)  # arg981_1
    buf982 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf982, (128,), is_leaf=True)  # arg982_1
    buf983 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf983, (128,), is_leaf=True)  # arg983_1
    buf984 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf984, (512, 128), is_leaf=True)  # arg984_1
    buf985 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf985, (512,), is_leaf=True)  # arg985_1
    buf986 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf986, (128, 512), is_leaf=True)  # arg986_1
    buf987 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf987, (128,), is_leaf=True)  # arg987_1
    buf988 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf988, (128,), is_leaf=True)  # arg988_1
    buf989 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf989, (128,), is_leaf=True)  # arg989_1
    buf990 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf990, (512, 128), is_leaf=True)  # arg990_1
    buf991 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf991, (512,), is_leaf=True)  # arg991_1
    buf992 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf992, (512,), is_leaf=True)  # arg992_1
    buf993 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf993, (512,), is_leaf=True)  # arg993_1
    buf994 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf994, (128, 512), is_leaf=True)  # arg994_1
    buf995 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf995, (128,), is_leaf=True)  # arg995_1
    buf996 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf996, (128,), is_leaf=True)  # arg996_1
    buf997 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf997, (128,), is_leaf=True)  # arg997_1
    buf998 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf998, (128, 512), is_leaf=True)  # arg998_1
    buf999 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf999, (128,), is_leaf=True)  # arg999_1
    buf1000 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1000, (128,), is_leaf=True)  # arg1000_1
    buf1001 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1001, (128,), is_leaf=True)  # arg1001_1
    buf1002 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1002, (512, 128), is_leaf=True)  # arg1002_1
    buf1003 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1003, (512,), is_leaf=True)  # arg1003_1
    buf1004 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1004, (128, 512), is_leaf=True)  # arg1004_1
    buf1005 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1005, (128,), is_leaf=True)  # arg1005_1
    buf1006 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1006, (128,), is_leaf=True)  # arg1006_1
    buf1007 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1007, (128,), is_leaf=True)  # arg1007_1
    buf1008 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1008, (512, 128), is_leaf=True)  # arg1008_1
    buf1009 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1009, (512,), is_leaf=True)  # arg1009_1
    buf1010 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1010, (128, 512), is_leaf=True)  # arg1010_1
    buf1011 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1011, (128,), is_leaf=True)  # arg1011_1
    buf1012 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1012, (128,), is_leaf=True)  # arg1012_1
    buf1013 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1013, (128,), is_leaf=True)  # arg1013_1
    buf1014 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1014, (512, 128), is_leaf=True)  # arg1014_1
    buf1015 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1015, (512,), is_leaf=True)  # arg1015_1
    buf1016 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1016, (128, 512), is_leaf=True)  # arg1016_1
    buf1017 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1017, (128,), is_leaf=True)  # arg1017_1
    buf1018 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1018, (128,), is_leaf=True)  # arg1018_1
    buf1019 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1019, (128,), is_leaf=True)  # arg1019_1
    buf1020 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1020, (128, 128), is_leaf=True)  # arg1020_1
    buf1021 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1021, (128,), is_leaf=True)  # arg1021_1
    buf1022 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1022, (128, 128), is_leaf=True)  # arg1022_1
    buf1023 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1023, (128,), is_leaf=True)  # arg1023_1
    buf1024 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1024, (128, 512), is_leaf=True)  # arg1024_1
    buf1025 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1025, (128,), is_leaf=True)  # arg1025_1
    buf1026 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1026, (128, 128), is_leaf=True)  # arg1026_1
    buf1027 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1027, (128,), is_leaf=True)  # arg1027_1
    buf1028 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1028, (128,), is_leaf=True)  # arg1028_1
    buf1029 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1029, (128,), is_leaf=True)  # arg1029_1
    buf1030 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1030, (512, 128), is_leaf=True)  # arg1030_1
    buf1031 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1031, (512,), is_leaf=True)  # arg1031_1
    buf1032 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1032, (128, 512), is_leaf=True)  # arg1032_1
    buf1033 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1033, (128,), is_leaf=True)  # arg1033_1
    buf1034 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1034, (128,), is_leaf=True)  # arg1034_1
    buf1035 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1035, (128,), is_leaf=True)  # arg1035_1
    buf1036 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1036, (512, 128), is_leaf=True)  # arg1036_1
    buf1037 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1037, (512,), is_leaf=True)  # arg1037_1
    buf1038 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1038, (512,), is_leaf=True)  # arg1038_1
    buf1039 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1039, (512,), is_leaf=True)  # arg1039_1
    buf1040 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1040, (128, 512), is_leaf=True)  # arg1040_1
    buf1041 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1041, (128,), is_leaf=True)  # arg1041_1
    buf1042 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1042, (128,), is_leaf=True)  # arg1042_1
    buf1043 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1043, (128,), is_leaf=True)  # arg1043_1
    buf1044 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1044, (128, 512), is_leaf=True)  # arg1044_1
    buf1045 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1045, (128,), is_leaf=True)  # arg1045_1
    buf1046 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1046, (128,), is_leaf=True)  # arg1046_1
    buf1047 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1047, (128,), is_leaf=True)  # arg1047_1
    buf1048 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1048, (512, 128), is_leaf=True)  # arg1048_1
    buf1049 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1049, (512,), is_leaf=True)  # arg1049_1
    buf1050 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1050, (128, 512), is_leaf=True)  # arg1050_1
    buf1051 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1051, (128,), is_leaf=True)  # arg1051_1
    buf1052 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1052, (128,), is_leaf=True)  # arg1052_1
    buf1053 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1053, (128,), is_leaf=True)  # arg1053_1
    buf1054 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1054, (512, 128), is_leaf=True)  # arg1054_1
    buf1055 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1055, (512,), is_leaf=True)  # arg1055_1
    buf1056 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1056, (128, 512), is_leaf=True)  # arg1056_1
    buf1057 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1057, (128,), is_leaf=True)  # arg1057_1
    buf1058 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1058, (128,), is_leaf=True)  # arg1058_1
    buf1059 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1059, (128,), is_leaf=True)  # arg1059_1
    buf1060 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1060, (512, 128), is_leaf=True)  # arg1060_1
    buf1061 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1061, (512,), is_leaf=True)  # arg1061_1
    buf1062 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1062, (128, 512), is_leaf=True)  # arg1062_1
    buf1063 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1063, (128,), is_leaf=True)  # arg1063_1
    buf1064 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1064, (128,), is_leaf=True)  # arg1064_1
    buf1065 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1065, (128,), is_leaf=True)  # arg1065_1
    buf1066 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1066, (128, 128), is_leaf=True)  # arg1066_1
    buf1067 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1067, (128,), is_leaf=True)  # arg1067_1
    buf1068 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1068, (128, 128), is_leaf=True)  # arg1068_1
    buf1069 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1069, (128,), is_leaf=True)  # arg1069_1
    buf1070 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1070, (128, 512), is_leaf=True)  # arg1070_1
    buf1071 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1071, (128,), is_leaf=True)  # arg1071_1
    buf1072 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1072, (128, 128), is_leaf=True)  # arg1072_1
    buf1073 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1073, (128,), is_leaf=True)  # arg1073_1
    buf1074 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1074, (128,), is_leaf=True)  # arg1074_1
    buf1075 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1075, (128,), is_leaf=True)  # arg1075_1
    buf1076 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1076, (512, 128), is_leaf=True)  # arg1076_1
    buf1077 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1077, (512,), is_leaf=True)  # arg1077_1
    buf1078 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1078, (128, 512), is_leaf=True)  # arg1078_1
    buf1079 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1079, (128,), is_leaf=True)  # arg1079_1
    buf1080 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1080, (128,), is_leaf=True)  # arg1080_1
    buf1081 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1081, (128,), is_leaf=True)  # arg1081_1
    buf1082 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1082, (512, 128), is_leaf=True)  # arg1082_1
    buf1083 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1083, (512,), is_leaf=True)  # arg1083_1
    buf1084 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1084, (512,), is_leaf=True)  # arg1084_1
    buf1085 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1085, (512,), is_leaf=True)  # arg1085_1
    buf1086 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1086, (128, 512), is_leaf=True)  # arg1086_1
    buf1087 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1087, (128,), is_leaf=True)  # arg1087_1
    buf1088 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1088, (128,), is_leaf=True)  # arg1088_1
    buf1089 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1089, (128,), is_leaf=True)  # arg1089_1
    buf1090 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1090, (128, 512), is_leaf=True)  # arg1090_1
    buf1091 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1091, (128,), is_leaf=True)  # arg1091_1
    buf1092 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1092, (128,), is_leaf=True)  # arg1092_1
    buf1093 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1093, (128,), is_leaf=True)  # arg1093_1
    buf1094 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1094, (512, 128), is_leaf=True)  # arg1094_1
    buf1095 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1095, (512,), is_leaf=True)  # arg1095_1
    buf1096 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1096, (128, 512), is_leaf=True)  # arg1096_1
    buf1097 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1097, (128,), is_leaf=True)  # arg1097_1
    buf1098 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1098, (128,), is_leaf=True)  # arg1098_1
    buf1099 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1099, (128,), is_leaf=True)  # arg1099_1
    buf1100 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1100, (512, 128), is_leaf=True)  # arg1100_1
    buf1101 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1101, (512,), is_leaf=True)  # arg1101_1
    buf1102 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1102, (128, 512), is_leaf=True)  # arg1102_1
    buf1103 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1103, (128,), is_leaf=True)  # arg1103_1
    buf1104 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1104, (128,), is_leaf=True)  # arg1104_1
    buf1105 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1105, (128,), is_leaf=True)  # arg1105_1
    buf1106 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1106, (512, 128), is_leaf=True)  # arg1106_1
    buf1107 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1107, (512,), is_leaf=True)  # arg1107_1
    buf1108 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1108, (128, 512), is_leaf=True)  # arg1108_1
    buf1109 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1109, (128,), is_leaf=True)  # arg1109_1
    buf1110 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1110, (128,), is_leaf=True)  # arg1110_1
    buf1111 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1111, (128,), is_leaf=True)  # arg1111_1
    buf1112 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1112, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1112_1
    buf1113 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1113, (2, 512), is_leaf=True)  # arg1113_1
    buf1114 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf1114, (2,), is_leaf=True)  # arg1114_1
    buf1115 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1115, (128,), dtype=torch.int64, is_leaf=True)  # arg1115_1
    buf1116 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1116, (128,), dtype=torch.int64, is_leaf=True)  # arg1116_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)